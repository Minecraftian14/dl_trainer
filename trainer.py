import os
from datetime import datetime
from typing import Any

import numpy as np
import json
import torch
from torch import nn
from torch.utils import data
from torch.utils._pytree import tree_map
from torch.utils.data.dataloader import default_collate

from .bench import TypedTimer


def recursive_collate(batch: list[tuple[Any, Any]], collate=default_collate):
    model_data = collate([x[0] for x in batch])
    loss_data = collate([x[1] for x in batch])
    return model_data, loss_data


def split_collate(batch: list[tuple[tuple[Any], tuple[Any]]], collate=default_collate):
    model_data = [collate(x[0]) for x in batch]
    loss_data = [collate(x[1]) for x in batch]
    return model_data, loss_data


def _ada_pad(item: np.ndarray, length):
    if item.ndim == 0 or len(item) == 1: return item
    pad_map = [(0, length - len(item))]
    if item.ndim > 1: pad_map.extend([(0, 0)] * (item.ndim - 1))
    return np.pad(item, pad_map)


def sequence_collate(batch: list[tuple[tuple[Any], tuple[Any]]], collate=default_collate):
    model_data, loss_data = zip(*batch)
    # print([x.shape for x in model_data[0]])
    first_seq_idx = [idx for idx, data in enumerate(model_data[0]) if data.ndim > 0 and len(data) != 1][0]
    max_len = max(len(sample[first_seq_idx]) for sample in model_data)
    lens = collate(np.asarray([len(sample[first_seq_idx]) for sample in model_data]))
    mask = collate(np.asarray([[i < len(sample[first_seq_idx]) for i in range(max_len)] for sample in model_data]))
    model_data = tuple(zip(*(tuple(_ada_pad(data, max_len) for data in sample) for sample in model_data)))
    model_data = tuple(collate(np.stack(data)) for data in model_data)
    # print([x.shape for x in model_data])
    loss_data = tuple(zip(*(tuple(data for data in sample) for sample in loss_data)))
    loss_data = tuple(collate(np.stack(data)) for data in loss_data)
    return (*model_data, mask, lens), loss_data


def create_sequence_collator(collation_map, collate=default_collate):
    def sequence_collate(batch: list[tuple[tuple[Any], tuple[Any]]]):
        model_data, loss_data = zip(*batch)

        non_seq_data = [tuple(data for is_seq, data in zip(collation_map, sample) if not is_seq) for sample in model_data]
        seq_data = [tuple(data for is_seq, data in zip(collation_map, sample) if is_seq) for sample in model_data]

        non_seq_data = tuple(zip(*(tuple(data for data in sample) for sample in non_seq_data)))
        non_seq_data = tuple(collate(np.stack(data)) for data in non_seq_data)

        max_len = max(len(data) for sample in seq_data for data in sample)
        mask = collate(np.asarray([[i < len(sample[0]) for i in range(max_len)] for sample in seq_data]))
        lens = collate(np.asarray([len(sample[0]) for sample in seq_data]))
        seq_data = tuple(zip(*(tuple(_ada_pad(data, max_len) for data in sample) for sample in seq_data)))
        seq_data = tuple(collate(np.stack(data)) for data in seq_data)

        loss_data = tuple(zip(*(tuple(data for data in sample) for sample in loss_data)))
        loss_data = tuple(collate(np.stack(data)) for data in loss_data)

        return (*non_seq_data, *seq_data, mask, lens), loss_data

    return sequence_collate


class Trainer:
    def __init__(
            self,

            # The model to be trained
            model: nn.Module,

            # The dataloader to be used for training
            train_dataloader: data.DataLoader,

            # The loss function and optimizer to be used for training
            criterion=nn.CrossEntropyLoss(ignore_index=0),
            regularization=None,
            optimizer=torch.optim.Adam,

            # How many times the dataset is trained through
            epochs: int = 5,
            # Override the epochs param and only use a fraction of the whole dataset
            dataset_fraction: float = None,

            # Optionally provide a validation dataloader
            val_dataloader: data.DataLoader = None,

            # How many steps between each saved checkpoint
            checkpoint_frequency: int = None,

            lr_scheduler=None,
            device='cpu',

            model_dir=None,
            model_name=None,

            model_outputs_adaptor=lambda x: x,
            model_train_step=lambda model, data: model(*data),

            record_per_epoch_training_loss=False,
            record_per_batch_training_loss=False,

            loss=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.regularization = regularization
        self.optimizer = optimizer(model.parameters())
        self.epochs = epochs
        self.dataset_fraction = dataset_fraction
        self.val_dataloader = val_dataloader
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir if model_dir else os.path.join('temp', 'checkpoints')
        os.path.exists(self.model_dir) or os.makedirs(self.model_dir)
        self.model_name = model_name if model_name else model.__class__.__name__
        self.model_outputs_adaptor = model_outputs_adaptor
        self.model_train_step = model_train_step
        self.record_per_epoch_training_loss = record_per_epoch_training_loss
        self.record_per_batch_training_loss = record_per_batch_training_loss

        self.loss = loss or {"train": [], "val": [], "epoch.train": [], "batch.train": []}
        self.model.to(self.device)

        self.timer = TypedTimer(self.model_name)

        try:
            # self.dataset_length = len(train_dataloader.dataset) // train_dataloader.batch_size
            self.dataset_length = len(train_dataloader)
            if dataset_fraction is not None:
                self.dataset_length = min(self.dataset_length, dataset_fraction)
        except:
            self.dataset_length = None

    def to(self, device):
        self.device = device
        self.model.to(device)

    def learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.timer.start("train")

        for epoch in range(1, 1 + self.epochs):
            self._train_step(epoch)
            self._validate_step()
            self._log_step(epoch, train_loss=self.loss["train"][-1], val_loss=self.loss["val"][-1] if len(self.loss["val"]) > 0 else None, tts=self.timer.since("train"))

            if self.lr_scheduler: self.lr_scheduler.step()

            if self.checkpoint_frequency and epoch % self.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)

        if not self.checkpoint_frequency or self.epochs % self.checkpoint_frequency != 0:
            self._save_checkpoint(self.epochs)

        self.timer.end("train")

    def _train_step(self, epoch=None):
        self.timer.start("_train_step")

        self.model.train()
        running_loss = []

        self.timer.start("train_dataloader")
        i = 1
        for batch_data in self.train_dataloader:
            self.timer.end("train_dataloader")
            self.timer.start("batch")
            batch_data = tree_map(lambda x: x.to(self.device) if torch.is_tensor(x) else x, batch_data)
            self.optimizer.zero_grad()
            predictions = self.model_train_step(self.model, batch_data[0])
            loss = self.criterion(predictions, batch_data[1])
            running_loss.append(loss.item())
            regularization = None
            if self.regularization:
                regularization = self.regularization(predictions)
                loss += regularization
                regularization = regularization.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.alpha.parameters(), max_norm=1.0)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.norm():.2e}")
                else: print(f"{name}: None")
            self.optimizer.step()
            # Grab one parameter to monitor
            # param = list(self.model.history_model.parameters())[0]
            # print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Weight sample: {param.data[0][0]:.6f}")
            # if self.record_per_batch_training_loss: self.loss["batch.train"].append(running_loss[-1])
            self.timer.end("batch")

            if self.dataset_fraction and i > self.dataset_fraction: break

            # if self.dataset_fraction and self.timer.drag("_train_step", 60):
            #     self._validate_step()
            #     self._log_step(epoch - 1 + i / self.dataset_fraction, running_loss[-1], tts=self.timer.since("train"))
            #     self.model.train()

            if self.timer.drag("_train_step", 1):
                # epoch_data = epoch - 1 if self.dataset_fraction is None else epoch - 1 + i / self.dataset_fraction
                epoch_data = epoch - 1 if self.dataset_length is None else epoch - 1 + i / self.dataset_length
                # if self.dataset_length is None: self._log_step(epoch - 1, running_loss[-1], tts=self.timer.since("train"))
                # else: self._log_step(epoch - 1 + i / self.dataset_length, running_loss[-1], tts=self.timer.since("train"))
                self._log_step(epoch_data, running_loss[-1], tts=self.timer.since("train"), dataset_fraction=i, regularization=regularization)

            self.timer.start("train_dataloader")
            i += 1

        epoch_loss = np.mean(running_loss)
        # Redundant, because "train" is doing the same thing
        if self.record_per_epoch_training_loss: self.loss["epoch.train"].append(epoch_loss)
        self.loss["train"].append(epoch_loss)
        self.timer.end("_train_step")

    def _validate_step(self):
        if not self.val_dataloader: return
        self.timer.start("_validate_step")

        self.model.eval()
        running_loss = []

        with torch.no_grad():
            self.timer.start("val_dataloader")
            for batch_data in self.val_dataloader:
                self.timer.end("val_dataloader")
                self.timer.start("val_batch")
                batch_data = tree_map(lambda x: x.to(self.device) if torch.is_tensor(x) else x, batch_data)
                predictions = self.model_train_step(self.model, batch_data[0])
                loss = self.criterion(predictions, batch_data[1])
                running_loss.append(loss.item())
                self.timer.end("val_batch")

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
        self.timer.end("_validate_step")

    def _log_step(self, epoch=None, train_loss=None, val_loss=None, tts=None, dataset_fraction=None, regularization=None):
        messages = []

        if dataset_fraction is not None and self.dataset_fraction is not None:
            messages.append(f"Fraction: {dataset_fraction:2d}/{self.dataset_fraction:2d}")

        if epoch is not None:
            messages.append(f"Epoch: {int(epoch):2d}/{self.epochs:2d}")

        if train_loss is None:
            if len(self.loss["train"]) > 0: train_loss = self.loss["train"][-1]

        if train_loss is not None:
            messages.append(f"Train Loss: {train_loss:.2f}")

        if regularization is not None:
            messages.append(f"Regularization: {regularization:.2f}")

        if val_loss is None:
            if len(self.loss["val"]) > 0: val_loss = self.loss["val"][-1]

        if val_loss is not None:
            messages.append(f"Val Loss: {val_loss:.2f}")

        if tts is not None:
            messages.append(f"TTS: {tts:.2f}")
            if epoch is not None:
                tpe = tts / epoch
                messages.append(f"ETA: {(self.epochs - epoch) * tpe:.2f}")

        print("    ".join(messages))

    def _save_checkpoint(self, epoch, running_loss=None):
        if isinstance(epoch, int): epoch = f"{epoch:03d}"
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss = running_loss or (self.loss["train"][-1] if len(self.loss["train"]) > 0 else None)
        model_path = f"checkpoint_{self.model_name}_{epoch}_{date_str}_{loss}.pt"
        model_path = os.path.join(self.model_dir, model_path)
        torch.save(self.model.state_dict(), model_path)

    def save_model(self, name=None):
        if name is None: name = self.model_name
        model_path = os.path.join(self.model_dir, f"model_{name}.pt")
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, name=None):
        if name is None: name = f"model_{self.model_name}.pt"
        model_path = os.path.join(self.model_dir, name)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def save_loss(self, name=None):
        if name is None: name = self.model_name
        loss_path = os.path.join(self.model_dir, f"loss_{name}.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)

    def load_checkpoint(self, name=None, save_backup=True):
        if save_backup: self._save_checkpoint("load_backup")
        if name is not None:
            self.load_model(name)
            return
        available_choices = [(x, os.path.join(self.model_dir, x)) for x in os.listdir(self.model_dir) if x.startswith("checkpoint_") and x.endswith(".pt")]
        if len(available_choices) == 0:
            print("No checkpoints found")
            return
        available_choices.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        print("Pick an option")
        if len(available_choices) > 3: print("  0> View All")
        if len(available_choices) > 0: print("  1> 1st Last Model", available_choices[0][0])
        if len(available_choices) > 1: print("  2> 2st Last Model", available_choices[1][0])
        if len(available_choices) > 2: print("  3> 3st Last Model", available_choices[2][0])
        choice = int(input("Choice: "))
        if choice == 0:
            for i, (name, _) in enumerate(available_choices, 1):
                print(f"{i:03d} {name}")
            choice = int(input("Choice: "))
        self.load_model(available_choices[choice - 1][0])

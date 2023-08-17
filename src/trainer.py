import os
from typing import Dict, Union

import torch
from torch import Tensor

from models.losses import CombinedMarginLoss
from models.networks import MMSERA, SERVER, AudioOnly, TextOnly
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self, network: Union[MMSERA, AudioOnly, TextOnly, SERVER], criterion: torch.nn.CrossEntropyLoss = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        # Forward pass
        output = self.network(input_ids, audio)
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, audio)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}


class MarginTrainer(TorchTrainer):
    def __init__(self, network: Union[MMSERA, AudioOnly, TextOnly, SERVER], criterion: CombinedMarginLoss, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(criterion, CombinedMarginLoss), "Criterion must be CombinedMarginLoss"
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.opt_criterion = torch.optim.SGD(
            params=[{"params": self.criterion.parameters()}],
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )

        self.scheduler_criterion = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_criterion,
            lr_lambda=lambda epoch: (
                ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
            ),
        )

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()
        self.opt_criterion.zero_grad()

        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        # Forward pass
        output = self.network(input_ids, audio)
        loss, logits = self.criterion(output[1], label)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.opt_criterion.step()

        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, audio)
            loss, logits = self.criterion(output[1], label)
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

    def lr_scheduler(self, step: int, epoch: int):
        if self.scheduler is not None:
            self.scheduler.step()
        self.scheduler_criterion.step()

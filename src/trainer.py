from typing import Dict, Union

import torch
from torch import Tensor
from transformers import BertTokenizer

from models.networks import MMSERA, MMSERALayerNorm
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self, network: Union[MMSERA, MMSERALayerNorm], tokenizer: BertTokenizer, criterion: torch.nn.CrossEntropyLoss, **kwargs
    ):
        super().__init__(**kwargs)
        self.network = network
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        text, label, sprectrome = batch["text"], batch["label"], batch["sprectrome"]
        label = torch.tensor([int(label)])
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        # Move inputs to cpu or gpu
        sprectrome = sprectrome.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        # Forward pass
        output = self.network(input_ids, sprectrome)
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output, 1)
        accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        text, label, sprectrome = batch["text"], batch["label"], batch["sprectrome"]
        label = torch.tensor([int(label)])
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        # Move inputs to cpu or gpu
        sprectrome = sprectrome.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, sprectrome)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output, 1)
            accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

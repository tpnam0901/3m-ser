from typing import Dict, Union

import librosa
import torch
from torch import Tensor
from transformers import BertTokenizer

from models.networks import MMSERA, MMSERALayerNorm
from utils.torch.trainer import TorchTrainer


def move_data_to_device(x, device):
    if "float" in str(x.dtype):
        x = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


class Trainer(TorchTrainer):
    def __init__(
        self,
        network: Union[MMSERA, MMSERALayerNorm],
        tokenizer: BertTokenizer,
        criterion: torch.nn.CrossEntropyLoss,
        use_waveform=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.network = network
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.use_waveform = use_waveform

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        text, label, audio = batch["text"], batch["label"], batch["sprectrome"]
        if self.use_waveform:
            # Load audio
            (waveform, _) = librosa.core.load(batch["fileName"], sr=32000, mono=True)
            waveform = waveform[None, :]  # (1, audio_length)
            waveform = move_data_to_device(waveform, self.device)
            audio = waveform

        label = torch.tensor([int(label)])
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

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
        _, preds = torch.max(output, 1)
        accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        text, label, audio = batch["text"], batch["label"], batch["sprectrome"]
        label = torch.tensor([int(label)])
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        if self.use_waveform:
            # Load audio
            (waveform, _) = librosa.core.load(batch["fileName"], sr=32000, mono=True)
            waveform = waveform[None, :]  # (1, audio_length)
            waveform = move_data_to_device(waveform, self.device)
            audio = waveform

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, audio)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output, 1)
            accuracy = torch.mean((preds == label).float())
        return {"loss": loss.detach().cpu().item(), "acc": accuracy.detach().cpu().item()}

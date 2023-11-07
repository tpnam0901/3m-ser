import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from torchvggish.vggish_input import waveform_to_examples


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, opt: str):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(opt.checkpoint_dir), exist_ok=True)
        out_opt = os.path.join(opt.checkpoint_dir, "opt.log")
        with open(out_opt, "w") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logging.info(message)

    def load(self, opt_path: str):
        def decode_value(value: str):
            value = value.strip()
            if "." in value and value.replace(".", "").isdigit():
                value = float(value)
            elif value.isdigit():
                value = int(value)
            elif value == "True":
                value = True
            elif value == "False":
                value = False
            elif value == "None":
                value = None
            elif (
                value.startswith("'")
                and value.endswith("'")
                or value.startswith('"')
                and value.endswith('"')
            ):
                value = value[1:-1]
            return value

        with open(opt_path, "r") as f:
            data = f.read().split("\n")
            # remove all empty strings
            data = list(filter(None, data))
            # convert to dict
            data_dict = {}
            for i in range(len(data)):
                key, value = (
                    data[i].split(":")[0].strip(),
                    data[i].split(":")[1].strip(),
                )
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    value = [decode_value(x) for x in value]
                else:
                    value = decode_value(value)

                data_dict[key] = value
        for key, value in data_dict.items():
            setattr(self, key, value)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # Training settings
        self.num_epochs: int = 250
        self.checkpoint_dir: str = "checkpoints"
        self.save_all_states: bool = True
        self.save_best_val: bool = True
        self.max_to_keep: int = 1
        self.save_freq: int = 4000
        self.batch_size: int = 1

        # Resume training
        self.resume: bool = False
        # path to checkpoint.pt file, only available when using save_all_states = True in previous training
        self.resume_path: str = None
        self.opt_path: str = None
        if self.resume:
            assert os.path.exists(self.resume_path), "Resume path not found"

        # [CrossEntropyLoss, CrossEntropyLoss_ContrastiveCenterLoss, CrossEntropyLoss_CenterLoss,
        #  CombinedMarginLoss, FocalLoss,CenterLossSER,ContrastiveCenterLossSER]
        self.loss_type: str = "CrossEntropyLoss"

        # For CrossEntropyLoss_ContrastiveCenterLoss
        self.lambda_c: float = 1.0
        self.feat_dim: int = 768

        # For combined margin loss
        self.margin_loss_m1: float = 1.0
        self.margin_loss_m2: float = 0.5
        self.margin_loss_m3: float = 0.0
        self.margin_loss_scale: float = 64.0

        # For focal loss
        self.focal_loss_gamma: float = 0.5
        self.focal_loss_alpha: float = None

        # Learning rate
        self.learning_rate: float = 0.0001
        self.learning_rate_step_size: int = 30
        self.learning_rate_gamma: float = 0.1

        # Dataset
        self.data_name: str = "IEMOCAP"  # [IEMOCAP, ESD, MELD, IEMOCAPAudio]
        self.data_root: str = "data/IEMOCAP"  # folder contains train.pkl and test.pkl
        # use for training with batch size > 1
        self.text_max_length: int = 297
        self.audio_max_length: int = 546220

        # Model
        self.num_classes: int = 4
        self.num_attention_head: int = 8
        self.dropout: float = 0.5
        self.model_type: str = "MMSERA"  # [MMSERA, AudioOnly, TextOnly, SERVER]
        self.text_encoder_type: str = "bert"  # [bert, roberta]
        self.text_encoder_dim: int = 768
        self.text_unfreeze: bool = False
        self.audio_encoder_type: str = (
            "panns"  # [vggish, panns, hubert_base, wav2vec2_base, wavlm_base, lstm]
        )
        self.audio_encoder_dim: int = 2048  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base,wavlm_base, 512 - lstm
        self.audio_norm_type: str = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze: bool = True

        self.fusion_head_output_type: str = "cls"  # [cls, mean, max]

        # For LSTM
        self.lstm_hidden_size = 512  # should be the same as audio_encoder_dim
        self.lstm_num_layers = 2

        # For hyperparameter search
        self.optim_attributes: List = None
        # Example of hyperparameter search for lambda_c.
        # self.lambda_c = [x / 10 for x in range(5, 21, 5)]
        # self.optim_attributes = ["lambda_c"]

        # Search for linear layer output dimension
        self.linear_layer_output: List = [256, 128]
        self.linear_layer_last_dim: int = 64

        for key, value in kwargs.items():
            setattr(self, key, value)


class LSTM_Mel(nn.Module):
    def __init__(
        self, feature_module, input_size=512, hidden_size=512, num_layers=2, **kwargs
    ):
        super(LSTM_Mel, self).__init__(**kwargs)
        self.feature_module = feature_module
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, audio):
        out = []
        for i in range(audio.size(0)):
            out_dim1 = []
            for j in range(audio.size(1)):
                x = self.feature_module(audio[i, j : j + 1, :])
                x = torch.transpose(x, 1, 3)
                x = x.reshape(x.size(0), -1, x.size(3))
                x = x.mean(dim=1)
                x = x.squeeze(0)
                out_dim1.append(x)
            out.append(torch.stack(out_dim1, axis=0))
        out = torch.stack(out, axis=0)

        x, _ = self.lstm(out)
        # take only the last output
        x = x[:, -1, :]
        return x


def build_lstm_mel_encoder(opt: Config) -> nn.Module:
    weights = "vggish_feature_extractor.pth"
    url = "https://github.com/namphuongtran9196/GitReleaseStorage/releases/download/wav2vec_base/feature_extractor_wav2vec_base.pth"

    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    if not os.path.exists(os.path.join("/tmp/{}".format(weights))):
        os.system("wget {} -O /tmp/{}".format(url, weights))
    feature_extractor = nn.Sequential(*layers)
    feature_extractor.to("cpu")
    state_dict = torch.load(os.path.join("/tmp/{}".format(weights)), map_location="cpu")
    feature_extractor.load_state_dict(state_dict)

    model = LSTM_Mel(
        feature_extractor,
        input_size=512,
        hidden_size=opt.lstm_hidden_size,
        num_layers=opt.lstm_num_layers,
    )

    return model


def build_audio_encoder(opt: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        opt (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = opt.audio_encoder_type

    encoders = {
        "lstm_mel": build_lstm_mel_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](opt)


class AudioOnly_v2(nn.Module):
    def __init__(
        self,
        opt: Config,
        device: str = "cpu",
    ):
        """Speech Emotion Recognition with Audio Only

        Args:
            opt (Config): Config object
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(AudioOnly_v2, self).__init__()

        # Audio module
        self.audio_encoder = build_audio_encoder(opt)
        self.audio_encoder.to(device)
        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = opt.audio_unfreeze

        self.dropout = nn.Dropout(opt.dropout)

        # self.linear = nn.Linear(opt.audio_encoder_dim, opt.audio_encoder_dim)
        # self.classifer = nn.Linear(opt.audio_encoder_dim, opt.num_classes)
        # Start testing #
        self.linear = nn.Linear(opt.audio_encoder_dim, opt.linear_layer_last_dim)
        self.classifer = nn.Linear(opt.linear_layer_last_dim, opt.num_classes)
        # End testing #

        self.fusion_head_output_type = opt.fusion_head_output_type

    def forward(self, audio: torch.Tensor):
        # Audio processing
        audio_embeddings = self.audio_encoder(audio)

        # Check if vggish outputs is (128) or (num_samples, 128)
        if len(audio_embeddings.size()) == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Expand the audio embeddings to match the text embeddings
        if len(audio_embeddings.size()) == 2:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            audio_embeddings = audio_embeddings[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            audio_embeddings = audio_embeddings.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            audio_embeddings = audio_embeddings.max(dim=1)
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = self.linear(audio_embeddings)
        x = self.dropout(x)
        out = self.classifer(x)

        return out


opt = Config()
opt.load("checkpoints/AudioOnly_v2/cls_bert_lstm_mel/20231026-121302/opt.log")

model = AudioOnly_v2(opt)
model.to("cpu")
model.load_state_dict(
    torch.load(
        "checkpoints/AudioOnly_v2/cls_bert_lstm_mel/20231026-121302/weights/best_acc/checkpoint_0_0.pt",
        map_location=torch.device("cpu"),
    )["state_dict_network"]
)

wav_path = "IEMOCAP_emotion/test/0/Ses01F_impro04_F028.wav"

wav_data, sr = sf.read(wav_path, dtype="int16")
samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]

samples = waveform_to_examples(samples, sr, return_tensor=False)  # num_samples, 96, 64
samples = np.expand_dims(samples, axis=1)  # num_samples, 1, 96, 64
samples = torch.from_numpy(samples.astype(np.float32))

index2label = ["neutral", "positive", "negative"]
with torch.no_grad():
    output = model(samples.unsqueeze(0))
    print(torch.argmax(output))

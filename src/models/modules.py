import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(file_dir, "audioset_tagging_cnn/pytorch"))
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel

from configs.base import Config
from torchvggish import vggish

from .audioset_tagging_cnn.pytorch.models import (
    Wavegram_Logmel_Cnn14 as Wavegram_Logmel_Cnn14_Base,
)
from .feature_extraction import get_feature_extractor


def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


def build_roberta_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
    roberta = RobertaModel.from_pretrained("roberta-base", config=config)
    return roberta


class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.vggish = vggish()

    def forward(self, x):
        out = []
        for i in range(x.size(0)):
            out.append(self.vggish(x[i]))
        x = torch.stack(out, axis=0)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        return x


def build_vggish_encoder(cfg: Config) -> nn.Module:
    """A function to build vggish encoder"""
    return VGGish()


class Wavegram_Logmel_Cnn14(Wavegram_Logmel_Cnn14_Base):
    def __init__(self, **kwargs):
        super(Wavegram_Logmel_Cnn14, self).__init__(**kwargs)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)

        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        # Fix mismatch dimension for concatenation
        if x.size(2) > a1.size(2):
            a1 = torch.cat((a1, a1[:, :, -(x.size(2) - a1.size(2)) :, :]), dim=2)
        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        return embedding


def build_panns_encoder(cfg: Config) -> nn.Module:
    type = "Wavegram_Logmel_Cnn14"
    """A function to build panns encoder"""
    panns = {
        "Wavegram_Logmel_Cnn14": Wavegram_Logmel_Cnn14,
    }
    weights = {
        "Wavegram_Logmel_Cnn14": (
            "Wavegram_Logmel_Cnn14_mAP=0.439.pth",
            "https://github.com/namphuongtran9196/GitReleaseStorage/releases/download/somethings/Wavegram_Logmel_Cnn14_mAP.0.439.pth",
        ),
    }
    assert type in ["Wavegram_Logmel_Cnn14"], f"Do not support {type}"
    sample_rate = 16000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 125  # 50
    fmax = 7500  # 14000
    classes_num = 527

    model = panns[type](
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        classes_num=classes_num,
    )

    weights, url = weights[type]
    if not os.path.exists(os.path.join("/tmp/{}".format(weights))):
        os.system("wget {} -O /tmp/{}".format(url, weights))

    model.to("cpu")
    state_dict = torch.load(os.path.join("/tmp/{}".format(weights)), map_location="cpu")
    model.load_state_dict(state_dict["model"])
    return model


class HuBertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HuBertBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    """A function to build hubert encoder"""
    return HuBertBase()


class Wav2Vec2Base(nn.Module):
    def __init__(self, **kwargs):
        super(Wav2Vec2Base, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_wav2vec2_base_encoder(cfg: Config) -> nn.Module:
    return Wav2Vec2Base()


class WavlmBase(nn.Module):
    def __init__(self, **kwargs):
        super(WavlmBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.WAVLM_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_wavlm_base_encoder(cfg: Config) -> nn.Module:
    return WavlmBase()


class LSTM(nn.Module):
    def __init__(
        self, feature_module, input_size=512, hidden_size=512, num_layers=2, **kwargs
    ):
        super(LSTM, self).__init__(**kwargs)
        self.feature_module = feature_module
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        x, lengths = self.feature_module(x, None)  # (samples, length)
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        return x


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


def build_lstm_encoder(cfg: Config) -> nn.Module:
    weights = "feature_extractor_wav2vec_base.pth"
    url = "https://github.com/namphuongtran9196/GitReleaseStorage/releases/download/wav2vec_base/feature_extractor_wav2vec_base.pth"

    extractor_mode = "group_norm"
    extractor_conv_layer_config = [
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ]
    extractor_conv_bias = False
    feature_extractor = get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias
    )

    if not os.path.exists(os.path.join("/tmp/{}".format(weights))):
        os.system("wget {} -O /tmp/{}".format(url, weights))

    feature_extractor.to("cpu")
    state_dict = torch.load(os.path.join("/tmp/{}".format(weights)), map_location="cpu")
    feature_extractor.load_state_dict(state_dict)

    model = LSTM(
        feature_extractor,
        input_size=512,
        hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.lstm_num_layers,
    )

    return model


def build_lstm_mel_encoder(cfg: Config) -> nn.Module:
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
        hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.lstm_num_layers,
    )

    return model


def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        cfg (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = cfg.audio_encoder_type

    encoders = {
        "vggish": build_vggish_encoder,
        "panns": build_panns_encoder,
        "hubert_base": build_hubert_base_encoder,
        "wav2vec2_base": build_wav2vec2_base_encoder,
        "wavlm_base": build_wavlm_base_encoder,
        "lstm": build_lstm_encoder,
        "lstm_mel": build_lstm_mel_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](cfg)


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {
        "bert": build_bert_encoder,
        "roberta": build_roberta_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()

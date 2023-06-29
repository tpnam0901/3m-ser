import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(file_dir, "audioset_tagging_cnn/pytorch"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvggish import vggish
from transformers import BertConfig, BertModel

from .audioset_tagging_cnn.pytorch.models import Wavegram_Logmel_Cnn14 as Wavegram_Logmel_Cnn14_Base


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


class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.vggish = vggish()

    def forward(self, x):
        out = []
        for i in range(x.size(0)):
            out.append(self.vggish(x[i]))
        x = torch.stack(out, axis=0)
        return x


def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {"bert": build_bert_encoder}
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()


def build_vggish_encoder() -> nn.Module:
    """A function to build vggish encoder"""
    return VGGish()


def build_panns_encoder(type: str = "Wavegram_Logmel_Cnn14") -> nn.Module:
    """A function to build panns encoder"""
    panns = {
        "Wavegram_Logmel_Cnn14": Wavegram_Logmel_Cnn14,
    }
    weights = {
        "Wavegram_Logmel_Cnn14": (
            "Wavegram_Logmel_Cnn14_mAP=0.439.pth",
            "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1",
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


def build_audio_encoder(type: str = "vggish") -> nn.Module:
    """A function to build audio encoder

    Args:
        type (str, optional): Type of audio encoder. Defaults to "vggish".

    Returns:
        nn.Module: Audio encoder
    """
    encoders = {
        "vggish": build_vggish_encoder,
        "panns": build_panns_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type]()

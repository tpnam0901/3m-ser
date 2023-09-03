import logging
import os
from abc import ABC, abstractmethod
from typing import List


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
            elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value

        with open(opt_path, "r") as f:
            data = f.read().split("\n")
            # remove all empty strings
            data = list(filter(None, data))
            # convert to dict
            data_dict = {}
            for i in range(len(data)):
                key, value = data[i].split(":")[0].strip(), data[i].split(":")[1].strip()
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
        self.feat_dim: int = 2048

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
        self.audio_encoder_type: str = "panns"  # [vggish, panns, hubert_base, wav2vec2_base, wavlm_base]
        self.audio_encoder_dim: int = 2048  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base,wavlm_base
        self.audio_norm_type: str = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze: bool = True

        self.fusion_head_output_type: str = "cls"  # [cls, mean, max]

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

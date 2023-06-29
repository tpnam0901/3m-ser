import logging
import os
from abc import ABC, abstractmethod


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
        super().__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, opt):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(opt.checkpoint_dir), exist_ok=True)
        out_opt = os.path.join(opt.checkpoint_dir, "opt.log")
        with open(out_opt, "w") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logging.info(message)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        self.num_epochs = 200
        self.checkpoint_dir = "checkpoints_panns"
        self.save_freq = 4000
        self.batch_size = 64

        # Learning rate
        self.learning_rate = 0.0001
        self.learning_rate_step_size = 30
        self.learning_rate_gamma = 0.1

        # Dataset
        self.data_root = "data/PANNS"  # [data/3M-SER, data/PANNS]
        self.text_max_length = 297
        self.audio_max_length = 546220

        # Model
        self.num_classes = 4
        self.num_attention_head = 8
        self.dropout = 0.5
        self.model_type = "MMSERALayerNorm"  # [MMSERA, MMSERAMinMax, MMSERALayerNorm, AudioModel]
        self.text_encoder_type = "bert"  # [bert]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "panns"  # [vggish, panns]
        self.audio_encoder_dim = 2048  # 2048 for panns, 128 for vggish
        self.audio_unfreeze = True

        for key, value in kwargs.items():
            setattr(self, key, value)

from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "bert_panns"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 16

        self.model_type = "MMSERA"  # [MMSERA, AudioModel]
        self.text_encoder_type = "bert"  # [bert]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "panns"  # [vggish, panns, hubert_base]
        self.audio_encoder_dim = 2048  # 2048 - panns, 128 - vggish, 768 - hubert_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = True

        for key, value in kwargs.items():
            setattr(self, key, value)

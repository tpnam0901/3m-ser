from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "bert_wav2vec2"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1

        self.model_type = "MMSERA"  # # [MMSERA, AudioOnly, TextOnly, MMSERA_without_fusion_module]
        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "wav2vec2_base"  # [vggish, panns, hubert_base, wav2vec2_base]
        self.audio_encoder_dim = 768  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = False

        for key, value in kwargs.items():
            setattr(self, key, value)

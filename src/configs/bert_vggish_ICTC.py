from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "bert_vggish_ICTC"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 1000

        self.loss_type = "CrossEntropyLoss"  # [CrossEntropyLoss, CrossEntropyLoss_ContrastiveCenterLoss]

        self.model_type = "MMSERA"  # # [MMSERA, AudioOnly, TextOnly, MMSERA_without_fusion_module]
        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "vggish"  # [vggish, panns, hubert_base]
        self.audio_encoder_dim = 128  # 2048 - panns, 128 - vggish, 768 - hubert_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = True

        self.fusion_head_output_type = "mean"  # [cls, mean, max]

        for key, value in kwargs.items():
            setattr(self, key, value)

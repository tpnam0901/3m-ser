from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "3m-ser_roberta_wav2vec2_cel_ccl"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 250

        self.loss_type = "CrossEntropyLoss_ContrastiveCenterLoss"  # [CrossEntropyLoss, CrossEntropyLoss_ContrastiveCenterLoss]
        self.feat_dim = 128

        # For contrastive-center loss
        self.lambda_c = 2.0

        # For combined margin loss
        self.margin_loss_m1 = 1.0
        self.margin_loss_m2 = 0.5
        self.margin_loss_m3 = 0.0
        self.margin_loss_scale = 64.0

        # For focal loss
        self.focal_loss_gamma = 0.5
        self.focal_loss_alpha = None

        self.model_type = "MMSERA"  # [MMSERA, AudioOnly, TextOnly, MMSERA_without_fusion_module]
        self.text_encoder_type = "roberta"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "wav2vec2_base"  # [vggish, panns, hubert_base, wav2vec2_base]
        self.audio_encoder_dim = 768  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = True

        self.fusion_head_output_type = "mean"  # [cls, mean, max]

        for key, value in kwargs.items():
            setattr(self, key, value)

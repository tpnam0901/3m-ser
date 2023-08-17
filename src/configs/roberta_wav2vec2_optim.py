from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "3m-ser_roberta_wav2vec2_cel_ccl_1.0"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 250

        self.loss_type = "CrossEntropyLoss_ContrastiveCenterLoss"  # [CrossEntropyLoss, CrossEntropyLoss_ContrastiveCenterLoss]
        self.feat_dim = 128

        self.model_type = "MMSERA"  # # [MMSERA, AudioOnly, TextOnly, MMSERA_without_fusion_module]
        self.text_encoder_type = "roberta"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "wav2vec2_base"  # [vggish, panns, hubert_base, wav2vec2_base]
        self.audio_encoder_dim = 768  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = False

        self.fusion_head_output_type = "mean"  # [cls, mean, max]

        # # Hyperparameter search
        # self.lambda_c = [0.5,1.0,1.5,2.0]  # For CrossEntropyLoss_ContrastiveCenterLoss
        # self.optim_attributes = ["lambda_c"]

        for key, value in kwargs.items():
            setattr(self, key, value)

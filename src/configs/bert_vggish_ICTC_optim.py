from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "SERVER_bert_vggish_ICTC_optim"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 250

        self.loss_type = "CrossEntropyLoss_ContrastiveCenterLoss"  # [CrossEntropyLoss, CrossEntropyLoss_ContrastiveCenterLoss]

        self.model_type = "SERVER"  # # [MMSERA, AudioOnly, TextOnly, SERVER]
        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "vggish"  # [vggish, panns, hubert_base]
        self.audio_encoder_dim = 128  # 2048 - panns, 128 - vggish, 768 - hubert_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = True

        self.fusion_head_output_type = "mean"  # [cls, mean, max]

        # Hyperparameter search
        self.lambda_c = [x / 10 for x in range(11, 20)]  # For CrossEntropyLoss_ContrastiveCenterLoss
        self.optim_attributes = ["lambda_c"]

        for key, value in kwargs.items():
            setattr(self, key, value)

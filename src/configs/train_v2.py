from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "roberta_wav2vec2_cel_linear_optim"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        # self.resume = True
        # self.resume_path = "/home/kuhaku/Code/EmotionClassification/code/3m-ser-private/scripts/checkpoints/text_roberta_cel/20230822-102000/weights/checkpoint_59_220000.pt"
        # self.opt_path = "/home/kuhaku/Code/EmotionClassification/code/3m-ser-private/scripts/checkpoints/text_roberta_cel/20230822-102000/opt.log"

        self.batch_size = 1
        self.num_epochs = 250

        self.loss_type = "CrossEntropyLoss"  # [CrossEntropyLoss, CrossEntropyLoss_ContrastiveCenterLoss]

        # For contrastive-center loss
        self.lambda_c = 1.0
        self.feat_dim = 128

        # For combined margin loss
        self.margin_loss_m1 = 1.0
        self.margin_loss_m2 = 0.5
        self.margin_loss_m3 = 0.0
        self.margin_loss_scale = 64.0

        # For focal loss
        self.focal_loss_gamma = 0.5
        self.focal_loss_alpha = None

        self.model_type = "MMSERA_v2"  # [MMSERA, AudioOnly, TextOnly, MMSERA_without_fusion_module]
        self.text_encoder_type = "roberta"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = "wav2vec2_base"  # [vggish, panns, hubert_base, wav2vec2_base]
        self.audio_encoder_dim = 768  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = False

        self.fusion_head_output_type = "cls"  # [cls, mean, max]

        # Search for linear layer output dimension
        self.linear_layer_output = [256, 128]
        self.linear_layer_last_dim = 64

        for key, value in kwargs.items():
            setattr(self, key, value)

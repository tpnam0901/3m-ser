from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 200

        self.loss_type = "CrossEntropyLoss"

        self.checkpoint_dir = "checkpoints_latest/IEMOCAP/MMER"

        # For contrastive-center loss
        self.lambda_c = 1.0
        self.feat_dim = 768

        # For combined margin loss
        self.margin_loss_m1 = 1.0
        self.margin_loss_m2 = 0.5
        self.margin_loss_m3 = 0.0
        self.margin_loss_scale = 64.0

        # For focal loss
        self.focal_loss_gamma = 0.5
        self.focal_loss_alpha = None

        self.model_type = "MSER"  # [MMSERA, AudioOnly, TextOnly, SERVER,  ]
        self.trainer = "MSER_Trainer"
        self.text_encoder_type = "roberta"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False
        self.audio_encoder_type = (
            "wav2vec2_base"  # [vggish, panns, hubert_base, wav2vec2_base, wavlm_base]
        )
        self.audio_encoder_dim = 768  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base, wavlm_base
        self.audio_norm_type = "layer_norm"  # [layer_norm, min_max, None]
        self.audio_unfreeze = True

        self.fusion_dim: int = 768
        self.fusion_head_output_type = "cls"  # [cls, mean, max]
        self.linear_layer_output = [128, 64]

        # Dataset
        self.data_name: str = "IEMOCAP_MSER"  # [IEMOCAP, ESD, MELD, IEMOCAP_MSER]
        self.data_root: str = (
            "data/IEMOCAP_preprocessed"  # folder contains train.pkl and test.pkl
        )
        self.data_valid: str = "val.pkl"  # change this to your validation subset name if you want to use validation dataset. If None, test.pkl will be use

        # use for training with batch size > 1
        self.text_max_length: int = 297
        self.audio_max_length: int = 546220

        # Config name
        self.name = f"{self.loss_type}_{self.text_encoder_type}_{self.audio_encoder_type}_{self.fusion_head_output_type}"
        # self.name = f"{self.fusion_head_output_type}_{self.text_encoder_type}_{self.audio_encoder_type}"

        for key, value in kwargs.items():
            setattr(self, key, value)

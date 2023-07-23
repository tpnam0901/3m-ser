from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "roberta_wav2vec2_optim"
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.model_type = "MMSERA"  # # [MMSERA, AudioOnly, TextOnly, MMSERA_without_fusion_module]
        self.text_encoder_type = "roberta"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.audio_encoder_type = "wav2vec2_base"  # [vggish, panns, hubert_base, wav2vec2_base]
        self.audio_encoder_dim = 768  # 2048 - panns, 128 - vggish, 768 - hubert_base,wav2vec2_base

        # Hyperparameter search
        self.batch_size = [1, 2]
        self.learning_rate = [0.1, 0.01, 0.001, 0.0001]
        self.num_attention_head = [2, 4, 6, 8]
        self.audio_norm_type = ["layer_norm", "min_max", "None"]
        self.fusion_head_output_type = ["cls", "mean", "max"]
        self.optim_attributes = [
            "batch_size",
            "learning_rate",
            "num_attention_head",
            "audio_norm_type",
            "fusion_head_output_type",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

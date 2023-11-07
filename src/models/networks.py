import torch
import torch.nn as nn

from configs.base import Config

from .modules import build_audio_encoder, build_text_encoder


class AudioOnly_v2(nn.Module):
    def __init__(
        self,
        opt: Config,
        device: str = "cpu",
    ):
        """Speech Emotion Recognition with Audio Only

        Args:
            opt (Config): Config object
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(AudioOnly_v2, self).__init__()

        # Audio module
        self.audio_encoder = build_audio_encoder(opt)
        self.audio_encoder.to(device)
        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = opt.audio_unfreeze

        self.dropout = nn.Dropout(opt.dropout)

        # self.linear = nn.Linear(opt.audio_encoder_dim, opt.audio_encoder_dim)
        # self.classifer = nn.Linear(opt.audio_encoder_dim, opt.num_classes)
        # Start testing #
        self.linear = nn.Linear(opt.audio_encoder_dim, opt.linear_layer_last_dim)
        self.classifer = nn.Linear(opt.linear_layer_last_dim, opt.num_classes)
        # End testing #

        self.fusion_head_output_type = opt.fusion_head_output_type

    def forward(
        self,
        input_ids: torch.Tensor,
        audio: torch.Tensor,
        output_attentions: bool = False,
    ):
        # Audio processing
        audio_embeddings = self.audio_encoder(audio)

        # Check if vggish outputs is (128) or (num_samples, 128)
        if len(audio_embeddings.size()) == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Expand the audio embeddings to match the text embeddings
        if len(audio_embeddings.size()) == 2:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            audio_embeddings = audio_embeddings[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            audio_embeddings = audio_embeddings.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            audio_embeddings = audio_embeddings.max(dim=1)
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = self.linear(audio_embeddings)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, audio_embeddings, None, None


# Create audio only model
class TextOnly_v2(nn.Module):
    def __init__(
        self,
        opt: Config,
        device: str = "cpu",
    ):
        """Speech Emotion Recognition with Text Only

        Args:
            opt (Config): Config object
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(TextOnly_v2, self).__init__()

        # Text module
        self.text_encoder = build_text_encoder(opt.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = opt.text_unfreeze

        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.text_encoder_dim, opt.linear_layer_last_dim)
        self.classifer = nn.Linear(opt.linear_layer_last_dim, opt.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio: torch.Tensor,
        output_attentions: bool = False,
    ):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).pooler_output
        # Classification head
        x = self.linear(text_embeddings)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, text_embeddings, None, None


class MMSERA_v2(nn.Module):
    def __init__(
        self,
        opt: Config,
        device: str = "cpu",
    ):
        """Summary: 3M-SER model version 2 for optimizing parameters"""
        super(MMSERA_v2, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(opt.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = opt.text_unfreeze

        # Audio module
        self.audio_norm_type = opt.audio_norm_type
        self.audio_encoder = build_audio_encoder(opt)
        self.audio_encoder.to(device)
        if opt.audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(opt.audio_encoder_dim)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = opt.audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=opt.text_encoder_dim,
            num_heads=opt.num_attention_head,
            dropout=opt.dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(opt.text_encoder_dim, opt.audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(opt.audio_encoder_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=opt.audio_encoder_dim,
            num_heads=opt.num_attention_head,
            dropout=opt.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(opt.audio_encoder_dim, opt.audio_encoder_dim)
        self.fusion_layer_norm = nn.LayerNorm(opt.audio_encoder_dim)

        self.dropout = nn.Dropout(opt.dropout)

        self.linear_layer_output = opt.linear_layer_output

        previous_dim = opt.audio_encoder_dim
        if len(opt.linear_layer_output) > 0:
            for i, linear_layer in enumerate(opt.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, opt.num_classes)

        self.fusion_head_output_type = opt.fusion_head_output_type

    def forward(
        self,
        input_ids: torch.Tensor,
        audio: torch.Tensor,
        output_attentions: bool = False,
    ):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).last_hidden_state

        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        if self.audio_norm_type == "layer_norm":
            audio_embeddings = self.audio_encoder_layer_norm(audio_embeddings)
        elif self.audio_norm_type == "min_max":
            # Min-max normalization
            audio_embeddings = (audio_embeddings - audio_embeddings.min()) / (
                audio_embeddings.max() - audio_embeddings.min()
            )

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_embeddings), 1)

        # Selt-attention module
        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings,
            fusion_embeddings,
            fusion_embeddings,
            average_attn_weights=False,
        )
        fusion_linear = self.fusion_linear(fusion_attention)
        fusion_norm = self.fusion_layer_norm(fusion_linear)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [
                text_attn_output_weights,
                fusion_attn_output_weights,
            ]

        return (
            out,
            cls_token_final_fusion_norm,
            text_norm.mean(dim=1),
            audio_embeddings.mean(dim=1),
        )


class SERVER_v2(nn.Module):
    def __init__(
        self,
        opt: Config,
        device: str = "cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(SERVER_v2, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(opt.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = opt.text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(opt)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = opt.audio_unfreeze

        self.dropout = nn.Dropout(opt.dropout)
        self.linear1 = nn.Linear(opt.text_encoder_dim, 128)
        self.linear2 = nn.Linear(256, 64)
        self.classifer = nn.Linear(64, opt.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio: torch.Tensor,
        output_attentions: bool = False,
    ):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).pooler_output
        text_embeddings = self.linear1(text_embeddings)
        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        # Get classification token from the audio module
        audio_embeddings = audio_embeddings.sum(dim=1)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_embeddings, audio_embeddings), 1)

        # Classification head
        x = self.dropout(fusion_embeddings)
        x = self.linear2(x)
        out = self.classifer(x)

        return out, fusion_embeddings, text_embeddings, audio_embeddings

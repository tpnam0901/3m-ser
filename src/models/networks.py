import torch
import torch.nn as nn

from .modules import build_audio_encoder, build_text_encoder


# Create audio only model
class AudioOnly(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        text_encoder_type="bert",
        text_encoder_dim=768,
        text_unfreeze=False,
        audio_encoder_type="vggish",
        audio_encoder_dim=128,
        audio_unfreeze=True,
        audio_norm_type="layer_norm",
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(AudioOnly, self).__init__()

        # Audio module
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)
        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(audio_encoder_dim, audio_encoder_dim)
        self.classifer = nn.Linear(audio_encoder_dim, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Audio processing
        audio_embeddings = self.audio_encoder(audio)

        # Check if vggish outputs is (128) or (num_samples, 128)
        if len(audio_embeddings.size()) == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Expand the audio embeddings to match the text embeddings
        audio_embeddings = audio_embeddings.unsqueeze(0)
        # Flatten the audio embeddings
        audio_embeddings = audio_embeddings.view(audio_embeddings.size(0), -1)
        # Classification head
        x = self.dropout(audio_embeddings)
        x = self.linear(x)
        x = nn.functional.leaky_relu(x)
        out = self.classifer(x)

        return out


# Create audio only model
class TextOnly(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        text_encoder_type="bert",
        text_encoder_dim=768,
        text_unfreeze=False,
        audio_encoder_type="vggish",
        audio_encoder_dim=128,
        audio_unfreeze=True,
        audio_norm_type="layer_norm",
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(AudioOnly, self).__init__()

        # Text module
        self.text_encoder = build_text_encoder(text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(audio_encoder_dim, audio_encoder_dim)
        self.classifer = nn.Linear(audio_encoder_dim, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).pooler_output
        # Classification head
        x = self.dropout(text_embeddings)
        x = self.linear(x)
        x = nn.functional.leaky_relu(x)
        out = self.classifer(x)

        return out


# Create Multi-modal model - layer norm
class MMSERA(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        text_encoder_type="bert",
        text_encoder_dim=768,
        text_unfreeze=False,
        audio_encoder_type="vggish",
        audio_encoder_dim=128,
        audio_unfreeze=True,
        audio_norm_type="layer_norm",
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(MMSERA, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        # Audio module
        self.audio_norm_type = audio_norm_type
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)
        if audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(audio_encoder_dim)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.text_linear = nn.Linear(text_encoder_dim, audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(audio_encoder_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.fusion_linear = nn.Linear(audio_encoder_dim, 128)
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).last_hidden_state

        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        if self.audio_norm_type == "layer_norm":
            audio_embeddings = self.audio_encoder_layer_norm(audio_embeddings)
        elif self.audio_norm_type == "min_max":
            # Min-max normalization
            audio_embeddings = (audio_embeddings - audio_embeddings.min()) / (audio_embeddings.max() - audio_embeddings.min())

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_embeddings), 1)

        # Selt-attention module
        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings, fusion_embeddings, fusion_embeddings, average_attn_weights=False
        )
        fusion_linear = self.fusion_linear(fusion_attention)
        fusion_norm = self.fusion_layer_norm(fusion_linear)

        # Get classification token from the fusion module
        cls_token_final_fusion_norm = fusion_norm[:, 0, :]

        # Classification head
        x = self.dropout(cls_token_final_fusion_norm)
        x = self.linear(x)
        x = nn.functional.leaky_relu(x)
        out = self.classifer(x)

        if output_attentions:
            return out, [text_attn_output_weights, fusion_attn_output_weights]

        return out


class MMSERA_without_fusion_module(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        text_encoder_type="bert",
        text_encoder_dim=768,
        text_unfreeze=False,
        audio_encoder_type="vggish",
        audio_encoder_dim=128,
        audio_unfreeze=True,
        audio_norm_type="layer_norm",
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(MMSERA, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = text_unfreeze

        # Audio module
        self.audio_norm_type = audio_norm_type
        self.audio_encoder = build_audio_encoder(audio_encoder_type)
        self.audio_encoder.to(device)
        if audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(audio_encoder_dim)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.text_linear = nn.Linear(text_encoder_dim, audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(audio_encoder_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=audio_encoder_dim, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.fusion_linear = nn.Linear(audio_encoder_dim, 128)
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        text_embeddings = self.text_encoder(input_ids).pooler_output

        # Audio processing
        audio_embeddings = self.audio_encoder(audio)
        # Get classification token from the audio module
        audio_embeddings = audio_embeddings[:, 0, :]

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_embeddings, audio_embeddings), 1)

        # Classification head
        x = self.dropout(fusion_embeddings)
        x = self.linear(x)
        x = nn.functional.leaky_relu(x)
        out = self.classifer(x)

        if output_attentions:
            return out, [text_attn_output_weights, fusion_attn_output_weights]

        return out

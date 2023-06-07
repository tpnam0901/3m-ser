import torch
import torch.nn as nn

# from cca_zoo.deepmodels import DCCA
# from cca_zoo.deepmodels.objectives import MCCA
from torchvggish import vggish
from transformers import BertConfig, BertModel


# Create Multi-modal model
class MMSERA(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
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
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.bert.to(device)
        # Freeze the text module
        for param in self.bert.parameters():
            param.requires_grad = False

        # Audio module
        self.vggish = vggish()
        self.vggish.to(device)

        # Fusion module
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=num_attention_head, dropout=dropout)
        self.text_linear = nn.Linear(768, 128)

        self.fusion_attention = nn.MultiheadAttention(embed_dim=128, num_heads=num_attention_head, dropout=dropout)
        self.fusion_linear = nn.Linear(128, 128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        input_ids = input_ids.long()
        text_embeddings = self.bert(input_ids).last_hidden_state

        # Audio processing
        audio_embeddings = self.vggish(audio)
        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)

        # Check if vggish outputs is (128) or (num_samples, 128)
        if len(audio_embeddings.size()) == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Expand the audio embeddings to match the text embeddings
        audio_embeddings = audio_embeddings.unsqueeze(0)
        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_linear, audio_embeddings), 1)

        # Selt-attention module
        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings, fusion_embeddings, fusion_embeddings, average_attn_weights=False
        )
        fusion_linear = self.fusion_linear(fusion_attention)

        # Get classification token from the fusion module
        cls_token_final_fusion_linear = fusion_linear[:, 0, :]

        # Classification head
        x = self.dropout(cls_token_final_fusion_linear)
        x = self.linear(x)
        x = nn.functional.leaky_relu(x)
        out = self.classifer(x)
        if output_attentions:
            return out, [text_attn_output_weights, fusion_attn_output_weights]
        return out


# Create Multi-modal model - layer norm
class MMSERALayerNorm(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(MMSERALayerNorm, self).__init__()
        # Text module
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.bert.to(device)
        # Freeze the text module
        for param in self.bert.parameters():
            param.requires_grad = False

        # Audio module
        self.vggish = vggish()
        self.vggish.to(device)
        self.vggish_layer_norm = nn.LayerNorm(128)
        # Freeze the audio module
        for param in self.vggish.parameters():
            param.requires_grad = True

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.text_linear = nn.Linear(768, 128)
        self.text_layer_norm = nn.LayerNorm(128)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.fusion_linear = nn.Linear(128, 128)
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        text_embeddings = self.bert(input_ids).last_hidden_state

        # Audio processing
        audio_embeddings = self.vggish(audio)
        audio_embeddings = self.vggish_layer_norm(audio_embeddings)

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Check if vggish outputs is (128) or (num_samples, 128)
        if len(audio_embeddings.size()) == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Expand the audio embeddings to match the text embeddings
        audio_embeddings = audio_embeddings.unsqueeze(0)

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


# Create Multi-modal model - min max
class MMSERAMinMax(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        device="cpu",
    ):
        """

        Args: MMSERA model extends from MMSER model in the paper
            num_classes (int, optional): The number of classes. Defaults to 4.
            num_attention_head (int, optional): The number of self-attention heads. Defaults to 8.
            dropout (float, optional): Whether to use dropout. Defaults to 0.5.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(MMSERAMinMax, self).__init__()
        # Text module
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.bert.to(device)
        # Freeze the text module
        for param in self.bert.parameters():
            param.requires_grad = False

        # Audio module
        self.vggish = vggish()
        self.vggish.to(device)

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.text_linear = nn.Linear(768, 128)
        self.text_layer_norm = nn.LayerNorm(128)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=num_attention_head, dropout=dropout, batch_first=True
        )
        self.fusion_linear = nn.Linear(128, 128)
        self.fusion_layer_norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 64)
        self.classifer = nn.Linear(64, num_classes)

    def forward(self, input_ids, audio, output_attentions=False):
        # Text processing
        text_embeddings = self.bert(input_ids).last_hidden_state

        # Audio processing
        audio_embeddings = self.vggish(audio)
        # Min-max normalization
        audio_embeddings = (audio_embeddings - audio_embeddings.min()) / (audio_embeddings.max() - audio_embeddings.min())

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Check if vggish outputs is (128) or (num_samples, 128)
        if len(audio_embeddings.size()) == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)

        # Expand the audio embeddings to match the text embeddings
        audio_embeddings = audio_embeddings.unsqueeze(0)

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

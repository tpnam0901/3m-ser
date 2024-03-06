# Get from https://github.com/Sreyan88/MMER/blob/684fb1ccbe9e3b3e86b4e2926e4fb76467bbd638/src/mmi_module.py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import tempfile
from collections import OrderedDict
from io import open

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from transformers import BertConfig as HFBertConfig
from transformers import BertModel, HubertModel, Wav2Vec2Model, WavLMModel


# from utils import create_mask
def create_mask(batch_size, seq_len, spec_len):
    with torch.no_grad():
        attn_mask = torch.ones((batch_size, seq_len))  # (batch_size, seq_len)

        for idx in range(batch_size):
            # zero vectors for padding dimension
            attn_mask[idx, spec_len[idx] :] = 0

    return attn_mask


# from torchcrf import CRF


# from .file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

# from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder, TransformerSentenceEncoderLayer

args = {
    "input_feat": 1024,
    "encoder_embed_dim": 1024,
    "encoder_layers": 3,
    "dropout": 0.1,
    "activation_dropout": 0,
    "dropout_input": 0.1,
    "attention_dropout": 0.1,
    "encoder_layerdrop": 0.05,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "encoder_ffn_embed_dim": 2048,
    "encoder_attention_heads": 4,
    "activation_fn": "gelu",
    "layer_norm_first": False,
}


class Config:
    pass


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = "bert_config.json"
TF_WEIGHTS_NAME = "model.ckpt"


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (
            sys.version_info[0] == 2
            and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertLastSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertLastSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size * 2 / config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size * 2, self.all_head_size)
        self.key = nn.Linear(config.hidden_size * 2, self.all_head_size)
        self.value = nn.Linear(config.hidden_size * 2, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(
            s1_hidden_states, s2_hidden_states, s2_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertSelfEncoder(nn.Module):
    def __init__(self, config):
        super(BertSelfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(
        self,
        s1_hidden_states,
        s2_hidden_states,
        s2_attention_mask,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(
                s1_hidden_states, s2_hidden_states, s2_attention_mask
            )
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class ActivateFun(nn.Module):
    def __init__(self, cfg):
        super(ActivateFun, self).__init__()
        self.activate_fun = cfg

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == "relu":
            return torch.relu(x)
        elif self.activate_fun == "gelu":
            return self._gelu(x)


class MMI_Model(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top."""

    def __init__(
        self,
        config,
        # ctc_output_size,
        label_output_size,
        layer_num1=1,
        layer_num2=1,
        layer_num3=1,
        # num_labels=2,
        # auxnum_labels=2,
    ):
        super(MMI_Model, self).__init__()
        # self.num_labels = num_labels
        config = HFBertConfig.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            output_hidden_states=True,
            return_dict=True,
            apply_spec_augment=False,
        )

        self.wav2vec2.feature_extractor._freeze_parameters()
        self.self_attention = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.img2txt_attention = BertCrossEncoder(config, layer_num2)
        self.txt2txt_attention = BertCrossEncoder(config, layer_num3)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Sequential(
            OrderedDict(
                [("linear3", nn.Linear(config.hidden_size * 2, label_output_size))]
            )
        )

        self.dropout_audio_input = nn.Dropout(0.1)

        self.downsample_final = nn.Linear(768 * 2, 768)

        self.weights = nn.Parameter(torch.zeros(13))

        self.fuse_type = "max"

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(768 * 2, 768),
            ActivateFun("gelu"),
            nn.Linear(768, 768),
        )

        if self.fuse_type == "att":
            self.output_attention_audio = nn.Sequential(
                nn.Linear(768, 768 // 2), ActivateFun("gelu"), nn.Linear(768 // 2, 1)
            )
            self.output_attention_multimodal = nn.Sequential(
                nn.Linear(768 * 2, 768 * 2 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 * 2 // 2, 1),
            )

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):
        loss = None
        if labels is not None:
            # # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            if attention_mask is not None:
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)
                ).type(torch.IntTensor)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                )

        return loss

    def _cls_loss(
        self, logits, cls_labels
    ):  # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def _weighted_sum(self, feature, normalize):
        stacked_feature = torch.stack(feature, dim=0)

        if normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(13, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def forward(
        self,
        bert_input_ids,
        bert_attention_mask,
        audio_input,
        audio_length,
        # ctc_labels,
        # emotion_labels,
        # augmentation=False,
    ):
        # text_output = self.bert(bert_input_ids,attention_mask=bert_attention_mask,token_type_ids=bert_segment_ids)
        text_output = self.bert(
            bert_input_ids, attention_mask=bert_attention_mask
        ).hidden_states
        text_output = self.dropout(text_output[0])

        audio_output_wav2vec2 = self.wav2vec2(audio_input)[0]  # imp

        # -----------------------------------------------------------------------------------------------------------#
        # create raw audio, FBank and wav2vec2 hidden state attention masks
        # create raw audio, FBank and wav2vec2 hidden state attention masks
        (
            audio_attention_mask,
            fbank_attention_mask,
            wav2vec2_attention_mask,
            input_lengths,
        ) = (None, None, None, None)

        audio_attention_mask = create_mask(
            audio_input.shape[0], audio_input.shape[1], audio_length
        )

        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            audio_attention_mask.sum(-1)
        ).type(torch.IntTensor)
        wav2vec2_attention_mask = create_mask(
            audio_output_wav2vec2.shape[0],
            audio_output_wav2vec2.shape[1],
            input_lengths,
        )

        wav2vec2_attention_mask = wav2vec2_attention_mask.cuda()

        # -----------------------------------------------------------------------------------------------------------#

        audio_output_dropout = self.dropout_audio_input(audio_output_wav2vec2)
        # logits_ctc = self.ctc_linear(audio_output_dropout)

        extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        main_addon_sequence_encoder = self.self_attention(
            text_output, extended_txt_mask
        )
        main_addon_sequence_output = main_addon_sequence_encoder[-1]

        wav2vec2_attention_mask_back = wav2vec2_attention_mask.clone()
        # subsample the frames to 1/4th of the number
        audio_output = audio_output_wav2vec2.clone()
        # project audio embeddings to a smaller space
        converted_vis_embed_map = self.vismap2text(audio_output)

        # --------------------applying txt2img attention mechanism to obtain image-based text representations----------------------------#
        # calculate added attention mask
        img_mask = wav2vec2_attention_mask.squeeze(1).clone()
        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(
            main_addon_sequence_output, converted_vis_embed_map, extended_img_mask
        )
        cross_output_layer = cross_encoder[
            -1
        ]  # self.batch_size * text_len * hidden_dim

        # ----------------------------------------------------------------------------------------------------------------------------------#

        # ----------------------apply img2txt attention mechanism to obtain multimodal-based text representations-------------------------#

        # project audio embeddings to a smaller space || left part of the image
        converted_vis_embed_map = self.vismap2text(audio_output)

        cross_txt_encoder = self.img2txt_attention(
            converted_vis_embed_map, main_addon_sequence_output, extended_txt_mask
        )
        cross_txt_output_layer = cross_txt_encoder[
            -1
        ]  # self.batch_size * audio_length * hidden_dim

        # ----------------------------------#

        cross_final_txt_encoder = self.txt2txt_attention(
            main_addon_sequence_output, cross_txt_output_layer, extended_img_mask
        )
        cross_final_txt_layer = cross_final_txt_encoder[
            -1
        ]  # self.batch_size * text_len * hidden_dim

        # ----------------------------------------------------------------------------------------------------------------------------------#

        # ---------------------------------------apply visual gate and get final representations---------------------------------------------#

        merge_representation = torch.cat(
            (cross_final_txt_layer, cross_output_layer), dim=-1
        )
        gate_value = torch.sigmoid(
            self.gate(merge_representation)
        )  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)

        final_output = torch.cat(
            (cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1
        )

        audio_output_pool = audio_output_wav2vec2.clone()  # change _2

        multimodal_output = final_output.clone()

        if self.fuse_type == "mean":
            if audio_attention_mask is None:
                classification_feats_audio = torch.mean(audio_output_wav2vec2, dim=1)
            else:
                padding_mask = self.wav2vec2._get_feature_vector_attention_mask(
                    audio_output_wav2vec2.shape[1], audio_attention_mask
                )
                padding_mask = padding_mask.to(audio_output_wav2vec2.device)
                audio_output_pool[~padding_mask] = 0.0  # mean
                classification_feats_audio = audio_output_pool.sum(
                    dim=1
                ) / padding_mask.sum(dim=1).view(
                    -1, 1
                )  # mean
        elif self.fuse_type == "max":
            padding_mask = self.wav2vec2._get_feature_vector_attention_mask(
                audio_output_wav2vec2.shape[1], audio_attention_mask
            )
            padding_mask = padding_mask.to(audio_output_wav2vec2.device)
            audio_output_pool[~padding_mask] = -9999.9999  # max
            classification_feats_audio, _ = torch.max(audio_output_pool, dim=1)  # max
        elif self.fuse_type == "att":
            text_image_mask = wav2vec2_attention_mask_back.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0 : audio_output_pool.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention_audio(audio_output_pool)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(
                text_image_mask == 0, -1e9
            )
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
            classification_feats_audio = (
                text_image_alpha.unsqueeze(-1) * audio_output_pool
            ).sum(dim=1)
        elif self.fuse_type == "stats":
            classification_feats_audio = torch.cat(
                (
                    torch.mean(audio_output_pool, dim=1),
                    torch.std(audio_output_pool, dim=1),
                ),
                dim=-1,
            )  # 768*2

        if self.fuse_type == "mean":
            padding_mask_text = bert_attention_mask > 0
            multimodal_output[~padding_mask_text] = 0.0  # mean
            classification_feats_multimodal = multimodal_output.sum(
                dim=1
            ) / padding_mask_text.sum(dim=1).view(
                -1, 1
            )  # mean
            # classification_feats_multimodal = torch.mean(final_output, dim=1)
        elif self.fuse_type == "max":
            padding_mask_text = bert_attention_mask > 0
            multimodal_output[~padding_mask_text] = -9999.9999  # max
            classification_feats_multimodal, _ = torch.max(
                multimodal_output, dim=1
            )  # max
        elif self.fuse_type == "att":
            multimodal_mask = bert_attention_mask.permute(1, 0).contiguous()
            multimodal_mask = multimodal_mask[0 : multimodal_output.size(1)]
            multimodal_mask = multimodal_mask.permute(1, 0).contiguous()

            multimodal_alpha = self.output_attention_multimodal(multimodal_output)
            multimodal_alpha = multimodal_alpha.squeeze(-1).masked_fill(
                multimodal_mask == 0, -1e9
            )
            multimodal_alpha = torch.softmax(multimodal_alpha, dim=-1)
            classification_feats_multimodal = (
                multimodal_alpha.unsqueeze(-1) * multimodal_output
            ).sum(dim=1)
        elif self.fuse_type == "stats":
            classification_feats_multimodal = torch.cat(
                (
                    torch.mean(multimodal_output, dim=1),
                    torch.std(multimodal_output, dim=1),
                ),
                dim=-1,
            )

        classification_feats_multimodal = self.downsample_final(
            classification_feats_multimodal
        )
        final_output = torch.cat(
            (classification_feats_audio, classification_feats_multimodal), dim=-1
        )
        classification_feats_pooled = self.classifier(final_output)

        # ------------------------------------------------------------------------------------------------------------------------------------#

        # ------------------------------------------------------calculate losses---------------------------------------------------------------#

        # loss = None
        # loss_ctc = None
        # loss_cls = None
        # if not augmentation:
        #     loss_ctc = self._ctc_loss(
        #         logits_ctc, ctc_labels, input_lengths, audio_attention_mask
        #     )  # ctc loss
        #     loss_cls = self._cls_loss(
        #         classification_feats_pooled, emotion_labels
        #     )  # cls loss

        orgin_res_change = self.orgin_linear_change(final_output)

        return (
            classification_feats_pooled,
            orgin_res_change,
            final_output,
        )  # , loss_cls, loss_ctc

# Copyright (c) 2023, salesforce.com, inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from LAVIS Qformer.py by Junnan Li, based on Hugging Face BERT.
# See LICENSE and SOURCES.
"""The two (unimodal) LAVIS Q-Former encoder passes needed for ITC."""

from copy import deepcopy

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertAttention, BertModel


class QFormer(nn.Module):
    def __init__(self, config, vocab_size, vision_width, bert_weights):
        super().__init__()
        # Reuse the matching BERT attention/MLP primitives, NOT stock BERT's
        # forward. Give HF the explicitly loaded .bin, never a Hub model ID.
        self.bert = BertModel.from_pretrained(
            str(bert_weights), config=config, add_pooling_layer=False, local_files_only=True,
            state_dict=torch.load(bert_weights, map_location="cpu", weights_only=True),
        )
        self.bert.resize_token_embeddings(vocab_size)  # BLIP's additional [DEC]
        del self.bert.embeddings.token_type_embeddings
        for i, layer in enumerate(self.bert.encoder.layer):
            # LAVIS initializes the separate query FFNs from the text FFNs.
            layer.intermediate_query = deepcopy(layer.intermediate)
            layer.output_query = deepcopy(layer.output)
            if i % 2 == 0:
                cross = BertAttention(config)
                cross.self.key = nn.Linear(vision_width, config.hidden_size)
                cross.self.value = nn.Linear(vision_width, config.hidden_size)
                layer.crossattention = cross

    def encode_queries(self, queries, image_embeds):
        embeds = self.bert.embeddings
        x = embeds.dropout(embeds.LayerNorm(queries))
        for layer in self.bert.encoder.layer:
            # No padding: the official query and image masks are all ones
            # (identically zero after conversion to additive attention masks).
            x = layer.attention(x)[0]
            if hasattr(layer, "crossattention"):
                x = layer.crossattention(x, encoder_hidden_states=image_embeds)[0]
            x = layer.output_query(layer.intermediate_query(x), x)
        return x

    def encode_text(self, input_ids, attention_mask):
        embeds = self.bert.embeddings
        positions = embeds.position_ids[:, :input_ids.shape[1]]
        # Q-Former has no token-type embeddings, unlike vanilla BertModel.
        x = embeds.word_embeddings(input_ids) + embeds.position_embeddings(positions)
        x = embeds.dropout(embeds.LayerNorm(x))
        mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * -10000.0
        for layer in self.bert.encoder.layer:
            x = layer.attention(x, attention_mask=mask)[0]
            x = layer.output(layer.intermediate(x), x)
        return x

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import DistilBertModel

class QAEmbeddingModel(nn.Module):
    """
    Model for embedding questions and answers using DistilBERT
    with a projection layer to reduce dimensionality.
    """
    def __init__(self, embed_dim=768, projection_dim=128):
        super(QAEmbeddingModel, self).__init__()
        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Projection layer to get embeddings of the desired dimension
        self.projection = nn.Linear(embed_dim, projection_dim)
        
    def forward(self, input_ids, attention_mask):
        """
        Generate embeddings for input text
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            normalized_embeddings: L2-normalized embeddings
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Project to lower dimension
        projected = self.projection(embeddings)
        # Normalize embeddings to unit length
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        return normalized
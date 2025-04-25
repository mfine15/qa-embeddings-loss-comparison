#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified loss functions that work with batch transforms directly.
Each loss function handles its specific batch format and model forward passes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any, Optional

from rank_test.losses import (
    FlexibleInfoNCELoss,
    RankInfoNCELoss,
    RelaxedHardNegativeInfoNCELoss,
    TripletLoss, 
    ListwiseRankingLoss
)


class UnifiedLoss(nn.Module):
    """Base class for unified loss functions"""
    
    def __init__(self, **kwargs):
        super(UnifiedLoss, self).__init__()
        self.name = "unified_base_loss"
        
    def forward(self, model, batch, device):
        """
        Process batch and calculate loss based on model outputs
        
        Args:
            model: Model to use for forward pass
            batch: Raw batch from dataloader 
                  (format depends on batch transform)
            device: Device to run on
            
        Returns:
            loss: Loss tensor with gradient
            metrics: Dictionary of metrics for logging
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_name(self):
        """Get loss function name for logging"""
        return self.name


class UnifiedInfoNCELoss(UnifiedLoss):
    """
    InfoNCE loss adapter for standard and multiple positives formats.
    Works with:
    - infonce_batch_transform: Standard InfoNCE with in-batch negatives
      Format: {'q_input_ids', 'a_input_ids', 'question_ids', 'ranks', 'scores'}
    
    - multiple_positives_batch_transform: Multiple positives per query
      Format: {'q_input_ids', 'a_input_ids', 'question_ids', 'ranks', 'scores'}
    """
    
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.name = f"unified_infonce_t{temperature}"
        # Create wrapped loss function
        self.core_loss = FlexibleInfoNCELoss(temperature=temperature)
        
    def forward(self, model, batch, device):
        """
        Process batch and calculate InfoNCE loss
        
        Args:
            model: QA Embedding model
            batch: Batch from infonce or multiple_positives transform
            device: Device to run on
            
        Returns:
            loss: Loss tensor with gradient
            metrics: Dictionary with accuracy metrics
        """
        # Move tensors to device
        q_input_ids = batch['q_input_ids'].to(device)
        q_attention_mask = batch['q_attention_mask'].to(device)
        a_input_ids = batch['a_input_ids'].to(device)
        a_attention_mask = batch['a_attention_mask'].to(device)
        question_ids = batch['question_ids']
        
        # Get embeddings
        q_embeddings = model(q_input_ids, q_attention_mask)
        a_embeddings = model(a_input_ids, a_attention_mask)
        
        # Calculate loss using core loss function
        loss, metrics = self.core_loss(
            q_embeddings, a_embeddings, question_ids=question_ids
        )
        
        return loss, metrics


class UnifiedRankInfoNCELoss(UnifiedLoss):
    """
    Rank-weighted InfoNCE loss adapter for infonce and multiple_positives formats.
    Uses rank/score information to weight the positive examples.
    Works with:
    - infonce_batch_transform: Standard InfoNCE with in-batch negatives
      Format: {'q_input_ids', 'a_input_ids', 'question_ids', 'ranks', 'scores'}
    
    - multiple_positives_batch_transform: Multiple positives per query
      Format: {'q_input_ids', 'a_input_ids', 'question_ids', 'ranks', 'scores'}
    """
    
    def __init__(self, temperature=0.1, use_ranks=True, use_scores=False,
                 rank_weight=0.1, score_weight=0.01, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.use_ranks = use_ranks
        self.use_scores = use_scores
        self.rank_weight = rank_weight
        self.score_weight = score_weight
        
        # Create descriptive name
        ranking_method = []
        if use_ranks:
            ranking_method.append(f"rank{rank_weight}")
        if use_scores:
            ranking_method.append(f"score{score_weight}")
        
        ranking_str = "_".join(ranking_method) if ranking_method else "standard"
        self.name = f"unified_rank_infonce_{ranking_str}_t{temperature}"
        
        # Create wrapped loss function
        self.core_loss = RankInfoNCELoss(
            temperature=temperature,
            use_ranks=use_ranks,
            use_scores=use_scores,
            rank_weight=rank_weight,
            score_weight=score_weight
        )
        
    def forward(self, model, batch, device):
        """
        Process batch and calculate rank-weighted InfoNCE loss
        
        Args:
            model: QA Embedding model
            batch: Batch from infonce or multiple_positives transform
            device: Device to run on
            
        Returns:
            loss: Loss tensor with gradient
            metrics: Dictionary with accuracy metrics
        """
        # Move tensors to device
        q_input_ids = batch['q_input_ids'].to(device)
        q_attention_mask = batch['q_attention_mask'].to(device)
        a_input_ids = batch['a_input_ids'].to(device)
        a_attention_mask = batch['a_attention_mask'].to(device)
        question_ids = batch['question_ids']
        
        # Get embeddings
        q_embeddings = model(q_input_ids, q_attention_mask)
        a_embeddings = model(a_input_ids, a_attention_mask)
        
        # Extract rank/score information if present and needed
        kwargs = {}
        if self.use_ranks and 'ranks' in batch:
            kwargs['ranks'] = batch['ranks'].to(device)
        if self.use_scores and 'scores' in batch:
            kwargs['scores'] = batch['scores'].to(device)
        
        # Calculate loss using core loss function
        loss, metrics = self.core_loss(
            q_embeddings, a_embeddings, 
            question_ids=question_ids,
            **kwargs
        )
        
        return loss, metrics


class UnifiedRelaxedHardNegativeLoss(UnifiedLoss):
    """
    Relaxed hard negative InfoNCE loss adapter.
    Works with:
    - hard_negative_batch_transform: Each question with multiple answers
      Format: List of dicts, each with {'q_input_ids', 'answers', 'question_id'}
      where answers is a list of {'input_ids', 'attention_mask', 'id', 'score', 'rank'}
    """
    
    def __init__(self, temperature=0.1, hard_neg_target_prob=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_target_prob = hard_neg_target_prob
        self.name = f"unified_relaxed_hn_t{temperature}_p{hard_neg_target_prob}"
        
        # Create wrapped loss function
        self.core_loss = RelaxedHardNegativeInfoNCELoss(
            temperature=temperature,
            hard_neg_target_prob=hard_neg_target_prob
        )
        
    def forward(self, model, batch_items, device):
        """
        Process hard negative batch and calculate loss
        
        Args:
            model: QA Embedding model
            batch_items: List of items from hard_negative_batch_transform
            device: Device to run on
            
        Returns:
            loss: Loss tensor with gradient
            metrics: Dictionary with accuracy metrics
        """
        # Initialize loss and metrics accumulators
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_metrics = {'loss': 0.0, 'acc': 0.0, 'groups': 0}
        total_items = 0
        
        # Process each question with its answers
        for item in batch_items:
            # Get question embedding
            q_input_ids = item['q_input_ids'].unsqueeze(0).to(device)
            q_attention_mask = item['q_attention_mask'].unsqueeze(0).to(device)
            q_embedding = model(q_input_ids, q_attention_mask)
            
            # Process all answers for this question
            a_embeddings = []
            question_ids = []
            
            for answer in item['answers']:
                a_input_ids = answer['input_ids'].unsqueeze(0).to(device)
                a_attention_mask = answer['attention_mask'].unsqueeze(0).to(device)
                a_embedding = model(a_input_ids, a_attention_mask)
                a_embeddings.append(a_embedding)
                question_ids.append(item['question_id'])
            
            # Stack answer embeddings
            if a_embeddings:
                a_embeddings = torch.cat(a_embeddings, dim=0)
                
                # Create repeated q_embedding to match a_embeddings
                q_repeated = q_embedding.repeat(len(a_embeddings), 1)
                
                # Calculate loss for this item
                item_loss, item_metrics = self.core_loss(
                    q_repeated, a_embeddings, question_ids=question_ids
                )
                
                # Accumulate loss and metrics
                if total_loss.item() == 0 and not total_loss.requires_grad:
                    # First item, replace the tensor
                    total_loss = item_loss
                else:
                    # Add to the existing tensor
                    total_loss = total_loss + item_loss
                
                for k, v in item_metrics.items():
                    total_metrics[k] += v
                
                total_items += 1
        
        # Average metrics
        if total_items > 0:
            loss = total_loss / total_items
            for k in total_metrics:
                total_metrics[k] /= total_items
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure loss is in metrics dict
        total_metrics['loss'] = loss.item()
        
        return loss, total_metrics


class UnifiedTripletLoss(UnifiedLoss):
    """
    Triplet loss adapter.
    Works with:
    - triplet_batch_transform: Query, positive, negative triplets
      Format: {'q_input_ids', 'a_pos_input_ids', 'a_neg_input_ids', 
              'q_attention_mask', 'a_pos_attention_mask', 'a_neg_attention_mask'}
    """
    
    def __init__(self, margin=0.3, **kwargs):
        super().__init__()
        self.margin = margin
        self.name = f"unified_triplet_m{margin}"
        
        # Create wrapped loss function
        self.core_loss = TripletLoss(margin=margin)
        
    def forward(self, model, batch, device):
        """
        Process triplet batch and calculate loss
        
        Args:
            model: QA Embedding model
            batch: Batch from triplet_batch_transform
            device: Device to run on
            
        Returns:
            loss: Loss tensor with gradient
            metrics: Dictionary with accuracy and similarity metrics
        """
        # Move tensors to device
        q_input_ids = batch['q_input_ids'].to(device)
        q_attention_mask = batch['q_attention_mask'].to(device)
        a_pos_input_ids = batch['a_pos_input_ids'].to(device)
        a_pos_attention_mask = batch['a_pos_attention_mask'].to(device)
        a_neg_input_ids = batch['a_neg_input_ids'].to(device)
        a_neg_attention_mask = batch['a_neg_attention_mask'].to(device)
        
        # Get embeddings
        q_embeddings = model(q_input_ids, q_attention_mask)
        a_pos_embeddings = model(a_pos_input_ids, a_pos_attention_mask)
        a_neg_embeddings = model(a_neg_input_ids, a_neg_attention_mask)
        
        # Calculate loss using core loss function
        loss, metrics = self.core_loss(
            q_embeddings, a_pos_embeddings, a_neg_embeddings
        )
        
        return loss, metrics


class UnifiedListwiseRankingLoss(UnifiedLoss):
    """
    Listwise ranking loss adapter.
    Works with:
    - listwise_batch_transform: Each question with multiple scored answers
      Format: List of dicts, each with {'q_input_ids', 'a_input_ids', 
              'q_attention_mask', 'a_attention_masks', 'scores', 'question_id'}
    """
    
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.name = f"unified_listwise_t{temperature}"
        
        # Create wrapped loss function
        self.core_loss = ListwiseRankingLoss(temperature=temperature)
        
    def forward(self, model, batch_items, device):
        """
        Process listwise batch and calculate loss
        
        Args:
            model: QA Embedding model
            batch_items: List of items from listwise_batch_transform
            device: Device to run on
            
        Returns:
            loss: Loss tensor with gradient
            metrics: Dictionary with NDCG and other metrics
        """
        # Prepare data for listwise loss
        q_embeddings = []
        a_list_embeddings = []
        a_list_scores = []
        
        # Process each item
        for item in batch_items:
            # Get question embedding
            q_input_ids = item['q_input_ids'].unsqueeze(0).to(device)
            q_attention_mask = item['q_attention_mask'].unsqueeze(0).to(device)
            q_embedding = model(q_input_ids, q_attention_mask)
            
            # Get answer embeddings
            a_input_ids = item['a_input_ids'].to(device)
            a_attention_masks = item['a_attention_masks'].to(device)
            a_embeddings = model(a_input_ids, a_attention_masks)
            
            # Add to lists
            q_embeddings.append(q_embedding)
            a_list_embeddings.append(a_embeddings)
            a_list_scores.append(item['scores'].to(device))
        
        # Skip empty batch
        if not q_embeddings:
            return torch.tensor(0.0, device=device, requires_grad=True), {'loss': 0.0, 'ndcg': 0.0}
        
        # Calculate loss using core loss function
        loss, metrics = self.core_loss(
            q_embeddings, a_list_embeddings, a_list_scores
        )
        
        return loss, metrics


def create_unified_loss(loss_name, **kwargs):
    """
    Factory function to create a unified loss by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss
        
    Returns:
        UnifiedLoss: Loss function instance
        
    Raises:
        ValueError: If the loss name is not recognized
    """
    # Define mappings from loss names to unified loss classes
    losses = {
        'infonce': UnifiedInfoNCELoss,
        'rank_infonce': UnifiedRankInfoNCELoss,
        'relaxed_hard_neg': UnifiedRelaxedHardNegativeLoss,
        'triplet': UnifiedTripletLoss,
        'listwise': UnifiedListwiseRankingLoss
    }
    
    # Handle named variants with parameters embedded in the name
    # rank_infonce_rank0.1_score0.05_t0.2
    if loss_name.startswith('rank_infonce_'):
        options = loss_name.split('_')[2:] # Skip 'rank_infonce'
        params = {
            'temperature': kwargs.get('temperature', 0.1),
            'use_ranks': False,
            'use_scores': False,
            'rank_weight': 0.1,
            'score_weight': 0.01
        }
        
        for option in options:
            if option.startswith('rank'):
                params['use_ranks'] = True
                try: params['rank_weight'] = float(option[4:])
                except (ValueError, IndexError): pass
            elif option.startswith('score'):
                params['use_scores'] = True
                try: params['score_weight'] = float(option[5:])
                except (ValueError, IndexError): pass
            elif option.startswith('t'):
                try: params['temperature'] = float(option[1:])
                except (ValueError, IndexError): pass
        
        params.update(kwargs) # Allow explicit kwargs to override parsed ones
        return UnifiedRankInfoNCELoss(**params)
    
    # relaxed_hard_neg_p0.2_t0.1
    if loss_name.startswith('relaxed_hard_neg_'):
        options = loss_name.split('_')[3:] # Skip 'relaxed_hard_neg'
        params = {
            'temperature': kwargs.get('temperature', 0.1),
            'hard_neg_target_prob': kwargs.get('hard_neg_target_prob', 0.1)
        }
        
        for option in options:
            if option.startswith('t'):
                try: params['temperature'] = float(option[1:])
                except (ValueError, IndexError): pass
            elif option.startswith('p'):
                try: params['hard_neg_target_prob'] = float(option[1:])
                except (ValueError, IndexError): pass
        
        params.update(kwargs)
        return UnifiedRelaxedHardNegativeLoss(**params)
    
    # Try simple lookup
    if loss_name in losses:
        return losses[loss_name](**kwargs)
    
    # If no match found
    available = list(losses.keys()) + ["rank_infonce_...", "relaxed_hard_neg_..."]
    raise ValueError(f"Unknown loss function: {loss_name}. Available losses: {available}")
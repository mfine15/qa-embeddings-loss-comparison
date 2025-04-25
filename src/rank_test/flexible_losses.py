#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flexible loss functions that work with various data sampling strategies.
These losses are designed to work with the output of the flexible dataset module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from collections import defaultdict


class BaseLoss(nn.Module):
    """Base class for all loss functions"""
    
    def __init__(self, **kwargs):
        super(BaseLoss, self).__init__()
        self.name = "base_loss"
        
    def forward(self, q_embeddings, a_embeddings, **kwargs):
        """
        Calculate loss
        
        Args:
            q_embeddings: Tensor of question embeddings (batch_size, embed_dim)
            a_embeddings: Tensor of answer embeddings (batch_size, embed_dim)
            **kwargs: Additional loss-specific parameters
            
        Returns:
            loss: Loss value
            metrics: Dictionary of metrics for logging
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_name(self):
        """Get loss function name for logging"""
        return self.name


class StandardInfoNCELoss(BaseLoss):
    """Standard InfoNCE loss with in-batch negatives"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.name = f"std_infonce_t{temperature}"
        
    def forward(self, q_embeddings, a_embeddings, **kwargs):
        """
        Calculate standard InfoNCE loss with in-batch negatives
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            
        Returns:
            loss: InfoNCE loss
            metrics: Dictionary with accuracy metrics
        """
        batch_size = q_embeddings.shape[0]
        
        # Calculate similarity matrix
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        
        # Labels are on the diagonal (each question matched with its answer)
        labels = torch.arange(batch_size, device=q_embeddings.device)
        
        # Compute loss (bidirectional)
        loss_q2a = F.cross_entropy(similarity, labels)
        loss_a2q = F.cross_entropy(similarity.T, labels)
        loss = (loss_q2a + loss_a2q) / 2
        
        # Calculate accuracy metrics
        q2a_preds = torch.argmax(similarity, dim=1)
        a2q_preds = torch.argmax(similarity.T, dim=1)
        q2a_acc = (q2a_preds == labels).float().mean()
        a2q_acc = (a2q_preds == labels).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'q2a_acc': q2a_acc.item(),
            'a2q_acc': a2q_acc.item(),
            'avg_acc': (q2a_acc.item() + a2q_acc.item()) / 2
        }
        
        return loss, metrics


class RankInfoNCELoss(BaseLoss):
    """
    InfoNCE loss that leverages rank or score information from the dataset.
    Works with standard batch format or multiple positives format.
    
    Can use either:
    - Ordinal ranks (position in the ranked list, 0 is best)
    - Cardinal scores (actual score values, higher is better)
    """
    
    def __init__(self, temperature=0.1, use_ranks=True, use_scores=False, 
                 rank_weight=0.1, score_weight=0.01):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarity scores
            use_ranks: Whether to use ordinal rank information in loss calculation
            use_scores: Whether to use cardinal score information in loss calculation
            rank_weight: Weight for rank-based penalties (higher ranks get lower weight)
            score_weight: Weight for score-based adjustments (higher scores get higher weight)
        """
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
        self.name = f"infonce_{ranking_str}_t{temperature}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids=None, ranks=None, scores=None, **kwargs):
        """
        Calculate InfoNCE loss with rank/score-based weighting
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: Question IDs to identify answers to the same question
            ranks: Ordinal rank of each answer (0 is highest ranked)
            scores: Cardinal score of each answer (higher is better)
            
        Returns:
            loss: Weighted InfoNCE loss
            metrics: Dictionary with accuracy and other metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        # Calculate similarity matrix
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        
        # Standard InfoNCE loss labels
        labels = torch.arange(batch_size, device=device)
        
        # Track whether we've applied any adjustments
        adjusted = False
        sim_weights = torch.ones_like(similarity)
        
        # Group answers by question if needed for adjustments
        if (self.use_ranks or self.use_scores) and question_ids is not None:
            question_groups = defaultdict(list)
            for i, q_id in enumerate(question_ids):
                item = [i]  # Always include index
                
                if self.use_ranks and ranks is not None:
                    item.append(ranks[i].item())  # Add rank
                else:
                    item.append(None)  # Placeholder
                    
                if self.use_scores and scores is not None:
                    item.append(scores[i].item())  # Add score
                else:
                    item.append(None)  # Placeholder
                    
                question_groups[q_id].append(item)
                
            # Apply adjustments for questions with multiple answers
            for q_id, items in question_groups.items():
                if len(items) <= 1:
                    continue  # Skip questions with only one answer
                
                # Apply rank-based adjustments
                if self.use_ranks and ranks is not None:
                    adjusted = True
                    for i, item1 in enumerate(items):
                        idx1, rank1, _ = item1
                        for j, item2 in enumerate(items):
                            if i == j:
                                continue  # Skip self
                                
                            idx2, rank2, _ = item2
                            # Apply penalty based on rank
                            # Higher ranks (worse answers) get lower weight
                            if rank2 is not None:
                                rank_penalty = 1.0 - self.rank_weight * rank2
                                sim_weights[idx1, idx2] *= rank_penalty
                
                # Apply score-based adjustments
                if self.use_scores and scores is not None:
                    adjusted = True
                    
                    # Find max score for this question (for normalization)
                    max_score = max(item[2] for item in items if item[2] is not None)
                    if max_score <= 0:
                        continue  # Skip if no positive scores
                        
                    for i, item1 in enumerate(items):
                        idx1, _, score1 = item1
                        for j, item2 in enumerate(items):
                            if i == j:
                                continue  # Skip self
                                
                            idx2, _, score2 = item2
                            # Apply weight based on normalized score
                            # Higher scores get higher weight
                            if score2 is not None and max_score > 0:
                                normalized_score = score2 / max_score
                                score_weight = 1.0 + self.score_weight * normalized_score
                                sim_weights[idx1, idx2] *= score_weight
        
        # Compute loss
        if adjusted:
            # Apply weights to similarity
            adjusted_similarity = similarity * sim_weights
            
            # Compute loss with adjusted similarity
            loss_q2a = F.cross_entropy(adjusted_similarity, labels)
            loss_a2q = F.cross_entropy(adjusted_similarity.T, labels)
            loss = (loss_q2a + loss_a2q) / 2
        else:
            # Standard InfoNCE without adjustments
            loss_q2a = F.cross_entropy(similarity, labels)
            loss_a2q = F.cross_entropy(similarity.T, labels)
            loss = (loss_q2a + loss_a2q) / 2
        
        # Calculate accuracy metrics
        q2a_preds = torch.argmax(similarity, dim=1)
        a2q_preds = torch.argmax(similarity.T, dim=1)
        q2a_acc = (q2a_preds == labels).float().mean()
        a2q_acc = (a2q_preds == labels).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'q2a_acc': q2a_acc.item(),
            'a2q_acc': a2q_acc.item(),
            'avg_acc': (q2a_acc.item() + a2q_acc.item()) / 2,
            'used_adjustment': adjusted
        }
        
        return loss, metrics


class HardNegativeInfoNCELoss(BaseLoss):
    """
    InfoNCE loss with additional penalty for hard negatives.
    Works with both standard batches and hard negative batches.
    """
    
    def __init__(self, temperature=0.1, hard_negative_weight=1.0):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarity scores
            hard_negative_weight: Weight for the hard negative component of the loss
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.name = f"hard_neg_infonce_t{temperature}_w{hard_negative_weight}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids=None, **kwargs):
        """
        Calculate InfoNCE loss with hard negative penalty
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: Question IDs to identify hard negatives
            
        Returns:
            loss: Combined InfoNCE loss with hard negative penalty
            metrics: Dictionary with accuracy and hard negative metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        # Standard InfoNCE loss with in-batch negatives
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        labels = torch.arange(batch_size, device=device)
        base_loss = F.cross_entropy(similarity, labels)
        
        # Additional penalty for hard negatives (answers to the same question)
        hard_negative_loss = torch.tensor(0.0, device=device)
        hard_neg_count = 0
        
        if question_ids is not None:
            # Group by question ID
            question_groups = defaultdict(list)
            for i, q_id in enumerate(question_ids):
                question_groups[q_id].append(i)
            
            # Calculate hard negative loss
            for q_id, indices in question_groups.items():
                if len(indices) <= 1:
                    continue  # Skip questions with only one answer
                
                # For each pair of answers to the same question
                for i, idx1 in enumerate(indices):
                    for j, idx2 in enumerate(indices):
                        if i != j:  # Different answers to same question
                            # Penalize high similarity between different answers to same question
                            sim = similarity[idx1, idx2]
                            hard_negative_loss += torch.exp(sim)
                            hard_neg_count += 1
        
        # Add weighted hard negative loss if found
        total_loss = base_loss
        if hard_neg_count > 0:
            hard_negative_loss = hard_negative_loss / hard_neg_count
            total_loss += self.hard_negative_weight * hard_negative_loss
        
        # Calculate metrics
        preds = torch.argmax(similarity, dim=1)
        accuracy = (preds == labels).float().mean()
        
        metrics = {
            'loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'hard_neg_loss': hard_negative_loss.item() if isinstance(hard_negative_loss, torch.Tensor) else 0,
            'accuracy': accuracy.item(),
            'hard_neg_count': hard_neg_count
        }
        
        return total_loss, metrics


class MultiplePositivesLoss(BaseLoss):
    """
    Loss function that handles multiple positive answers per question.
    Works with the multiple_positives batch transform.
    """
    
    def __init__(self, temperature=0.1, rank_weight=0.1):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarity scores
            rank_weight: Weight for rank-based penalties (higher ranks get lower weight)
        """
        super().__init__()
        self.temperature = temperature
        self.rank_weight = rank_weight
        self.name = f"multi_pos_t{temperature}_w{rank_weight}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids, ranks=None, **kwargs):
        """
        Calculate loss with multiple positives per question
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: Question IDs to identify answers to the same question
            ranks: Rank of each answer (optional)
            
        Returns:
            loss: Combined loss accounting for multiple positives
            metrics: Dictionary with accuracy and other metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        # Calculate similarity matrix
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        
        # Group by question ID
        question_groups = defaultdict(list)
        for i, q_id in enumerate(question_ids):
            question_groups[q_id].append(i)
        
        # Calculate loss
        total_loss = torch.tensor(0.0, device=device)
        correct_count = 0
        total_count = 0
        
        # For each question
        for q_id, indices in question_groups.items():
            if len(indices) <= 0:
                continue
                
            for i, idx in enumerate(indices):
                # Get query embedding
                q_embed = q_embeddings[idx]
                
                # Calculate similarity to all answers
                sims = torch.matmul(q_embed.unsqueeze(0), a_embeddings.T).squeeze(0)
                
                # Create mask for positive samples (answers to same question)
                pos_mask = torch.zeros(batch_size, device=device)
                for pos_idx in indices:
                    weight = 1.0
                    # Apply rank weighting if available
                    if ranks is not None:
                        # Higher ranked answers get higher weight
                        weight = 1.0 - self.rank_weight * ranks[pos_idx].item()
                    pos_mask[pos_idx] = weight
                
                # Create masked loss - encourage high similarity with positives
                neg_mask = 1.0 - (pos_mask > 0).float()
                
                # Compute InfoNCE-style loss for this query
                numerator = torch.sum(pos_mask * torch.exp(sims))
                denominator = torch.sum(torch.exp(sims))
                loss_i = -torch.log(numerator / denominator + 1e-8)
                total_loss += loss_i
                
                # Calculate accuracy
                pred = torch.argmax(sims).item()
                if pos_mask[pred] > 0:
                    correct_count += 1
                total_count += 1
        
        # Average loss
        loss = total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=device)
        
        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'total_samples': total_count
        }
        
        return loss, metrics


class TripletLoss(BaseLoss):
    """
    Triplet loss for query-positive-negative triplets.
    Works with the triplet batch transform.
    """
    
    def __init__(self, margin=0.3):
        """
        Initialize the triplet loss.
        
        Args:
            margin: Margin between positive and negative similarities
        """
        super().__init__()
        self.margin = margin
        self.name = f"triplet_m{margin}"
        
    def forward(self, q_embeddings, a_pos_embeddings, a_neg_embeddings, **kwargs):
        """
        Calculate triplet loss
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_pos_embeddings: Positive answer embeddings (batch_size, embed_dim)
            a_neg_embeddings: Negative answer embeddings (batch_size, embed_dim)
            
        Returns:
            loss: Triplet loss
            metrics: Dictionary with accuracy and similarity metrics
        """
        # Calculate similarity between queries and answers
        pos_sim = torch.sum(q_embeddings * a_pos_embeddings, dim=1)
        neg_sim = torch.sum(q_embeddings * a_neg_embeddings, dim=1)
        
        # Calculate triplet loss: enforce margin between pos_sim and neg_sim
        losses = F.relu(neg_sim - pos_sim + self.margin)
        loss = torch.mean(losses)
        
        # Calculate accuracy (how often pos_sim > neg_sim)
        acc = (pos_sim > neg_sim).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'acc': acc.item(),
            'avg_pos_sim': pos_sim.mean().item(),
            'avg_neg_sim': neg_sim.mean().item(),
            'margin_violations': (losses > 0).float().mean().item()
        }
        
        return loss, metrics


class ListwiseRankingLoss(BaseLoss):
    """
    Listwise ranking loss for learning to rank multiple answers.
    Works with the listwise batch transform.
    """
    
    def __init__(self, temperature=1.0):
        """
        Initialize the listwise ranking loss.
        
        Args:
            temperature: Temperature for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
        self.name = f"listwise_t{temperature}"
        
    def forward(self, q_embeddings, a_list_embeddings, a_list_scores, **kwargs):
        """
        Calculate listwise ranking loss
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
                          or list of embeddings, one per question
            a_list_embeddings: List of answer embeddings tensors
                              [(num_answers, embed_dim), ...]
            a_list_scores: List of scores for each answer per question
                          [(num_answers,), ...]
            
        Returns:
            loss: Listwise ranking loss
            metrics: Dictionary with NDCG and other metrics
        """
        # Determine if we're using a device (for GPU support)
        if isinstance(q_embeddings, list):
            device = q_embeddings[0].device if q_embeddings else torch.device('cpu')
        else:
            device = q_embeddings.device
            
        total_loss = torch.tensor(0.0, device=device)
        total_ndcg = 0.0
        
        # Check if we have a batch or individual questions
        if isinstance(q_embeddings, list):
            q_embeds = q_embeddings
        else:
            # Assume we have a batch with one embedding per question
            q_embeds = [q_embeddings[i].unsqueeze(0) for i in range(q_embeddings.shape[0])]
        
        # Make sure all lists have the same length
        min_length = min(len(q_embeds), len(a_list_embeddings), len(a_list_scores))
        
        # Process each question separately
        for i in range(min_length):
            q_embed = q_embeds[i]
            answers_embed = a_list_embeddings[i]
            scores = a_list_scores[i]
            
            # Calculate similarity between question and all answers
            sim = torch.matmul(q_embed, answers_embed.T).squeeze(0)  # (num_answers,)
            sim = sim / self.temperature
            
            # Convert to probabilities
            sim_probs = F.softmax(sim, dim=0)  # (num_answers,)
            
            # Create target probabilities from normalized scores
            target_probs = F.softmax(scores / self.temperature, dim=0)  # (num_answers,)
            
            # KL divergence loss
            loss_i = F.kl_div(torch.log(sim_probs + 1e-8), target_probs, reduction='sum')
            total_loss += loss_i
            
            # Calculate NDCG
            # Sort predicted and target rankings
            _, pred_indices = torch.sort(sim, descending=True)
            _, ideal_indices = torch.sort(scores, descending=True)
            
            # Calculate DCG
            pred_dcg = self._calculate_dcg(pred_indices, scores)
            ideal_dcg = self._calculate_dcg(ideal_indices, scores)
            
            # Calculate NDCG
            ndcg = pred_dcg / (ideal_dcg + 1e-8)
            total_ndcg += ndcg
        
        # Average loss and metrics
        batch_size = min_length
        loss = total_loss / batch_size if batch_size > 0 else total_loss
        avg_ndcg = total_ndcg / batch_size if batch_size > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'ndcg': avg_ndcg
        }
        
        return loss, metrics
    
    def _calculate_dcg(self, indices, scores):
        """Helper function to calculate DCG"""
        device = indices.device
        ranks = torch.arange(1, len(indices) + 1, dtype=torch.float, device=device)
        gain = scores[indices]
        return torch.sum(gain / torch.log2(ranks + 1))


# Factory function to create a loss by name
def create_flexible_loss(loss_name, **kwargs):
    """
    Factory function to create a loss by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If the loss name is not recognized
    """
    # Basic loss functions
    losses = {
        'infonce': StandardInfoNCELoss,
        'rank_infonce': RankInfoNCELoss,
        'hard_negative': HardNegativeInfoNCELoss,
        'multiple_positives': MultiplePositivesLoss,
        'triplet': TripletLoss,
        'listwise': ListwiseRankingLoss
    }
    
    # Handle specialized versions of InfoNCE
    if loss_name.startswith('infonce_'):
        # Parse options from name
        options = loss_name.split('_')[1:]
        
        # Default parameters
        params = {
            'temperature': kwargs.get('temperature', 0.1),
            'use_ranks': False,
            'use_scores': False,
            'rank_weight': 0.1,
            'score_weight': 0.01
        }
        
        # Update based on options
        for option in options:
            if option.startswith('rank'):
                params['use_ranks'] = True
                try:
                    params['rank_weight'] = float(option[4:])
                except (ValueError, IndexError):
                    pass
            elif option.startswith('score'):
                params['use_scores'] = True
                try:
                    params['score_weight'] = float(option[5:])
                except (ValueError, IndexError):
                    pass
            elif option.startswith('t'):
                try:
                    params['temperature'] = float(option[1:])
                except (ValueError, IndexError):
                    pass
        
        # Update with provided kwargs (override parsed values)
        params.update(kwargs)
        
        # Create RankInfoNCE with parsed parameters
        return RankInfoNCELoss(**params)
    
    # For standard loss names
    if loss_name not in losses:
        raise ValueError(f"Unknown loss function: {loss_name}. Available losses: {list(losses.keys())}")
    
    return losses[loss_name](**kwargs)
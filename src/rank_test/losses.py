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

# Define base loss classes

class BaseLoss(nn.Module):
    """Base class for all loss functions"""
    
    def __init__(self, **kwargs):
        super(BaseLoss, self).__init__()
        self.name = "base_loss"
        
    def forward(self, **kwargs):
        """
        Calculate loss
            
        Returns:
            loss: Loss value
            metrics: Dictionary of metrics for logging
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_name(self):
        """Get loss function name for logging"""
        return self.name


class FlexibleInfoNCELoss(BaseLoss):
    """
    Flexible InfoNCE loss that handles single or multiple positives per query.
    Relies on question_ids to identify positive pairs.
    Assumes batch is structured such that for a given question_id, all its 
    corresponding items (q_embeddings and a_embeddings) are positives for each other.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.name = f"flex_infonce_t{temperature}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids, **kwargs):
        """
        Calculate flexible InfoNCE loss.
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: List or Tensor of question IDs corresponding to each q/a pair.
                          Items with the same question_id are considered positives for each other.
            
        Returns:
            loss: InfoNCE loss value
            metrics: Dictionary with accuracy metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        if batch_size <= 1:
             return torch.tensor(0.0, device=device, requires_grad=True), {'loss': 0.0, 'acc': 0.0, 'groups': 0}

        # Calculate similarity matrix (Q x A)
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        
        # Group indices by question ID
        question_groups = defaultdict(list)
        for i, q_id in enumerate(question_ids):
            # Ensure q_id is hashable (e.g., convert tensor to int/str if needed)
            hashable_q_id = q_id.item() if isinstance(q_id, torch.Tensor) else q_id
            question_groups[hashable_q_id].append(i)

        total_loss = torch.tensor(0.0, device=device)
        correct_count = 0
        group_count = 0

        # Calculate loss per query
        for q_id, indices in question_groups.items():
            if len(indices) == 0:
                continue
            
            group_count += 1
            
            # Loss for each item in the group (treating others in the group as positive)
            for i in indices:
                # Log-softmax scores for the current query q_i against all answers
                log_probs = F.log_softmax(similarity[i], dim=0)
                
                # Positive indices for this query are the *other* items in the same group
                # If only one item for this q_id, it's contrasted against all others (standard InfoNCE)
                positive_indices = indices # All items from the same question are positives
                
                if len(positive_indices) > 0:
                    # Average the negative log probabilities of positive targets
                    loss_i = -log_probs[positive_indices].mean()
                    total_loss += loss_i

                # Accuracy: Check if the highest scoring item is one of the positives
                # Note: This definition might need adjustment based on specific eval needs
                # For simplicity, we check if the argmax is *any* of the items from the same group.
                if len(indices) > 0: # Only calculate accuracy if positives exist
                    pred_idx = torch.argmax(similarity[i]).item()
                    if pred_idx in indices:
                         correct_count += 1 # Count prediction for this item

        # Average loss over all items in the batch
        loss = total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)
        
        # Average accuracy over all items
        accuracy = correct_count / batch_size if batch_size > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'acc': accuracy,
            'groups': group_count # Number of unique questions processed
        }
        
        return loss, metrics


class RankInfoNCELoss(FlexibleInfoNCELoss):
    """
    InfoNCE loss that weights positive examples based on rank or score.
    Inherits from FlexibleInfoNCELoss and modifies the positive target weighting.
    Lower ranks (better) or higher scores (better) receive higher weight.
    """
    
    def __init__(self, temperature=0.1, use_ranks=True, use_scores=False, 
                 rank_weight=0.1, score_weight=0.01):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarity scores
            use_ranks: Whether to use ordinal rank information (lower is better)
            use_scores: Whether to use cardinal score information (higher is better)
            rank_weight: Controls how much weight decreases for higher ranks.
                         Weight = exp(-rank_weight * rank)
            score_weight: Controls how much weight increases for higher scores.
                          Weight = exp(score_weight * normalized_score)
        """
        super().__init__(temperature=temperature)
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
        self.name = f"rank_infonce_{ranking_str}_t{temperature}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids, ranks=None, scores=None, **kwargs):
        """
        Calculate rank/score-weighted InfoNCE loss.
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: List or Tensor of question IDs.
            ranks: Ordinal rank of each answer (0 is highest ranked). Tensor expected.
            scores: Cardinal score of each answer (higher is better). Tensor expected.
            
        Returns:
            loss: Weighted InfoNCE loss
            metrics: Dictionary with accuracy and other metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        if batch_size <= 1:
             return torch.tensor(0.0, device=device, requires_grad=True), {'loss': 0.0, 'acc': 0.0, 'groups': 0}

        # Calculate similarity matrix (Q x A)
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        
        # Group indices by question ID
        question_groups = defaultdict(list)
        group_data = defaultdict(lambda: {'indices': [], 'ranks': [], 'scores': []})
        
        for i, q_id in enumerate(question_ids):
            hashable_q_id = q_id.item() if isinstance(q_id, torch.Tensor) else q_id
            group_data[hashable_q_id]['indices'].append(i)
            if self.use_ranks and ranks is not None:
                group_data[hashable_q_id]['ranks'].append(ranks[i].item())
            if self.use_scores and scores is not None:
                group_data[hashable_q_id]['scores'].append(scores[i].item())

        total_loss = torch.tensor(0.0, device=device)
        correct_count = 0
        group_count = 0

        # Calculate loss per query
        for q_id, data in group_data.items():
            indices = data['indices']
            if len(indices) == 0:
                continue
                
            group_count += 1
            
            group_ranks = torch.tensor(data['ranks'], device=device) if self.use_ranks and data['ranks'] else None
            group_scores = torch.tensor(data['scores'], device=device) if self.use_scores and data['scores'] else None
            
            # Calculate weights for positives within this group
            pos_weights = torch.ones(len(indices), device=device)
            
            if self.use_ranks and group_ranks is not None and len(group_ranks) == len(indices):
                # Exponential decay for ranks: lower rank (better) -> higher weight
                pos_weights *= torch.exp(-self.rank_weight * group_ranks)
                
            if self.use_scores and group_scores is not None and len(group_scores) == len(indices):
                # Normalize scores within the group (optional, but helps scale weight)
                max_score = torch.max(group_scores)
                min_score = torch.min(group_scores)
                if max_score > min_score:
                     norm_scores = (group_scores - min_score) / (max_score - min_score)
                else:
                     norm_scores = torch.zeros_like(group_scores) # Handle case where all scores are the same

                # Exponential increase for scores: higher score -> higher weight
                pos_weights *= torch.exp(self.score_weight * norm_scores)

            # Normalize weights to sum to 1 (important for weighted average)
            if torch.sum(pos_weights) > 0:
                 pos_weights /= torch.sum(pos_weights)
            else:
                 pos_weights = torch.ones_like(pos_weights) / len(pos_weights) # Equal weight if sum is zero

            # Loss for each item in the group
            for i_idx, i in enumerate(indices): # i_idx is the index within the group, i is the global index
                log_probs = F.log_softmax(similarity[i], dim=0)
                
                # Positive indices for this query are all items in the same group
                positive_indices = indices 
                
                if len(positive_indices) > 0:
                    # Weighted average of negative log probabilities of positive targets
                    # We use the weights calculated for the *targets* (positives)
                    loss_i = -torch.sum(pos_weights * log_probs[positive_indices])
                    total_loss += loss_i

                # Accuracy: Check if the highest scoring item is one of the positives
                if len(indices) > 0:
                    pred_idx = torch.argmax(similarity[i]).item()
                    if pred_idx in indices:
                         correct_count += 1

        # Average loss over all items
        loss = total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)
        accuracy = correct_count / batch_size if batch_size > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'acc': accuracy,
            'groups': group_count
        }
        
        return loss, metrics


class RelaxedHardNegativeInfoNCELoss(FlexibleInfoNCELoss):
    """
    InfoNCE loss variant that reduces the penalty for confusing positives
    with hard negatives (other answers from the same question).
    Uses KL divergence with a soft target distribution where hard negatives
    receive a small target probability.
    """
    
    def __init__(self, temperature=0.1, hard_neg_target_prob=0.1):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarity scores.
            hard_neg_target_prob: Target probability mass to distribute among 
                                  hard negatives for a given query. Should be < 1.0.
        """
        super().__init__(temperature=temperature)
        if not (0.0 <= hard_neg_target_prob < 1.0):
             raise ValueError("hard_neg_target_prob must be between 0.0 and 1.0 (exclusive of 1.0)")
        self.hard_neg_target_prob = hard_neg_target_prob
        self.name = f"relaxed_hn_infonce_t{temperature}_p{hard_neg_target_prob}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids, **kwargs):
        """
        Calculate relaxed hard negative InfoNCE loss using KL divergence.
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: List or Tensor of question IDs.
            
        Returns:
            loss: Relaxed hard negative loss value
            metrics: Dictionary with accuracy metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        if batch_size <= 1:
             return torch.tensor(0.0, device=device, requires_grad=True), {'loss': 0.0, 'acc': 0.0, 'groups': 0}

        # Calculate similarity matrix (Q x A)
        similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
        
        # Group indices by question ID
        question_groups = defaultdict(list)
        for i, q_id in enumerate(question_ids):
            hashable_q_id = q_id.item() if isinstance(q_id, torch.Tensor) else q_id
            question_groups[hashable_q_id].append(i)

        total_loss = torch.tensor(0.0, device=device)
        correct_count = 0
        group_count = 0

        # Calculate loss per query using KL divergence
        for q_id, indices in question_groups.items():
            if len(indices) == 0:
                continue
            
            group_count += 1
            
            # Loss for each item `i` in the group
            for i in indices:
                # Identify positive and hard negative indices for query i
                # In this setup, all items from the same group are treated symmetrically.
                # We consider 'i' as the anchor/positive for this specific calculation,
                # and the *other* items in the group as hard negatives.
                positive_indices = [i] # Anchor item is the positive target
                hard_negative_indices = [j for j in indices if j != i]
                
                # Create the soft target distribution for query i
                target_probs = torch.zeros(batch_size, device=device)
                
                # Assign probability to the positive anchor
                positive_mass = 1.0 - self.hard_neg_target_prob
                target_probs[positive_indices] = positive_mass / len(positive_indices) # Should always be len 1 here

                # Distribute remaining probability mass among hard negatives
                if hard_negative_indices and self.hard_neg_target_prob > 0:
                     target_probs[hard_negative_indices] = self.hard_neg_target_prob / len(hard_negative_indices)
                
                # Ensure target sums to 1 (handles edge case of no hard negs)
                if not torch.isclose(torch.sum(target_probs), torch.tensor(1.0, device=device)):
                    # Fallback if no hard negs, put all mass on positive
                     target_probs = torch.zeros(batch_size, device=device)
                     target_probs[positive_indices] = 1.0 / len(positive_indices)

                # Calculate KL divergence loss for query i
                # log_softmax of similarities vs target distribution
                log_probs = F.log_softmax(similarity[i], dim=0)
                loss_i = F.kl_div(log_probs, target_probs, reduction='sum') # kl_div expects log input
                total_loss += loss_i

                # Accuracy: Check if the highest scoring item is the anchor 'i'
                # (Stricter than FlexibleInfoNCE, checks if the *intended* positive wins)
                if len(indices) > 0:
                    pred_idx = torch.argmax(similarity[i]).item()
                    if pred_idx == i: # Check if prediction matches the anchor
                         correct_count += 1

        # Average loss over all items in the batch
        loss = total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)
        
        # Average accuracy over all items
        # Note: Accuracy definition here is stricter (predicting the anchor 'i')
        accuracy = correct_count / batch_size if batch_size > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'acc': accuracy, # Accuracy of predicting the specific anchor 'i'
            'groups': group_count
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
            # Handle case where input is already per-question lists
            if not q_embeddings: return torch.tensor(0.0), {'loss': 0.0, 'ndcg': 0.0}
            device = q_embeddings[0].device
            q_embeds = q_embeddings
            batch_size_eff = len(q_embeddings)
        elif isinstance(q_embeddings, torch.Tensor):
            # Handle standard batch tensor input (assuming one query embedding per list entry)
            if q_embeddings.nelement() == 0: return torch.tensor(0.0), {'loss': 0.0, 'ndcg': 0.0}
            device = q_embeddings.device
            # This assumes the batch dimension of q_embeddings corresponds to the lists
            q_embeds = [q_embeddings[i] for i in range(q_embeddings.shape[0])]
            batch_size_eff = q_embeddings.shape[0]
        else:
            raise TypeError("q_embeddings must be a list of tensors or a single tensor.")

        if not isinstance(a_list_embeddings, list) or not isinstance(a_list_scores, list):
             raise TypeError("a_list_embeddings and a_list_scores must be lists.")
        
        # Ensure consistent lengths
        min_len = min(len(q_embeds), len(a_list_embeddings), len(a_list_scores))
        if min_len == 0: return torch.tensor(0.0, device=device), {'loss': 0.0, 'ndcg': 0.0}
        
        total_loss = torch.tensor(0.0, device=device)
        total_ndcg = 0.0
        valid_items = 0

        for i in range(min_len):
            q_embed = q_embeds[i] # Shape might be (embed_dim,) or (1, embed_dim)
            answers_embed = a_list_embeddings[i] # Shape (num_answers, embed_dim)
            scores = a_list_scores[i] # Shape (num_answers,)

            if answers_embed.nelement() == 0 or scores.nelement() == 0 or q_embed.nelement() == 0:
                continue
            
            # Ensure q_embed is 2D for matmul: (1, embed_dim)
            if q_embed.dim() == 1:
                q_embed = q_embed.unsqueeze(0)
                
            # Ensure answers_embed and scores are on the correct device
            answers_embed = answers_embed.to(device)
            scores = scores.to(device)

            num_answers = answers_embed.shape[0]
            if num_answers < 1: # Need at least one answer
                 continue

            # Calculate similarity: (1, embed_dim) @ (embed_dim, num_answers) -> (1, num_answers)
            sim = torch.matmul(q_embed, answers_embed.T).squeeze(0)  # Result shape: (num_answers,)
            sim = sim / self.temperature
            
            # Convert similarities to probabilities
            sim_probs = F.softmax(sim, dim=0)
            
            # Create target probabilities from normalized scores (normalize to avoid large values)
            # Temperature scaling for targets as well
            norm_scores = scores / self.temperature
            target_probs = F.softmax(norm_scores, dim=0)
            
            # KL divergence loss: KL(target || predicted) = sum(target * log(target / predicted))
            # Note: F.kl_div expects input=log(predicted), target=target
            # Use reduction='batchmean' or 'sum' and average later
            loss_i = F.kl_div(torch.log(sim_probs + 1e-9), target_probs, reduction='sum')
            
             # Avoid NaN loss if target_probs has zeros where sim_probs is non-zero due to temp scaling
            if torch.isnan(loss_i):
                # Fallback or alternative handling needed, e.g., skip or use different loss
                print(f"Warning: NaN loss encountered for item {i}. Skipping.")
                continue
                
            total_loss += loss_i
            valid_items += 1
            
            # Calculate NDCG
            if num_answers > 1:
                try:
                    # Sort predicted and target rankings
                    _, pred_indices = torch.sort(sim, descending=True)
                    _, ideal_indices = torch.sort(scores, descending=True) # Use original scores for ideal ranking
                    
                    # Calculate DCG
                    pred_dcg = self._calculate_dcg(pred_indices, scores) # Use original scores for gain
                    ideal_dcg = self._calculate_dcg(ideal_indices, scores)
                    
                    # Calculate NDCG
                    ndcg = pred_dcg / (ideal_dcg + 1e-8) if ideal_dcg > 0 else 0.0
                    total_ndcg += ndcg.item() # Add scalar value
                except Exception as e:
                    print(f"Warning: Error calculating NDCG for item {i}: {e}")
                    # NDCG calculation might fail if scores are constant, etc.


        # Average loss and metrics over valid items processed
        loss = total_loss / valid_items if valid_items > 0 else torch.tensor(0.0, device=device)
        avg_ndcg = total_ndcg / valid_items if valid_items > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'ndcg': avg_ndcg
        }
        
        return loss, metrics
    
    def _calculate_dcg(self, indices, scores):
        """Helper function to calculate DCG"""
        device = indices.device
        # Ranks are 1-based: 1, 2, 3, ...
        ranks = torch.arange(1, len(indices) + 1, dtype=torch.float, device=device)
        # Gain is the score of the item at that rank
        gain = scores[indices] # Use original scores
        # DCG = sum(gain_i / log2(rank_i + 1))
        return torch.sum(gain / torch.log2(ranks + 1))


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
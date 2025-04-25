#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class InfoNCELoss(BaseLoss):
    """InfoNCE contrastive loss with configurable options for negatives"""
    
    def __init__(self, temperature=0.1, in_batch_negatives=True, hard_negatives=True):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.in_batch_negatives = in_batch_negatives
        self.hard_negatives = hard_negatives
        self.name = f"infonce_t{temperature}_ibn{int(in_batch_negatives)}_hn{int(hard_negatives)}"
        
    def forward(self, q_embeddings, a_embeddings, a_negatives=None, **kwargs):
        """
        Calculate InfoNCE loss with configurable negatives
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            a_negatives: Optional hard negative answer embeddings (batch_size, num_negs, embed_dim)
        
        Returns:
            loss: Loss value
            metrics: Dictionary with accuracy and other metrics
        """
        batch_size = q_embeddings.shape[0]
        
        # Create similarity matrix between questions and answers
        if self.in_batch_negatives:
            # With in-batch negatives: compute full similarity matrix
            similarity = torch.matmul(q_embeddings, a_embeddings.T) / self.temperature
            # Labels are on the diagonal (each question matched with its answer)
            labels = torch.arange(batch_size, device=q_embeddings.device)
            
            # Compute InfoNCE loss (bidirectional)
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
        else:
            # Without in-batch negatives: compare each question only with its own answers
            # Each question has a positive answer and optionally hard negatives
            
            # Initialize logits and labels for each question
            all_logits = []
            all_labels = []
            
            # For each question, calculate its similarity with positive and negative answers
            for i in range(batch_size):
                q_embed = q_embeddings[i:i+1]  # (1, embed_dim)
                
                if self.hard_negatives and a_negatives is not None:
                    # Combine positive answer with hard negatives for this question
                    a_pos = a_embeddings[i:i+1]  # (1, embed_dim)
                    a_negs = a_negatives[i]  # (num_negs, embed_dim)
                    
                    # Combine positive and negatives
                    a_all = torch.cat([a_pos, a_negs], dim=0)  # (1+num_negs, embed_dim)
                    
                    # Calculate similarity between question and all answers
                    sim = torch.matmul(q_embed, a_all.T) / self.temperature  # (1, 1+num_negs)
                    
                    # First answer is positive, rest are negative
                    logits = sim.view(-1)  # (1+num_negs,)
                    labels = torch.zeros(1, dtype=torch.long, device=q_embed.device)
                    
                    all_logits.append(logits)
                    all_labels.append(labels)
                else:
                    # Without hard negatives, just use the positive pair
                    # This becomes a binary classification task
                    a_embed = a_embeddings[i:i+1]  # (1, embed_dim)
                    sim = torch.matmul(q_embed, a_embed.T) / self.temperature  # (1, 1)
                    
                    # Sigmoid loss (binary classification)
                    target = torch.ones_like(sim)
                    loss_i = F.binary_cross_entropy_with_logits(sim, target)
                    
                    if i == 0:
                        loss = loss_i
                    else:
                        loss += loss_i
            
            if self.hard_negatives and a_negatives is not None:
                # Concatenate all logits and labels
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                
                # Cross entropy loss
                loss = F.cross_entropy(all_logits, all_labels)
                
                # Calculate accuracy
                preds = torch.argmax(all_logits.view(batch_size, -1), dim=1)
                acc = (preds == 0).float().mean()  # 0 is the index of positive answer
                
                metrics = {
                    'loss': loss.item(),
                    'acc': acc.item()
                }
            else:
                # Binary case
                loss = loss / batch_size
                metrics = {
                    'loss': loss.item(),
                    'acc': 0.5  # Placeholder, no meaningful accuracy for binary case
                }
        
        return loss, metrics


class MSELoss(BaseLoss):
    """MSE loss on normalized upvote scores"""
    
    def __init__(self, normalize=True):
        super(MSELoss, self).__init__()
        self.normalize = normalize
        self.mse = nn.MSELoss()
        self.name = f"mse_norm{int(normalize)}"
        
    def forward(self, q_embeddings, a_embeddings, upvote_scores=None, **kwargs):
        """
        Calculate MSE loss between similarity and normalized upvote scores
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            upvote_scores: Upvote scores for each answer (batch_size,)
        
        Returns:
            loss: Loss value
            metrics: Dictionary with metrics
        """
        if upvote_scores is None:
            raise ValueError("MSELoss requires upvote_scores")
            
        # Calculate similarity between paired questions and answers
        similarity = torch.sum(q_embeddings * a_embeddings, dim=1)  # (batch_size,)
        
        # Normalize similarity to [0, 1] range
        if self.normalize:
            similarity = (similarity + 1) / 2
            
        # Normalize upvote scores
        if self.normalize:
            upvote_scores = upvote_scores / torch.max(upvote_scores)
            
        # Calculate MSE loss
        loss = self.mse(similarity, upvote_scores)
        
        # Calculate correlation for tracking
        sim_np = similarity.detach().cpu().numpy()
        score_np = upvote_scores.detach().cpu().numpy()
        correlation = np.corrcoef(sim_np, score_np)[0, 1]
        
        metrics = {
            'loss': loss.item(),
            'correlation': correlation
        }
        
        return loss, metrics


class TripletLoss(BaseLoss):
    """Triplet loss with upvote-based selection of positives and negatives"""
    
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
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
            loss: Loss value
            metrics: Dictionary with metrics
        """
        # Calculate similarity between questions and answers
        pos_sim = torch.sum(q_embeddings * a_pos_embeddings, dim=1)  # (batch_size,)
        neg_sim = torch.sum(q_embeddings * a_neg_embeddings, dim=1)  # (batch_size,)
        
        # Calculate triplet loss: sim(q, a_neg) - sim(q, a_pos) + margin
        losses = F.relu(neg_sim - pos_sim + self.margin)
        loss = torch.mean(losses)
        
        # Calculate accuracy (how often pos_sim > neg_sim)
        acc = (pos_sim > neg_sim).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'acc': acc.item(),
            'avg_pos_sim': pos_sim.mean().item(),
            'avg_neg_sim': neg_sim.mean().item()
        }
        
        return loss, metrics


class ListwiseRankingLoss(BaseLoss):
    """Listwise ranking loss for learning to rank answers"""
    
    def __init__(self, in_batch_negatives=True, temperature=1.0):
        super(ListwiseRankingLoss, self).__init__()
        self.in_batch_negatives = in_batch_negatives
        self.temperature = temperature
        self.name = f"listwise_ibn{int(in_batch_negatives)}_t{temperature}"
        
    def forward(self, q_embeddings, a_embeddings, upvote_scores=None, 
                a_list_embeddings=None, a_list_scores=None, **kwargs):
        """
        Calculate listwise ranking loss
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            upvote_scores: Upvote scores for answers (batch_size,)
            a_list_embeddings: List of answer embeddings per question [(num_answers, embed_dim), ...]
            a_list_scores: List of scores for answers per question [(num_answers,), ...]
        
        Returns:
            loss: Loss value
            metrics: Dictionary with metrics
        """
        batch_size = q_embeddings.shape[0]
        
        if self.in_batch_negatives:
            # With in-batch negatives: compute full similarity matrix
            # and compare with ranking from upvote scores
            
            if upvote_scores is None:
                raise ValueError("ListwiseRankingLoss with in-batch negatives requires upvote_scores")
            
            # Compute similarity matrix
            similarity = torch.matmul(q_embeddings, a_embeddings.T)  # (batch_size, batch_size)
            
            # Scale similarity by temperature
            similarity = similarity / self.temperature
            
            # Convert to softmax probabilities
            sim_probs = F.softmax(similarity, dim=1)  # (batch_size, batch_size)
            
            # Create target probabilities from normalized upvote scores
            target_probs = F.softmax(upvote_scores.view(1, -1) / self.temperature, dim=1)
            target_probs = target_probs.repeat(batch_size, 1)  # (batch_size, batch_size)
            
            # Create mask to set target probs to 0 for answers to other questions
            mask = torch.eye(batch_size, device=q_embeddings.device)
            masked_target_probs = target_probs * mask
            
            # Renormalize masked target probs
            row_sums = masked_target_probs.sum(dim=1, keepdim=True)
            masked_target_probs = masked_target_probs / (row_sums + 1e-8)
            
            # KL divergence loss
            loss = F.kl_div(torch.log(sim_probs + 1e-8), masked_target_probs, reduction='batchmean')
            
            # Calculate ranking metrics
            metrics = {
                'loss': loss.item()
            }
            
        else:
            # Without in-batch negatives: compare each question with its list of answers
            
            if a_list_embeddings is None or a_list_scores is None:
                raise ValueError("ListwiseRankingLoss without in-batch negatives requires answer lists")
            
            total_loss = 0.0
            total_ndcg = 0.0
            
            # Process each question separately
            for i in range(batch_size):
                q_embed = q_embeddings[i:i+1]  # (1, embed_dim)
                a_list = a_list_embeddings[i]  # (num_answers, embed_dim)
                
                # Calculate similarity between question and all answers
                sim = torch.matmul(q_embed, a_list.T).squeeze(0)  # (num_answers,)
                sim = sim / self.temperature
                
                # Convert to probabilities
                sim_probs = F.softmax(sim, dim=0)  # (num_answers,)
                
                # Create target probabilities from normalized scores
                scores = a_list_scores[i]  # (num_answers,)
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
            loss = total_loss / batch_size
            avg_ndcg = total_ndcg / batch_size
            
            metrics = {
                'loss': loss.item(),
                'ndcg': avg_ndcg.item()
            }
            
        return loss, metrics
    
    def _calculate_dcg(self, indices, scores):
        """Helper function to calculate DCG"""
        ranks = torch.arange(1, len(indices) + 1, dtype=torch.float, device=indices.device)
        gain = scores[indices]
        return torch.sum(gain / torch.log2(ranks + 1))


# Factory function to create loss based on name
class EnhancedInfoNCELoss(BaseLoss):
    """InfoNCE loss with extra penalty for hard negatives (answers to same question)"""
    
    def __init__(self, temperature=0.1, hard_negative_weight=2.0):
        super(EnhancedInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.name = f"enhanced_infonce_t{temperature}_hnw{hard_negative_weight}"
        
    def forward(self, q_embeddings, a_embeddings, question_ids=None, **kwargs):
        """
        Calculate InfoNCE loss with extra penalty for hard negatives
        
        Args:
            q_embeddings: Question embeddings (batch_size, embed_dim)
            a_embeddings: Answer embeddings (batch_size, embed_dim)
            question_ids: Question IDs for identifying hard negatives
            
        Returns:
            loss: Loss value
            metrics: Dictionary with accuracy and hard negative metrics
        """
        batch_size = q_embeddings.shape[0]
        device = q_embeddings.device
        
        # Calculate similarity matrix
        similarity = torch.matmul(q_embeddings, a_embeddings.T)
        
        # Scale by temperature
        logits = similarity / self.temperature
        
        # Standard InfoNCE loss - use diagonal as positive examples
        labels = torch.arange(batch_size, device=device)
        infonce_loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy: whether correct answer has highest similarity
        predictions = torch.argmax(similarity, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        # Initialize metrics
        metrics = {
            'base_loss': infonce_loss.item(),
            'acc': accuracy,
            'hard_neg_loss': 0.0
        }
        
        # Add extra penalty for hard negatives if question IDs are provided
        if question_ids is not None:
            # Group query indices by question ID
            question_groups = defaultdict(list)
            for i, q_id in enumerate(question_ids):
                # Handle different question ID formats
                if isinstance(q_id, torch.Tensor):
                    q_id = q_id.item()
                elif hasattr(q_id, 'decode'):
                    q_id = q_id.decode()
                    
                question_groups[q_id].append(i)
            
            # Calculate hard negative loss for questions with multiple answers
            questions_with_hard_negatives = 0
            hard_negative_pairs = 0
            hard_negative_loss = 0.0
            
            for q_id, indices in question_groups.items():
                if len(indices) <= 1:
                    continue
                    
                questions_with_hard_negatives += 1
                
                # For each query in this group
                for idx in indices:
                    # Get similarities between this query and other answers to the same question
                    other_indices = [j for j in indices if j != idx]
                    hard_negative_pairs += len(other_indices)
                    
                    # Skip if no other answers 
                    if not other_indices:
                        continue
                        
                    # Calculate similarity to hard negatives
                    hard_neg_sim = similarity[idx, other_indices]
                    
                    # Add penalty (we want to minimize similarity to hard negatives)
                    hard_negative_loss += torch.mean(hard_neg_sim)
            
            # Only add to loss if we found hard negatives
            if hard_negative_pairs > 0:
                # Scale the hard negative loss
                scaled_hard_negative_loss = (hard_negative_loss / hard_negative_pairs) * self.hard_negative_weight
                total_loss = infonce_loss + scaled_hard_negative_loss
                
                metrics['hard_neg_loss'] = scaled_hard_negative_loss.item()
                metrics['hard_neg_count'] = hard_negative_pairs
                metrics['questions_with_hard_negs'] = questions_with_hard_negatives
            else:
                total_loss = infonce_loss
        else:
            total_loss = infonce_loss
            
        metrics['loss'] = total_loss.item()
        return total_loss, metrics


def create_loss(loss_name, **kwargs):
    """Factory function for creating loss functions"""
    losses = {
        "infonce": InfoNCELoss,
        "infonce_no_batch_neg": lambda **kw: InfoNCELoss(in_batch_negatives=False, **kw),
        "infonce_no_hard_neg": lambda **kw: InfoNCELoss(hard_negatives=False, **kw),
        "mse": MSELoss,
        "triplet": TripletLoss,
        "listwise": ListwiseRankingLoss,
        "listwise_no_batch_neg": lambda **kw: ListwiseRankingLoss(in_batch_negatives=False, **kw),
        "enhanced_infonce": EnhancedInfoNCELoss
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return losses[loss_name](**kwargs)
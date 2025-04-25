#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for flexible loss functions.
Tests each loss function with appropriate input.
"""

import pytest
import torch
import numpy as np

from rank_test.flexible_losses import (
    StandardInfoNCELoss,
    RankInfoNCELoss,
    HardNegativeInfoNCELoss,
    MultiplePositivesLoss,
    TripletLoss,
    ListwiseRankingLoss,
    create_flexible_loss
)

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing"""
    # Create random embeddings and normalize them
    torch.manual_seed(42)  # For reproducibility
    q_embeddings = torch.randn(8, 128)
    a_embeddings = torch.randn(8, 128)
    
    # Normalize
    q_embeddings = q_embeddings / torch.norm(q_embeddings, dim=1, keepdim=True)
    a_embeddings = a_embeddings / torch.norm(a_embeddings, dim=1, keepdim=True)
    
    return q_embeddings, a_embeddings

@pytest.fixture
def mock_question_ids():
    """Create mock question IDs for testing"""
    # 8 embeddings with duplication pattern
    return ["q1", "q2", "q3", "q1", "q4", "q2", "q5", "q3"]

@pytest.fixture
def mock_ranks():
    """Create mock ranks for testing"""
    # 8 ranks corresponding to the embeddings
    return torch.tensor([0, 0, 0, 1, 0, 1, 0, 1], dtype=torch.long)

@pytest.fixture
def mock_triplet_data():
    """Create mock data for triplet loss"""
    torch.manual_seed(42)
    q_embeddings = torch.randn(5, 128)
    a_pos_embeddings = torch.randn(5, 128)
    a_neg_embeddings = torch.randn(5, 128)
    
    # Normalize
    q_embeddings = q_embeddings / torch.norm(q_embeddings, dim=1, keepdim=True)
    a_pos_embeddings = a_pos_embeddings / torch.norm(a_pos_embeddings, dim=1, keepdim=True)
    a_neg_embeddings = a_neg_embeddings / torch.norm(a_neg_embeddings, dim=1, keepdim=True)
    
    # Make positives more similar to queries than negatives for some examples
    for i in range(3):
        # Positive should be more similar
        a_pos_embeddings[i] = 0.8 * q_embeddings[i] + 0.2 * a_pos_embeddings[i]
        a_pos_embeddings[i] = a_pos_embeddings[i] / torch.norm(a_pos_embeddings[i])
        
        # Negative should be less similar
        a_neg_embeddings[i] = 0.2 * q_embeddings[i] + 0.8 * a_neg_embeddings[i]
        a_neg_embeddings[i] = a_neg_embeddings[i] / torch.norm(a_neg_embeddings[i])
    
    return q_embeddings, a_pos_embeddings, a_neg_embeddings

@pytest.fixture
def mock_listwise_data():
    """Create mock data for listwise ranking loss"""
    torch.manual_seed(42)
    
    # Create 3 questions, each with multiple answers
    q_embeddings = []
    a_list_embeddings = []
    a_list_scores = []
    
    for i in range(3):
        # Number of answers for this question
        num_answers = i + 2  # 2, 3, 4 answers
        
        # Create question embedding
        q_embed = torch.randn(1, 128)
        q_embed = q_embed / torch.norm(q_embed)
        
        # Create answer embeddings
        a_embeds = torch.randn(num_answers, 128)
        a_embeds = a_embeds / torch.norm(a_embeds, dim=1, keepdim=True)
        
        # Create scores
        scores = torch.tensor([float(num_answers - j) for j in range(num_answers)])
        
        q_embeddings.append(q_embed)
        a_list_embeddings.append(a_embeds)
        a_list_scores.append(scores)
    
    # Make sure all lists are same length
    # First item will have 3 elements in each list
    return q_embeddings[:1], a_list_embeddings[:1], a_list_scores[:1]


def test_standard_infonce_loss(mock_embeddings):
    """Test standard InfoNCE loss"""
    q_embeddings, a_embeddings = mock_embeddings
    
    # Create loss function
    loss_fn = StandardInfoNCELoss(temperature=0.1)
    
    # Calculate loss
    loss, metrics = loss_fn(q_embeddings, a_embeddings)
    
    # Basic validation
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0  # Loss should be positive
    
    # Check metrics
    assert 'loss' in metrics
    assert 'q2a_acc' in metrics
    assert 'a2q_acc' in metrics
    assert 'avg_acc' in metrics
    
    # Check that metrics are reasonable
    assert 0 <= metrics['q2a_acc'] <= 1
    assert 0 <= metrics['a2q_acc'] <= 1
    assert metrics['avg_acc'] == (metrics['q2a_acc'] + metrics['a2q_acc']) / 2


def test_rank_infonce_loss(mock_embeddings, mock_question_ids, mock_ranks):
    """Test rank-based InfoNCE loss"""
    q_embeddings, a_embeddings = mock_embeddings
    
    # Create loss function with rank weighting
    loss_fn = RankInfoNCELoss(temperature=0.1, use_ranks=True, rank_weight=0.1)
    
    # Calculate loss with rank information
    loss, metrics = loss_fn(q_embeddings, a_embeddings, 
                            question_ids=mock_question_ids,
                            ranks=mock_ranks)
    
    # Basic validation
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0  # Loss should be positive
    
    # Check metrics
    assert 'loss' in metrics
    assert 'q2a_acc' in metrics
    assert 'a2q_acc' in metrics
    
    # Create loss function without rank weighting
    loss_fn_no_rank = RankInfoNCELoss(temperature=0.1, use_ranks=False)
    
    # Calculate loss without rank information
    loss_no_rank, metrics_no_rank = loss_fn_no_rank(q_embeddings, a_embeddings)
    
    # Basic validation
    assert isinstance(loss_no_rank, torch.Tensor)
    assert loss_no_rank.dim() == 0  # Scalar tensor
    assert loss_no_rank.item() > 0  # Loss should be positive


def test_hard_negative_infonce_loss(mock_embeddings, mock_question_ids):
    """Test hard negative InfoNCE loss"""
    q_embeddings, a_embeddings = mock_embeddings
    
    # Create loss function
    loss_fn = HardNegativeInfoNCELoss(temperature=0.1, hard_negative_weight=1.0)
    
    # Calculate loss with hard negatives
    loss, metrics = loss_fn(q_embeddings, a_embeddings, question_ids=mock_question_ids)
    
    # Basic validation
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0  # Loss should be positive
    
    # Check metrics
    assert 'loss' in metrics
    assert 'base_loss' in metrics
    assert 'hard_neg_loss' in metrics
    assert 'accuracy' in metrics
    assert 'hard_neg_count' in metrics
    
    # Check that we found hard negatives
    assert metrics['hard_neg_count'] > 0
    assert metrics['hard_neg_loss'] > 0
    
    # Calculate loss without hard negatives
    loss_no_hn, metrics_no_hn = loss_fn(q_embeddings, a_embeddings)  # No question_ids
    
    # Should fall back to standard InfoNCE
    assert metrics_no_hn['hard_neg_count'] == 0
    assert metrics_no_hn['hard_neg_loss'] == 0
    assert metrics_no_hn['base_loss'] == metrics_no_hn['loss']


def test_multiple_positives_loss(mock_embeddings, mock_question_ids, mock_ranks):
    """Test multiple positives loss"""
    q_embeddings, a_embeddings = mock_embeddings
    
    # Create loss function
    loss_fn = MultiplePositivesLoss(temperature=0.1, rank_weight=0.1)
    
    # Calculate loss
    loss, metrics = loss_fn(q_embeddings, a_embeddings, 
                          question_ids=mock_question_ids,
                          ranks=mock_ranks)
    
    # Basic validation
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0  # Loss should be positive
    
    # Check metrics
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'total_samples' in metrics
    
    # Check that metrics are reasonable
    assert 0 <= metrics['accuracy'] <= 1
    assert metrics['total_samples'] == len(mock_question_ids)


def test_triplet_loss(mock_triplet_data):
    """Test triplet loss"""
    q_embeddings, a_pos_embeddings, a_neg_embeddings = mock_triplet_data
    
    # Create loss function
    loss_fn = TripletLoss(margin=0.3)
    
    # Calculate loss
    loss, metrics = loss_fn(q_embeddings, a_pos_embeddings, a_neg_embeddings)
    
    # Basic validation
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() >= 0  # Loss should be non-negative
    
    # Check metrics
    assert 'loss' in metrics
    assert 'acc' in metrics
    assert 'avg_pos_sim' in metrics
    assert 'avg_neg_sim' in metrics
    assert 'margin_violations' in metrics
    
    # Check that metrics are reasonable
    assert 0 <= metrics['acc'] <= 1
    assert metrics['avg_pos_sim'] > metrics['avg_neg_sim']  # Should be true on average
    
    # Test with different margin
    loss_fn_large_margin = TripletLoss(margin=1.0)
    loss_large, metrics_large = loss_fn_large_margin(q_embeddings, a_pos_embeddings, a_neg_embeddings)
    
    # Larger margin should lead to more violations and potentially higher loss
    assert metrics_large['margin_violations'] >= metrics['margin_violations']


def test_listwise_ranking_loss(mock_listwise_data):
    """Test listwise ranking loss"""
    q_embeddings, a_list_embeddings, a_list_scores = mock_listwise_data
    
    # Create loss function
    loss_fn = ListwiseRankingLoss(temperature=1.0)
    
    # Calculate loss
    loss, metrics = loss_fn(q_embeddings, a_list_embeddings, a_list_scores)
    
    # Basic validation
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0  # Loss should be positive
    
    # Check metrics
    assert 'loss' in metrics
    assert 'ndcg' in metrics
    
    # Check that metrics are reasonable
    assert 0 <= metrics['ndcg'] <= 1


def test_create_flexible_loss():
    """Test loss factory function"""
    # Test creating all loss types
    losses = {
        'infonce': StandardInfoNCELoss,
        'rank_infonce': RankInfoNCELoss,
        'hard_negative': HardNegativeInfoNCELoss,
        'multiple_positives': MultiplePositivesLoss,
        'triplet': TripletLoss,
        'listwise': ListwiseRankingLoss
    }
    
    for name, loss_class in losses.items():
        loss = create_flexible_loss(name)
        assert isinstance(loss, loss_class)
    
    # Test with parameters
    loss = create_flexible_loss('infonce', temperature=0.5)
    assert loss.temperature == 0.5
    
    # Test with invalid name
    with pytest.raises(ValueError):
        create_flexible_loss('invalid_loss')
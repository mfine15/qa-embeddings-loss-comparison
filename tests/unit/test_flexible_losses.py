#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for flexible loss functions.
Tests each loss function with appropriate input, focusing on semantic behavior.
"""

import pytest
import torch
import numpy as np
from collections import defaultdict

# Assuming losses are in src/rank_test/losses.py relative to project root
# Adjust path if necessary
from rank_test.losses import (
    FlexibleInfoNCELoss,
    RankInfoNCELoss,
    RelaxedHardNegativeInfoNCELoss, # New Relaxed HN loss
    TripletLoss,
    ListwiseRankingLoss,
    create_loss # Updated factory function name
)

# Helper to create normalized embeddings
def create_normalized_embeddings(shape, seed=42):
    torch.manual_seed(seed)
    embeddings = torch.randn(shape)
    return embeddings / torch.norm(embeddings, dim=-1, keepdim=True)

@pytest.fixture
def mock_embeddings_basic():
    """Basic (Batch Size, Embed Dim) embeddings for simple tests"""
    q = create_normalized_embeddings((4, 64), seed=1)
    # Make diagonal entries similar (like standard InfoNCE setup)
    a = torch.zeros_like(q)
    a[0] = 0.9 * q[0] + 0.1 * create_normalized_embeddings((1, 64), seed=10)
    a[1] = 0.9 * q[1] + 0.1 * create_normalized_embeddings((1, 64), seed=11)
    a[2] = 0.9 * q[2] + 0.1 * create_normalized_embeddings((1, 64), seed=12)
    a[3] = 0.9 * q[3] + 0.1 * create_normalized_embeddings((1, 64), seed=13)
    a = a / torch.norm(a, dim=-1, keepdim=True)
    return q, a

@pytest.fixture
def mock_data_multi_positive():
    """Data with multiple positives per question for Flex/Rank/Relaxed tests"""
    # Q1: idx 0 (pos), idx 3 (pos/hn)
    # Q2: idx 1 (pos), idx 5 (pos/hn)
    # Q3: idx 2 (pos), idx 7 (pos/hn)
    # Q4: idx 4 (pos)
    # Q5: idx 6 (pos)
    question_ids = ["q1", "q2", "q3", "q1", "q4", "q2", "q5", "q3"]
    ranks = torch.tensor([0, 0, 0, 1, 0, 1, 0, 1], dtype=torch.long) # Lower is better
    scores = torch.tensor([10, 9, 8, 5, 10, 4, 9, 3], dtype=torch.float) # Higher is better
    batch_size = len(question_ids)
    embed_dim = 64

    q_embeddings = create_normalized_embeddings((batch_size, embed_dim), seed=20)
    a_embeddings = torch.zeros_like(q_embeddings)

    # Group indices by question ID
    q_groups = defaultdict(list)
    for i, q_id in enumerate(question_ids):
        q_groups[q_id].append(i)

    # Create answers such that answers for the same question are somewhat similar
    # and answers for different questions are less similar.
    # Make rank 0 answers closer to the query than rank 1 answers.
    noise_level_pos = 0.1 # Noise for the best positive (rank 0)
    noise_level_hn = 0.3  # Noise for the hard negative (rank 1)
    noise_level_neg = 0.8 # Noise relative to unrelated Qs

    for q_id, indices in q_groups.items():
        anchor_q_idx = indices[0] # Use the first query occurrence as anchor for simplicity
        anchor_q_embed = q_embeddings[anchor_q_idx]

        for i in indices:
             rank = ranks[i].item()
             if rank == 0: # Primary positive
                 noise = create_normalized_embeddings((1, embed_dim), seed=30+i)
                 a_embeddings[i] = (1-noise_level_pos) * anchor_q_embed + noise_level_pos * noise
             else: # Hard negative / lower-ranked positive
                 noise = create_normalized_embeddings((1, embed_dim), seed=40+i)
                 a_embeddings[i] = (1-noise_level_hn) * anchor_q_embed + noise_level_hn * noise

    # Add noise relative to *other* questions to simulate negatives
    for i in range(batch_size):
        for j in range(batch_size):
             if question_ids[i] != question_ids[j]: # Unrelated question
                 noise = create_normalized_embeddings((1, embed_dim), seed=50+i*batch_size+j)
                 # Add a small component from an unrelated query's direction
                 # Fix the broadcasting by ensuring consistent shapes
                 noise_component = noise_level_neg * (q_embeddings[j].unsqueeze(0) * 0.1 + noise * 0.9)
                 a_embeddings[i] += noise_component.squeeze(0)

    a_embeddings = a_embeddings / torch.norm(a_embeddings, dim=-1, keepdim=True)

    return q_embeddings, a_embeddings, question_ids, ranks, scores


def test_flexible_infonce_loss_single_positive(mock_embeddings_basic):
    """Test FlexibleInfoNCE behaving like standard InfoNCE (one positive per Q)."""
    q_embeddings, a_embeddings = mock_embeddings_basic
    # Each item has a unique question ID
    question_ids = [f"q{i}" for i in range(q_embeddings.shape[0])]
    
    loss_fn = FlexibleInfoNCELoss(temperature=0.1)
    loss, metrics = loss_fn(q_embeddings, a_embeddings, question_ids=question_ids)
    
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert loss.item() > 0
    assert 'loss' in metrics and 'acc' in metrics and 'groups' in metrics
    assert metrics['groups'] == q_embeddings.shape[0] # Each qid is its own group
    # Accuracy should be high because diagonal elements were made similar
    assert metrics['acc'] > 0.5


def test_flexible_infonce_loss_multi_positive(mock_data_multi_positive):
    """Test FlexibleInfoNCE with multiple positives per question."""
    q_embeddings, a_embeddings, question_ids, _, _ = mock_data_multi_positive
    
    loss_fn = FlexibleInfoNCELoss(temperature=0.1)
    loss, metrics = loss_fn(q_embeddings, a_embeddings, question_ids=question_ids)
    
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert loss.item() > 0
    assert 'loss' in metrics and 'acc' in metrics and 'groups' in metrics
    
    # Count unique question IDs
    unique_qids = len(set(question_ids))
    assert metrics['groups'] == unique_qids
    
    # Accuracy check: Since rank 0 answers were made closer, expect decent accuracy
    # where accuracy means predicting *any* answer from the same group.
    assert metrics['acc'] > 0 # Should get some right


def test_rank_infonce_loss_weights(mock_data_multi_positive):
    """Test RankInfoNCE applies weighting based on ranks/scores."""
    q_embeddings, a_embeddings, question_ids, ranks, scores = mock_data_multi_positive
    
    # --- Test with Rank Weighting ---
    loss_fn_rank = RankInfoNCELoss(temperature=0.1, use_ranks=True, use_scores=False, rank_weight=0.5)
    loss_rank, metrics_rank = loss_fn_rank(q_embeddings, a_embeddings, question_ids=question_ids, ranks=ranks)

    # --- Test with Score Weighting ---
    loss_fn_score = RankInfoNCELoss(temperature=0.1, use_ranks=False, use_scores=True, score_weight=0.1)
    loss_score, metrics_score = loss_fn_score(q_embeddings, a_embeddings, question_ids=question_ids, scores=scores)

    # --- Test without Weighting (using FlexibleInfoNCE for comparison) ---
    loss_fn_base = FlexibleInfoNCELoss(temperature=0.1)
    loss_base, metrics_base = loss_fn_base(q_embeddings, a_embeddings, question_ids=question_ids)

    # --- Assertions ---
    assert isinstance(loss_rank, torch.Tensor) and loss_rank.dim() == 0
    assert isinstance(loss_score, torch.Tensor) and loss_score.dim() == 0
    assert loss_rank.item() > 0
    assert loss_score.item() > 0
    
    # Semantic Check: Weighted losses should differ from the base flexible loss
    # because ranks/scores vary within groups. The exact difference depends on data,
    # but they shouldn't be identical unless all ranks/scores within groups are the same.
    assert not np.isclose(loss_rank.item(), loss_base.item()), "Rank weighting should change loss"
    assert not np.isclose(loss_score.item(), loss_base.item()), "Score weighting should change loss"
    
    # Check metrics structure
    assert 'acc' in metrics_rank and 'groups' in metrics_rank
    assert 'acc' in metrics_score and 'groups' in metrics_score
    # Accuracy definition is the same as FlexibleInfoNCE (predicting any item in group)
    assert np.isclose(metrics_rank['acc'], metrics_base['acc'])
    assert np.isclose(metrics_score['acc'], metrics_base['acc'])


def test_relaxed_hard_negative_loss():
    """Test RelaxedHardNegative loss penalizes HN less than FlexibleInfoNCE."""
    # Create specialized test data that will highlight the difference
    # between the two accuracy definitions
    batch_size = 6
    embed_dim = 64
    
    # Setup: 3 questions, 2 answers each
    # q1: idx 0,1
    # q2: idx 2,3
    # q3: idx 4,5
    question_ids = ["q1", "q1", "q2", "q2", "q3", "q3"]
    
    # Create embeddings with specific similarity patterns
    q_embeddings = torch.zeros((batch_size, embed_dim))
    a_embeddings = torch.zeros((batch_size, embed_dim))
    
    # Initialize with random normalized vectors
    torch.manual_seed(42)
    for i in range(batch_size):
        q_embeddings[i] = torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
        a_embeddings[i] = torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Now create a specific pattern where:
    # - For indices 0,2,4: highest similarity is with another answer from same question
    # - For indices 1,3,5: highest similarity is with itself
    
    # Make answer 0's highest similarity be with q1 (itself) but also high with a1
    a_embeddings[0] = 0.7 * q_embeddings[0] + 0.3 * torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Make answer 1's highest similarity be with q0 (not itself)
    a_embeddings[1] = 0.8 * q_embeddings[0] + 0.2 * torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Make answer 2's highest similarity be with q3 (not itself)
    a_embeddings[2] = 0.8 * q_embeddings[3] + 0.2 * torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Make answer 3's highest similarity be with q2 (itself)
    a_embeddings[3] = 0.7 * q_embeddings[3] + 0.3 * torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Make answer 4's highest similarity be with q5 (not itself)
    a_embeddings[4] = 0.8 * q_embeddings[5] + 0.2 * torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Make answer 5's highest similarity be with q4 (itself)
    a_embeddings[5] = 0.7 * q_embeddings[5] + 0.3 * torch.nn.functional.normalize(torch.randn(embed_dim), dim=0)
    
    # Normalize all embeddings
    q_embeddings = torch.nn.functional.normalize(q_embeddings, dim=1)
    a_embeddings = torch.nn.functional.normalize(a_embeddings, dim=1)
    
    # Set up the losses
    temp = 0.1
    hard_neg_prob = 0.2 # Assign 20% target probability to hard negatives
    
    # --- Relaxed Hard Negative Loss ---
    loss_fn_relaxed = RelaxedHardNegativeInfoNCELoss(temperature=temp, hard_neg_target_prob=hard_neg_prob)
    loss_relaxed, metrics_relaxed = loss_fn_relaxed(q_embeddings, a_embeddings, question_ids=question_ids)

    # --- Flexible InfoNCE Loss (for comparison) ---
    loss_fn_flex = FlexibleInfoNCELoss(temperature=temp)
    loss_flex, metrics_flex = loss_fn_flex(q_embeddings, a_embeddings, question_ids=question_ids)
    
    # --- Assertions ---
    assert isinstance(loss_relaxed, torch.Tensor) and loss_relaxed.dim() == 0
    assert loss_relaxed.item() >= 0 # KL divergence loss >= 0
    assert 'acc' in metrics_relaxed and 'groups' in metrics_relaxed
    
    # Semantic Check: Loss is lower for relaxed version because it doesn't
    # penalize hard negatives as strongly
    assert loss_relaxed.item() < loss_flex.item(), \
        "RelaxedHN loss should be lower than FlexibleInfoNCE when hard negatives are somewhat similar"

    # Check accuracy definitions are different:
    # - FlexibleInfoNCE counts correct if highest similarity is with ANY item in same group
    # - RelaxedHardNegative counts correct only if highest similarity is with ITSELF
    # With our constructed test data, these should give different results
    assert 'acc' in metrics_relaxed and 'acc' in metrics_flex
    assert 0 <= metrics_relaxed['acc'] <= 1 and 0 <= metrics_flex['acc'] <= 1
    
    # FlexibleInfoNCE should have higher accuracy as its definition is more lenient
    assert metrics_flex['acc'] > metrics_relaxed['acc'], \
        f"FlexibleInfoNCE acc ({metrics_flex['acc']}) should be higher than RelaxedHN acc ({metrics_relaxed['acc']})"


# --- Tests for Triplet and Listwise Losses (mostly unchanged, maybe slightly more robust) ---

@pytest.fixture
def mock_triplet_data():
    """Create mock data for triplet loss with clearer separation"""
    batch_size = 10
    embed_dim = 64
    q = create_normalized_embeddings((batch_size, embed_dim), seed=60)
    a_pos = torch.zeros_like(q)
    a_neg = torch.zeros_like(q)

    # Make positives clearly similar, negatives clearly dissimilar
    for i in range(batch_size):
        noise_pos = create_normalized_embeddings((1, embed_dim), seed=70+i)
        noise_neg = create_normalized_embeddings((1, embed_dim), seed=80+i)
        # Ensure positive is aligned, negative is less aligned
        a_pos[i] = 0.95 * q[i] + 0.05 * noise_pos
        a_neg[i] = 0.1 * q[i] + 0.9 * noise_neg # Make negative mostly noise

    a_pos = a_pos / torch.norm(a_pos, dim=-1, keepdim=True)
    a_neg = a_neg / torch.norm(a_neg, dim=-1, keepdim=True)

    return q, a_pos, a_neg

def test_triplet_loss(mock_triplet_data):
    """Test triplet loss ensures margin"""
    q_embeddings, a_pos_embeddings, a_neg_embeddings = mock_triplet_data
    margin = 0.3
    loss_fn = TripletLoss(margin=margin)
    loss, metrics = loss_fn(q_embeddings, a_pos_embeddings, a_neg_embeddings)

    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert loss.item() >= 0
    assert 'loss' in metrics and 'acc' in metrics and 'avg_pos_sim' in metrics
    assert 'avg_neg_sim' in metrics and 'margin_violations' in metrics
    
    # Semantic checks
    assert metrics['acc'] > 0.8, "Accuracy should be high with clear separation"
    assert metrics['avg_pos_sim'] > metrics['avg_neg_sim'] + margin * 0.5, \
        "Avg pos sim should be significantly > avg neg sim"
    # Some violations might occur if noise pushes neg closer occasionally
    assert 0 <= metrics['margin_violations'] <= 1


@pytest.fixture
def mock_listwise_data():
    """Create mock data for listwise ranking loss with varying list lengths"""
    q_embeddings = []
    a_list_embeddings = []
    a_list_scores = []
    embed_dim = 64

    # Question 1: 3 answers
    q1 = create_normalized_embeddings((1, embed_dim), seed=90)
    a1 = create_normalized_embeddings((3, embed_dim), seed=91)
    s1 = torch.tensor([5.0, 2.0, 0.0]) # Clear score separation
    q_embeddings.append(q1.squeeze(0)) # Store as (embed_dim,)
    a_list_embeddings.append(a1)
    a_list_scores.append(s1)

    # Question 2: 5 answers
    q2 = create_normalized_embeddings((1, embed_dim), seed=100)
    a2 = create_normalized_embeddings((5, embed_dim), seed=101)
    s2 = torch.tensor([10.0, 8.0, 8.0, 3.0, 1.0]) # Include ties
    q_embeddings.append(q2.squeeze(0))
    a_list_embeddings.append(a2)
    a_list_scores.append(s2)
    
    # Question 3: 2 answers
    q3 = create_normalized_embeddings((1, embed_dim), seed=110)
    a3 = create_normalized_embeddings((2, embed_dim), seed=111)
    s3 = torch.tensor([7.0, 6.0]) 
    q_embeddings.append(q3.squeeze(0))
    a_list_embeddings.append(a3)
    a_list_scores.append(s3)

    # Structure expected by loss: list of tensors for Q, list of lists for A/Scores
    return q_embeddings, a_list_embeddings, a_list_scores


def test_listwise_ranking_loss(mock_listwise_data):
    """Test listwise ranking loss computes KL divergence and NDCG."""
    q_embeddings, a_list_embeddings, a_list_scores = mock_listwise_data
    loss_fn = ListwiseRankingLoss(temperature=1.0) # Use temp=1 for direct score comparison
    
    # --- Calculate Loss ---
    # Note: Listwise loss expects a list of Q embeddings if A/Scores are lists
    loss, metrics = loss_fn(q_embeddings, a_list_embeddings, a_list_scores)

    # --- Assertions ---
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert loss.item() >= 0 # KL divergence >= 0
    assert 'loss' in metrics and 'ndcg' in metrics
    
    # Semantic checks
    # Since embeddings are random, NDCG will likely be low, but should be between 0 and 1
    assert 0 <= metrics['ndcg'] <= 1, "NDCG must be in [0, 1]"
    # We could craft embeddings to test NDCG=1, but random check is simpler


# --- Test Factory Function ---

def test_create_loss_factory():
    """Test the updated loss factory function."""
    # Test creating new base/derived losses
    loss_flex = create_loss('infonce', temperature=0.5)
    assert isinstance(loss_flex, FlexibleInfoNCELoss)
    assert loss_flex.temperature == 0.5

    loss_rank = create_loss('rank_infonce', use_ranks=True, rank_weight=0.2)
    assert isinstance(loss_rank, RankInfoNCELoss)
    assert loss_rank.use_ranks is True and loss_rank.rank_weight == 0.2

    loss_relaxed = create_loss('relaxed_hard_neg', hard_neg_target_prob=0.15)
    assert isinstance(loss_relaxed, RelaxedHardNegativeInfoNCELoss)
    assert loss_relaxed.hard_neg_target_prob == 0.15

    loss_triplet = create_loss('triplet', margin=0.5)
    assert isinstance(loss_triplet, TripletLoss)
    assert loss_triplet.margin == 0.5

    loss_listwise = create_loss('listwise', temperature=2.0)
    assert isinstance(loss_listwise, ListwiseRankingLoss)
    assert loss_listwise.temperature == 2.0

    # Test creating named variants
    loss_rank_named = create_loss('rank_infonce_rank0.1_score0.05_t0.2')
    assert isinstance(loss_rank_named, RankInfoNCELoss)
    assert loss_rank_named.use_ranks is True and loss_rank_named.rank_weight == 0.1
    assert loss_rank_named.use_scores is True and loss_rank_named.score_weight == 0.05
    assert loss_rank_named.temperature == 0.2

    loss_relaxed_named = create_loss('relaxed_hard_neg_p0.25_t0.05')
    assert isinstance(loss_relaxed_named, RelaxedHardNegativeInfoNCELoss)
    assert loss_relaxed_named.hard_neg_target_prob == 0.25
    assert loss_relaxed_named.temperature == 0.05
    
    # Test overriding named params with kwargs
    loss_override = create_loss('rank_infonce_rank0.1_t0.2', temperature=0.9, rank_weight=0.99)
    assert loss_override.temperature == 0.9
    assert loss_override.rank_weight == 0.99


    # Test invalid name
    with pytest.raises(ValueError):
        create_loss('nonexistent_loss_type')
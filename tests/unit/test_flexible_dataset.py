#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for flexible dataset module.
Tests the different batch transformation strategies.
"""

import os
import pytest
import torch
import json
import tempfile
from transformers import DistilBertTokenizer

from rank_test.flexible_dataset import (
    FlexibleQADataset,
    infonce_batch_transform,
    multiple_positives_batch_transform,
    hard_negative_batch_transform,
    triplet_batch_transform,
    listwise_batch_transform,
    get_batch_transform
)

# Create a mock dataset for testing
@pytest.fixture
def mock_qa_data():
    """Create a mock dataset for testing"""
    data = [
        {
            "question": {
                "id": "q1",
                "title": "Question 1",
                "body": "What is the best way to test code?"
            },
            "answers": [
                {
                    "id": "a1",
                    "body": "Use pytest for testing",
                    "score": 95  # Cardinal score (e.g., user rating or upvotes)
                },
                {
                    "id": "a2",
                    "body": "Use unittest for testing",
                    "score": 82
                },
                {
                    "id": "a3",
                    "body": "Use pytest with coverage for testing",
                    "score": 78
                },
                {
                    "id": "a4",
                    "body": "Use a combination of pytest and unittest",
                    "score": 65
                },
                {
                    "id": "a5",
                    "body": "Manual testing works too",
                    "score": 35
                }
            ]
        },
        {
            "question": {
                "id": "q2",
                "title": "Question 2",
                "body": "How to implement a dataset in PyTorch?"
            },
            "answers": [
                {
                    "id": "a6",
                    "body": "Inherit from torch.utils.data.Dataset",
                    "score": 92
                },
                {
                    "id": "a7",
                    "body": "Override __getitem__ and __len__",
                    "score": 88
                },
                {
                    "id": "a8",
                    "body": "Create a class with __getitem__, __len__, and collate_fn",
                    "score": 75
                },
                {
                    "id": "a9",
                    "body": "Use DataLoader with your custom Dataset class",
                    "score": 62
                }
            ]
        },
        {
            "question": {
                "id": "q3",
                "title": "Question 3",
                "body": "What is a loss function?"
            },
            "answers": [
                {
                    "id": "a10",
                    "body": "A function that measures model error",
                    "score": 90
                },
                {
                    "id": "a11",
                    "body": "A metric that quantifies the difference between predicted and actual values",
                    "score": 85
                },
                {
                    "id": "a12",
                    "body": "A mathematical function that evaluates how well a model performs",
                    "score": 70
                }
            ]
        }
    ]
    
    # Create a temporary file with the data
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def mock_tokenizer():
    """Return a mock tokenizer for testing"""
    # For testing, we can use a simple tokenizer
    class MockTokenizer:
        def __call__(self, text, max_length=10, padding=None, truncation=None, return_tensors=None):
            # Just return a fixed tensor for testing
            return {
                'input_ids': torch.ones(1, max_length, dtype=torch.long),
                'attention_mask': torch.ones(1, max_length, dtype=torch.long)
            }
    
    return MockTokenizer()


def test_get_batch_transform():
    """Test getting batch transform functions by name"""
    # Test valid transform names
    assert get_batch_transform('infonce') == infonce_batch_transform
    assert get_batch_transform('multiple_positives') == multiple_positives_batch_transform
    assert get_batch_transform('hard_negative') == hard_negative_batch_transform
    assert get_batch_transform('triplet') == triplet_batch_transform
    assert get_batch_transform('listwise') == listwise_batch_transform
    
    # Test invalid transform name
    with pytest.raises(ValueError):
        get_batch_transform('invalid_transform')


def test_infonce_batch_transform(mock_qa_data, mock_tokenizer):
    """Test infonce batch transform"""
    with open(mock_qa_data, 'r') as f:
        data = json.load(f)
    
    # Test with top answers
    batch = infonce_batch_transform(data, mock_tokenizer, max_length=10, take_top=True)
    
    # Validate batch format
    assert 'q_input_ids' in batch
    assert 'a_input_ids' in batch
    assert 'question_ids' in batch
    assert 'ranks' in batch
    assert 'scores' in batch
    
    # Check shapes
    assert batch['q_input_ids'].shape[0] == len(data)
    assert batch['a_input_ids'].shape[0] == len(data)
    assert len(batch['question_ids']) == len(data)
    
    # Test all ranks are 0 (top answers)
    assert (batch['ranks'] == 0).all()
    
    # Test with random answers
    batch_random = infonce_batch_transform(data, mock_tokenizer, max_length=10, take_top=False)
    
    # Validate batch format
    assert 'q_input_ids' in batch_random
    assert 'a_input_ids' in batch_random
    assert 'question_ids' in batch_random
    assert 'ranks' in batch_random
    
    # Check shapes
    assert batch_random['q_input_ids'].shape[0] == len(data)
    assert batch_random['a_input_ids'].shape[0] == len(data)
    assert len(batch_random['question_ids']) == len(data)


def test_multiple_positives_batch_transform(mock_qa_data, mock_tokenizer):
    """Test multiple positives batch transform"""
    with open(mock_qa_data, 'r') as f:
        data = json.load(f)
    
    # Test with up to 2 positives per question
    batch = multiple_positives_batch_transform(data, mock_tokenizer, max_length=10, pos_count=2)
    
    # Validate batch format
    assert 'q_input_ids' in batch
    assert 'a_input_ids' in batch
    assert 'question_ids' in batch
    assert 'ranks' in batch
    assert 'scores' in batch  # Check that scores are included
    
    # Check number of samples (with 2 positives per question)
    # Q1: 2 answers, Q2: 2 answers, Q3: 2 answers = 6 total samples
    expected_samples = 6
    assert batch['q_input_ids'].shape[0] == expected_samples
    assert batch['a_input_ids'].shape[0] == expected_samples
    assert len(batch['question_ids']) == expected_samples
    
    # Verify ranks are correct
    assert torch.sum(batch['ranks'] == 0) > 0  # Some rank 0
    assert torch.sum(batch['ranks'] == 1) > 0  # Some rank 1
    
    # Check that we have scores
    assert batch['scores'].shape[0] == expected_samples


def test_hard_negative_batch_transform(mock_qa_data, mock_tokenizer):
    """Test hard negative batch transform"""
    with open(mock_qa_data, 'r') as f:
        data = json.load(f)
    
    batch = hard_negative_batch_transform(data, mock_tokenizer, max_length=10)
    
    # Validate batch format - this returns a list of items
    assert isinstance(batch, list)
    assert len(batch) == 3  # All questions now have multiple answers
    
    # Check first question
    assert 'q_input_ids' in batch[0]
    assert 'q_attention_mask' in batch[0]
    assert 'answers' in batch[0]
    assert 'question_id' in batch[0]
    
    # Check answers format
    answers = batch[0]['answers']
    assert isinstance(answers, list)
    assert len(answers) == 5  # Q1 has 5 answers
    
    # Verify answer properties
    assert 'input_ids' in answers[0]
    assert 'attention_mask' in answers[0]
    assert 'id' in answers[0]
    assert 'score' in answers[0]
    assert 'rank' in answers[0]
    
    # Check second question
    answers = batch[1]['answers']
    assert len(answers) == 4  # Q2 has 4 answers
    
    # Check third question
    answers = batch[2]['answers']
    assert len(answers) == 3  # Q3 has 3 answers


def test_triplet_batch_transform(mock_qa_data, mock_tokenizer):
    """Test triplet batch transform"""
    with open(mock_qa_data, 'r') as f:
        data = json.load(f)
    
    # Test with hard negatives
    batch = triplet_batch_transform(data, mock_tokenizer, max_length=10, neg_strategy="hard_negative")
    
    # Validate batch format
    assert 'q_input_ids' in batch
    assert 'a_pos_input_ids' in batch
    assert 'a_neg_input_ids' in batch
    assert 'question_ids' in batch
    assert 'pos_scores' in batch
    assert 'neg_scores' in batch
    
    # Check shapes - all 3 questions have multiple answers in our updated mock data
    assert batch['q_input_ids'].shape[0] == 3
    assert batch['a_pos_input_ids'].shape[0] == 3
    assert batch['a_neg_input_ids'].shape[0] == 3
    assert len(batch['question_ids']) == 3
    
    # Test with in-batch negatives
    batch_in_batch = triplet_batch_transform(data, mock_tokenizer, max_length=10, neg_strategy="in_batch")
    
    # Should have samples for all questions
    assert batch_in_batch['q_input_ids'].shape[0] == 3


def test_listwise_batch_transform(mock_qa_data, mock_tokenizer):
    """Test listwise batch transform"""
    with open(mock_qa_data, 'r') as f:
        data = json.load(f)
    
    batch = listwise_batch_transform(data, mock_tokenizer, max_length=10, max_answers=3)
    
    # Validate batch format - this returns a list of items
    assert isinstance(batch, list)
    assert len(batch) == 3  # All questions have multiple answers
    
    # Check first question
    assert 'q_input_ids' in batch[0]
    assert 'q_attention_mask' in batch[0]
    assert 'a_input_ids' in batch[0]
    assert 'a_attention_masks' in batch[0]
    assert 'scores' in batch[0]
    assert 'question_id' in batch[0]
    
    # Check shapes
    assert batch[0]['a_input_ids'].shape[0] == 3  # Limited to max_answers=3 (Q1 has 5 total)
    assert batch[1]['a_input_ids'].shape[0] == 3  # Limited to max_answers=3 (Q2 has 4 total)
    assert batch[2]['a_input_ids'].shape[0] == 3  # Q3 has 3 answers
    
    # Check scores are normalized
    assert torch.max(batch[0]['scores']).item() == 1.0
    assert torch.max(batch[1]['scores']).item() == 1.0
    assert torch.max(batch[2]['scores']).item() == 1.0


def test_flexible_dataset(mock_qa_data, mock_tokenizer):
    """Test flexible dataset class"""
    # Test with default transform (infonce)
    dataset = FlexibleQADataset(
        mock_qa_data,
        batch_size=2,
        tokenizer=mock_tokenizer,
        max_length=10
    )
    
    # Check dataset length
    # With batch_size=2 and 3 questions, should have 2 batches
    assert len(dataset) == 2
    
    # Check first batch
    batch = dataset[0]
    assert 'q_input_ids' in batch
    assert 'a_input_ids' in batch
    assert 'question_ids' in batch
    
    # Test with multiple positives transform
    multi_pos_dataset = FlexibleQADataset(
        mock_qa_data,
        batch_transform_fn=multiple_positives_batch_transform,
        batch_size=2,
        tokenizer=mock_tokenizer,
        max_length=10,
        pos_count=2
    )
    
    # Check dataset
    assert len(multi_pos_dataset) > 0
    
    # Test dataloader creation
    dataloader = FlexibleQADataset.get_dataloader(multi_pos_dataset)
    assert dataloader is not None
    
    # Get a batch from the dataloader
    for batch in dataloader:
        assert 'q_input_ids' in batch
        assert 'a_input_ids' in batch
        assert 'question_ids' in batch
        assert 'ranks' in batch
        break


def test_dataset_with_all_transforms(mock_qa_data, mock_tokenizer):
    """Test dataset with all transform types"""
    # List of all transform functions to test
    transforms = [
        infonce_batch_transform,
        multiple_positives_batch_transform,
        hard_negative_batch_transform,
        triplet_batch_transform,
        listwise_batch_transform
    ]
    
    for transform_fn in transforms:
        # Create dataset with this transform
        dataset = FlexibleQADataset(
            mock_qa_data,
            batch_transform_fn=transform_fn,
            batch_size=2,
            tokenizer=mock_tokenizer,
            max_length=10
        )
        
        # Basic validation
        assert len(dataset) > 0
        
        # Get first batch
        batch = dataset[0]
        assert batch is not None
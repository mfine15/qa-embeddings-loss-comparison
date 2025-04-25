# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Integration tests for the flexible dataset and loss function pipeline.
# Tests that the different components work well together.
# """

# import os
# import pytest
# import torch
# import json
# import tempfile
# from torch.utils.data import DataLoader

# from rank_test.flexible_dataset import (
#     FlexibleQADataset,
#     infonce_batch_transform,
#     multiple_positives_batch_transform,
#     hard_negative_batch_transform,
#     triplet_batch_transform,
#     listwise_batch_transform
# )

# from rank_test.flexible_losses import (
#     StandardInfoNCELoss,
#     RankInfoNCELoss,
#     HardNegativeInfoNCELoss,
#     MultiplePositivesLoss,
#     TripletLoss,
#     ListwiseRankingLoss
# )

# from rank_test.models import QAEmbeddingModel

# # Create a mock dataset for testing
# @pytest.fixture
# def mock_qa_data():
#     """Create a mock dataset for testing"""
#     data = [
#         {
#             "question": {
#                 "id": "q1",
#                 "title": "Question 1",
#                 "body": "What is the best way to test code?"
#             },
#             "answers": [
#                 {
#                     "id": "a1",
#                     "body": "Use pytest for testing",
#                     "score": 95  # Cardinal score (e.g., user rating or upvotes)
#                 },
#                 {
#                     "id": "a2",
#                     "body": "Use unittest for testing",
#                     "score": 82
#                 },
#                 {
#                     "id": "a3",
#                     "body": "Use pytest with coverage for testing",
#                     "score": 78
#                 },
#                 {
#                     "id": "a4",
#                     "body": "Use a combination of pytest and unittest",
#                     "score": 65
#                 },
#                 {
#                     "id": "a5",
#                     "body": "Manual testing works too",
#                     "score": 35
#                 }
#             ]
#         },
#         {
#             "question": {
#                 "id": "q2",
#                 "title": "Question 2",
#                 "body": "How to implement a dataset in PyTorch?"
#             },
#             "answers": [
#                 {
#                     "id": "a6",
#                     "body": "Inherit from torch.utils.data.Dataset",
#                     "score": 92
#                 },
#                 {
#                     "id": "a7",
#                     "body": "Override __getitem__ and __len__",
#                     "score": 88
#                 },
#                 {
#                     "id": "a8",
#                     "body": "Create a class with __getitem__, __len__, and collate_fn",
#                     "score": 75
#                 },
#                 {
#                     "id": "a9",
#                     "body": "Use DataLoader with your custom Dataset class",
#                     "score": 62
#                 }
#             ]
#         },
#         {
#             "question": {
#                 "id": "q3",
#                 "title": "Question 3",
#                 "body": "What is a loss function?"
#             },
#             "answers": [
#                 {
#                     "id": "a10",
#                     "body": "A function that measures model error",
#                     "score": 90
#                 },
#                 {
#                     "id": "a11",
#                     "body": "A metric that quantifies the difference between predicted and actual values",
#                     "score": 85
#                 },
#                 {
#                     "id": "a12",
#                     "body": "A mathematical function that evaluates how well a model performs",
#                     "score": 70
#                 }
#             ]
#         }
#     ]
    
#     # Create a temporary file with the data
#     with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
#         json.dump(data, f)
#         temp_path = f.name
    
#     yield temp_path
    
#     # Cleanup
#     os.unlink(temp_path)

# @pytest.fixture
# def mock_tokenizer():
#     """Return a mock tokenizer for testing"""
#     # For testing, we can use a simple tokenizer
#     class MockTokenizer:
#         def __call__(self, text, max_length=10, padding=None, truncation=None, return_tensors=None):
#             # Just return a fixed tensor for testing
#             return {
#                 'input_ids': torch.ones(1, max_length, dtype=torch.long),
#                 'attention_mask': torch.ones(1, max_length, dtype=torch.long)
#             }
    
#     return MockTokenizer()

# @pytest.fixture
# def mock_model():
#     """Return a mock model for testing"""
#     model = QAEmbeddingModel(embed_dim=768, projection_dim=128)
    
#     # Replace forward method with a simple mock
#     def mock_forward(input_ids, attention_mask=None):
#         batch_size = input_ids.shape[0]
#         # Return normalized random embeddings
#         embeddings = torch.randn(batch_size, 128)
#         return embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    
#     model.forward = mock_forward
#     return model


# def test_infonce_pipeline(mock_qa_data, mock_tokenizer, mock_model):
#     """Test InfoNCE pipeline with standard dataset"""
#     # Create dataset
#     dataset = FlexibleQADataset(
#         mock_qa_data,
#         batch_transform_fn=infonce_batch_transform,
#         batch_size=3,
#         tokenizer=mock_tokenizer,
#         max_length=10
#     )
    
#     # Create dataloader
#     dataloader = FlexibleQADataset.get_dataloader(dataset)
    
#     # Create loss function
#     loss_fn = StandardInfoNCELoss(temperature=0.1)
    
#     # Test full pipeline
#     for batch in dataloader:
#         # Get embeddings
#         q_embeddings = mock_model(batch['q_input_ids'])
#         a_embeddings = mock_model(batch['a_input_ids'])
        
#         # Calculate loss
#         loss, metrics = loss_fn(q_embeddings, a_embeddings)
        
#         # Basic validation
#         assert isinstance(loss, torch.Tensor)
#         assert loss.dim() == 0  # Scalar tensor
#         assert loss.item() > 0  # Loss should be positive
        
#         # Check metrics
#         assert 'loss' in metrics
#         assert 'q2a_acc' in metrics
#         assert 'a2q_acc' in metrics
        
#         # Only need to test one batch
#         break


# def test_rank_infonce_pipeline(mock_qa_data, mock_tokenizer, mock_model):
#     """Test rank-aware InfoNCE pipeline"""
#     # Create dataset with multiple positives
#     dataset = FlexibleQADataset(
#         mock_qa_data,
#         batch_transform_fn=multiple_positives_batch_transform,
#         batch_size=3,
#         tokenizer=mock_tokenizer,
#         max_length=10,
#         pos_count=2
#     )
    
#     # Create dataloader
#     dataloader = FlexibleQADataset.get_dataloader(dataset)
    
#     # Create loss function
#     loss_fn = RankInfoNCELoss(temperature=0.1, use_ranks=True, rank_weight=0.1)
    
#     # Test full pipeline
#     for batch in dataloader:
#         # Get embeddings
#         q_embeddings = mock_model(batch['q_input_ids'])
#         a_embeddings = mock_model(batch['a_input_ids'])
        
#         # Calculate loss
#         loss, metrics = loss_fn(
#             q_embeddings, 
#             a_embeddings,
#             question_ids=batch['question_ids'],
#             ranks=batch['ranks']
#         )
        
#         # Basic validation
#         assert isinstance(loss, torch.Tensor)
#         assert loss.dim() == 0  # Scalar tensor
#         assert loss.item() > 0  # Loss should be positive
        
#         # Only need to test one batch
#         break


# def test_hard_negative_pipeline(mock_qa_data, mock_tokenizer, mock_model):
#     """Test hard negative pipeline"""
#     # Create dataset
#     dataset = FlexibleQADataset(
#         mock_qa_data,
#         batch_transform_fn=hard_negative_batch_transform,
#         batch_size=3,
#         tokenizer=mock_tokenizer,
#         max_length=10
#     )
    
#     # Create dataloader
#     dataloader = FlexibleQADataset.get_dataloader(dataset)
    
#     # Create loss function
#     loss_fn = HardNegativeInfoNCELoss(temperature=0.1, hard_negative_weight=1.0)
    
#     # Test full pipeline
#     for batch_items in dataloader:
#         for item in batch_items:
#             # Process each question with its answers
#             q_embedding = mock_model(item['q_input_ids'].unsqueeze(0))
            
#             # Get embeddings for all answers
#             a_embeddings = []
#             a_ids = []
#             for answer in item['answers']:
#                 a_embedding = mock_model(answer['input_ids'].unsqueeze(0))
#                 a_embeddings.append(a_embedding)
#                 a_ids.append(answer['id'])
            
#             a_embeddings = torch.cat(a_embeddings, dim=0)
            
#             # Create question_ids list (all same ID for this question's answers)
#             question_ids = [item['question_id']] * len(a_ids)
            
#             # Calculate loss
#             loss, metrics = loss_fn(
#                 q_embedding.repeat(len(a_ids), 1),  # Repeat question embedding
#                 a_embeddings,
#                 question_ids=question_ids
#             )
            
#             # Basic validation
#             assert isinstance(loss, torch.Tensor)
#             assert loss.dim() == 0  # Scalar tensor
#             assert loss.item() > 0  # Loss should be positive
            
#             # Check metrics
#             assert 'loss' in metrics
#             assert 'hard_neg_count' in metrics
            
#             # Only need to test one item
#             break
#         break


# def test_triplet_pipeline(mock_qa_data, mock_tokenizer, mock_model):
#     """Test triplet loss pipeline"""
#     # Create dataset
#     dataset = FlexibleQADataset(
#         mock_qa_data,
#         batch_transform_fn=triplet_batch_transform,
#         batch_size=3,
#         tokenizer=mock_tokenizer,
#         max_length=10,
#         neg_strategy="hard_negative"
#     )
    
#     # Create dataloader
#     dataloader = FlexibleQADataset.get_dataloader(dataset)
    
#     # Create loss function
#     loss_fn = TripletLoss(margin=0.3)
    
#     # Test full pipeline
#     for batch in dataloader:
#         # Get embeddings
#         q_embeddings = mock_model(batch['q_input_ids'])
#         a_pos_embeddings = mock_model(batch['a_pos_input_ids'])
#         a_neg_embeddings = mock_model(batch['a_neg_input_ids'])
        
#         # Calculate loss
#         loss, metrics = loss_fn(q_embeddings, a_pos_embeddings, a_neg_embeddings)
        
#         # Basic validation
#         assert isinstance(loss, torch.Tensor)
#         assert loss.dim() == 0  # Scalar tensor
#         assert loss.item() >= 0  # Loss should be non-negative
        
#         # Check metrics
#         assert 'loss' in metrics
#         assert 'acc' in metrics
#         assert 'avg_pos_sim' in metrics
#         assert 'avg_neg_sim' in metrics
        
#         # Only need to test one batch
#         break


# def test_listwise_pipeline(mock_qa_data, mock_tokenizer, mock_model):
#     """Test listwise ranking pipeline"""
#     # Create dataset
#     dataset = FlexibleQADataset(
#         mock_qa_data,
#         batch_transform_fn=listwise_batch_transform,
#         batch_size=3,
#         tokenizer=mock_tokenizer,
#         max_length=10,
#         max_answers=3
#     )
    
#     # Create dataloader
#     dataloader = FlexibleQADataset.get_dataloader(dataset)
    
#     # Create loss function
#     loss_fn = ListwiseRankingLoss(temperature=1.0)
    
#     # Test full pipeline
#     for batch_items in dataloader:
#         # Prepare inputs for listwise loss
#         q_embeddings = []
#         a_list_embeddings = []
#         a_list_scores = []
        
#         for item in batch_items:
#             # Get question embedding
#             q_embedding = mock_model(item['q_input_ids'].unsqueeze(0))
            
#             # Get embeddings for all answers
#             a_embeddings = mock_model(item['a_input_ids'])
            
#             q_embeddings.append(q_embedding)
#             a_list_embeddings.append(a_embeddings)
#             a_list_scores.append(item['scores'])
        
#         # Calculate loss
#         loss, metrics = loss_fn(q_embeddings, a_list_embeddings, a_list_scores)
        
#         # Basic validation
#         assert isinstance(loss, torch.Tensor)
#         assert loss.dim() == 0  # Scalar tensor
#         assert loss.item() > 0  # Loss should be positive
        
#         # Check metrics
#         assert 'loss' in metrics
#         assert 'ndcg' in metrics
#         assert 0 <= metrics['ndcg'] <= 1
        
#         # Only need to test one batch
#         break


# def test_pipeline_combinations(mock_qa_data, mock_tokenizer, mock_model):
#     """Test different combinations of dataset transformations and loss functions"""
#     # Define combinations to test
#     combinations = [
#         (infonce_batch_transform, StandardInfoNCELoss(temperature=0.1)),
#         (multiple_positives_batch_transform, RankInfoNCELoss(temperature=0.1, use_ranks=True)),
#         (hard_negative_batch_transform, HardNegativeInfoNCELoss(temperature=0.1)),
#         (triplet_batch_transform, TripletLoss(margin=0.3)),
#         (listwise_batch_transform, ListwiseRankingLoss(temperature=1.0))
#     ]
    
#     for transform_fn, loss_fn in combinations:
#         # Create dataset
#         dataset = FlexibleQADataset(
#             mock_qa_data,
#             batch_transform_fn=transform_fn,
#             batch_size=3,
#             tokenizer=mock_tokenizer,
#             max_length=10
#         )
        
#         # Verify dataset was created
#         assert len(dataset) > 0
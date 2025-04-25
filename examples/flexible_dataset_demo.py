#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for the flexible dataset and loss functions.
Shows how to create different dataset configurations and train with them.
"""

import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from rank_test.flexible_dataset import (
    FlexibleQADataset,
    infonce_batch_transform, 
    multiple_positives_batch_transform,
    hard_negative_batch_transform,
    triplet_batch_transform,
    listwise_batch_transform
)

from rank_test.flexible_losses import (
    StandardInfoNCELoss,
    RankInfoNCELoss,
    HardNegativeInfoNCELoss,
    MultiplePositivesLoss,
    TripletLoss,
    ListwiseRankingLoss,
    create_flexible_loss
)

from rank_test.models import QAEmbeddingModel


def print_separator(title):
    """Print a separator with title"""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def create_sample_data(output_path="data/sample_qa.json"):
    """Create a sample dataset for demonstration"""
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
                    "score": 95 
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
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write data to file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created sample dataset at {output_path}")
    return output_path


def demo_infonce_dataset(data_path, tokenizer):
    """Demonstrate InfoNCE dataset and loss"""
    print_separator("InfoNCE Dataset and Loss")
    
    # Create dataset
    dataset = FlexibleQADataset(
        data_path,
        batch_transform_fn=infonce_batch_transform,
        batch_size=3,
        tokenizer=tokenizer,
        max_length=128,
        take_top=True  # Use top-ranked answer for each question
    )
    
    print(f"Created InfoNCE dataset with {len(dataset)} batches")
    
    # Create dataloader
    dataloader = FlexibleQADataset.get_dataloader(dataset)
    
    # Create loss function
    loss_fn = StandardInfoNCELoss(temperature=0.1)
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Sample model (for demonstration)
    model = QAEmbeddingModel(embed_dim=768, projection_dim=128)
    
    # Process a batch
    for batch in dataloader:
        print("\nProcessing batch:")
        print(f"  Questions: {len(batch['question_ids'])}")
        print(f"  Input shapes: {batch['q_input_ids'].shape}, {batch['a_input_ids'].shape}")
        
        # Get embeddings
        with torch.no_grad():
            q_embeddings = model(batch['q_input_ids'])
            a_embeddings = model(batch['a_input_ids'])
        
        # Calculate loss
        loss, metrics = loss_fn(q_embeddings, a_embeddings)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Metrics: {metrics}")
        break


def demo_hard_negative_dataset(data_path, tokenizer):
    """Demonstrate hard negative dataset and loss"""
    print_separator("Hard Negative Dataset and Loss")
    
    # Create dataset
    dataset = FlexibleQADataset(
        data_path,
        batch_transform_fn=hard_negative_batch_transform,
        batch_size=3,
        tokenizer=tokenizer,
        max_length=128
    )
    
    print(f"Created hard negative dataset with {len(dataset)} batches")
    
    # Create dataloader
    dataloader = FlexibleQADataset.get_dataloader(dataset)
    
    # Create loss function
    loss_fn = HardNegativeInfoNCELoss(temperature=0.1, hard_negative_weight=1.0)
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Sample model (for demonstration)
    model = QAEmbeddingModel(embed_dim=768, projection_dim=128)
    
    # Process a batch
    for batch_items in dataloader:
        print("\nProcessing batch with questions:")
        for item in batch_items:
            print(f"  Question ID: {item['question_id']} with {item['answer_count']} answers")
            
            # Process this question with its answers
            q_embedding = model(item['q_input_ids'].unsqueeze(0))
            
            # Get embeddings for all answers
            a_embeddings = []
            a_ids = []
            for answer in item['answers']:
                a_embedding = model(answer['input_ids'].unsqueeze(0))
                a_embeddings.append(a_embedding)
                a_ids.append(answer['id'])
                print(f"    Answer ID: {answer['id']}, Score: {answer['score']:.1f}, Rank: {answer['rank']}")
            
            a_embeddings = torch.cat(a_embeddings, dim=0)
            
            # Create a batch with this question and all its answers
            question_ids = [item['question_id']] * len(a_ids)
            
            # Calculate loss
            loss, metrics = loss_fn(
                q_embedding.repeat(len(a_ids), 1),  # Repeat question embedding
                a_embeddings,
                question_ids=question_ids
            )
            
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Metrics: {metrics}")
        break


def demo_multiple_positives_dataset(data_path, tokenizer):
    """Demonstrate multiple positives dataset and loss"""
    print_separator("Multiple Positives Dataset and Loss")
    
    # Create dataset
    dataset = FlexibleQADataset(
        data_path,
        batch_transform_fn=multiple_positives_batch_transform,
        batch_size=3,
        tokenizer=tokenizer,
        max_length=128,
        pos_count=3  # Include up to 3 positive answers per question
    )
    
    print(f"Created multiple positives dataset with {len(dataset)} batches")
    
    # Create dataloader
    dataloader = FlexibleQADataset.get_dataloader(dataset)
    
    # Sample model (for demonstration)
    model = QAEmbeddingModel(embed_dim=768, projection_dim=128)
    
    # Process a batch
    for batch in dataloader:
        print("\nProcessing batch:")
        
        # Group by question ID
        question_groups = {}
        for i, q_id in enumerate(batch['question_ids']):
            if q_id not in question_groups:
                question_groups[q_id] = []
            question_groups[q_id].append((i, batch['ranks'][i].item(), batch['scores'][i].item()))
        
        # Print batch structure
        for q_id, indices in question_groups.items():
            print(f"  Question {q_id} with {len(indices)} answers:")
            for idx, rank, score in indices:
                print(f"    Answer at position {idx}: Rank {rank}, Score {score:.1f}")
        
        # Get embeddings
        with torch.no_grad():
            q_embeddings = model(batch['q_input_ids'])
            a_embeddings = model(batch['a_input_ids'])
        
        # Test with different loss configurations
        
        # 1. Rank-based loss
        rank_loss_fn = RankInfoNCELoss(
            temperature=0.1, 
            use_ranks=True, 
            use_scores=False, 
            rank_weight=0.1
        )
        print(f"\nUsing rank-based loss: {rank_loss_fn.get_name()}")
        
        rank_loss, rank_metrics = rank_loss_fn(
            q_embeddings, 
            a_embeddings,
            question_ids=batch['question_ids'],
            ranks=batch['ranks']
        )
        
        print(f"Loss: {rank_loss.item():.4f}")
        print(f"Metrics: {rank_metrics}")
        
        # 2. Score-based loss
        score_loss_fn = RankInfoNCELoss(
            temperature=0.1, 
            use_ranks=False, 
            use_scores=True, 
            score_weight=0.05
        )
        print(f"\nUsing score-based loss: {score_loss_fn.get_name()}")
        
        score_loss, score_metrics = score_loss_fn(
            q_embeddings, 
            a_embeddings,
            question_ids=batch['question_ids'],
            scores=batch['scores']
        )
        
        print(f"Loss: {score_loss.item():.4f}")
        print(f"Metrics: {score_metrics}")
        
        # 3. Combined rank and score loss
        combined_loss_fn = RankInfoNCELoss(
            temperature=0.1, 
            use_ranks=True, 
            use_scores=True, 
            rank_weight=0.1,
            score_weight=0.05
        )
        print(f"\nUsing combined loss: {combined_loss_fn.get_name()}")
        
        combined_loss, combined_metrics = combined_loss_fn(
            q_embeddings, 
            a_embeddings,
            question_ids=batch['question_ids'],
            ranks=batch['ranks'],
            scores=batch['scores']
        )
        
        print(f"Loss: {combined_loss.item():.4f}")
        print(f"Metrics: {combined_metrics}")
        
        # 4. Standard InfoNCE (no rank or score information)
        std_loss_fn = StandardInfoNCELoss(temperature=0.1)
        print(f"\nUsing standard InfoNCE: {std_loss_fn.get_name()}")
        
        std_loss, std_metrics = std_loss_fn(q_embeddings, a_embeddings)
        
        print(f"Loss: {std_loss.item():.4f}")
        print(f"Metrics: {std_metrics}")
        
        # Alternative: Use factory function to create loss from name
        print("\nCreating loss via factory function:")
        factory_loss = create_flexible_loss(
            "infonce_rank0.2_score0.1_t0.1", 
            temperature=0.05  # Override temperature
        )
        print(f"Created: {factory_loss.get_name()}")
        
        break


def demo_triplet_dataset(data_path, tokenizer):
    """Demonstrate triplet dataset and loss"""
    print_separator("Triplet Dataset and Loss")
    
    # Create dataset
    dataset = FlexibleQADataset(
        data_path,
        batch_transform_fn=triplet_batch_transform,
        batch_size=3,
        tokenizer=tokenizer,
        max_length=128,
        neg_strategy="hard_negative"  # Use lower-ranked answers as negatives
    )
    
    print(f"Created triplet dataset with {len(dataset)} batches")
    
    # Create dataloader
    dataloader = FlexibleQADataset.get_dataloader(dataset)
    
    # Create loss function
    loss_fn = TripletLoss(margin=0.3)
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Sample model (for demonstration)
    model = QAEmbeddingModel(embed_dim=768, projection_dim=128)
    
    # Process a batch
    for batch in dataloader:
        print("\nProcessing batch:")
        print(f"  Questions: {len(batch['question_ids'])}")
        print(f"  Input shapes:")
        print(f"    Questions: {batch['q_input_ids'].shape}")
        print(f"    Positive answers: {batch['a_pos_input_ids'].shape}")
        print(f"    Negative answers: {batch['a_neg_input_ids'].shape}")
        
        # Get embeddings
        with torch.no_grad():
            q_embeddings = model(batch['q_input_ids'])
            a_pos_embeddings = model(batch['a_pos_input_ids'])
            a_neg_embeddings = model(batch['a_neg_input_ids'])
        
        # Print score information
        for i, q_id in enumerate(batch['question_ids']):
            print(f"  Question {q_id}:")
            print(f"    Positive score: {batch['pos_scores'][i].item():.1f}")
            print(f"    Negative score: {batch['neg_scores'][i].item():.1f}")
        
        # Calculate loss
        loss, metrics = loss_fn(q_embeddings, a_pos_embeddings, a_neg_embeddings)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Metrics: {metrics}")
        break


def demo_listwise_dataset(data_path, tokenizer):
    """Demonstrate listwise dataset and loss"""
    print_separator("Listwise Dataset and Loss")
    
    # Create dataset
    dataset = FlexibleQADataset(
        data_path,
        batch_transform_fn=listwise_batch_transform,
        batch_size=3,
        tokenizer=tokenizer,
        max_length=128,
        max_answers=4  # Include up to 4 answers per question
    )
    
    print(f"Created listwise dataset with {len(dataset)} batches")
    
    # Create dataloader
    dataloader = FlexibleQADataset.get_dataloader(dataset)
    
    # Create loss function
    loss_fn = ListwiseRankingLoss(temperature=1.0)
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Sample model (for demonstration)
    model = QAEmbeddingModel(embed_dim=768, projection_dim=128)
    
    # Process a batch
    for batch_items in dataloader:
        print("\nProcessing batch with questions:")
        
        # Prepare inputs for listwise loss
        q_embeddings = []
        a_list_embeddings = []
        a_list_scores = []
        
        for item in batch_items:
            print(f"  Question ID: {item['question_id']} with {item['answer_count']} answers")
            print(f"  Normalized scores: {item['scores'].tolist()}")
            
            # Get question embedding
            q_embedding = model(item['q_input_ids'].unsqueeze(0))
            
            # Get embeddings for all answers
            a_embeddings = model(item['a_input_ids'])
            
            q_embeddings.append(q_embedding)
            a_list_embeddings.append(a_embeddings)
            a_list_scores.append(item['scores'])
        
        # Calculate loss
        loss, metrics = loss_fn(q_embeddings, a_list_embeddings, a_list_scores)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"NDCG: {metrics['ndcg']:.4f}")
        break


def main():
    parser = argparse.ArgumentParser(description="Demo for flexible datasets and losses")
    parser.add_argument("--data-path", type=str, default=None, 
                       help="Path to QA dataset (will create sample if not provided)")
    args = parser.parse_args()
    
    # Create or use dataset
    data_path = args.data_path
    if data_path is None:
        data_path = create_sample_data()
    
    # Create tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Run demos
    demo_infonce_dataset(data_path, tokenizer)
    demo_hard_negative_dataset(data_path, tokenizer)
    demo_multiple_positives_dataset(data_path, tokenizer)
    demo_triplet_dataset(data_path, tokenizer)
    demo_listwise_dataset(data_path, tokenizer)
    
    print_separator("Demo Complete")
    print("This demo showcased different dataset transformations and loss functions.")
    print("For actual training, you would combine these components with a training loop.")


if __name__ == "__main__":
    main()
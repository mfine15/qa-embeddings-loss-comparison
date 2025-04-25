#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flexible dataset module that supports various sampling strategies
for QA ranking tasks. This module provides configurable approaches
for creating training data from ranked QA pairs.
"""

import json
import re
import os
import random
import torch
import time
import csv
import sys
from collections import defaultdict
from typing import List, Dict, Callable, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from transformers import DistilBertTokenizer

def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class FlexibleQADataset(Dataset):
    """
    Flexible QA dataset that supports various batch transformation strategies.
    This dataset provides configurable approaches for creating training data
    from ranked QA pairs for different loss functions.
    """
    
    def __init__(
        self, 
        data_path: str, 
        batch_transform_fn: Callable = None, 
        batch_size: int = 16, 
        tokenizer = None,
        max_length: int = 128,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Initialize the dataset with a specific batch transformation strategy.
        
        Args:
            data_path: Path to JSON dataset with ranked QA pairs
            batch_transform_fn: Function that transforms raw data to model-ready batches
            batch_size: Batch size for pre-processing
            tokenizer: Tokenizer to use (will create DistilBERT tokenizer if None)
            max_length: Maximum sequence length for tokenization
            shuffle: Whether to shuffle data during batch creation
            **kwargs: Additional parameters for the batch transform function
        """
        # Load raw data
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
            
        # Store tokenizer and other parameters
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length
        
        # Use provided transform or default
        self.batch_transform_fn = batch_transform_fn or infonce_batch_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        
        # Pre-process into batches
        self.batches = self._create_batches()
        
    def _create_batches(self):
        """Transform raw data into batches based on the strategy"""
        # Create mini-batches of raw data
        num_items = len(self.raw_data)
        indices = list(range(num_items))
        
        if self.shuffle:
            random.shuffle(indices)
        
        batches = []
        total_docs = 0
        
        for i in trange(0, num_items, self.batch_size, desc="Creating batches"):
            batch_indices = indices[i:i+self.batch_size]
            batch_data = [self.raw_data[idx] for idx in batch_indices]
            
            # Apply the transform to create model-ready batch
            result = self.batch_transform_fn(
                batch_data, 
                self.tokenizer, 
                self.max_length, 
                **self.kwargs
            )
            
            # Handle return value (either just batch or batch with doc count)
            if isinstance(result, tuple) and len(result) == 2:
                processed_batch, batch_docs = result
                total_docs += batch_docs
            else:
                processed_batch = result
                # Estimate docs if not provided (backward compatibility)
                total_docs += len(batch_data) * 2  # Rough estimate: 1 question + 1 answer per item
            
            if processed_batch:  # Skip empty batches
                # Store batch with cumulative doc count
                batches.append((processed_batch, total_docs))
                
        return batches
    
    def __len__(self):
        """Return number of batches"""
        return len(self.batches)
        
    def __getitem__(self, idx):
        """Return a pre-processed batch with document count"""
        return self.batches[idx]
    
    @staticmethod
    def get_dataloader(dataset, shuffle=False):
        """
        Create a DataLoader for this dataset
        
        Since the dataset already returns batches, use batch_size=1
        """
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=lambda x: x[0]  # Extract single batch from list
        )


# ----- BATCH TRANSFORMATION STRATEGIES -----

def infonce_batch_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int, 
    take_top: bool = True,
    **kwargs
) -> Dict:
    """
    Standard InfoNCE transform with batched tokenization for better performance.
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        take_top: If True, use the highest-ranked answer, otherwise random
        
    Returns:
        Tuple of (batch dictionary with tokenized inputs, document count)
    """
    # Collect all texts first
    questions_to_tokenize = []
    answers_to_tokenize = []
    question_ids = []
    answer_ids = []
    scores = []
    ranks = []
    
    # Document counter
    doc_count = 0
    
    for item in batch_data:
        question = item['question']
        q_text = question['title'] + " " + clean_html(question['body'])
        
        answers = item['answers']
        if not answers:
            continue
            
        # Sort answers by score (descending) to establish ranks
        sorted_answers = sorted(answers, key=lambda a: float(a['score']), reverse=True)
        
        if not take_top and len(sorted_answers) > 1:
            selected_answer = random.choice(sorted_answers)
            rank = sorted_answers.index(selected_answer)
        else:
            selected_answer = sorted_answers[0]
            rank = 0
        
        a_text = clean_html(selected_answer['body'])
        
        # Collect texts and metadata
        questions_to_tokenize.append(q_text)
        answers_to_tokenize.append(a_text)
        question_ids.append(question['id'])
        answer_ids.append(selected_answer['id'])
        scores.append(float(selected_answer['score']))
        ranks.append(rank)
        
        # Count documents (1 question + 1 answer)
        doc_count += 2
    
    if not questions_to_tokenize:
        return None, 0
        
    # Batch tokenize questions
    q_encodings = tokenizer(
        questions_to_tokenize,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Batch tokenize answers
    a_encodings = tokenizer(
        answers_to_tokenize,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Create final batch dictionary
    batch = {
        'q_input_ids': q_encodings['input_ids'],
        'q_attention_mask': q_encodings['attention_mask'],
        'a_input_ids': a_encodings['input_ids'],
        'a_attention_mask': a_encodings['attention_mask'],
        'question_ids': question_ids,
        'answer_ids': answer_ids,
        'scores': torch.tensor(scores, dtype=torch.float32),
        'ranks': torch.tensor(ranks, dtype=torch.long)
    }
    
    return batch, doc_count


def multiple_positives_batch_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int, 
    pos_count: int = 3,
    **kwargs
) -> Dict:
    """
    Multiple positives transform for contrastive learning.
    
    Creates batches where:
    - Each question appears multiple times with different positive answers
    - Each question-answer pair maintains both cardinal scores and ordinal ranks
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        pos_count: Maximum number of positive answers to include per question
        
    Returns:
        Tuple of (batch dictionary with tokenized inputs, document count)
    """
    q_input_ids, q_attention_mask = [], []
    a_input_ids, a_attention_mask = [], []
    question_ids, answer_ids = [], []
    scores, ranks = [], []
    
    # Document counter
    doc_count = 0
    
    for item in batch_data:
        question = item['question']
        q_text = question['title'] + " " + clean_html(question['body'])
        q_id = question['id']
        
        answers = item['answers']
        if not answers:
            continue
            
        # Sort answers by score (descending) to establish ranks
        sorted_answers = sorted(answers, key=lambda a: float(a['score']), reverse=True)
        
        # Take top K answers as positives (or fewer if not enough answers)
        positives = sorted_answers[:min(pos_count, len(sorted_answers))]
        
        # Count documents (1 question + N answers)
        doc_count += 1 + len(positives)
            
        # For each positive answer
        for rank, answer in enumerate(positives):
            a_text = clean_html(answer['body'])
            
            # Tokenize
            q_encoding = tokenizer(q_text, max_length=max_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
            a_encoding = tokenizer(a_text, max_length=max_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
            
            # Add to batch
            q_input_ids.append(q_encoding['input_ids'])
            q_attention_mask.append(q_encoding['attention_mask'])
            a_input_ids.append(a_encoding['input_ids'])
            a_attention_mask.append(a_encoding['attention_mask'])
            question_ids.append(q_id)
            answer_ids.append(answer['id'])
            scores.append(float(answer['score']))  # Cardinal score
            ranks.append(rank)                    # Ordinal rank (position)
    
    if not q_input_ids:
        return None, 0
        
    # Create final batch
    batch = {
        'q_input_ids': torch.cat(q_input_ids, dim=0),
        'q_attention_mask': torch.cat(q_attention_mask, dim=0),
        'a_input_ids': torch.cat(a_input_ids, dim=0),
        'a_attention_mask': torch.cat(a_attention_mask, dim=0),
        'question_ids': question_ids,
        'answer_ids': answer_ids,
        'scores': torch.tensor(scores, dtype=torch.float32),  # Cardinal scores 
        'ranks': torch.tensor(ranks, dtype=torch.long)       # Ordinal ranks
    }
    
    return batch, doc_count


def hard_negative_batch_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int, 
    **kwargs
) -> Dict:
    """
    Hard negative transform that explicitly includes lower-ranked 
    answers as hard negatives.
    
    Creates batches where:
    - Each question includes its top answer and lower-ranked answers
    - Each answer maintains both cardinal scores and ordinal ranks
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        
    Returns:
        Batch dictionary with tokenized inputs for all answers per question
    """
    batch_items = []
    
    for item in batch_data:
        question = item['question']
        q_text = question['title'] + " " + clean_html(question['body'])
        q_id = question['id']
        
        answers = item['answers']
        if len(answers) <= 1:
            continue  # Need multiple answers
        
        # Sort answers by score to establish ranks
        sorted_answers = sorted(answers, key=lambda a: float(a['score']), reverse=True)
            
        # Tokenize question once
        q_encoding = tokenizer(q_text, max_length=max_length, 
                             padding='max_length', truncation=True, return_tensors='pt')
        
        # Tokenize all answers
        a_encodings = []
        for rank, answer in enumerate(sorted_answers):
            a_text = clean_html(answer['body'])
            a_encoding = tokenizer(a_text, max_length=max_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
            
            # Store answer data
            a_encodings.append({
                'input_ids': a_encoding['input_ids'].squeeze(0),
                'attention_mask': a_encoding['attention_mask'].squeeze(0),
                'id': answer['id'],
                'score': float(answer['score']),    # Cardinal score (actual value)
                'rank': rank                       # Ordinal rank (position)
            })
        
        # Create batch item with question and all its answers
        batch_items.append({
            'q_input_ids': q_encoding['input_ids'].squeeze(0),
            'q_attention_mask': q_encoding['attention_mask'].squeeze(0),
            'answers': a_encodings,
            'question_id': q_id,
            'answer_count': len(a_encodings)
        })
    
    return batch_items


def triplet_batch_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int, 
    neg_strategy: str = "hard_negative",
    **kwargs
) -> Dict:
    """
    Triplet transform for triplet loss learning.
    
    Creates batches of triplets where:
    - Each question is paired with a positive and negative answer
    - Different strategies for selecting negatives are supported
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        neg_strategy: Strategy for selecting negatives:
                     "hard_negative" - use lower-ranked answer to same question
                     "in_batch" - use answer from different question
                     "mixed" - randomly mix both strategies
        
    Returns:
        Batch dictionary with tokenized triplets
    """
    q_input_ids, q_attention_mask = [], []
    a_pos_input_ids, a_pos_attention_mask = [], []
    a_neg_input_ids, a_neg_attention_mask = [], []
    question_ids, pos_scores, neg_scores = [], [], []
    
    # Group answers by question for sampling
    question_answers = {}
    for item in batch_data:
        q_id = item['question']['id']
        question_answers[q_id] = item
    
    # Question IDs in this batch
    batch_q_ids = list(question_answers.keys())
    
    for item in batch_data:
        question = item['question']
        q_text = question['title'] + " " + clean_html(question['body'])
        q_id = question['id']
        
        answers = item['answers']
        if not answers:
            continue
            
        # Get positive (top answer)
        pos_answer = answers[0]
        pos_text = clean_html(pos_answer['body'])
        
        # Get negative based on strategy
        neg_text = None
        neg_score = 0
        
        if neg_strategy == "hard_negative":
            # Use lower ranked answer from same question
            if len(answers) > 1:
                neg_answer = answers[-1]  # Lowest ranked
                neg_text = clean_html(neg_answer['body'])
                neg_score = float(neg_answer['score'])
                
        elif neg_strategy == "in_batch":
            # Use answer from different question
            other_questions = [q for q in batch_q_ids if q != q_id]
            if other_questions:
                other_q_id = random.choice(other_questions)
                other_answers = question_answers[other_q_id]['answers']
                if other_answers:
                    other_answer = other_answers[0]  # Top answer from other question
                    neg_text = clean_html(other_answer['body'])
                    neg_score = float(other_answer['score'])
                    
        elif neg_strategy == "mixed":
            # 50% chance to use each strategy
            if random.random() < 0.5 and len(answers) > 1:
                # Hard negative
                neg_answer = answers[-1]
                neg_text = clean_html(neg_answer['body'])
                neg_score = float(neg_answer['score'])
            else:
                # Other question negative
                other_questions = [q for q in batch_q_ids if q != q_id]
                if other_questions:
                    other_q_id = random.choice(other_questions)
                    other_answers = question_answers[other_q_id]['answers']
                    if other_answers:
                        other_answer = other_answers[0]
                        neg_text = clean_html(other_answer['body'])
                        neg_score = float(other_answer['score'])
        
        # Skip if no negative found
        if not neg_text:
            continue
            
        # Tokenize
        q_encoding = tokenizer(q_text, max_length=max_length, 
                             padding='max_length', truncation=True, return_tensors='pt')
        pos_encoding = tokenizer(pos_text, max_length=max_length, 
                               padding='max_length', truncation=True, return_tensors='pt')
        neg_encoding = tokenizer(neg_text, max_length=max_length, 
                               padding='max_length', truncation=True, return_tensors='pt')
        
        # Add to batch
        q_input_ids.append(q_encoding['input_ids'])
        q_attention_mask.append(q_encoding['attention_mask'])
        a_pos_input_ids.append(pos_encoding['input_ids'])
        a_pos_attention_mask.append(pos_encoding['attention_mask'])
        a_neg_input_ids.append(neg_encoding['input_ids'])
        a_neg_attention_mask.append(neg_encoding['attention_mask'])
        question_ids.append(q_id)
        pos_scores.append(float(pos_answer['score']))
        neg_scores.append(neg_score)
    
    if not q_input_ids:
        return None
        
    # Create final batch
    batch = {
        'q_input_ids': torch.cat(q_input_ids, dim=0),
        'q_attention_mask': torch.cat(q_attention_mask, dim=0),
        'a_pos_input_ids': torch.cat(a_pos_input_ids, dim=0),
        'a_pos_attention_mask': torch.cat(a_pos_attention_mask, dim=0),
        'a_neg_input_ids': torch.cat(a_neg_input_ids, dim=0),
        'a_neg_attention_mask': torch.cat(a_neg_attention_mask, dim=0),
        'question_ids': question_ids,
        'pos_scores': torch.tensor(pos_scores, dtype=torch.float32),
        'neg_scores': torch.tensor(neg_scores, dtype=torch.float32)
    }
    
    return batch


def listwise_batch_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int, 
    max_answers: int = 5,
    **kwargs
) -> Dict:
    """
    Listwise transform for listwise ranking losses.
    
    Creates batches where:
    - Each question has multiple answers with scores
    - Answers are ranked by their original scores
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        max_answers: Maximum number of answers to include per question
        
    Returns:
        Batch dictionary with tokenized questions and multiple answers
    """
    batch_items = []
    
    for item in batch_data:
        question = item['question']
        q_text = question['title'] + " " + clean_html(question['body'])
        q_id = question['id']
        
        answers = item['answers']
        if len(answers) < 2:  # Need at least 2 answers for ranking
            continue
            
        # Limit number of answers
        answers = answers[:min(max_answers, len(answers))]
        
        # Tokenize question
        q_encoding = tokenizer(q_text, max_length=max_length, 
                             padding='max_length', truncation=True, return_tensors='pt')
        
        # Tokenize all answers
        a_input_ids = []
        a_attention_masks = []
        scores = []
        
        for answer in answers:
            a_text = clean_html(answer['body'])
            a_encoding = tokenizer(a_text, max_length=max_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
            
            a_input_ids.append(a_encoding['input_ids'])
            a_attention_masks.append(a_encoding['attention_mask'])
            scores.append(float(answer['score']))
        
        # Stack answer tensors
        a_input_ids = torch.cat(a_input_ids, dim=0)
        a_attention_masks = torch.cat(a_attention_masks, dim=0)
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Normalize scores to [0, 1]
        if torch.max(scores) > 0:
            scores = scores / torch.max(scores)
        
        # Add to batch
        batch_items.append({
            'q_input_ids': q_encoding['input_ids'].squeeze(0),
            'q_attention_mask': q_encoding['attention_mask'].squeeze(0),
            'a_input_ids': a_input_ids,
            'a_attention_masks': a_attention_masks,
            'scores': scores,
            'question_id': q_id,
            'answer_count': len(scores)
        })
    
    return batch_items


def standardized_test_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int,
    **kwargs
) -> Dict:
    """
    Creates a standardized test batch with positive, hard negative, and 
    normal negative samples for each question.
    
    This test transform is strategy-agnostic and provides consistent
    evaluation across all training approaches.
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        
    Returns:
        Batch dictionary with tokenized inputs for standardized evaluation
    """
    test_items = []
    
    for item in batch_data:
        question = item['question']
        q_id = question['id']
        q_text = question['title'] + " " + clean_html(question['body'])
        
        # Get all answers for this question, sorted by score
        answers = item['answers']
        if len(answers) < 2:
            continue
            
        sorted_answers = sorted(answers, key=lambda a: float(a['score']), reverse=True)
        
        # Tokenize question
        q_encoding = tokenizer(q_text, max_length=max_length, 
                             padding='max_length', truncation=True, return_tensors='pt')
        
        # Get all answers
        all_answers = []
        
        # Top answer is positive
        if sorted_answers:
            pos_answer = sorted_answers[0]
            pos_text = clean_html(pos_answer['body'])
            pos_encoding = tokenizer(pos_text, max_length=max_length, 
                                  padding='max_length', truncation=True, return_tensors='pt')
            
            all_answers.append({
                'input_ids': pos_encoding['input_ids'].squeeze(0),
                'attention_mask': pos_encoding['attention_mask'].squeeze(0),
                'score': float(pos_answer['score']),
                'rank': 0,
                'answer_id': pos_answer['id'],
                'is_positive': True,
                'is_hard_negative': False
            })
        
        # Add hard negatives (lower-ranked answers to same question)
        for i, answer in enumerate(sorted_answers[1:min(6, len(sorted_answers))]):  # Up to 5 hard negatives
            a_text = clean_html(answer['body'])
            a_encoding = tokenizer(a_text, max_length=max_length, 
                                padding='max_length', truncation=True, return_tensors='pt')
            
            all_answers.append({
                'input_ids': a_encoding['input_ids'].squeeze(0),
                'attention_mask': a_encoding['attention_mask'].squeeze(0),
                'score': float(answer['score']),
                'rank': i+1,
                'answer_id': answer['id'],
                'is_positive': False,
                'is_hard_negative': True
            })
        
        # Add test item
        test_items.append({
            'q_input_ids': q_encoding['input_ids'].squeeze(0),
            'q_attention_mask': q_encoding['attention_mask'].squeeze(0),
            'question_id': q_id,
            'answers': all_answers
        })
    
    # For batch-level negative sampling, add answers from other questions
    # as normal negatives to each question's answer pool
    for i, item in enumerate(test_items):
        for j, other_item in enumerate(test_items):
            if i != j:  # Different question
                # Add top answer from other question as a normal negative
                other_answers = other_item['answers']
                if other_answers:
                    other_top = other_answers[0]  # Top answer
                    
                    # Add to this question's answer pool
                    item['answers'].append({
                        'input_ids': other_top['input_ids'],
                        'attention_mask': other_top['attention_mask'],
                        'score': 0.0,  # Lower score for negatives
                        'rank': 999,  # High rank for negatives
                        'answer_id': other_top.get('answer_id', 'unknown'),
                        'is_positive': False,
                        'is_hard_negative': False,
                        'from_question_id': other_item['question_id']
                    })
    
    # Count documents - for eval we don't track this as closely
    doc_count = 0
    for item in test_items:
        # One question + all its answers
        doc_count += 1 + len(item['answers'])
    
    return test_items, doc_count


# Factory function to get the transform function by name
def get_batch_transform(transform_name: str) -> Callable:
    """
    Get a batch transform function by name
    
    Args:
        transform_name: Name of the transform function
        
    Returns:
        Batch transform function
        
    Raises:
        ValueError: If the transform name is not recognized
    """
    transforms = {
        'infonce': infonce_batch_transform,
        'multiple_positives': multiple_positives_batch_transform,
        'hard_negative': hard_negative_batch_transform,
        'triplet': triplet_batch_transform,
        'listwise': listwise_batch_transform,
        'standardized_test': standardized_test_transform
    }
    
    if transform_name not in transforms:
        raise ValueError(f"Unknown transform: {transform_name}. Available transforms: {list(transforms.keys())}")
    
    return transforms[transform_name]


def parse_from_json(data_path: str, limit: int = None) -> list:
    """
    Load QA pairs from a JSON file
    
    Args:
        data_path: Path to the JSON file
        limit: Maximum number of items to load
        
    Returns:
        List of QA pairs
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    if limit is not None:
        data = data[:limit]
        
    return data


def export_to_json(data: list, output_path: str = "data/ranked_qa.json") -> None:
    """
    Export QA pairs to a JSON file
    
    Args:
        data: List of QA pairs
        output_path: Path to save the JSON file
    """
    print(f"Exporting data to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} question-answer sets to {output_path}")


def download_dataset():
    """
    Download the StackExchange dataset from Kaggle.
    Uses Kaggle API to download and extract the dataset.
    """
    # Create a directory for the dataset if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if files already exist to avoid re-downloading
    if os.path.exists("data/Questions.csv") and os.path.exists("data/Answers.csv"):
        q_size = os.path.getsize("data/Questions.csv")
        a_size = os.path.getsize("data/Answers.csv")
        # If files are reasonable size, assume they're valid
        if q_size > 10000 and a_size > 10000:
            print(f"Found existing dataset files:")
            print(f"  Questions.csv: {q_size/1_000_000:.1f} MB")
            print(f"  Answers.csv: {a_size/1_000_000:.1f} MB")
            return "data"
    
    print("Downloading dataset from Kaggle...")
    
    try:
        # Import the Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        dataset_name = 'stackoverflow/statsquestions'
        print(f"Downloading dataset {dataset_name}...")
        api.dataset_download_files(
            dataset_name, 
            path='data',
            unzip=True
        )
        print("Dataset downloaded and extracted successfully.")
        
        # Verify files were downloaded and are not empty
        if os.path.exists("data/Questions.csv") and os.path.exists("data/Answers.csv"):
            q_size = os.path.getsize("data/Questions.csv")
            a_size = os.path.getsize("data/Answers.csv")
            print(f"Dataset files:")
            print(f"  Questions.csv: {q_size/1_000_000:.1f} MB")
            print(f"  Answers.csv: {a_size/1_000_000:.1f} MB")
            return "data"
        else:
            raise FileNotFoundError("Dataset files not found after download")
            
    except (ImportError, ModuleNotFoundError):
        print("Error: Kaggle API not found. Installing kaggle package...")
        os.system("uv add kaggle")
        print("Kaggle package installed. Please run the script again.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Fall back to using sample data for testing
        print("Falling back to sample data for testing")
        return "data"

def parse_posts(data_dir, limit=None):
    """
    Parse questions and answers from CSV files.
    
    Args:
        data_dir: Path to the directory containing Questions.csv and Answers.csv
        limit: Optional limit on the number of questions to process
        
    Returns:
        questions, answers: Dictionaries containing questions and their answers
    """
    questions_file = os.path.join(data_dir, "Questions.csv")
    answers_file = os.path.join(data_dir, "Answers.csv")
    
    # If files don't exist, return sample data
    if not os.path.exists(questions_file) or not os.path.exists(answers_file):
        print("CSV files not found. Using sample data instead.")
        return _create_sample_data()
    
    print(f"Parsing data from {data_dir}...")
    
    questions = {}  # Dictionary to store questions by ID
    answers = defaultdict(list)  # Dictionary to store answers by parent ID
    
    try:
        # First pass: collect all answers
        answer_count = 0
        start_time = time.time()
        last_update_time = start_time
        
        print("First pass: collecting answers...")
        
        with open(answers_file, 'r', encoding='latin-1') as f:
            # Skip header
            header = next(f)
            
            # Count lines for progress bar (subtract 1 for header)
            total_lines = sum(1 for _ in f) 
            f.seek(0)
            next(f)  # Skip header again
            
            # Create CSV reader
            reader = csv.reader(f)
            
            # Create progress bar
            for row in tqdm(reader, total=total_lines, desc="Parsing answers"):
                # Skip empty rows
                if not row:
                    continue
                
                answer_id, owner_id, creation_date, parent_id, score, body = row[:6]
                
                # Store answer data
                answers[parent_id].append({
                    "id": answer_id,
                    "body": body,
                    "score": int(score),
                    "is_accepted": False  # Will be set later
                })
                answer_count += 1
                
                # Print progress every 1000 answers or every 5 seconds
                current_time = time.time()
                if answer_count % 1000 == 0 or (current_time - last_update_time >= 5 and answer_count % 100 == 0):
                    elapsed = current_time - start_time
                    rate = answer_count / elapsed if elapsed > 0 else 0
                    print(f"  Processed {answer_count} answers... ({rate:.1f} answers/sec)")
                    last_update_time = current_time
        
        elapsed = time.time() - start_time
        print(f"Collected {answer_count} answers for {len(answers)} questions in {elapsed:.1f} seconds.")
        
        # Second pass: collect questions that have answers
        print("Second pass: collecting questions with answers...")
        start_time = time.time()
        last_update_time = start_time
        question_count = 0
        
        with open(questions_file, 'r', encoding='latin-1') as f:
            # Skip header
            header = next(f)
            
            # Count lines for progress bar (subtract 1 for header)
            total_lines = sum(1 for _ in f) 
            f.seek(0)
            next(f)  # Skip header again
            
            # Create CSV reader
            reader = csv.reader(f)
            
            # Create progress bar
            for row in tqdm(reader, total=total_lines, desc="Parsing questions"):
                # Skip empty rows
                if not row:
                    continue
                
                # CSV may not have all columns for every row
                row_data = row[:6] if len(row) >= 6 else row + [''] * (6 - len(row))
                question_id, owner_id, creation_date, score, title, body = row_data
                
                # Only process questions that have answers
                if question_id in answers:
                    questions[question_id] = {
                        "id": question_id,
                        "title": title,
                        "body": body,
                        "score": int(score),
                        "view_count": 0,  # Not available in this dataset
                        "tags": "",       # Not available in this dataset
                        "accepted_answer_id": None  # Will try to determine later
                    }
                    
                    question_count += 1
                    
                    # Print progress every 100 questions or every 5 seconds
                    current_time = time.time()
                    if question_count % 100 == 0 or (current_time - last_update_time >= 5 and question_count % 10 == 0):
                        elapsed = current_time - start_time
                        rate = question_count / elapsed if elapsed > 0 else 0
                        print(f"  Processed {question_count} questions with answers... ({rate:.1f} questions/sec)")
                        last_update_time = current_time
                    
                    # Apply limit if specified
                    if limit and question_count >= limit:
                        print(f"Reached limit of {limit} questions.")
                        break
        
        elapsed = time.time() - start_time
        print(f"Collected {question_count} questions with answers in {elapsed:.1f} seconds.")
        
        # Since we don't have accepted answers in this dataset, 
        # we'll consider the highest scored answer as the accepted one
        print("Ranking answers by score...")
        start_time = time.time()
        
        # Rank answers by score for each question
        for q_id in answers:
            # Sort answers by score (highest first)
            answers[q_id].sort(key=lambda x: x["score"], reverse=True)
            
            # If the question exists in our collection and has answers
            if q_id in questions and answers[q_id]:
                # Set the top-scored answer as the accepted answer
                top_answer = answers[q_id][0]
                top_answer["is_accepted"] = True
                questions[q_id]["accepted_answer_id"] = top_answer["id"]
        
        elapsed = time.time() - start_time
        print(f"Ranked all answers and marked highest-scored answers as accepted in {elapsed:.1f} seconds.")
        
        # If no questions were collected or limit is 0, use sample data
        if not questions or (limit is not None and limit <= 0):
            print("No questions collected. Using sample data instead.")
            return _create_sample_data()
        
        return questions, answers
    
    except Exception as e:
        print(f"Error parsing posts: {e}")
        import traceback
        traceback.print_exc()
        
        # Return sample data in case of error
        print("Falling back to sample data due to error.")
        return _create_sample_data()

def _create_sample_data():
    """Create sample QA data for testing"""
    questions = {}
    answers = {}
    
    # Add a few sample Q&A pairs for testing
    sample_q = {
        "id": "q1",
        "title": "Sample Question",
        "body": "This is a sample question for testing"
    }
    
    sample_a1 = {
        "id": "a1",
        "body": "This is a sample answer",
        "score": 5
    }
    
    sample_a2 = {
        "id": "a2",
        "body": "This is another sample answer",
        "score": 3
    }
    
    questions["q1"] = sample_q
    answers["q1"] = [sample_a1, sample_a2]
    
    return questions, answers

def ensure_dataset_exists(data_path: str = 'data/ranked_qa.json', 
                           data_limit: int = None, 
                           force_regenerate: bool = False) -> None:
    """
    Ensure the dataset exists, generating it if necessary.
    
    Args:
        data_path: Path where the JSON dataset should be stored
        data_limit: Limit the number of questions to process
        force_regenerate: Force regeneration even if file exists
    """
    # Check current limit if file exists
    current_limit = None
    regenerate_needed = force_regenerate
    
    if os.path.exists(data_path) and not force_regenerate:
        # Try to determine the current limit from the dataset
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                current_limit = len(data)
                print(f"Found existing dataset at {data_path} with {current_limit} items")
                
                # If data_limit is specified and different from current, regenerate
                if data_limit is not None and data_limit != current_limit:
                    print(f"Requested limit ({data_limit}) differs from current dataset size ({current_limit})")
                    regenerate_needed = True
                else:
                    return  # Dataset exists with correct limit
        except Exception as e:
            print(f"Error reading existing dataset: {e}")
            regenerate_needed = True  # Regenerate if there's an issue with the file
    else:
        regenerate_needed = True
    
    if regenerate_needed:
        if os.path.exists(data_path):
            action = "Regenerating" if force_regenerate else "Updating"
            print(f"{action} dataset at {data_path} with limit={data_limit}")
        else:
            print(f"Dataset not found at {data_path}. Generating it with limit={data_limit}")
        
        # Get path for data directory
        data_dir = download_dataset()
            
        # Parse posts from CSV files
        print(f"Using CSV files in {data_dir}")
        questions, answers = parse_posts(data_dir, limit=data_limit)
        
        # Convert to the format expected by our JSON dataset
        data = []
        for q_id, question in questions.items():
            if q_id in answers:
                item = {
                    "question": question,
                    "answers": answers[q_id]
                }
                data.append(item)
        
        # Export to JSON
        export_to_json(data, data_path)
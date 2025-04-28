#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch transformation strategies for QA ranking tasks.

This module provides various batch transformation strategies for QA ranking tasks.
These transformations prepare data in different formats suitable for different loss functions:

1. infonce - Standard InfoNCE contrastive learning with in-batch negatives
2. multiple_positives - Handles multiple positive answers per question with rank weighting
3. hard_negative - Explicitly incorporates hard negatives (low-ranked answers to same question)
4. triplet - Creates triplets of (query, positive, negative) for triplet loss
5. listwise - Prepares data for listwise ranking losses, handling multiple ranked answers per query
6. standardized_test - Creates a standardized test batch format for consistent evaluation

Each transformation can be used with the QADataset by setting the appropriate batch_transform_fn.
"""

import torch
from typing import List, Dict, Callable
import re

def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def infonce_batch_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int, 
    take_top: bool = True,
    device="cpu",
    **kwargs
) -> Dict:
    """
    Standard InfoNCE transform with batched tokenization for better performance.
    
    Creates batches where:
    - Each question is paired with its top (or random) answer
    - During training, negatives come from other questions in the same batch
    - Original ranks and scores are preserved for potential weighting
    
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
        return {"items": []}
        
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
    
    return {"batch": batch, "doc_count": doc_count}


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
    - Useful for models that need to learn subtle differences between answers
    
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
        return {"items": []}
        
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
    
    return {"batch": batch, "doc_count": doc_count}


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
    - Designed for loss functions that need to handle hard negatives explicitly
    
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
    
    return {"items": batch_items}


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
    - Specifically designed for triplet loss functions
    
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
    import random
    
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
        return {"items": []}
        
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
    
    return {"batch": batch}


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
    - Designed for listwise ranking loss functions
    
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
    
    return {"items": batch_items}


def standardized_test_transform(
    batch_data: List[Dict], 
    tokenizer, 
    max_length: int,
    device="cpu",
    **kwargs
) -> Dict:
    """
    Creates a standardized test batch with positive, hard negative, and 
    normal negative samples for each question.
    
    This test transform is strategy-agnostic and provides consistent
    evaluation across all training approaches. The format includes:
    - Original positive answer (highest scored)
    - Hard negative answers (lower-ranked answers to same question)
    - Normal negatives (answers from other questions)
    
    Args:
        batch_data: List of raw data items with questions and ranked answers
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (batch dict for standardized evaluation, document count)
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
    
    return {"items": test_items, "doc_count": doc_count}


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

# Utility function to unpack standardized batch dicts
def unpack_batch(batch_data):
    """Given a standardized batch dict, return a list of batch dicts to process."""
    if batch_data is None:
        return []
    if isinstance(batch_data, tuple):
        batch_data = batch_data[0]
    if isinstance(batch_data, dict):
        if "batch" in batch_data:
            return [batch_data["batch"]]
        elif "items" in batch_data:
            return batch_data["items"]
    return []
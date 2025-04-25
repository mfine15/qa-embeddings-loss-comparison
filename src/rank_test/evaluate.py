#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import wandb

def ndcg_at_k(relevances, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k
    
    Args:
        relevances: List of relevance scores in rank order
        k: Cutoff for relevance calculation
        
    Returns:
        NDCG@k score
    """
    relevances = np.array(relevances)
    if len(relevances) < k:
        # Pad with zeros if not enough relevances
        relevances = np.pad(relevances, (0, k - len(relevances)))
    relevances = relevances[:k]
    
    # Calculate DCG
    dcg = np.sum(relevances / np.log2(np.arange(2, k + 2)))
    
    # Calculate ideal DCG
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = np.sum(ideal_relevances / np.log2(np.arange(2, k + 2)))
    
    # Return NDCG
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(relevances, k):
    """
    Calculate Mean Average Precision (MAP) at k
    
    Args:
        relevances: List of relevance scores in rank order
        k: Cutoff for MAP calculation
        
    Returns:
        MAP@k score
    """
    relevances = np.array(relevances)
    if len(relevances) < k:
        # Pad with zeros if not enough relevances
        relevances = np.pad(relevances, (0, k - len(relevances)))
    relevances = relevances[:k]
    
    # Calculate precision at each position
    precisions = np.cumsum(relevances) / np.arange(1, k + 1)
    
    # Calculate average precision
    ap = np.sum(precisions * relevances) / np.sum(relevances) if np.sum(relevances) > 0 else 0.0
    
    return ap

def mrr(relevances):
    """
    Calculate Reciprocal Rank (RR) for a single query
    
    Args:
        relevances: List of relevance scores in rank order
                   (sorted by similarity/relevance in descending order)
        
    Returns:
        Reciprocal Rank score (1/rank of first relevant item)
    """
    relevances = np.array(relevances)
    # Find index of first relevant result
    indices = np.where(relevances > 0)[0]
    if len(indices) > 0:
        rank = indices[0] + 1  # +1 because indices are 0-based
        return 1.0 / rank
    else:
        return 0.0

def evaluate_model(model, test_dataloader, device, k_values=[1, 5, 10]):
    """
    Evaluate model performance on test data with three accuracy types:
    1. Overall accuracy - considering all answers in the test set
    2. In-batch negative accuracy - ignoring hard negatives from same question
    3. Hard negative accuracy - only comparing with answers to the same question
    
    Args:
        model: QAEmbeddingModel
        test_dataloader: DataLoader with test data
        device: Device to run evaluation on
        k_values: List of k values for @k metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Collect all embeddings, relevance scores, and question IDs
    all_q_embeddings = []
    all_a_embeddings = []
    all_question_ids = []  # To track which answers belong to which questions
    
    # Extract data to track which answers belong to which questions
    question_data = {}  # Map question ID to list of answer indices
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Calculating embeddings")):
            # Get embeddings
            q_embeddings = model(batch['q_input_ids'].to(device), 
                                batch['q_attention_mask'].to(device))
            a_embeddings = model(batch['a_input_ids'].to(device), 
                                batch['a_attention_mask'].to(device))
            
            # Store embeddings
            all_q_embeddings.append(q_embeddings.cpu())
            all_a_embeddings.append(a_embeddings.cpu())
            
            # Track question IDs (create indices if not provided)
            batch_size = q_embeddings.shape[0]
            q_ids = batch.get('question_id', [f"{batch_idx}_{i}" for i in range(batch_size)])
            all_question_ids.extend(q_ids)
            
            # Store indices of this batch
            start_idx = sum(len(e) for e in all_a_embeddings[:-1])
            for i, q_id in enumerate(q_ids):
                if q_id not in question_data:
                    question_data[q_id] = []
                question_data[q_id].append(start_idx + i)
    
    # Concatenate all embeddings
    all_q_embeddings = torch.cat(all_q_embeddings, dim=0)
    all_a_embeddings = torch.cat(all_a_embeddings, dim=0)
    
    # Calculate similarity matrix for all question-answer pairs
    similarity = torch.matmul(all_q_embeddings, all_a_embeddings.T)
    
    # Initialize metrics
    metrics = {
        # Overall metrics (all answers)
        'mrr': 0.0,
        'accuracy@1': 0.0,
        
        # In-batch negative metrics (ignoring hard negatives)
        'mrr_in_batch': 0.0,
        'accuracy@1_in_batch': 0.0,
        
        # Hard negative metrics (only answers for same question)
        'mrr_hard_neg': 0.0,
        'accuracy@1_hard_neg': 0.0,
    }
    
    for k in k_values:
        metrics[f'ndcg@{k}'] = 0.0
        metrics[f'map@{k}'] = 0.0
    
    # Dictionary to track metrics for each query
    per_query_metrics = {
        'mrr': [],
        'mrr_in_batch': [],
        'mrr_hard_neg': [],
    }
    
    for k in k_values:
        per_query_metrics[f'ndcg@{k}'] = []
        per_query_metrics[f'map@{k}'] = []
        
    # IMPORTANT: How the three types of accuracy work:
    # 1. Overall accuracy - Ranks across ALL answers in the test set
    #    - For each question, how well its correct answer ranks among all possible answers
    # 2. In-batch negative accuracy - Only considers answers NOT from the same question
    #    - For each question, how well its answer ranks among answers to other questions
    # 3. Hard negative accuracy - Only considers answers to the same question
    #    - For each question, how well its correct answer ranks among all answers for that question
    
    # Calculate metrics for each query
    n_queries = similarity.shape[0]
    for i in range(n_queries):
        # Get similarities for this query
        query_similarities = similarity[i].numpy()
        
        # Create binary relevance (1 for matches, 0 for non-matches)
        relevance = np.zeros_like(query_similarities)
        relevance[i] = 1  # Correct answer is at index i
        
        # Get question ID for this query
        q_id = all_question_ids[i]
        
        # Get indices of all answers to this question (hard negatives)
        hard_negative_indices = question_data[q_id]
        
        # 1. Overall ranking (all answers)
        sorted_indices = np.argsort(-query_similarities)
        sorted_relevances = relevance[sorted_indices]
        
        # Calculate overall MRR and metrics
        query_mrr = mrr(sorted_relevances)
        metrics['mrr'] += query_mrr
        per_query_metrics['mrr'].append(query_mrr)
        
        # Add accuracy@1 - whether the correct answer is ranked first
        metrics['accuracy@1'] += 1.0 if sorted_indices[0] == i else 0.0
        
        # Calculate NDCG@k and MAP@k for overall ranking
        for k in k_values:
            query_ndcg = ndcg_at_k(sorted_relevances, k)
            query_map = map_at_k(sorted_relevances, k)
            
            metrics[f'ndcg@{k}'] += query_ndcg
            metrics[f'map@{k}'] += query_map
            
            per_query_metrics[f'ndcg@{k}'].append(query_ndcg)
            per_query_metrics[f'map@{k}'].append(query_map)
        
        # 2. In-batch negative ranking (excluding hard negatives)
        # Create mask that excludes hard negatives
        in_batch_mask = np.ones_like(query_similarities, dtype=bool)
        for idx in hard_negative_indices:
            if idx != i:  # Keep the correct answer
                in_batch_mask[idx] = False
        
        # Only consider in-batch negatives
        in_batch_similarities = query_similarities[in_batch_mask]
        in_batch_relevance = relevance[in_batch_mask]
        
        # Get correct answer index in this reduced set
        correct_idx = np.where(in_batch_mask)[0].tolist().index(i)
        
        # Sort by similarity (descending)
        in_batch_sorted_indices = np.argsort(-in_batch_similarities)
        in_batch_sorted_relevances = in_batch_relevance[in_batch_sorted_indices]
        
        # Calculate MRR for in-batch negatives
        query_mrr_in_batch = mrr(in_batch_sorted_relevances)
        metrics['mrr_in_batch'] += query_mrr_in_batch
        per_query_metrics['mrr_in_batch'].append(query_mrr_in_batch)
        
        # Add accuracy@1 for in-batch negatives
        metrics['accuracy@1_in_batch'] += 1.0 if in_batch_sorted_indices[0] == correct_idx else 0.0
        
        # 3. Hard negative ranking (only answers to same question)
        # Skip if there's only one answer for this question
        if len(hard_negative_indices) > 1:
            # Get similarities for just the answers to this question
            hard_neg_similarities = query_similarities[hard_negative_indices]
            
            # Create relevance scores for just these answers
            hard_neg_relevance = np.zeros_like(hard_neg_similarities)
            hard_neg_relevance[hard_negative_indices.index(i)] = 1  # Mark correct answer
            
            # Sort by similarity (descending)
            hard_neg_sorted_indices = np.argsort(-hard_neg_similarities)
            hard_neg_sorted_relevances = hard_neg_relevance[hard_neg_sorted_indices]
            
            # Calculate MRR for hard negatives
            query_mrr_hard_neg = mrr(hard_neg_sorted_relevances)
            metrics['mrr_hard_neg'] += query_mrr_hard_neg
            per_query_metrics['mrr_hard_neg'].append(query_mrr_hard_neg)
            
            # Add accuracy@1 for hard negatives
            correct_idx_in_hard_neg = hard_negative_indices.index(i)
            metrics['accuracy@1_hard_neg'] += 1.0 if hard_neg_sorted_indices[0] == correct_idx_in_hard_neg else 0.0
    
    # Average metrics across all queries
    for key in metrics:
        if key == 'mrr_hard_neg' or key == 'accuracy@1_hard_neg':
            # Only average over questions that have multiple answers
            num_questions_with_multiple_answers = sum(len(indices) > 1 for indices in question_data.values())
            if num_questions_with_multiple_answers > 0:
                metrics[key] /= num_questions_with_multiple_answers
        else:
            metrics[key] /= n_queries
    
    # Add std dev for each metric
    for key in per_query_metrics:
        if per_query_metrics[key]:
            metrics[f'{key}_std'] = np.std(per_query_metrics[key])
    
    return metrics

def compare_models(results, output_dir='results'):
    """
    Compare metrics across multiple models
    
    Args:
        results: Dictionary mapping model names to their evaluation metrics
        output_dir: Directory to save comparison results
        
    Returns:
        DataFrame with model comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    models = []
    for model_name, metrics in results.items():
        row = {'model': model_name}
        row.update(metrics)
        models.append(row)
    
    df = pd.DataFrame(models)
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create comparison plots
    plt.figure(figsize=(12, 8))
    
    # Compare MRR
    plt.subplot(2, 2, 1)
    plt.bar(df['model'], df['mrr'])
    plt.title('Mean Reciprocal Rank (MRR)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Compare NDCG@10
    plt.subplot(2, 2, 2)
    plt.bar(df['model'], df['ndcg@10'])
    plt.title('NDCG@10')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Compare MAP@10
    plt.subplot(2, 2, 3)
    plt.bar(df['model'], df['map@10'])
    plt.title('MAP@10')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    
    print(f"Model comparison saved to {output_dir}")
    return df


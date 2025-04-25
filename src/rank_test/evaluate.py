#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import torch
import numpy as np
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Import model and dataset classes
from rank_test.models import QAEmbeddingModel
from rank_test.dataset import QADataset, create_dataloaders

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
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        relevances: List of relevance scores in rank order
        
    Returns:
        MRR score
    """
    relevances = np.array(relevances)
    # Find index of first relevant result
    indices = np.where(relevances > 0)[0]
    if len(indices) > 0:
        return 1.0 / (indices[0] + 1)
    else:
        return 0.0

def evaluate_model(model, test_dataloader, device, k_values=[1, 5, 10]):
    """
    Evaluate model performance on test data
    
    Args:
        model: QAEmbeddingModel
        test_dataloader: DataLoader with test data
        device: Device to run evaluation on
        k_values: List of k values for @k metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Collect all embeddings and relevance scores
    all_q_embeddings = []
    all_a_embeddings = []
    all_relevance_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Calculating embeddings"):
            # Get embeddings
            q_embeddings = model(batch['q_input_ids'].to(device), 
                                batch['q_attention_mask'].to(device))
            a_embeddings = model(batch['a_input_ids'].to(device), 
                                batch['a_attention_mask'].to(device))
            
            # Store embeddings and scores
            all_q_embeddings.append(q_embeddings.cpu())
            all_a_embeddings.append(a_embeddings.cpu())
            if 'relevance' in batch:
                all_relevance_scores.append(batch['relevance'].cpu())
    
    # Concatenate all embeddings
    all_q_embeddings = torch.cat(all_q_embeddings, dim=0)
    all_a_embeddings = torch.cat(all_a_embeddings, dim=0)
    if all_relevance_scores:
        all_relevance_scores = torch.cat(all_relevance_scores, dim=0)
    
    # Calculate similarity matrix for all question-answer pairs
    similarity = torch.matmul(all_q_embeddings, all_a_embeddings.T)
    
    # Initialize metrics
    metrics = {
        'mrr': 0.0,
    }
    for k in k_values:
        metrics[f'ndcg@{k}'] = 0.0
        metrics[f'map@{k}'] = 0.0
    
    # Dictionary to track metrics for each query
    per_query_metrics = {
        'mrr': [],
    }
    for k in k_values:
        per_query_metrics[f'ndcg@{k}'] = []
        per_query_metrics[f'map@{k}'] = []
    
    # Calculate metrics for each query
    n_queries = similarity.shape[0]
    for i in range(n_queries):
        # Get similarities for this query
        query_similarities = similarity[i].numpy()
        
        # Get relevance scores (either from dataset or from ground truth)
        if all_relevance_scores:
            # Use provided relevance scores
            relevance = all_relevance_scores[i].numpy()
        else:
            # Create binary relevance (1 for matches, 0 for non-matches)
            relevance = np.zeros_like(query_similarities)
            relevance[i] = 1  # Correct answer is at index i
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-query_similarities)
        sorted_relevances = relevance[sorted_indices]
        
        # Calculate MRR
        query_mrr = mrr(sorted_relevances)
        metrics['mrr'] += query_mrr
        per_query_metrics['mrr'].append(query_mrr)
        
        # Calculate NDCG@k and MAP@k
        for k in k_values:
            query_ndcg = ndcg_at_k(sorted_relevances, k)
            query_map = map_at_k(sorted_relevances, k)
            
            metrics[f'ndcg@{k}'] += query_ndcg
            metrics[f'map@{k}'] += query_map
            
            per_query_metrics[f'ndcg@{k}'].append(query_ndcg)
            per_query_metrics[f'map@{k}'].append(query_map)
    
    # Average metrics across all queries
    for key in metrics:
        metrics[key] /= n_queries
    
    # Add std dev for each metric
    for key in per_query_metrics:
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

def run_evaluation(config):
    """
    Run evaluation for a given experiment configuration
    
    Args:
        config: Dictionary with experiment configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model_path = config.get('model_path')
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model not found at {model_path}")
    
    model = QAEmbeddingModel(config.get('embed_dim', 768), config.get('projection_dim', 128))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create test dataloader
    dataset = QADataset(
        data_path=config.get('data_path', 'data/ranked_qa.json'),
        split='test',
        test_size=config.get('test_size', 0.2),
        seed=config.get('seed', 42)
    )
    
    _, test_loader = create_dataloaders(
        dataset, 
        batch_size=config.get('batch_size', 32),
        split='test'
    )
    
    # Run evaluation
    metrics = evaluate_model(
        model=model,
        test_dataloader=test_loader,
        device=device,
        k_values=config.get('k_values', [1, 5, 10])
    )
    
    # Print results
    print(f"\nEvaluation results for {os.path.basename(model_path)}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    """Main function for model evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate QA embedding models")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or directory with multiple models")
    parser.add_argument("--data", type=str, default="data/ranked_qa.json", help="Path to the dataset")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project="qa-embeddings-evaluation")
    
    # Determine if evaluating a single model or comparing multiple models
    if os.path.isdir(args.model):
        # Find all model files in the directory
        model_files = [os.path.join(args.model, f) for f in os.listdir(args.model) 
                      if f.endswith('.pt') or f.endswith('.pth')]
        
        if not model_files:
            print(f"No model files found in {args.model}")
            return
        
        # Evaluate each model
        results = {}
        for model_path in model_files:
            model_name = os.path.basename(model_path).replace('.pt', '').replace('.pth', '')
            print(f"\nEvaluating model: {model_name}")
            
            config = {
                'model_path': model_path,
                'data_path': args.data,
                'batch_size': args.batch_size,
                'test_size': args.test_size,
                'seed': args.seed
            }
            
            metrics = run_evaluation(config)
            results[model_name] = metrics
            
            # Log to wandb
            if args.wandb:
                wandb_metrics = {f"{model_name}/{k}": v for k, v in metrics.items()}
                wandb.log(wandb_metrics)
        
        # Compare models
        comparison = compare_models(results, args.output)
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
    else:
        # Evaluate a single model
        config = {
            'model_path': args.model,
            'data_path': args.data,
            'batch_size': args.batch_size,
            'test_size': args.test_size,
            'seed': args.seed
        }
        
        metrics = run_evaluation(config)
        
        # Log to wandb
        if args.wandb:
            wandb.log(metrics)
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
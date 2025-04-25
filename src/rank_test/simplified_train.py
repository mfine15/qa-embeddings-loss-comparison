#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified training module for QA ranking models.
Provides a streamlined training process with unified loss functions.
"""

import os
import time
import torch
import torch.optim as optim
import wandb
import json
import random
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

from rank_test.config import ExperimentConfig, PREDEFINED_CONFIGS
from rank_test.models import QAEmbeddingModel
from rank_test.dataset import (
    QADataset, 
    get_batch_transform,
    ensure_dataset_exists
)
from rank_test.unified_losses import create_unified_loss
from rank_test.evaluate import evaluate_model


def get_device():
    """Get appropriate device for training"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Mac")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU (No GPU acceleration available)")
    
    return device


def create_dataloaders(config: ExperimentConfig):
    """
    Create train and test dataloaders based on configuration
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Ensure dataset exists
    data_path = config.data_path
    ensure_dataset_exists(
        data_path=data_path,
        data_limit=config.get_limit(),
        force_regenerate=config.force_regenerate
    )
    
    # Create tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Get batch transform function
    batch_transform_fn = get_batch_transform(config.batch_transform)
    
    # Load all data
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    
    # Split data based on test size
    num_items = len(all_data)
    test_size = config.test_size
    test_count = int(num_items * test_size)
    
    # Create indices and shuffle
    indices = list(range(num_items))
    random.seed(config.seed)  # Use seed for reproducibility
    random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Create train and test datasets
    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]
    
    print(f"Splitting dataset: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Create training dataset
    print("Creating training dataset")
    train_dataset = QADataset(
        data_path=data_path,
        batch_transform_fn=batch_transform_fn,
        batch_size=config.get_batch_size(),
        tokenizer=tokenizer,
        max_length=128,
        **config.get_batch_transform_kwargs()
    )
    # Override raw data with train split
    train_dataset.raw_data = train_data
    # Recreate batches with train data
    train_dataset.batches = train_dataset._create_batches()
    
    # Create dataloader
    train_loader = QADataset.get_dataloader(train_dataset, shuffle=True)
    
    # Create standardized test dataset
    print("Creating standardized test dataset")
    # Always use the standardized test transform regardless of training strategy
    test_transform_fn = get_batch_transform("standardized_test")
    test_batch_size = min(len(test_data), config.get_batch_size() * 4)  # Use larger batches for testing
    
    test_dataset = QADataset(
        data_path=data_path,
        batch_transform_fn=test_transform_fn,
        batch_size=test_batch_size,
        tokenizer=tokenizer,
        max_length=128
    )
    # Override raw data with test split
    test_dataset.raw_data = test_data
    # Recreate batches with test data
    test_dataset.batches = test_dataset._create_batches()
    
    test_loader = QADataset.get_dataloader(test_dataset, shuffle=False)
    
    print(f"Created training dataset with {len(train_dataset)} batches")
    print(f"Test dataset: {len(test_dataset)} batches")
    
    return train_loader, test_loader


def train(config: ExperimentConfig):
    """
    Simplified training function that uses unified loss functions
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Trained model
    """
    # Setup device, dataset, model
    device = get_device()
    
    # Initialize wandb if enabled
    if config.log_to_wandb and wandb.run is None:
        run_name = f"{config.loss_type}-{config.batch_transform}-{time.strftime('%Y%m%d-%H%M%S')}"
        if config.name:
            run_name = f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Create a clean config for wandb logging
        wandb_config = config.dict()
        
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=wandb_config
        )
        
        # Log loss parameters as a table
        loss_kwargs = config.get_loss_kwargs()
        if loss_kwargs:
            loss_params = [[k, str(v)] for k, v in loss_kwargs.items()]
            wandb.log({"loss_parameters": wandb.Table(columns=["Parameter", "Value"], 
                                                    data=loss_params)})
    
    # Create model
    print(f"Creating model with embed_dim={config.embed_dim} and projection_dim={config.projection_dim}")
    model = QAEmbeddingModel(
        embed_dim=config.embed_dim,
        projection_dim=config.projection_dim
    ).to(device)
    
    # Create dataset and dataloader
    train_loader, test_loader = create_dataloaders(config)
    
    # Create loss function with unified interface
    loss_fn = create_unified_loss(config.loss_type, **config.get_loss_kwargs())
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initial evaluation (if enabled)
    if config.eval_at_zero and test_loader is not None:
        print("\nRunning evaluation at step 0 (initial model)...")
        try:
            model.eval()
            
            # Run evaluation
            initial_metrics = evaluate_model(model, test_loader, device)
            
            # Print metrics
            print("Step 0 evaluation results:")
            for metric_name, metric_value in initial_metrics.items():
                if 'hard_neg' in metric_name:
                    print(f"  {metric_name}: {metric_value:.4f}*")  # Mark fixed metrics
                else:
                    print(f"  {metric_name}: {metric_value:.4f}")
            
            # Log metrics to wandb
            if wandb.run is not None:
                initial_metrics_log = {f"eval/{k}": v for k, v in initial_metrics.items()}
                initial_metrics_log["step"] = 0
                wandb.log(initial_metrics_log)
                
            model.train()
        except Exception as e:
            print(f"Error during initial evaluation: {e}")
            model.train()
    
    # Training loop
    global_step = 0
    cumulative_docs = 0
    
    # Main training loop - epoch based with step counter
    for epoch in range(config.get_epochs()):
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Extract batch and document count
            batch_doc_count = 0
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                batch, batch_doc_count = batch_data
            else:
                batch = batch_data
                batch_doc_count = len(batch) * 2  # Estimate
            
            # Add batch doc count to running total
            cumulative_docs += batch_doc_count
            
            # Process batch
            model.train()
            batch_start = time.time()
            
            # The key simplification: Unified interface for all loss functions
            # Loss function handles different batch formats internally
            loss, batch_metrics = loss_fn(model, batch, device)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update global step
            global_step += 1
            
            # Calculate batch time
            batch_time = time.time() - batch_start
            
            # Log metrics for this batch
            batch_metrics_log = {f"batch/{k}": v for k, v in batch_metrics.items()}
            batch_metrics_log["batch/loss"] = loss.item()
            batch_metrics_log["step"] = global_step
            batch_metrics_log["epoch"] = epoch + 1
            batch_metrics_log["cumulative_docs_seen"] = cumulative_docs
            batch_metrics_log["docs_per_step"] = cumulative_docs / max(global_step, 1)
            batch_metrics_log["batch/time"] = batch_time
            
            # Update progress bar with latest metrics
            progress_desc = f"Epoch {epoch+1} | Step {global_step} | "
            if 'acc' in batch_metrics:
                progress_desc += f"Acc: {batch_metrics['acc']:.4f} | "
            elif 'accuracy' in batch_metrics:
                progress_desc += f"Acc: {batch_metrics['accuracy']:.4f} | "
            elif 'ndcg' in batch_metrics:
                progress_desc += f"NDCG: {batch_metrics['ndcg']:.4f} | "
            progress_desc += f"Loss: {loss.item():.4f}"
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log(batch_metrics_log)
            
            # Run evaluation if requested
            if config.eval_steps and test_loader and global_step % config.eval_steps == 0:
                # Temporarily switch to eval mode
                model.eval()
                
                print(f"\nRunning evaluation at step {global_step}...")
                try:
                    # Run evaluation
                    test_metrics = evaluate_model(model, test_loader, device)
                    
                    # Print metrics
                    print(f"Step {global_step} evaluation results:")
                    for metric_name, metric_value in test_metrics.items():
                        if 'hard_neg' in metric_name:
                            print(f"  {metric_name}: {metric_value:.4f}*") # Mark fixed metrics
                        else:
                            print(f"  {metric_name}: {metric_value:.4f}")
                    
                    # Log metrics
                    if wandb.run is not None:
                        test_metrics_log = {f"eval/{k}": v for k, v in test_metrics.items()}
                        test_metrics_log["step"] = global_step
                        wandb.log(test_metrics_log)
                
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                
                # Switch back to training mode
                model.train()
    
    # Final evaluation
    if test_loader is not None:
        print("\nRunning final evaluation...")
        model.eval()
        final_metrics = evaluate_model(model, test_loader, device)
        
        # Print final metrics
        print("\nFinal metrics:")
        for k, v in final_metrics.items():
            if 'hard_neg' in k:
                print(f"  {k}: {v:.4f}*")  # Mark fixed metrics with an asterisk
            else:
                print(f"  {k}: {v:.4f}")
        
        # Log final metrics to wandb
        if wandb.run is not None:
            wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
            wandb.log({"step": global_step})
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
    
    return model


def main():
    """Main function to run training from command line with simplified API"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train QA model with unified loss functions")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--preset", type=str, choices=list(PREDEFINED_CONFIGS.keys()),
                       help="Use a predefined configuration preset")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with minimal data")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = ExperimentConfig.from_file(args.config)
    elif args.preset:
        print(f"Using preset configuration: {args.preset}")
        config = PREDEFINED_CONFIGS[args.preset]
    else:
        print("Using default configuration")
        config = ExperimentConfig()
    
    # Override configuration with command line arguments
    if args.debug:
        config.debug = True
    if args.no_wandb:
        config.log_to_wandb = False
    
    # Train model
    model = train(config)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
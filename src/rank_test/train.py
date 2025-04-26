#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training module for QA ranking models using HuggingFace datasets.
Provides a streamlined training process with unified loss functions.
"""

import json
import random
import time
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from rank_test.config import ExperimentConfig
from transformers import DistilBertTokenizerFast
from rank_test.transforms import get_batch_transform
from rank_test.models import QAEmbeddingModel
from rank_test.dataset import QADataset, ensure_dataset_exists
from rank_test.losses import create_unified_loss
from rank_test.evaluate import evaluate_model
from pydargs import parse

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
    Training function using HuggingFace datasets
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Trained model
    """
    # Setup device
    device = get_device()
    
    # Initialize wandb if enabled
    if config.log_to_wandb and wandb.run is None:
        run_name = f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.__dict__
        )
        
        # Log loss parameters
        if config.get_loss_kwargs():
            loss_params = [[k, str(v)] for k, v in config.get_loss_kwargs().items()]
            wandb.log({"loss_parameters": wandb.Table(
                columns=["Parameter", "Value"], 
                data=loss_params
            )})
    
    # Create model
    print(f"Creating model with embed_dim={config.embed_dim} and projection_dim={config.projection_dim}")
    model = QAEmbeddingModel(
        embed_dim=config.embed_dim,
        projection_dim=config.projection_dim
    ).to(device)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config)
    
    # Create loss function
    loss_fn = create_unified_loss(config.loss_type, **config.get_loss_kwargs())
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initial evaluation
    if config.eval_at_zero and test_loader is not None:
        print("\nRunning evaluation at step 0 (initial model)...")
        model.eval()
        with torch.no_grad():
            initial_metrics = evaluate_model(model, test_loader, device)
            if wandb.run is not None:
                wandb.log({f"eval/{k}": v for k, v in initial_metrics.items()})
        model.train()
    
    # Training loop
    global_step = 0
    total_docs_processed = 0
    
    for epoch in range(config.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            loss, batch_metrics = loss_fn(model, batch, device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update counters
            global_step += 1
            total_docs_processed += len(batch['labels'])
            
            # Log metrics
            if wandb.run is not None:
                metrics = {
                    'train/loss': loss.item(),
                    'train/step': global_step,
                    'train/epoch': epoch + 1,
                    'train/docs_processed': total_docs_processed,
                    **{f'train/{k}': v for k, v in batch_metrics.items()}
                }
                wandb.log(metrics)
            
            # Evaluation
            if config.eval_steps and global_step % config.eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    eval_metrics = evaluate_model(model, test_loader, device)
                    if wandb.run is not None:
                        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                model.train()
    
    # Final evaluation
    if test_loader is not None:
        print("\nRunning final evaluation...")
        model.eval()
        with torch.no_grad():
            final_metrics = evaluate_model(model, test_loader, device)
            if wandb.run is not None:
                wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
    
    return model

def cli():
    config = parse(ExperimentConfig, add_config_file_argument=True)
    train(config)

if __name__ == "__main__":
    cli()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training module for QA ranking models using HuggingFace datasets.
Provides a streamlined training process with unified loss functions.
"""

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
from rank_test.optimized_dataset import OptimizedQADataset as QADataset, ensure_dataset_exists, parse_from_json
from rank_test.losses import create_unified_loss
from rank_test.evaluate import evaluate_model
from pydargs import parse
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler



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
    Optimized function to create train and test dataloaders.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    start_time = time.time()
    device = get_device()
    
    # Ensure dataset exists
    data_path = config.data_path
    ensure_dataset_exists(
        data_path=data_path,
        data_limit=config.get_limit(),
        force_regenerate=config.force_regenerate
    )
    
    # Create tokenizer (shared between datasets)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Get batch transform functions
    train_transform_fn = get_batch_transform(config.batch_transform)
    test_transform_fn = get_batch_transform("standardized_test")
    
    # Load data once
    all_data = parse_from_json(data_path)
    
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
    
    # Extract train and test data
    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]
    
    print(f"Splitting dataset: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Create training dataset
    print("Creating training dataset")
    train_dataset = QADataset(
        data=train_data,
        batch_transform_fn=train_transform_fn,
        batch_size=config.get_batch_size(),
        tokenizer=tokenizer,
        max_length=128,
        limit=config.get_limit(),
        device=device,
        **config.get_batch_transform_kwargs()
    )
    
    # Create test dataset with standardized format
    print("Creating standardized test dataset")
    test_batch_size = min(len(test_data), config.get_batch_size() * 4)  # Use larger batches for testing
    
    test_dataset = QADataset(
        data=test_data,
        batch_transform_fn=test_transform_fn,
        batch_size=test_batch_size,
        tokenizer=tokenizer,
        max_length=128,
        device=device,
        limit=None  # Use all test data
    )
    
    # Get dataloaders (optimized dataset returns itself)
    train_loader = train_dataset
    test_loader = test_dataset
    
    print(f"Created training dataset with {len(train_dataset)} batches")
    print(f"Test dataset: {len(test_dataset)} batches")
    print(f"Dataloader creation completed in {time.time() - start_time:.2f} seconds")
    
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
    if config.log_to_wandb:
        print("Logging to wandb")
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
    else:
        print("Not logging to wandb")

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

    # Training loop with timing
    global_step = 0
    total_docs_processed = 0
    total_train_time = 0
    
    print(f"Starting training for {config.epochs} epochs")
    
    # Setup profiler
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=tensorboard_trace_handler("./log/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(config.epochs):
            epoch_start = time.time()
            batch_times = []
            
            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                if batch_idx % config.eval_steps == 0 and (batch_idx > 0 or config.eval_at_zero):
                    model.eval()
                    with torch.no_grad():
                        eval_metrics = evaluate_model(model, test_loader, device)
                        if wandb.run is not None:
                            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                    model.train()
                    
                batch_start = time.time()
                
                # Unpack batch data
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    batch, batch_docs = batch_data
                else:
                    batch, batch_docs = batch_data, None
                
                # Forward pass with profiling
                with record_function("forward_pass"):
                    loss, batch_metrics = loss_fn(model, batch, device)
                
                # Backward pass with profiling
                with record_function("backward_pass"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Update counters
                global_step += 1
                total_docs_processed += len(batch['scores']) if 'scores' in batch else 1
                
                # Record timing
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                
                # Log metrics
                if wandb.run is not None:
                    metrics = {
                        'train/loss': loss.item(),
                        'train/step': global_step,
                        'train/epoch': epoch + 1,
                        'train/docs_processed': total_docs_processed,
                        'train/docs_per_second': len(batch['scores']) / batch_time if 'scores' in batch else 1/batch_time,
                        'train/batch_time': batch_time,
                        **{f'train/{k}': v for k, v in batch_metrics.items()}
                    }
                    wandb.log(metrics)
                
                # Step profiler
                prof.step()
                
                # Print occasional timing stats
                if batch_idx % 50 == 0 and batch_idx > 0:
                    avg_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                    speed = len(batch['scores']) / avg_time if 'scores' in batch else 1/avg_time
                    print(f"  Batch {batch_idx}: {avg_time:.4f}s/batch, ~{speed:.1f} examples/s")
                    
                    # Print profiling stats
                    print(prof.key_averages().table(
                        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                        row_limit=50
                    ))
    
            # End of epoch stats
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s ({avg_batch_time:.4f}s/batch avg)")
            
            # Export profiling data
            prof.export_chrome_trace(f"trace_epoch_{epoch}.json")
            
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
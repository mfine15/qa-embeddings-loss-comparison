#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training module for the flexible QA ranking dataset.
Supports various dataset strategies and loss functions with Pydantic-based configuration.
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
from rank_test.losses import create_loss
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


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, config, 
               test_dataloader=None, step_offset=0):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader with training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Configuration object
        test_dataloader: DataLoader with test data for periodic evaluation
        step_offset: Global step count offset from previous epochs
        
    Returns:
        Dictionary of training metrics and global step count
    """
    model.train()
    
    # Training metrics
    epoch_loss = 0
    epoch_metrics = {}
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    # Track batch time for logging
    batch_times = []
    start_time = time.time()
    
    # Global step counter and doc counter
    global_step = step_offset
    cumulative_docs = 0
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Extract batch and document count
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            batch, cumulative_docs = batch_data
        else:
            batch = batch_data
        
        # Update global step
        global_step += 1
        
        # Calculate batch processing time
        batch_start = time.time()
        
        # Process batch based on transform type
        batch_transform = config.batch_transform
        
        if batch_transform == "triplet":
            # Process triplet format
            q_embeddings = model(batch['q_input_ids'].to(device), 
                               batch['q_attention_mask'].to(device))
            a_pos_embeddings = model(batch['a_pos_input_ids'].to(device), 
                                   batch['a_pos_attention_mask'].to(device))
            a_neg_embeddings = model(batch['a_neg_input_ids'].to(device), 
                                   batch['a_neg_attention_mask'].to(device))
            
            # Calculate loss
            loss, batch_metrics = loss_fn(q_embeddings, a_pos_embeddings, a_neg_embeddings)
            
        elif batch_transform == "listwise":
            # Process listwise format - each item in batch has a question and multiple answers
            loss_total = 0
            batch_metrics_sum = {}
            
            for item in batch:
                # Process question
                q_embedding = model(item['q_input_ids'].unsqueeze(0).to(device), 
                                  item['q_attention_mask'].unsqueeze(0).to(device))
                
                # Process all answers for this question
                a_embeddings = model(item['a_input_ids'].to(device), 
                                   item['a_attention_masks'].to(device))
                
                # Calculate loss for this question
                item_loss, item_metrics = loss_fn(
                    [q_embedding], 
                    [a_embeddings], 
                    [item['scores'].to(device)]
                )
                
                # Accumulate loss and metrics
                loss_total += item_loss
                for k, v in item_metrics.items():
                    if k in batch_metrics_sum:
                        batch_metrics_sum[k] += v
                    else:
                        batch_metrics_sum[k] = v
            
            # Average metrics across all items in batch
            batch_count = len(batch)
            if batch_count > 0:
                loss = loss_total / batch_count
                batch_metrics = {k: v / batch_count for k, v in batch_metrics_sum.items()}
            else:
                loss = torch.tensor(0.0, device=device)
                batch_metrics = {'loss': 0.0}
            
        elif batch_transform == "hard_negative":
            # Process hard negative format - each item has a question and multiple answers
            loss_total = 0
            batch_metrics_sum = {}
            total_items = 0
            
            for item in batch:
                # Process question
                q_input_ids = item['q_input_ids'].unsqueeze(0).to(device)
                q_attention_mask = item['q_attention_mask'].unsqueeze(0).to(device)
                q_embedding = model(q_input_ids, q_attention_mask)
                
                # Process answers
                answers = item['answers']
                a_embeddings = []
                
                for answer in answers:
                    a_input_ids = answer['input_ids'].unsqueeze(0).to(device)
                    a_attention_mask = answer['attention_mask'].unsqueeze(0).to(device)
                    a_embedding = model(a_input_ids, a_attention_mask)
                    a_embeddings.append(a_embedding)
                
                a_embeddings = torch.cat(a_embeddings, dim=0)
                
                # Create question_ids for loss function (all same ID)
                question_ids = [item['question_id']] * len(answers)
                
                # Calculate loss
                item_loss, item_metrics = loss_fn(
                    q_embedding.repeat(len(answers), 1), 
                    a_embeddings,
                    question_ids=question_ids
                )
                
                # Accumulate loss and metrics
                loss_total += item_loss
                for k, v in item_metrics.items():
                    if k in batch_metrics_sum:
                        batch_metrics_sum[k] += v
                    else:
                        batch_metrics_sum[k] = v
                
                total_items += 1
            
            # Average metrics
            if total_items > 0:
                loss = loss_total / total_items
                batch_metrics = {k: v / total_items for k, v in batch_metrics_sum.items()}
            else:
                loss = torch.tensor(0.0, device=device)
                batch_metrics = {'loss': 0.0}
            
        else:
            # Standard format (infonce, multiple_positives)
            q_embeddings = model(batch['q_input_ids'].to(device), 
                               batch['q_attention_mask'].to(device))
            a_embeddings = model(batch['a_input_ids'].to(device), 
                               batch['a_attention_mask'].to(device))
            
            # Additional parameters for different loss functions
            kwargs = {}
            if 'question_ids' in batch:
                kwargs['question_ids'] = batch['question_ids']
            if 'ranks' in batch:
                kwargs['ranks'] = batch['ranks'].to(device)
            if 'scores' in batch:
                kwargs['scores'] = batch['scores'].to(device)
            
            # Calculate loss
            loss, batch_metrics = loss_fn(q_embeddings, a_embeddings, **kwargs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        for k, v in batch_metrics.items():
            if k in epoch_metrics:
                epoch_metrics[k] += v
            else:
                epoch_metrics[k] = v
        
        # Log metrics for this batch
        batch_metrics_log = {f"batch/{k}": v for k, v in batch_metrics.items()}
        batch_metrics_log["batch/loss"] = loss.item()
        batch_metrics_log["global_step"] = global_step
        batch_metrics_log["cumulative_docs_seen"] = cumulative_docs
        batch_metrics_log["docs_per_step"] = cumulative_docs / max(global_step, 1)
        
        # Calculate batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        batch_metrics_log["batch/time"] = batch_time
        
        # Update progress bar with latest metrics
        progress_desc = f"Epoch {epoch+1} | Step {global_step} | "
        if 'acc' in batch_metrics:
            progress_desc += f"Acc: {batch_metrics['acc']:.4f} | "
        elif 'accuracy' in batch_metrics:
            progress_desc += f"Acc: {batch_metrics['accuracy']:.4f} | "
        elif 'avg_acc' in batch_metrics:
            progress_desc += f"Acc: {batch_metrics['avg_acc']:.4f} | "
        progress_desc += f"Loss: {loss.item():.4f}"
        progress_bar.set_description(progress_desc)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log(batch_metrics_log)
        
        # Run evaluation if requested
        if config.eval_steps and test_dataloader and global_step % config.eval_steps == 0:
            # Temporarily switch to eval mode
            model.eval()
            
            print(f"\nRunning evaluation at step {global_step}...")
            try:
                # Run evaluation
                test_metrics = evaluate_model(model, test_dataloader, device)
                
                # Print metrics
                print(f"Step {global_step} evaluation results:")
                for metric_name, metric_value in test_metrics.items():
                    if 'hard_neg' in metric_name:
                        print(f"  {metric_name}: {metric_value:.4f}*") # Mark fixed metrics
                    else:
                        print(f"  {metric_name}: {metric_value:.4f}")
                
                # Log metrics
                if wandb.run is not None:
                    test_metrics_log = {f"step_eval/{k}": v for k, v in test_metrics.items()}
                    test_metrics_log["global_step"] = global_step
                    wandb.log(test_metrics_log)
            
            except Exception as e:
                print(f"Error during evaluation: {e}")
            
            # Switch back to training mode
            model.train()
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    epoch_loss /= max(num_batches, 1)
    for k in epoch_metrics:
        epoch_metrics[k] /= max(num_batches, 1)
    
    # Add loss to metrics
    epoch_metrics['loss'] = epoch_loss
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    epoch_metrics['time'] = epoch_time
    epoch_metrics['avg_batch_time'] = sum(batch_times) / max(len(batch_times), 1)
    
    # Print epoch summary
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")
    for k, v in epoch_metrics.items():
        if k not in ['loss', 'time', 'avg_batch_time']:
            print(f"  {k}: {v:.4f}")
    
    # Log epoch metrics to wandb
    if wandb.run is not None:
        wandb.log({f"epoch/{k}": v for k, v in epoch_metrics.items()})
        wandb.log({"epoch": epoch+1})
    
    return epoch_metrics, global_step


def train_model(config: ExperimentConfig):
    """
    Train model with the given configuration
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Trained model and evaluation metrics
    """
    # Initialize training
    print("Using enhanced evaluation with standardized test format")
    
    # Ensure dataset exists
    data_path = config.data_path
    ensure_dataset_exists(
        data_path=data_path,
        data_limit=config.get_limit(),
        force_regenerate=config.force_regenerate
    )
    
    # Get device
    device = get_device()
    
    # Initialize wandb
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
    
    # Create tokenizer
    print("Creating tokenizer")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Create dataset based on strategy
    if config.dataset_strategy == "flexible":
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
        
        # Create flexible dataset for training
        print("Creating flexible dataset")
        dataset = QADataset(
            data_path=data_path,
            batch_transform_fn=batch_transform_fn,
            batch_size=config.get_batch_size(),
            tokenizer=tokenizer,
            max_length=128,
            **config.get_batch_transform_kwargs()
        )
        # Override raw data with train split
        dataset.raw_data = train_data
        # Recreate batches with train data
        dataset.batches = dataset._create_batches()
        
        # Create dataloader
        print("Creating train dataloader")
        train_loader = QADataset.get_dataloader(dataset, shuffle=True)
        
        # Create standardized test dataset that's consistent across all training methods
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
        
        print(f"Created flexible dataset with {len(dataset)} batches")
        print(f"Test dataset: {len(test_dataset)} batches")
    else:
        # TODO: Use standard dataset approach from original code
        raise NotImplementedError("Standard dataset strategy not implemented in this module")
    
    # Create loss function
    loss_kwargs = config.get_loss_kwargs()
    loss_fn = create_loss(config.loss_type, **loss_kwargs)
    print(f"Using loss function: {loss_fn.get_name()}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    model_dir = os.path.join(config.output_dir, run_name if wandb.run is not None else "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(model_dir, "config.json")
    config.save_to_file(config_path)
    print(f"Saved configuration to {config_path}")
    
    # Training loop
    print(f"Starting training for {config.get_epochs()} epochs")
    
    best_metrics = None
    best_loss = float('inf')
    global_step = 0
    
    # Run evaluation at step 0 if requested
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
                initial_metrics_log = {f"step_eval/{k}": v for k, v in initial_metrics.items()}
                initial_metrics_log["global_step"] = 0
                wandb.log(initial_metrics_log)
            
            # Save initial metrics
            initial_metrics_path = os.path.join(model_dir, "initial_metrics.json")
            with open(initial_metrics_path, 'w') as f:
                json.dump(initial_metrics, f, indent=2)
                
            model.train()
        except Exception as e:
            print(f"Error during initial evaluation: {e}")
            model.train()
    
    for epoch in range(config.get_epochs()):
        # Train one epoch with step-based evaluation
        metrics, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            test_dataloader=test_loader,
            step_offset=global_step
        )
        
        # Save checkpoint if requested
        if (epoch + 1) % config.checkpoint_interval == 0 or epoch == config.get_epochs() - 1:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Update best model if loss improved
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            best_metrics = metrics
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            
            # Add flag in wandb
            if wandb.run is not None:
                wandb.log({"best_model": True, "global_step": global_step})
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate on test set
    if test_loader is not None:
        print("\nEvaluating on test set...")
        # Run evaluation
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Print test metrics
        print("\nTest metrics:")
        for k, v in test_metrics.items():
            if 'hard_neg' in k:
                print(f"  {k}: {v:.4f}*")  # Mark fixed metrics with an asterisk
            else:
                print(f"  {k}: {v:.4f}")
        
        # Log test metrics to wandb
        if wandb.run is not None:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        
        # Save test metrics
        test_metrics_path = os.path.join(model_dir, "test_metrics.json")
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
    
    return model, best_metrics


def main():
    """Main function to run training from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train QA model with flexible dataset strategies")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--preset", type=str, choices=list(PREDEFINED_CONFIGS.keys()),
                       help="Use a predefined configuration preset")
    parser.add_argument("--output", type=str, help="Override output directory")
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
    if args.output:
        config.output_dir = args.output
    if args.debug:
        config.debug = True
    if args.no_wandb:
        config.log_to_wandb = False
    
    # Train model
    model, metrics = train_model(config)
    
    print("\nTraining complete!")
    print(f"Best metrics: {metrics}")


if __name__ == "__main__":
    main()
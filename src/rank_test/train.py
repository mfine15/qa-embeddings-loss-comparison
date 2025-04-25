#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.optim as optim
from transformers import DistilBertTokenizer
from tqdm import tqdm
import wandb
import json
import numpy as np

from rank_test.models import QAEmbeddingModel
from rank_test.losses import create_loss
from rank_test.dataset import QADataset, QABatchedDataset, QATripletDataset, create_dataloaders
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

def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, test_dataloader=None, eval_steps=None, step_offset=0):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader with training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        test_dataloader: DataLoader with test data for periodic evaluation
        eval_steps: Number of steps between evaluations
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
    
    # Global step counter
    global_step = step_offset
    
    for batch_idx, batch in enumerate(progress_bar):
        # Update global step
        global_step += 1
        
        # Calculate batch processing time
        batch_start = time.time()
        
        # Move data to device
        if isinstance(dataloader.dataset, QATripletDataset):
            # Handle triplet dataset
            q_embeddings = model(batch['q_input_ids'].to(device), 
                                batch['q_attention_mask'].to(device))
            pos_embeddings = model(batch['pos_input_ids'].to(device), 
                                  batch['pos_attention_mask'].to(device))
            neg_embeddings = model(batch['neg_input_ids'].to(device), 
                                  batch['neg_attention_mask'].to(device))
            
            # Calculate loss
            loss, batch_metrics = loss_fn(q_embeddings, pos_embeddings, neg_embeddings)
            
        elif isinstance(dataloader.dataset, QABatchedDataset):
            # Handle batched dataset
            q_embeddings = model(batch['q_input_ids'].to(device), 
                                batch['q_attention_mask'].to(device))
            
            # For listwise losses, handle multiple answers per question
            batch_size = q_embeddings.shape[0]
            a_list_embeddings = []
            a_list_scores = []
            
            # Process each answer set separately
            for i in range(batch_size):
                a_ids = batch['a_input_ids'][i].to(device)
                a_mask = batch['a_attention_masks'][i].to(device)
                a_emb = model(a_ids, a_mask)
                
                a_list_embeddings.append(a_emb)
                a_list_scores.append(batch['scores'][i])
            
            # Calculate loss
            loss, batch_metrics = loss_fn(
                q_embeddings, None,
                a_list_embeddings=a_list_embeddings,
                a_list_scores=a_list_scores
            )
            
        else:
            # Handle standard QA dataset
            q_embeddings = model(batch['q_input_ids'].to(device), 
                                batch['q_attention_mask'].to(device))
            a_embeddings = model(batch['a_input_ids'].to(device), 
                                batch['a_attention_mask'].to(device))
            
            # Additional parameters for different loss functions
            loss_kwargs = {}
            if hasattr(loss_fn, 'name') and 'mse' in loss_fn.name.lower():
                loss_kwargs['upvote_scores'] = batch['score'].to(device)
            
            # Calculate loss
            loss, batch_metrics = loss_fn(q_embeddings, a_embeddings, **loss_kwargs)
        
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
        
        # Calculate batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        batch_metrics_log["batch/time"] = batch_time
        
        # Update progress bar with latest metrics
        progress_desc = f"Epoch {epoch+1} | Step {global_step} | "
        if 'acc' in batch_metrics:
            progress_desc += f"Acc: {batch_metrics['acc']:.4f} | "
        elif 'avg_acc' in batch_metrics:
            progress_desc += f"Acc: {batch_metrics['avg_acc']:.4f} | "
        progress_desc += f"Loss: {loss.item():.4f}"
        progress_bar.set_description(progress_desc)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log(batch_metrics_log)
        
        # Run evaluation if requested
        if eval_steps and test_dataloader and global_step % eval_steps == 0:
            # Temporarily switch to eval mode
            model.eval()
            
            print(f"\nRunning evaluation at step {global_step}...")
            try:
                # Run evaluation
                test_metrics = evaluate_model(model, test_dataloader, device)
                
                # Print metrics
                print(f"Step {global_step} evaluation results:")
                for metric_name, metric_value in test_metrics.items():
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
    epoch_loss /= num_batches
    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches
    
    # Add loss to metrics
    epoch_metrics['loss'] = epoch_loss
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    epoch_metrics['time'] = epoch_time
    epoch_metrics['avg_batch_time'] = np.mean(batch_times)
    
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

def train_model(config):
    """
    Train model with given configuration
    
    Args:
        config: Dictionary with training configuration
        
    Returns:
        Trained model and evaluation metrics
    """
    # Extract configuration
    data_path = config.get('data_path', 'data/ranked_qa.json')
    output_dir = config.get('output_dir', 'models')
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 5)
    lr = config.get('learning_rate', 2e-5)
    limit = config.get('limit', None)
    debug = config.get('debug', False)
    loss_name = config.get('loss', 'infonce')
    loss_kwargs = config.get('loss_kwargs', {})
    test_size = config.get('test_size', 0.2)
    seed = config.get('seed', 42)
    checkpoint_interval = config.get('checkpoint_interval', 1)
    eval_steps = config.get('eval_steps', None)  # Number of steps between evaluations
    eval_at_zero = config.get('eval_at_zero', False)  # Whether to evaluate before training
    
    # Get device
    device = get_device()
    
    # Initialize wandb
    log_to_wandb = config.get('log_to_wandb', True)
    if log_to_wandb and wandb.run is None:
        run_name = f"{loss_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Create a clean config for wandb logging
        wandb_config = {
            # Data settings
            'data_path': data_path,
            'limit': limit,
            'test_size': test_size,
            
            # Model settings
            'embed_dim': config.get('embed_dim', 768),
            'projection_dim': config.get('projection_dim', 128),
            
            # Training settings
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'loss_type': loss_name,
            'loss_kwargs': loss_kwargs,
            'eval_steps': eval_steps,
            'eval_at_zero': config.get('eval_at_zero', False),
            
            # Hardware
            'device': str(device),
            
            # Debug settings
            'debug': debug
        }
        
        wandb.init(
            project=config.get('wandb_project', "qa-embeddings-comparison"),
            name=run_name,
            config=wandb_config
        )
        
        # Log additional config details as a table
        if isinstance(loss_kwargs, dict) and loss_kwargs:
            loss_params = [[k, str(v)] for k, v in loss_kwargs.items()]
            wandb.log({"loss_parameters": wandb.Table(columns=["Parameter", "Value"], 
                                                     data=loss_params)})
    
    # Create model
    model = QAEmbeddingModel(
        embed_dim=config.get('embed_dim', 768),
        projection_dim=config.get('projection_dim', 128)
    ).to(device)
    
    # Create tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create appropriate dataset based on loss
    if loss_name == 'triplet':
        dataset = QATripletDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            limit=limit,
            split='all',
            test_size=test_size,
            seed=seed
        )
    elif loss_name in ['listwise', 'listwise_no_batch_neg']:
        dataset = QABatchedDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            limit=limit,
            split='all',
            test_size=test_size,
            seed=seed,
            answers_per_question=config.get('answers_per_question', 5)
        )
    else:
        dataset = QADataset(
            data_path=data_path,
            tokenizer=tokenizer,
            limit=limit,
            split='all',
            test_size=test_size,
            seed=seed
        )
    
    # Create train/test split
    train_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        split='both'
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Training set: {len(train_loader.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Create loss function
    loss_fn = create_loss(loss_name, **loss_kwargs)
    print(f"Using loss function: {loss_name}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, run_name if wandb.run is not None else "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    
    best_metrics = None
    best_loss = float('inf')
    global_step = 0
    
    # Run evaluation at step 0 if requested
    if eval_at_zero and test_loader is not None:
        print("\nRunning evaluation at step 0 (initial model)...")
        try:
            model.eval()
            initial_metrics = evaluate_model(model, test_loader, device)
            
            # Print metrics
            print("Step 0 evaluation results:")
            for metric_name, metric_value in initial_metrics.items():
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
    
    for epoch in range(epochs):
        # Train one epoch with step-based evaluation
        metrics, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            test_dataloader=test_loader if eval_steps else None,
            eval_steps=eval_steps,
            step_offset=global_step
        )
        
        # Save checkpoint if requested
        if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
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
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Print test metrics
        print("\nTest metrics:")
        for k, v in test_metrics.items():
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
    """Main function to run training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train QA embedding model with various losses")
    parser.add_argument("--data", type=str, default="data/ranked_qa.json", 
                        help="Path to the QA dataset")
    parser.add_argument("--output", type=str, default="models",
                        help="Directory to save the model")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Limit number of samples to use (for fast iteration)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode with minimal samples")
    parser.add_argument("--loss", type=str, default="infonce",
                        choices=["infonce", "infonce_no_batch_neg", "infonce_no_hard_neg", 
                                "mse", "triplet", "listwise", "listwise_no_batch_neg"],
                        help="Loss function to use")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for InfoNCE and listwise losses")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Margin for triplet loss")
    parser.add_argument("--test-size", type=float, default=0.01,
                        help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="Run evaluation after every N steps (default: None, evaluate only at end of training)")
    parser.add_argument("--eval-at-zero", action="store_true",
                        help="Run evaluation at step 0, before any training")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = {
        'data_path': args.data,
        'output_dir': args.output,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'limit': args.limit,
        'debug': args.debug,
        'loss': args.loss,
        'test_size': args.test_size,
        'seed': args.seed,
        'eval_steps': args.eval_steps,
        'eval_at_zero': args.eval_at_zero,
        'log_to_wandb': not args.no_wandb,
    }
    
    # Add loss-specific parameters
    loss_kwargs = {}
    if args.loss.startswith('infonce'):
        loss_kwargs['temperature'] = args.temperature
    elif args.loss == 'triplet':
        loss_kwargs['margin'] = args.margin
    elif args.loss.startswith('listwise'):
        loss_kwargs['temperature'] = args.temperature
    
    config['loss_kwargs'] = loss_kwargs
    
    # If debug mode, limit samples
    if args.debug:
        config['limit'] = min(config.get('limit', 1000), 50)
    
    # Train model
    model, metrics = train_model(config)
    
    print("\nTraining complete!")
    print(f"Best metrics: {metrics}")

if __name__ == "__main__":
    main()
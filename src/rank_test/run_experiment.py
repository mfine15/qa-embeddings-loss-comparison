#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import pandas as pd
from datetime import datetime

# Use absolute imports
from rank_test.config import ExperimentConfig, default_configs
from rank_test.train import train_model
from rank_test.evaluate import compare_models
from rank_test.dataset import ensure_dataset_exists

def run_single_experiment(config_path=None, config_dict=None, config_name=None):
    """
    Run a single experiment with the given configuration
    
    Args:
        config_path: Path to config file
        config_dict: Configuration dictionary
        config_name: Name of default configuration
        
    Returns:
        Path to the trained model
    """
    # Get configuration
    if config_path:
        config = ExperimentConfig.load(config_path)
    elif config_dict:
        config = ExperimentConfig(**config_dict)
    elif config_name:
        if config_name not in default_configs:
            raise ValueError(f"Unknown configuration: {config_name}")
        config = default_configs[config_name]
    else:
        config = ExperimentConfig()
    
    # Ensure dataset exists
    data_path = config.get('data_path', 'data/ranked_qa.json')
    ensure_dataset_exists(
        data_path=data_path,
        data_limit=config.get_limit()
    )
    
    # Train model
    model, metrics = train_model(config.as_dict())
    
    # Return path to the trained model
    run_name = f"{config.get('loss')}-{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = os.path.join(config.get('output_dir', 'models'), run_name)
    
    return model_dir

def run_experiment_suite(config_names=None, output_dir='results'):
    """
    Run multiple experiments and compare results
    
    Args:
        config_names: List of configuration names to run (None for all)
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    # Determine which configurations to run
    if config_names is None:
        configs_to_run = default_configs.keys()
    else:
        configs_to_run = [name for name in config_names if name in default_configs]
        if not configs_to_run:
            raise ValueError(f"No valid configurations found in: {config_names}")
    
    print(f"Running experiment suite with {len(configs_to_run)} configurations:")
    for name in configs_to_run:
        print(f"  - {name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Result tracking
    results = {}
    model_paths = {}
    
    # Run each experiment
    for name in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}\n")
        
        # Run experiment
        config = default_configs[name]
        model_dir = run_single_experiment(config_dict=config.as_dict())
        model_paths[name] = model_dir
        
        # Load test metrics
        test_metrics_path = os.path.join(model_dir, "test_metrics.json")
        if os.path.exists(test_metrics_path):
            with open(test_metrics_path, 'r') as f:
                test_metrics = json.load(f)
            results[name] = test_metrics
        else:
            print(f"Warning: No test metrics found for {name}")
    
    # Compare results
    if results:
        comparison = compare_models(results, output_dir)
        
        # Save model paths
        paths_df = pd.DataFrame([
            {"model": name, "path": path}
            for name, path in model_paths.items()
        ])
        paths_df.to_csv(os.path.join(output_dir, "model_paths.csv"), index=False)
        
        return comparison
    else:
        print("No results to compare")
        return None

def main():
    """Main function for running experiments"""
    parser = argparse.ArgumentParser(description="Run QA retrieval experiments")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--loss", type=str, help="Loss function to use (default configuration)")
    parser.add_argument("--suite", action="store_true", help="Run experiment suite with all loss functions")
    parser.add_argument("--losses", type=str, nargs='+', help="Run suite with specific loss functions")
    parser.add_argument("--limit", type=int, help="Override sample limit")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--debug", action="store_true", help="Debug mode with minimal samples")
    
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Run experiment suite
    if args.suite or args.losses:
        # Determine losses to run
        losses = args.losses if args.losses else None
        
        # Create output directory with timestamp
        suite_output = os.path.join(args.output, f"suite-{timestamp}")
        
        print(f"Running experiment suite, saving results to {suite_output}")
        results = run_experiment_suite(losses, suite_output)
        
        if results is not None:
            print("\nExperiment Suite Results:")
            print(results.to_string(index=False))
    
    # Run single experiment
    elif args.config or args.loss:
        # Determine configuration
        if args.config:
            config = ExperimentConfig.load(args.config)
        elif args.loss:
            if args.loss in default_configs:
                config = default_configs[args.loss]
            else:
                raise ValueError(f"Unknown loss function: {args.loss}")
        else:
            config = ExperimentConfig()
        
        # Override configuration if specified
        if args.limit is not None:
            config['limit'] = args.limit
        if args.epochs is not None:
            config['epochs'] = args.epochs
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size
        if args.debug:
            config['debug'] = True
        
        # Create output directory with timestamp
        single_output = os.path.join(args.output, f"{config.get('loss')}-{timestamp}")
        config['output_dir'] = single_output
        
        print(f"Running experiment with {config.get('loss')} loss, saving to {single_output}")
        model_dir = run_single_experiment(config_dict=config.as_dict())
        
        print(f"Experiment complete! Model saved to {model_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
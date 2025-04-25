#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the simplified training module.
"""

import torch
from src.rank_test.config import ExperimentConfig
from src.rank_test.train import train, create_unified_loss

# Configure a minimal test run
config = ExperimentConfig(
    name="Simple Test",
    data_path="data/ranked_qa.json",
    limit=50,  # Use only 50 items
    test_size=0.2,
    seed=42,
    
    embed_dim=768,
    projection_dim=128,
    
    batch_size=4,
    epochs=1,
    learning_rate=2e-5,
    
    eval_steps=10,
    eval_at_zero=True,
    debug=True,
    
    log_to_wandb=False,  # Disable wandb for test
    
    dataset_strategy="flexible",
    batch_transform="infonce",  # Test with simple InfoNCE
    loss_type="infonce",
    temperature=0.1
)

# Run training
if __name__ == "__main__":
    print("Testing simplified training module...")
    model = train(config)
    print("Test complete!")
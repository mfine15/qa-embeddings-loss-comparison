#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized dataset module for QA ranking tasks.
This is a high-performance implementation that supports various
batch transformation strategies while minimizing processing overhead.
"""

import json
import os
import random
import time
import csv
import sys
from collections import defaultdict
from typing import Callable, List, Dict, Tuple, Optional, Union
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizerFast
import re

# Utility function to clean HTML tags from text
def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to process a single batch
def process_batch(
    batch_data: List[Dict],
    batch_transform_fn: Callable,
    tokenizer,
    max_length: int,
    device: str = "cpu",
    **kwargs
) -> Tuple[Dict, int]:
    """
    Process a single batch of data with the transform function.
    
    Args:
        batch_data: List of data items for this batch
        batch_transform_fn: Function to transform the batch into model-ready format
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        device: Device to place tensors on
        **kwargs: Additional parameters for the batch transform
        
    Returns:
        Tuple of (processed_batch, doc_count)
    """
    # Apply the transform to create model-ready batch
    result = batch_transform_fn(
        batch_data,
        tokenizer,
        max_length,
        device=device,
        **kwargs
    )
    
    # Handle return value (either just batch or doc count)
    if isinstance(result, tuple) and len(result) == 2:
        processed_batch, batch_docs = result
    else:
        processed_batch = result
        # Estimate docs if not provided (backward compatibility) 
        batch_docs = len(batch_data) * 2  # Rough estimate: 1 question + 1 answer per item
        
    if processed_batch:  # Skip empty batches
        return (processed_batch, batch_docs)
    return None

class OptimizedQADataset(Dataset):
    """
    Optimized QA dataset with improved performance.
    This dataset efficiently processes QA pairs for ranking tasks.
    """
    
    def __init__(
        self, 
        data: list, 
        batch_transform_fn: Callable, 
        batch_size: int = 16, 
        tokenizer = None,
        max_length: int = 128,
        shuffle: bool = True,
        limit: int = None,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize the dataset with a specific batch transformation strategy.
        
        Args:
            data: List of raw data items
            batch_transform_fn: Function that transforms raw data to model-ready batches
            batch_size: Batch size for pre-processing
            tokenizer: Tokenizer to use (will create DistilBERT tokenizer if None)
            max_length: Maximum sequence length for tokenization
            shuffle: Whether to shuffle data during batch creation
            limit: Maximum number of data items to use (applied before batching)
            device: Device to place tensors on
            **kwargs: Additional parameters for the batch transform function
        """
        start_time = time.time()
        
        # Store configuration
        self.data = data
        self.batch_transform_fn = batch_transform_fn
        self.batch_size = batch_size
        self.tokenizer = tokenizer or DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length
        self.shuffle = shuffle
        self.limit = limit
        self.device = device
        self.kwargs = kwargs
        
        # Process the data
        self.batches = self._create_batches()
        
        print(f"Dataset initialization completed in {time.time() - start_time:.2f} seconds")
    
    def _create_batches(self):
        """
        Create model-ready batches from raw data.
        This implementation is optimized for performance.
        """
        start_time = time.time()
        
        # Apply limit if specified
        data = self.data
        if self.limit is not None:
            data = data[:self.limit]
            print(f"Limiting dataset to {self.limit} items")
        
        # Create batch indices
        num_items = len(data)
        indices = list(range(num_items))
        
        if self.shuffle:
            random.shuffle(indices)
        
        # Group items into batches
        batches = []
        for i in range(0, num_items, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_data = [data[idx] for idx in batch_indices]
            batches.append(batch_data)
        
        print(f"Created {len(batches)} batches from {num_items} items (batch size: {self.batch_size})")
        
        # Process batches
        results = []
        for batch_data in tqdm(batches, desc="Processing batches"):
            batch_result = process_batch(
                batch_data,
                self.batch_transform_fn,
                self.tokenizer,
                self.max_length,
                self.device,
                **self.kwargs
            )
            if batch_result:  # Skip empty batches
                results.append(batch_result)
        
        print(f"Batch processing completed in {time.time() - start_time:.2f} seconds")
        return results
    
    def __len__(self):
        """Return number of batches"""
        return len(self.batches)
    
    def __getitem__(self, idx):
        """Return a pre-processed batch"""
        return self.batches[idx]
    
    @staticmethod
    def get_dataloader(dataset, shuffle=False):
        """
        Return the dataset directly since batches are already pre-processed.
        This removes the PyTorch DataLoader overhead.
        """
        return dataset


# Utility functions for handling data files

def parse_from_json(data_path: str, limit: int = None) -> list:
    """
    Load QA pairs from a JSON file.
    
    Args:
        data_path: Path to the JSON file
        limit: Maximum number of items to load
        
    Returns:
        List of QA pairs
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    if limit is not None:
        data = data[:limit]
        
    return data

def export_to_json(data: list, output_path: str = "data/ranked_qa.json") -> None:
    """
    Export QA pairs to a JSON file.
    
    Args:
        data: List of QA pairs
        output_path: Path to save the JSON file
    """
    print(f"Exporting data to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} question-answer sets to {output_path}")

# Import necessary functions from original dataset module
from rank_test.dataset import ensure_dataset_exists
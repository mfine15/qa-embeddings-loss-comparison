#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flexible dataset module that supports various sampling strategies
for QA ranking tasks. This module provides configurable approaches
for creating training data from ranked QA pairs.
"""

import json
import os
import random
import csv
import sys
from collections import defaultdict
from typing import Callable
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

# Function to process a single batch
def process_batch(args):
    """Process a single batch of data and apply the transform function"""
    batch_idx, raw_data, batch_transform_fn, tokenizer, max_length, device, kwargs = args
    # Gather the data for this batch
    batch_data = [raw_data[idx] for idx in batch_idx]
    
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

class QADataset(Dataset):
    """
    Optimized dataset that supports various batch transformation strategies.
    This dataset provides configurable approaches for creating training data
    from ranked QA pairs for different loss functions with improved performance.
    """
    
    def __init__(
        self, 
        data: list, 
        batch_transform_fn: Callable = None, 
        batch_size: int = 16, 
        tokenizer = None,
        max_length: int = 128,
        shuffle: bool = True,
        limit: int = None,
        device = "cpu",
        **kwargs
    ):
        """
        Initialize the dataset with a specific batch transformation strategy.
        
        Args:
            data: List of data items to process
            batch_transform_fn: Function that transforms raw data to model-ready batches
            batch_size: Batch size for pre-processing
            tokenizer: Tokenizer to use (will create DistilBERT tokenizer if None)
            max_length: Maximum sequence length for tokenization
            shuffle: Whether to shuffle data during batch creation
            limit: Maximum number of data items to use (applied before batching)
            device: Device to place tensors on
            **kwargs: Additional parameters for the batch transform function
        """
        # Store input data
        self.raw_data = data
            
        # Store tokenizer and other parameters
        self.tokenizer = tokenizer or DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length
        self.batch_transform_fn = batch_transform_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.limit = limit
        self.device = device
        self.kwargs = kwargs
        
        # Process data into batches
        self.batches = self._create_batches()
        
    def _create_batches(self):
        """Transform raw data into batches based on the strategy"""
        # Create mini-batches of raw data
        raw_data = self.raw_data
        
        # Apply limit before batching if specified
        if self.limit is not None:
            raw_data = raw_data[:self.limit]
            print(f"Limiting dataset to {self.limit} items")
        
        num_items = len(raw_data)
        indices = list(range(num_items))
        
        if self.shuffle:
            random.shuffle(indices)
        
        batches = []
        print(f"Creating batches from {num_items} items with batch size {self.batch_size}")
        print(f"Using tokenizer: {self.tokenizer}")
        
        # Create list of batch indices
        batch_indices = [
            indices[i:i+self.batch_size] 
            for i in range(0, num_items, self.batch_size)
        ]
        
        print(f"Created {len(batch_indices)} batches")
        
        
    
        # results = process_map(process_batch, [(batch_idx, self.raw_data, self.batch_transform_fn, self.tokenizer, self.max_length, self.kwargs) for batch_idx in batch_indices], max_workers=30)
        results = [process_batch((batch_idx, raw_data, self.batch_transform_fn, self.tokenizer, self.max_length, self.kwargs)) for batch_idx in tqdm(batch_indices)]
        # Filter out None results and return
        return [r for r in results if r is not None]
    
    def __len__(self):
        """Return number of batches"""
        return len(self.batches)
        
    def __getitem__(self, idx):
        """Return a pre-processed batch with document count"""
        return self.batches[idx]
    
    @staticmethod
    def get_dataloader(dataset, shuffle=False):
        """
        Create a DataLoader for this dataset
        
        Since the dataset already returns batches, use batch_size=1
        """
        return dataset

def parse_from_json(data_path: str, limit: int = None) -> list:
    """
    Load QA pairs from a JSON file
    
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
    Export QA pairs to a JSON file
    
    Args:
        data: List of QA pairs
        output_path: Path to save the JSON file
    """
    print(f"Exporting data to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} question-answer sets to {output_path}")


def download_dataset():
    """
    Download the StackExchange dataset from Kaggle.
    Uses Kaggle API to download and extract the dataset.
    """
    # Create a directory for the dataset if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if files already exist to avoid re-downloading
    if os.path.exists("data/Questions.csv") and os.path.exists("data/Answers.csv"):
        q_size = os.path.getsize("data/Questions.csv")
        a_size = os.path.getsize("data/Answers.csv")
        # If files are reasonable size, assume they're valid
        if q_size > 10000 and a_size > 10000:
            print("Found existing dataset files:")
            print(f"  Questions.csv: {q_size/1_000_000:.1f} MB")
            print(f"  Answers.csv: {a_size/1_000_000:.1f} MB")
            return "data"
    
    print("Downloading dataset from Kaggle...")
    
    try:
        # Import the Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        dataset_name = 'stackoverflow/statsquestions'
        print(f"Downloading dataset {dataset_name}...")
        api.dataset_download_files(
            dataset_name, 
            path='data',
            unzip=True
        )
        print("Dataset downloaded and extracted successfully.")
        
        # Verify files were downloaded and are not empty
        if os.path.exists("data/Questions.csv") and os.path.exists("data/Answers.csv"):
            q_size = os.path.getsize("data/Questions.csv")
            a_size = os.path.getsize("data/Answers.csv")
            print("Dataset files:")
            print(f"  Questions.csv: {q_size/1_000_000:.1f} MB")
            print(f"  Answers.csv: {a_size/1_000_000:.1f} MB")
            return "data"
        else:
            raise FileNotFoundError("Dataset files not found after download")
            
    except (ImportError, ModuleNotFoundError):
        print("Error: Kaggle API not found. Installing kaggle package...")
        os.system("uv add kaggle")
        print("Kaggle package installed. Please run the script again.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Fall back to using sample data for testing
        print("Falling back to sample data for testing")
        return "data"
def parse_posts(data_dir, limit=None):
    """
    Parse questions and answers from CSV files.
    
    Args:
        data_dir: Path to the directory containing Questions.csv and Answers.csv
        limit: Optional limit on the number of questions to process
        
    Returns:
        questions, answers: Dictionaries containing questions and their answers
    """
    questions_file = os.path.join(data_dir, "Questions.csv")
    answers_file = os.path.join(data_dir, "Answers.csv")
    
    # If files don't exist, return sample data
    if not os.path.exists(questions_file) or not os.path.exists(answers_file):
        print("CSV files not found. Using sample data instead.")
        return _create_sample_data()
    
    print(f"Parsing data from {data_dir}...")
    
    questions = {}  # Dictionary to store questions by ID
    answers = defaultdict(list)  # Dictionary to store answers by parent ID
    
    try:
        # First pass: collect all answers
        print("First pass: collecting answers...")
        
        with open(answers_file, 'r', encoding='latin-1') as f:
            # Skip header
            header = next(f)
            
            # Count lines for progress bar (subtract 1 for header)
            total_lines = sum(1 for _ in f) 
            f.seek(0)
            next(f)  # Skip header again
            
            # Create CSV reader
            reader = csv.reader(f)
            
            # Process answers with progress bar
            for row in tqdm(reader, total=total_lines, desc="Parsing answers"):
                # Skip empty rows
                if not row:
                    continue
                
                answer_id, owner_id, creation_date, parent_id, score, body = row[:6]
                
                # Store answer data
                answers[parent_id].append({
                    "id": answer_id,
                    "body": body,
                    "score": int(score),
                    "is_accepted": False  # Will be set later
                })
        
        # Second pass: collect questions that have answers
        print("Second pass: collecting questions with answers...")
        
        with open(questions_file, 'r', encoding='latin-1') as f:
            # Skip header
            header = next(f)
            
            # Count lines for progress bar (subtract 1 for header)
            total_lines = sum(1 for _ in f) 
            f.seek(0)
            next(f)  # Skip header again
            
            # Create CSV reader
            reader = csv.reader(f)
            
            # Process questions with progress bar
            for row in tqdm(reader, total=total_lines, desc="Parsing questions"):
                # Skip empty rows
                if not row:
                    continue
                
                # CSV may not have all columns for every row
                row_data = row[:6] if len(row) >= 6 else row + [''] * (6 - len(row))
                question_id, owner_id, creation_date, score, title, body = row_data
                
                # Only process questions that have answers
                if question_id in answers:
                    questions[question_id] = {
                        "id": question_id,
                        "title": title,
                        "body": body,
                        "score": int(score),
                        "view_count": 0,  # Not available in this dataset
                        "tags": "",       # Not available in this dataset
                        "accepted_answer_id": None  # Will try to determine later
                    }
                    
                    # Apply limit if specified
                    if limit and len(questions) >= limit:
                        print(f"Reached limit of {limit} questions.")
                        break
        
        # Since we don't have accepted answers in this dataset, 
        # we'll consider the highest scored answer as the accepted one
        print("Ranking answers by score...")
        
        # Rank answers by score for each question
        for q_id in tqdm(answers.keys(), desc="Ranking answers"):
            # Sort answers by score (highest first)
            answers[q_id].sort(key=lambda x: x["score"], reverse=True)
            
            # If the question exists in our collection and has answers
            if q_id in questions and answers[q_id]:
                # Set the top-scored answer as the accepted answer
                top_answer = answers[q_id][0]
                top_answer["is_accepted"] = True
                questions[q_id]["accepted_answer_id"] = top_answer["id"]
        
        # If no questions were collected or limit is 0, use sample data
        if not questions or (limit is not None and limit <= 0):
            print("No questions collected. Using sample data instead.")
            return _create_sample_data()
        
        return questions, answers
    
    except Exception as e:
        print(f"Error parsing posts: {e}")
        import traceback
        traceback.print_exc()
        
        # Return sample data in case of error
        print("Falling back to sample data due to error.")
        return _create_sample_data()

def _create_sample_data():
    """Create sample QA data for testing"""
    questions = {}
    answers = {}
    
    # Add a few sample Q&A pairs for testing
    sample_q = {
        "id": "q1",
        "title": "Sample Question",
        "body": "This is a sample question for testing"
    }
    
    sample_a1 = {
        "id": "a1",
        "body": "This is a sample answer",
        "score": 5
    }
    
    sample_a2 = {
        "id": "a2",
        "body": "This is another sample answer",
        "score": 3
    }
    
    questions["q1"] = sample_q
    answers["q1"] = [sample_a1, sample_a2]
    
    return questions, answers

def ensure_dataset_exists(data_path: str = 'data/ranked_qa.json', 
                           data_limit: int = None, 
                           force_regenerate: bool = False) -> None:
    """
    Ensure the dataset exists, generating it if necessary.
    
    Args:
        data_path: Path where the JSON dataset should be stored
        data_limit: Limit the number of questions to process
        force_regenerate: Force regeneration even if file exists
    """
    # Check current limit if file exists
    current_limit = None
    regenerate_needed = force_regenerate
    
    if os.path.exists(data_path) and not force_regenerate:
        # Try to determine the current limit from the dataset
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                current_limit = len(data)
                print(f"Found existing dataset at {data_path} with {current_limit} items")
                
                # If data_limit is specified and different from current, regenerate
                if data_limit is not None and data_limit != current_limit:
                    print(f"Requested limit ({data_limit}) differs from current dataset size ({current_limit})")
                    regenerate_needed = True
                else:
                    return  # Dataset exists with correct limit
        except Exception as e:
            print(f"Error reading existing dataset: {e}")
            regenerate_needed = True  # Regenerate if there's an issue with the file
    else:
        regenerate_needed = True
    
    if regenerate_needed:
        if os.path.exists(data_path):
            action = "Regenerating" if force_regenerate else "Updating"
            print(f"{action} dataset at {data_path} with limit={data_limit}")
        else:
            print(f"Dataset not found at {data_path}. Generating it with limit={data_limit}")
        
        # Get path for data directory
        data_dir = download_dataset()
            
        # Parse posts from CSV files
        print(f"Using CSV files in {data_dir}")
        questions, answers = parse_posts(data_dir, limit=data_limit)
        
        # Convert to the format expected by our JSON dataset
        data = []
        for q_id, question in questions.items():
            if q_id in answers:
                item = {
                    "question": question,
                    "answers": answers[q_id]
                }
                data.append(item)
        
        # Export to JSON
        export_to_json(data, data_path)
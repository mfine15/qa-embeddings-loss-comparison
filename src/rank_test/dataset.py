#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import torch
import csv
import time
import sys
import sqlite3
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer
from tqdm import tqdm

class QADataset(Dataset):
    """
    Dataset for question-answer pairs with support for train/test splits
    and different loss function requirements.
    """
    def __init__(self, data_path, tokenizer=None, max_length=128, limit=None, 
                 split='train', test_size=0.2, seed=42, include_scores=True):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to JSON dataset
            tokenizer: Optional pre-initialized tokenizer (will load if None)
            max_length: Maximum token length for sequences
            limit: Optional limit on dataset size
            split: 'train', 'test', or 'all'
            test_size: Proportion of data to use for test set
            seed: Random seed for reproducibility
            include_scores: Whether to include upvote scores
        """
        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            self.tokenizer = tokenizer
            
        self.max_length = max_length
        self.include_scores = include_scores
        self.data_path = data_path
        self.limit = limit
        self.test_size = test_size
        self.seed = seed
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        if limit:
            data = data[:limit]
            
        # Process QA pairs
        self.qa_pairs = []
        self.process_data(data)
        
        # Handle train/test splitting
        if split != 'all':
            indices = list(range(len(self.qa_pairs)))
            test_count = int(len(indices) * test_size)
            train_count = len(indices) - test_count
            
            # Use random_split for reproducible splits
            torch.manual_seed(seed)
            train_indices, test_indices = random_split(
                indices, [train_count, test_count]
            )
            
            # Select appropriate indices based on split
            if split == 'train':
                self.qa_pairs = [self.qa_pairs[i] for i in train_indices]
            elif split == 'test':
                self.qa_pairs = [self.qa_pairs[i] for i in test_indices]
    
    def process_data(self, data):
        """Process raw data into QA pairs"""
        for item in data:
            question = item['question']
            answers = item['answers']
            
            # Skip questions with no answers
            if not answers:
                continue
                
            # Clean text
            q_text = question['title'] + " " + self._clean_html(question['body'])
            q_id = question['id']  # Store question ID for tracking
            
            # Process each answer
            for i, answer in enumerate(answers):
                a_text = self._clean_html(answer['body'])
                
                # Create a pair
                pair = {
                    'question': q_text,
                    'question_id': q_id,  # Add question ID for evaluation
                    'answer': a_text,
                    'answer_id': answer['id'],
                    'score': float(answer['score']),
                    'is_best': i == 0  # First answer is highest scored
                }
                self.qa_pairs.append(pair)
    
    def _clean_html(self, text):
        """Remove HTML tags from text"""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def __len__(self):
        return len(self.qa_pairs)
        
    def __getitem__(self, idx):
        """Get a single QA pair with tokenization"""
        pair = self.qa_pairs[idx]
        
        # Tokenize question and answer
        q_encoding = self.tokenizer(
            pair['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        a_encoding = self.tokenizer(
            pair['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create item dict with required fields
        item = {
            'q_input_ids': q_encoding['input_ids'].squeeze(0),
            'q_attention_mask': q_encoding['attention_mask'].squeeze(0),
            'a_input_ids': a_encoding['input_ids'].squeeze(0),
            'a_attention_mask': a_encoding['attention_mask'].squeeze(0),
            'is_best': torch.tensor(pair['is_best'], dtype=torch.float32),
            'question_id': pair['question_id'],  # Add question ID for evaluation
            'answer_id': pair['answer_id'],
        }
        
        # Include scores if requested
        if self.include_scores:
            item['score'] = torch.tensor(pair['score'], dtype=torch.float32)
            
        return item

class QABatchedDataset(QADataset):
    """
    Dataset that creates batches with one question and multiple ranked answers.
    Useful for listwise ranking losses.
    """
    def __init__(self, data_path, tokenizer=None, max_length=128, limit=None,
                 split='train', test_size=0.2, seed=42, answers_per_question=5):
        """Initialize with batched structure"""
        super().__init__(data_path, tokenizer, max_length, limit, split, test_size, seed)
        self.answers_per_question = answers_per_question
        
        # Reorganize data by question
        self.questions = {}
        for pair in self.qa_pairs:
            q_text = pair['question']
            if q_text not in self.questions:
                self.questions[q_text] = []
            self.questions[q_text].append(pair)
        
        # Keep only questions with enough answers
        self.valid_questions = [
            q for q, answers in self.questions.items() 
            if len(answers) >= self.answers_per_question
        ]
        
    def __len__(self):
        return len(self.valid_questions)
    
    def __getitem__(self, idx):
        """Get a question with multiple answers"""
        q_text = self.valid_questions[idx]
        answers = self.questions[q_text]
        
        # Sort answers by score (descending)
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        
        # Limit to the required number of answers
        answers = answers[:self.answers_per_question]
        
        # Tokenize question once
        q_encoding = self.tokenizer(
            q_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize all answers
        a_input_ids = []
        a_attention_masks = []
        scores = []
        
        for answer in answers:
            a_encoding = self.tokenizer(
                answer['answer'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            a_input_ids.append(a_encoding['input_ids'])
            a_attention_masks.append(a_encoding['attention_mask'])
            scores.append(answer['score'])
        
        # Stack all answer tensors
        a_input_ids = torch.cat(a_input_ids, dim=0)
        a_attention_masks = torch.cat(a_attention_masks, dim=0)
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Normalize scores to [0, 1]
        if torch.max(scores) > 0:
            scores = scores / torch.max(scores)
        
        return {
            'q_input_ids': q_encoding['input_ids'].squeeze(0),
            'q_attention_mask': q_encoding['attention_mask'].squeeze(0),
            'a_input_ids': a_input_ids,
            'a_attention_masks': a_attention_masks,
            'scores': scores
        }

class QATripletDataset(QADataset):
    """
    Dataset that creates triplets (query, positive answer, negative answer)
    for triplet loss.
    """
    def __init__(self, data_path, tokenizer=None, max_length=128, limit=None,
                 split='train', test_size=0.2, seed=42):
        """Initialize with triplet structure"""
        super().__init__(data_path, tokenizer, max_length, limit, split, test_size, seed)
        
        # Reorganize data by question
        self.questions = {}
        for pair in self.qa_pairs:
            q_text = pair['question']
            if q_text not in self.questions:
                self.questions[q_text] = []
            self.questions[q_text].append(pair)
        
        # Keep only questions with multiple answers for triplets
        self.valid_questions = [
            q for q, answers in self.questions.items() 
            if len(answers) >= 2
        ]
        
    def __len__(self):
        return len(self.valid_questions)
    
    def __getitem__(self, idx):
        """Get a question with positive and negative answers"""
        q_text = self.valid_questions[idx]
        answers = self.questions[q_text]
        
        # Sort answers by score (descending)
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        
        # Take highest scored as positive, lowest as negative
        pos_answer = answers[0]
        neg_answer = answers[-1]
        
        # Tokenize question
        q_encoding = self.tokenizer(
            q_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize positive answer
        pos_encoding = self.tokenizer(
            pos_answer['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize negative answer
        neg_encoding = self.tokenizer(
            neg_answer['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'q_input_ids': q_encoding['input_ids'].squeeze(0),
            'q_attention_mask': q_encoding['attention_mask'].squeeze(0),
            'pos_input_ids': pos_encoding['input_ids'].squeeze(0),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(0),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(0),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(0),
            'pos_score': torch.tensor(pos_answer['score'], dtype=torch.float32),
            'neg_score': torch.tensor(neg_answer['score'], dtype=torch.float32)
        }

def create_dataloaders(dataset, batch_size=16, shuffle=True, split='train'):
    """
    Create appropriate dataloaders based on dataset type
    
    Args:
        dataset: QADataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        split: 'train', 'test', or 'all'
        
    Returns:
        train_loader, test_loader: DataLoader objects (or None if not applicable)
    """
    if split == 'all':
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        ), None
    
    elif split == 'train':
        # Create dataset with train split
        train_dataset = dataset.__class__(
            dataset.data_path,
            dataset.tokenizer,
            dataset.max_length,
            dataset.limit,
            'train',
            dataset.test_size,
            dataset.seed
        )
        
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        ), None
    
    elif split == 'test':
        # Create dataset with test split
        test_dataset = dataset.__class__(
            dataset.data_path,
            dataset.tokenizer,
            dataset.max_length,
            dataset.limit,
            'test',
            dataset.test_size,
            dataset.seed
        )
        
        return None, DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    else:
        # Create both train and test dataloaders
        train_dataset = dataset.__class__(
            dataset.data_path,
            dataset.tokenizer,
            dataset.max_length,
            dataset.limit,
            'train',
            dataset.test_size,
            dataset.seed
        )
        
        test_dataset = dataset.__class__(
            dataset.data_path,
            dataset.tokenizer,
            dataset.max_length,
            dataset.limit,
            'test',
            dataset.test_size,
            dataset.seed
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, test_loader
        
# Data generation and processing functions

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
        print("Please check your Kaggle credentials and internet connection.")
        sys.exit(1)


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
    
    print(f"Parsing data from {data_dir}...")
    
    questions = {}  # Dictionary to store questions by ID
    answers = defaultdict(list)  # Dictionary to store answers by parent ID
    
    try:
        # First pass: collect all answers
        answer_count = 0
        start_time = time.time()
        last_update_time = start_time
        
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
            
            # Create progress bar
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
                answer_count += 1
                
                # Print progress every 1000 answers or every 5 seconds
                current_time = time.time()
                if answer_count % 1000 == 0 or (current_time - last_update_time >= 5 and answer_count % 100 == 0):
                    elapsed = current_time - start_time
                    rate = answer_count / elapsed if elapsed > 0 else 0
                    print(f"  Processed {answer_count} answers... ({rate:.1f} answers/sec)")
                    last_update_time = current_time
        
        elapsed = time.time() - start_time
        print(f"Collected {answer_count} answers for {len(answers)} questions in {elapsed:.1f} seconds.")
        
        # Second pass: collect questions that have answers
        print("Second pass: collecting questions with answers...")
        start_time = time.time()
        last_update_time = start_time
        question_count = 0
        
        with open(questions_file, 'r', encoding='latin-1') as f:
            # Skip header
            header = next(f)
            
            # Count lines for progress bar (subtract 1 for header)
            total_lines = sum(1 for _ in f) 
            f.seek(0)
            next(f)  # Skip header again
            
            # Create CSV reader
            reader = csv.reader(f)
            
            # Create progress bar
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
                    
                    question_count += 1
                    
                    # Print progress every 100 questions or every 5 seconds
                    current_time = time.time()
                    if question_count % 100 == 0 or (current_time - last_update_time >= 5 and question_count % 10 == 0):
                        elapsed = current_time - start_time
                        rate = question_count / elapsed if elapsed > 0 else 0
                        print(f"  Processed {question_count} questions with answers... ({rate:.1f} questions/sec)")
                        last_update_time = current_time
                    
                    # Apply limit if specified
                    if limit and question_count >= limit:
                        print(f"Reached limit of {limit} questions.")
                        break
        
        elapsed = time.time() - start_time
        print(f"Collected {question_count} questions with answers in {elapsed:.1f} seconds.")
        
        # Since we don't have accepted answers in this dataset, 
        # we'll consider the highest scored answer as the accepted one
        print("Ranking answers by score...")
        start_time = time.time()
        
        # Rank answers by score for each question
        for q_id in answers:
            # Sort answers by score (highest first)
            answers[q_id].sort(key=lambda x: x["score"], reverse=True)
            
            # If the question exists in our collection and has answers
            if q_id in questions and answers[q_id]:
                # Set the top-scored answer as the accepted answer
                top_answer = answers[q_id][0]
                top_answer["is_accepted"] = True
                questions[q_id]["accepted_answer_id"] = top_answer["id"]
        
        elapsed = time.time() - start_time
        print(f"Ranked all answers and marked highest-scored answers as accepted in {elapsed:.1f} seconds.")
        
        return questions, answers
    
    except Exception as e:
        print(f"Error parsing posts: {e}")
        import traceback
        traceback.print_exc()
        return {}, defaultdict(list)


def parse_from_sqlite(db_path, limit=None):
    """
    Parse questions and answers directly from the SQLite database.
    This is more efficient than parsing CSV files.
    
    Args:
        db_path: Path to the SQLite database
        limit: Optional limit on the number of questions to process
        
    Returns:
        questions, answers: Dictionaries containing questions and their answers
    """
    print(f"Parsing data from SQLite database: {db_path}")
    
    questions = {}  # Dictionary to store questions by ID
    answers = defaultdict(list)  # Dictionary to store answers by parent ID
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First check if the tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if 'Answers' not in tables or 'Questions' not in tables:
            print("Required tables not found in SQLite database.")
            conn.close()
            return {}, defaultdict(list)
        
        # First pass: collect all answers
        print("Collecting answers from database...")
        start_time = time.time()
        last_update_time = start_time
        answer_count = 0
        
        # Use limit clause if specified
        limit_clause = f"LIMIT {limit * 10}" if limit else ""  # Get more answers to ensure we have enough for our questions
        
        # First count total rows for progress bar
        count_query = f"SELECT COUNT(*) FROM Answers {limit_clause}"
        total_rows = cursor.execute(count_query).fetchone()[0]
        
        # Execute the actual query
        query = f"SELECT Id, OwnerUserId, CreationDate, ParentId, Score, Body FROM Answers {limit_clause}"
        for row in tqdm(cursor.execute(query), total=total_rows, desc="Processing answers"):
            answer_id, owner_id, creation_date, parent_id, score, body = row
            
            # Store answer data
            answers[str(parent_id)].append({
                "id": str(answer_id),
                "body": body,
                "score": int(score),
                "is_accepted": False  # Will be set later
            })
            answer_count += 1
            
            # Print progress every 1000 answers or every 5 seconds
            current_time = time.time()
            if answer_count % 1000 == 0 or (current_time - last_update_time >= 5 and answer_count % 100 == 0):
                elapsed = current_time - start_time
                rate = answer_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {answer_count} answers... ({rate:.1f} answers/sec)")
                last_update_time = current_time
        
        elapsed = time.time() - start_time
        print(f"Collected {answer_count} answers for {len(answers)} questions in {elapsed:.1f} seconds.")
        
        # Second pass: collect questions that have answers
        print("Collecting questions with answers...")
        start_time = time.time()
        last_update_time = start_time
        question_count = 0
        
        # Use a join to get only questions with answers, and add a limit if specified
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT q.Id, q.OwnerUserId, q.CreationDate, q.Score, q.Title, q.Body
        FROM Questions q
        WHERE q.Id IN ({','.join(['?'] * len(answers))})
        {limit_clause}
        """
        
        for row in cursor.execute(query, list(answers.keys())):
            question_id, owner_id, creation_date, score, title, body = row
            
            # Store question data
            questions[str(question_id)] = {
                "id": str(question_id),
                "title": title,
                "body": body,
                "score": int(score),
                "view_count": 0,  # Not available in this dataset
                "tags": "",       # Tag info might be in a separate table
                "accepted_answer_id": None  # Will try to determine later
            }
            
            question_count += 1
            
            # Print progress every 100 questions or every 5 seconds
            current_time = time.time()
            if question_count % 100 == 0 or (current_time - last_update_time >= 5 and question_count % 10 == 0):
                elapsed = current_time - start_time
                rate = question_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {question_count} questions with answers... ({rate:.1f} questions/sec)")
                last_update_time = current_time
            
            # Apply limit if specified
            if limit and question_count >= limit:
                print(f"Reached limit of {limit} questions.")
                break
        
        conn.close()
        
        elapsed = time.time() - start_time
        print(f"Collected {question_count} questions with answers in {elapsed:.1f} seconds.")
        
        # Since we don't have accepted answers in this dataset, 
        # we'll consider the highest scored answer as the accepted one
        print("Ranking answers by score...")
        start_time = time.time()
        
        # Rank answers by score for each question
        for q_id in list(answers.keys()):
            # Keep only answers for questions we've actually loaded
            if q_id not in questions:
                del answers[q_id]
                continue
                
            # Sort answers by score (highest first)
            answers[q_id].sort(key=lambda x: x["score"], reverse=True)
            
            # If the question exists in our collection and has answers
            if answers[q_id]:
                # Set the top-scored answer as the accepted answer
                top_answer = answers[q_id][0]
                top_answer["is_accepted"] = True
                questions[q_id]["accepted_answer_id"] = top_answer["id"]
        
        elapsed = time.time() - start_time
        print(f"Ranked all answers and marked highest-scored answers as accepted in {elapsed:.1f} seconds.")
        
        return questions, answers
    
    except Exception as e:
        print(f"Error parsing from SQLite: {e}")
        import traceback
        traceback.print_exc()
        return {}, defaultdict(list)


def export_to_json(questions, answers, output_path="data/ranked_qa.json"):
    """Export questions and their ranked answers to a JSON file."""
    print(f"Exporting data to {output_path}...")
    
    data = []
    # Use progress bar for export
    for q_id, question in tqdm(questions.items(), desc="Exporting QA pairs", total=len(questions)):
        if q_id in answers:
            item = {
                "question": question,
                "answers": answers[q_id]
            }
            data.append(item)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} question-answer sets to {output_path}")


def ensure_dataset_exists(data_path='data/ranked_qa.json', data_limit=None, use_sqlite=False, force_regenerate=False):
    """
    Ensure the dataset exists, generating it if necessary.
    
    Args:
        data_path: Path where the JSON dataset should be stored
        data_limit: Limit the number of questions to process
        use_sqlite: Whether to use SQLite instead of CSV files
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
        
        # Choose parser based on user preference
        if use_sqlite and os.path.exists(os.path.join(data_dir, "database.sqlite")):
            db_path = os.path.join(data_dir, "database.sqlite")
            print(f"Using SQLite database at {db_path}")
            questions, answers = parse_from_sqlite(db_path, limit=data_limit)
        else:
            print(f"Using CSV files in {data_dir}")
            questions, answers = parse_posts(data_dir, limit=data_limit)
        
        # Export to JSON
        export_to_json(questions, answers, data_path)
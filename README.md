# QA Embeddings Loss Comparison

A system to compare different loss functions for embedding models on question-answer retrieval tasks.

## Overview

This project compares how different loss functions perform when training embedding models for retrieving relevant answers to questions. The project uses StackExchange data to train models on real-world question-answer pairs with upvote-based relevance scores.

## Features

- Flexible dataset with configurable batch transformations:
  - InfoNCE with in-batch negatives
  - Multiple positives per question
  - Hard negative sampling
  - Triplet format
  - Listwise ranking format
- Comprehensive loss function implementations:
  - Standard InfoNCE (with temperature scaling)
  - Rank-weighted InfoNCE
  - Hard negative InfoNCE
  - Multiple positives loss
  - Triplet loss
  - Listwise ranking loss
- Standardized evaluation metrics:
  - MRR (Mean Reciprocal Rank)
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - MAP@k (Mean Average Precision)
  - Hard negative accuracy
- Modular architecture for easy extension and experimentation
- Integration with Weights & Biases for experiment tracking
- Hardware acceleration detection (MPS, CUDA, CPU)

## Getting Started

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (Python package management)
- Kaggle API credentials (for dataset download)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/qa-embeddings-loss-comparison.git
cd qa-embeddings-loss-comparison
uv pip install -e .
```

### Running Experiments

To train a model with a predefined configuration:

```bash
uv run rank-test --config configs/test.json
```

To create a custom configuration, you can copy and modify one of the existing configuration files in `configs/`.

## Batch Transformation Strategies

The system uses a flexible data processing approach with various batch transformation strategies:

| Strategy | Description | Best Used With |
|----------|-------------|----------------|
| `infonce` | Standard InfoNCE format with each question paired with a top answer | StandardInfoNCELoss, RankInfoNCELoss |
| `multiple_positives` | Each question appears multiple times with different positive answers | MultiplePositivesLoss |
| `hard_negative` | Explicitly includes lower-ranked answers as hard negatives | HardNegativeInfoNCELoss |
| `triplet` | Creates triplets of (query, positive answer, negative answer) | TripletLoss |
| `listwise` | Prepares multiple ranked answers per question for listwise ranking | ListwiseRankingLoss |
| `standardized_test` | Standard evaluation format for fair comparison | Used automatically for evaluation |

Each transformation prepares data in a specific format suitable for its corresponding loss function. The transforms can be configured with additional parameters:

- `take_top`: Whether to use the highest-ranked answer (True) or a random answer (False)
- `pos_count`: Maximum number of positive answers to include per question
- `neg_strategy`: Strategy for selecting negatives in triplet format ("hard_negative", "in_batch", "mixed")
- `max_answers`: Maximum number of answers to include per question in listwise format

## Loss Functions

The project implements the following loss functions:

### StandardInfoNCELoss

Standard InfoNCE contrastive loss that contrasts positive pairs against in-batch negatives.

**Parameters:**
- `temperature`: Temperature parameter for scaling similarity scores (default: 0.1)

### RankInfoNCELoss

InfoNCE loss that leverages rank or score information from the dataset.

**Parameters:**
- `temperature`: Temperature parameter for scaling similarity scores (default: 0.1)
- `use_ranks`: Whether to use ordinal rank information (default: True)
- `use_scores`: Whether to use cardinal score information (default: False)
- `rank_weight`: Weight for rank-based penalties (default: 0.1)
- `score_weight`: Weight for score-based adjustments (default: 0.01)

### HardNegativeInfoNCELoss

InfoNCE loss with additional penalty for hard negatives.

**Parameters:**
- `temperature`: Temperature parameter for scaling similarity scores (default: 0.1)
- `hard_negative_weight`: Weight for the hard negative component (default: 1.0)

### MultiplePositivesLoss

Loss function that handles multiple positive answers per question.

**Parameters:**
- `temperature`: Temperature parameter for scaling similarity scores (default: 0.1)
- `rank_weight`: Weight for rank-based penalties (default: 0.1)

### TripletLoss

Triplet loss for query-positive-negative triplets.

**Parameters:**
- `margin`: Margin between positive and negative similarities (default: 0.3)

### ListwiseRankingLoss

Listwise ranking loss for learning to rank multiple answers.

**Parameters:**
- `temperature`: Temperature for scaling similarities (default: 1.0)

## Configuration Options

The system uses a strong typing configuration system based on Pydantic. Key configuration options include:

**Dataset and Data Processing:**
- `data_path`: Path to the JSON dataset with ranked QA pairs
- `limit`: Maximum number of questions to process
- `test_size`: Fraction of data to use for testing
- `seed`: Random seed for reproducibility
- `batch_size`: Batch size for training
- `dataset_strategy`: Dataset strategy to use (currently supports "flexible")
- `batch_transform`: Batch transformation strategy to use

**Model Architecture:**
- `embed_dim`: Embedding dimension for the base model
- `projection_dim`: Projection dimension for the final embeddings

**Training Parameters:**
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `eval_steps`: Number of steps between evaluations
- `eval_at_zero`: Whether to evaluate before training
- `checkpoint_interval`: Number of epochs between checkpoints

**Loss Function Parameters:**
- `loss_type`: Type of loss function to use
- `temperature`: Temperature parameter for scaling similarity scores
- `margin`: Margin for triplet loss
- `use_ranks`: Whether to use rank information
- `use_scores`: Whether to use score information
- `rank_weight`: Weight for rank-based adjustments
- `score_weight`: Weight for score-based adjustments
- `hard_negative_weight`: Weight for hard negative penalty

**Output and Logging:**
- `output_dir`: Directory to save outputs
- `log_to_wandb`: Whether to log to Weights & Biases
- `wandb_project`: Weights & Biases project name

Example configuration:
```json
{
  "name": "Test",
  "data_path": "data/ranked_qa.json",
  "limit": 100,
  "test_size": 0.02,
  "seed": 42,
  "embed_dim": 768,
  "projection_dim": 128,
  "batch_size": 16,
  "epochs": 1,
  "learning_rate": 2e-5,
  "eval_steps": 50,
  "eval_at_zero": true,
  "debug": false,
  "output_dir": "models/flexible",
  "log_to_wandb": false,
  "dataset_strategy": "flexible",
  "batch_transform": "infonce",
  "pos_count": 3,
  "loss_type": "infonce",
  "temperature": 0.1,
  "use_ranks": true,
  "use_scores": false,
  "rank_weight": 0.1
}
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **MRR**: Mean Reciprocal Rank - measures how highly the first relevant document is ranked
- **Accuracy@k**: Percentage of queries where a relevant document is ranked in the top k
- **NDCG@k**: Normalized Discounted Cumulative Gain - measures ranking quality considering the graded relevance of documents
- **MAP@k**: Mean Average Precision - measures precision at different recall levels
- **Hard Negative Accuracy**: Measures the model's ability to distinguish between positive answers and hard negative answers (lower-ranked answers to the same question)

## Project Structure

- `src/rank_test/`
  - `dataset.py`: Dataset class and data loading utilities
  - `transforms.py`: Batch transformation strategies
  - `models.py`: Model architecture definitions
  - `losses.py`: Loss function implementations
  - `train.py`: Training loop and utilities
  - `evaluate.py`: Evaluation metrics and utilities
  - `config.py`: Experiment configuration
  - `run_experiment.py`: Experiment runner for comparing multiple losses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
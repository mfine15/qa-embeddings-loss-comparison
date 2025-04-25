# QA Embeddings Loss Comparison

A system to compare different loss functions for embedding models on question-answer retrieval tasks.

## Overview

This project compares how different loss functions perform when training embedding models for retrieving relevant answers to questions. The project uses StackExchange data to train models on real-world question-answer pairs with upvote-based relevance scores.

## Features

- Implementation of multiple loss functions:
  - InfoNCE (with variations: standard, no in-batch negatives, no hard negatives)
  - MSE Loss on normalized upvotes
  - Triplet Loss for pair ranking
  - Listwise Ranking Loss (with and without in-batch negatives)
- Evaluation metrics: NDCG@k, MAP@k, MRR
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

To train a model with InfoNCE loss:

```bash
uv run src/rank_test/train.py --loss infonce --limit 1000
```

To evaluate a model on the test set:

```bash
uv run src/rank_test/evaluate.py --model models/infonce-20250425/final_model.pt
```

To run a full experiment suite comparing all loss functions:

```bash
uv run src/rank_test/run_experiment.py --suite
```

## Loss Functions

The project implements the following loss functions:

- **InfoNCE Loss**: Contrastive learning loss that contrasts positive pairs against negative pairs
- **MSE Loss**: Mean squared error on normalized upvote scores
- **Triplet Loss**: Learns embeddings by minimizing distance between query-positive pairs and maximizing distance to negative examples
- **Listwise Ranking Loss**: Learns to rank multiple answers for a single question

## Evaluation Metrics

We use standard information retrieval metrics:

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank

## Project Structure

- `src/rank_test/`
  - `dataset.py`: Dataset classes and data loading utilities
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
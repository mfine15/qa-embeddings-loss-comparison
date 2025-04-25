# Implementation Plan for Comparing Loss Functions in QA Retrieval

## Overview
This plan outlines how we'll compare different loss functions for retrieving relevant answers from a list of answers. We'll implement a system that allows for easy comparison of models trained with different loss functions.

## Loss Functions to Implement
1. InfoNCE loss (current implementation)
2. InfoNCE without in-batch negatives
3. InfoNCE with no hard negatives (only query-top answer pairs)
4. MSE loss on normalized upvotes
5. Triplet loss with upvote-based positives
6. Listwise ranking losses (with and without in-batch negatives)

## Evaluation Metrics
- NDCG@k
- MAP@k
- MRR (Mean Reciprocal Rank)

## Architecture Design

### 1. Core Components
- **Loss Module**: Implement all loss functions with a consistent interface
- **Dataset Module**: Support different batching strategies required by each loss
- **Model Module**: Flexible encoding architecture that works with all losses
- **Evaluation Module**: Standardized metrics and evaluation pipeline
- **Training Module**: Unified training loop for all losses
- **Experiment Config**: Configuration system for easy experimentation

### 2. Directory Structure
```
src/rank_test/
  ├── dataset.py      # Dataset loading and preprocessing
  ├── losses.py       # Loss function implementations
  ├── models.py       # Model architecture
  ├── train.py        # Training loop
  ├── evaluate.py     # Evaluation metrics
  ├── config.py       # Experiment configuration
  ├── run_experiment.py # Main experiment runner
```

## Implementation Steps

### Phase 1: Loss Functions (2 days)
- Create a base loss class with a common interface
- Implement all required loss functions
- Add tests to verify each loss function works correctly

### Phase 2: Dataset Handling (1 day)
- Extend the existing dataset loader to support train/test splits
- Create specialized dataset classes for different losses
- Implement batching strategies for each loss type

### Phase 3: Evaluation Pipeline (1 day)
- Implement evaluation metrics
- Create standardized evaluation procedures
- Add support for model comparison

### Phase 4: Training Pipeline (1 day)
- Create a unified training loop that works with any loss
- Add support for early stopping and checkpointing
- Integrate with wandb for experiment tracking

### Phase 5: Experiment System (1 day)
- Create a configuration system for experiment parameters
- Implement experiment running script
- Add support for hyperparameter tuning

### Phase 6: Experimentation and Analysis (2 days)
- Run experiments with different losses
- Compare results and analyze performance
- Create visualizations to illustrate differences

## Specific Design Details

### Loss Functions Interface
```python
class BaseLoss:
    def __call__(self, q_embeddings, a_embeddings, **kwargs):
        """Calculate loss and metrics"""
        raise NotImplementedError
        
    def get_batch_metrics(self, batch_output):
        """Get metrics for logging during training"""
        raise NotImplementedError
```

### Dataset Design
```python
class QADataset:
    """Base dataset with train/test/val splitting"""
    
    def get_train_dataloader(self, batch_size):
        """Return dataloader for training data"""
        
    def get_test_dataloader(self, batch_size):
        """Return dataloader for test data"""
```

### Evaluation Design
```python
def evaluate_model(model, test_dataloader, metrics=['ndcg@5', 'ndcg@10', 'map@5', 'mrr']):
    """Evaluate model performance with given metrics"""
    
def compare_models(model_results):
    """Compare multiple models across metrics"""
```

### Experiment Configuration
```python
class ExperimentConfig:
    """Configuration for a single experiment"""
    
    def __init__(self, loss_type, **kwargs):
        # Set defaults and override with provided values
        
    def create_loss(self):
        """Create loss function based on configuration"""
```

## Testing Strategy
1. Create a small test dataset subset for quick validation
2. Compare loss values against expected values with known inputs
3. Verify gradient flow and model updates during training
4. Check evaluation metrics against known ranking orders

## Success Criteria
- All loss functions successfully implemented and tested
- Standard evaluation metrics correctly implemented
- Easy to run experiments with different losses
- Clear comparison of loss function performance
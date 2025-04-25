# Flexible QA Ranking Dataset

This module provides a flexible framework for working with ranked question-answer datasets. It allows for various sampling strategies and loss functions to experiment with different approaches to QA ranking.

## Key Features

- **Flexible Data Sampling**: Control which QA pairs are included in training
- **Multiple Loss Functions**: Support for InfoNCE, Triplet, Listwise and other losses
- **Rank vs. Score Handling**: Properly distinguishes between ordinal ranks and cardinal scores
- **Customizable Batching**: Different batch structures for different loss functions

## Quick Start

Here's a simple example of how to use the flexible dataset:

```python
from rank_test.flexible_dataset import FlexibleQADataset, infonce_batch_transform
from rank_test.flexible_losses import StandardInfoNCELoss

# Create dataset with standard InfoNCE format
dataset = FlexibleQADataset(
    'data/ranked_qa.json',
    batch_transform_fn=infonce_batch_transform,
    batch_size=16,
    take_top=True  # Use top-ranked answer for each question
)

# Create dataloader
dataloader = FlexibleQADataset.get_dataloader(dataset)

# Create loss function
loss_fn = StandardInfoNCELoss(temperature=0.1)

# In your training loop...
for batch in dataloader:
    # Get embeddings
    q_embeddings = model(batch['q_input_ids'])
    a_embeddings = model(batch['a_input_ids'])
    
    # Calculate loss
    loss, metrics = loss_fn(q_embeddings, a_embeddings)
    
    # Backpropagate etc.
```

## Transformation Strategies

### 1. InfoNCE Batch Transform

Standard contrastive learning approach where each question is paired with one answer (default: top-ranked).

```python
dataset = FlexibleQADataset(
    data_path,
    batch_transform_fn=infonce_batch_transform,
    take_top=True  # or False to randomly select answers
)
```

### 2. Multiple Positives Batch Transform

Each question appears multiple times with different positive answers.

```python
dataset = FlexibleQADataset(
    data_path,
    batch_transform_fn=multiple_positives_batch_transform,
    pos_count=3  # Number of positives per question
)
```

### 3. Hard Negative Batch Transform

Explicitly includes lower-ranked answers as hard negatives.

```python
dataset = FlexibleQADataset(
    data_path,
    batch_transform_fn=hard_negative_batch_transform
)
```

### 4. Triplet Batch Transform

Creates query-positive-negative triplets for triplet loss training.

```python
dataset = FlexibleQADataset(
    data_path,
    batch_transform_fn=triplet_batch_transform,
    neg_strategy="hard_negative"  # or "in_batch" or "mixed"
)
```

### 5. Listwise Batch Transform

Groups multiple answers per question for listwise ranking.

```python
dataset = FlexibleQADataset(
    data_path,
    batch_transform_fn=listwise_batch_transform,
    max_answers=5  # Maximum answers per question
)
```

## Loss Functions

### 1. Standard InfoNCE Loss

```python
loss_fn = StandardInfoNCELoss(temperature=0.1)
```

### 2. Rank-Aware InfoNCE Loss

Supports both ordinal rank and cardinal score information:

```python
# Rank-based weighting
loss_fn = RankInfoNCELoss(
    temperature=0.1, 
    use_ranks=True, 
    rank_weight=0.1
)

# Score-based weighting
loss_fn = RankInfoNCELoss(
    temperature=0.1, 
    use_scores=True, 
    score_weight=0.05
)

# Combined approach
loss_fn = RankInfoNCELoss(
    temperature=0.1, 
    use_ranks=True, 
    use_scores=True,
    rank_weight=0.1,
    score_weight=0.05
)
```

### 3. Hard Negative InfoNCE Loss

```python
loss_fn = HardNegativeInfoNCELoss(
    temperature=0.1, 
    hard_negative_weight=1.0
)
```

### 4. Multiple Positives Loss

```python
loss_fn = MultiplePositivesLoss(
    temperature=0.1, 
    rank_weight=0.1
)
```

### 5. Triplet Loss

```python
loss_fn = TripletLoss(margin=0.3)
```

### 6. Listwise Ranking Loss

```python
loss_fn = ListwiseRankingLoss(temperature=1.0)
```

## Using the Factory Function

You can also create losses using the factory function:

```python
from rank_test.flexible_losses import create_flexible_loss

# Basic losses
loss_fn = create_flexible_loss('infonce', temperature=0.1)
loss_fn = create_flexible_loss('triplet', margin=0.3)

# Specialized InfoNCE with options in the name
loss_fn = create_flexible_loss('infonce_rank0.2_score0.1_t0.1')
```

## Best Practices

1. **Ranks vs. Scores**:
   - **Ordinal Ranks**: Position in sorted order (0 = best)
   - **Cardinal Scores**: Actual score values (higher = better)

2. **Batch Size**: 
   - Make sure your batch contains enough variety for contrastive learning
   - For techniques like InfoNCE, larger batches provide more negatives

3. **Temperature**:
   - Lower values make model more confident
   - Common values range from 0.05 to 1.0
   - Experiment to find the right balance

4. **Negative Mining**:
   - Hard negatives improve model discrimination
   - Consider mixing in-batch and hard negatives

## Demo Example

See `examples/flexible_dataset_demo.py` for a complete demonstration of the different dataset strategies and loss functions.

```bash
# Run the demo
uv run examples/flexible_dataset_demo.py
```
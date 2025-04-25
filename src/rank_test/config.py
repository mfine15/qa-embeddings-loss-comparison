#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module using Pydantic for strongly typed configuration.
Provides a simple, flat structure for experiment configuration.
"""

from typing import Dict, Any, Optional, Union, Literal
from pathlib import Path
import json
from pydantic import BaseModel, Field
from argdantic.sources import from_file, JsonFileLoader
# Type definitions using literals instead of enums
DatasetStrategyType = Literal["standard", "flexible"]
BatchTransformType = Literal["infonce", "multiple_positives", "hard_negative", "triplet", "listwise"]
NegativeStrategyType = Literal["hard_negative", "in_batch", "mixed"]
LossType = Literal["infonce", "rank_infonce", "relaxed_hard_neg", "triplet", "listwise", "mse"]

@from_file(loader=JsonFileLoader)
class ExperimentConfig(BaseModel):
    """Complete experiment configuration with flat structure"""
    # Experiment metadata
    name: Optional[str] = Field(default=None, description="Name of the experiment")
    
    # Data settings
    data_path: str = Field(default="data/ranked_qa.json", description="Path to the dataset JSON file")
    limit: Optional[int] = Field(default=None, description="Limit the number of samples to process")
    test_size: float = Field(default=0.2, description="Proportion of data to use for testing")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    force_regenerate: bool = Field(default=False, description="Force dataset regeneration")
    
    # Model settings
    embed_dim: int = Field(default=768, description="Embedding dimension from BERT")
    projection_dim: int = Field(default=128, description="Dimension for embeddings projection")
    
    # Training settings
    batch_size: int = Field(default=64, description="Training batch size")
    epochs: int = Field(default=5, description="Number of training epochs")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    checkpoint_interval: int = Field(default=1, description="Save checkpoint every N epochs")
    eval_steps: Optional[int] = Field(default=None, description="Evaluate every N steps")
    eval_at_zero: bool = Field(default=False, description="Evaluate before training")
    debug: bool = Field(default=False, description="Debug mode with minimal data")
    
    # Output settings
    output_dir: str = Field(default="models", description="Directory to save models and results")
    log_to_wandb: bool = Field(default=True, description="Whether to log metrics to W&B")
    wandb_project: str = Field(default="qa-embeddings-comparison", description="W&B project name")
    use_fixed_evaluation: bool = Field(default=True, description="Use the fixed evaluation implementation")
    
    # Dataset strategy settings
    dataset_strategy: DatasetStrategyType = Field(default="standard", description="Dataset strategy to use")
    batch_transform: BatchTransformType = Field(default="infonce", description="Batch transformation strategy")
    
    # InfoNCE transform options
    take_top: bool = Field(default=True, description="For InfoNCE: use top-ranked answer (True) or random (False)")
    
    # Multiple positives options
    pos_count: int = Field(default=3, description="For multiple_positives: number of positives per question")
    
    # Triplet options
    neg_strategy: NegativeStrategyType = Field(default="hard_negative", description="For triplet: negative sampling strategy")
    
    # Listwise options
    max_answers: int = Field(default=5, description="For listwise: maximum answers per question")
    
    # Loss settings
    loss_type: LossType = Field(default="infonce", description="Loss function to use")
    temperature: float = Field(default=0.1, description="Temperature for InfoNCE and listwise losses")
    margin: float = Field(default=0.2, description="Margin for triplet loss")
    normalize: bool = Field(default=True, description="For MSE: normalize scores")
    
    # Loss weighting options
    use_ranks: bool = Field(default=True, description="Use ordinal ranks in loss calculation")
    use_scores: bool = Field(default=False, description="Use cardinal scores in loss calculation")
    rank_weight: float = Field(default=0.1, description="Weight for rank-based penalties")
    score_weight: float = Field(default=0.05, description="Weight for score-based adjustments")
    hard_negative_weight: float = Field(default=1.0, description="Weight for hard negative penalty")
    
    def auto_select_loss_type(cls, values):
        """Automatically select loss type based on batch transform if not explicitly set"""
        if 'loss_type' not in values or not values['loss_type']:
            batch_transform = values.get('batch_transform')
            if batch_transform:
                if batch_transform == "infonce":
                    values['loss_type'] = "infonce"
                elif batch_transform == "multiple_positives":
                    values['loss_type'] = "rank_infonce"
                elif batch_transform == "hard_negative":
                    values['loss_type'] = "relaxed_hard_neg"
                elif batch_transform == "triplet":
                    values['loss_type'] = "triplet"
                elif batch_transform == "listwise":
                    values['loss_type'] = "listwise"
        return values
    
    def validate_debug_settings(cls, values):
        """Reduce settings in debug mode"""
        if values.get('debug', False):
            # Reduce batch size and epochs in debug mode
            values['batch_size'] = min(values.get('batch_size', 16), 4)
            values['epochs'] = min(values.get('epochs', 5), 2)
            
            # Limit data
            if values.get('limit') is None or values.get('limit', 0) > 50:
                values['limit'] = 50
                
        return values
    
    def get_loss_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for the loss function"""
        kwargs = {}
        
        loss_type = self.loss_type
        
        # Add loss-specific parameters
        if loss_type in ["infonce", "rank_infonce", "relaxed_hard_neg", "listwise"]:
            kwargs['temperature'] = self.temperature
            
        if loss_type == "rank_infonce":
            kwargs['use_ranks'] = self.use_ranks
            kwargs['use_scores'] = self.use_scores
            kwargs['rank_weight'] = self.rank_weight
            kwargs['score_weight'] = self.score_weight
            
        elif loss_type == "relaxed_hard_neg":
            kwargs['hard_neg_target_prob'] = self.hard_negative_weight
            
        elif loss_type == "triplet":
            kwargs['margin'] = self.margin
            
        elif loss_type == "mse":
            kwargs['normalize'] = self.normalize
            
        return kwargs
    
    def get_batch_transform_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for the batch transformation function"""
        kwargs = {}
        
        transform = self.batch_transform
        
        if transform == "infonce":
            kwargs['take_top'] = self.take_top
        elif transform == "multiple_positives":
            kwargs['pos_count'] = self.pos_count
        elif transform == "triplet":
            kwargs['neg_strategy'] = self.neg_strategy
        elif transform == "listwise":
            kwargs['max_answers'] = self.max_answers
            
        return kwargs
    
    def get_limit(self) -> Optional[int]:
        """Get effective data limit, accounting for debug mode"""
        if self.debug:
            return min(self.limit or 50, 50)
        return self.limit
    
    def get_batch_size(self) -> int:
        """Get effective batch size, accounting for debug mode"""
        if self.debug:
            return min(self.batch_size, 4)
        return self.batch_size
    
    def get_epochs(self) -> int:
        """Get effective number of epochs, accounting for debug mode"""
        if self.debug:
            return min(self.epochs, 2)
        return self.epochs
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from a JSON file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls.parse_obj(config_data)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)


# Predefined experiment configurations
PREDEFINED_CONFIGS = {
    # Standard InfoNCE with in-batch negatives
    'infonce_standard': ExperimentConfig(
        name="InfoNCE Standard",
        dataset_strategy="flexible",
        batch_transform="infonce",
        take_top=True,
        loss_type="infonce",
        temperature=0.1
    ),
    
    # InfoNCE with random answer selection
    'infonce_random': ExperimentConfig(
        name="InfoNCE Random",
        dataset_strategy="flexible",
        batch_transform="infonce",
        take_top=False,  # Choose random answer instead of top
        loss_type="infonce",
        temperature=0.1
    ),
    
    # Multiple positives with rank weighting
    'multiple_positives_rank': ExperimentConfig(
        name="Multiple Positives with Ranks",
        dataset_strategy="flexible",
        batch_transform="multiple_positives",
        pos_count=3,
        use_ranks=True,
        use_scores=False,
        rank_weight=0.1,
        loss_type="rank_infonce",
        temperature=0.1
    ),
    
    # Multiple positives with score weighting
    'multiple_positives_score': ExperimentConfig(
        name="Multiple Positives with Scores",
        dataset_strategy="flexible",
        batch_transform="multiple_positives",
        pos_count=3,
        use_ranks=False,
        use_scores=True,
        score_weight=0.05,
        loss_type="rank_infonce",
        temperature=0.1
    ),
    
    # Hard negative approach
    'hard_negative': ExperimentConfig(
        name="Relaxed Hard Negative InfoNCE",
        dataset_strategy="flexible",
        batch_transform="hard_negative",
        hard_negative_weight=0.1,  # Target probability for hard negatives
        loss_type="relaxed_hard_neg",
        temperature=0.1
    ),
    
    # Triplet loss with hard negatives
    'triplet_hard_neg': ExperimentConfig(
        name="Triplet with Hard Negatives",
        dataset_strategy="flexible",
        batch_transform="triplet",
        neg_strategy="hard_negative",
        loss_type="triplet",
        margin=0.2
    ),
    
    # Listwise ranking
    'listwise': ExperimentConfig(
        name="Listwise Ranking",
        dataset_strategy="flexible",
        batch_transform="listwise",
        max_answers=5,
        loss_type="listwise",
        temperature=1.0
    )
}
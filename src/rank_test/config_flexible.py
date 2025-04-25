#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced configuration module for experiments with flexible datasets.
Extends the base ExperimentConfig with support for different data sampling strategies.
"""

import os
import yaml
import json
from rank_test.config import ExperimentConfig as BaseConfig

class FlexibleConfig(BaseConfig):
    """
    Enhanced configuration class with support for flexible dataset strategies.
    Extends the base ExperimentConfig with additional options for data sampling and
    different loss function configurations.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize experiment configuration with flexible dataset options
        
        Args:
            **kwargs: Configuration parameters
        """
        # Initialize with base configuration
        super().__init__(**kwargs)
        
        # Add default dataset configuration
        dataset_defaults = {
            # Dataset settings
            'dataset_strategy': 'standard',  # standard, flexible
            'batch_transform': 'infonce',    # infonce, multiple_positives, hard_negative, triplet, listwise
            'take_top': True,                # For infonce: use top-ranked or random answer
            'pos_count': 3,                  # For multiple_positives: number of positives per question
            'neg_strategy': 'hard_negative', # For triplet: hard_negative, in_batch, mixed
            'max_answers': 5,                # For listwise: max answers per question
            
            # Flexible loss settings
            'use_ranks': True,               # Use ordinal ranks in loss
            'use_scores': False,             # Use cardinal scores in loss
            'rank_weight': 0.1,              # Weight for rank-based penalties
            'score_weight': 0.05,            # Weight for score-based adjustments
            'hard_negative_weight': 1.0,     # Weight for hard negative component
        }
        
        # Update with defaults if not provided
        for key, value in dataset_defaults.items():
            if key not in self.config:
                self.config[key] = value
                
        # Map batch_transform to loss if not explicitly provided
        if 'loss' not in kwargs and 'batch_transform' in kwargs:
            transform = kwargs['batch_transform']
            if transform in ['infonce', 'multiple_positives']:
                self.config['loss'] = 'infonce'
            elif transform == 'hard_negative':
                self.config['loss'] = 'hard_negative' 
            elif transform == 'triplet':
                self.config['loss'] = 'triplet'
            elif transform == 'listwise':
                self.config['loss'] = 'listwise'
        
        # Update loss settings based on flexible configuration
        self._update_flexible_loss_settings()
    
    def _update_flexible_loss_settings(self):
        """Update loss settings based on flexible configuration"""
        batch_transform = self.config['batch_transform']
        loss = self.config['loss']
        
        # Configure loss kwargs based on the selected loss and transform
        loss_kwargs = self.config.get('loss_kwargs', {})
        
        # InfoNCE with ranks/scores
        if loss == 'infonce' and batch_transform in ['infonce', 'multiple_positives']:
            if 'temperature' not in loss_kwargs:
                loss_kwargs['temperature'] = 0.1
                
            # Add rank/score settings for RankInfoNCE
            if batch_transform == 'multiple_positives':
                loss_kwargs['use_ranks'] = self.config['use_ranks']
                loss_kwargs['use_scores'] = self.config['use_scores']
                loss_kwargs['rank_weight'] = self.config['rank_weight']
                loss_kwargs['score_weight'] = self.config['score_weight']
                
                # Set loss name for clarity
                if self.config['use_ranks'] and self.config['use_scores']:
                    self.config['loss'] = 'rank_infonce'
                elif self.config['use_ranks']:
                    self.config['loss'] = 'rank_infonce'
                elif self.config['use_scores']:
                    self.config['loss'] = 'score_infonce'
        
        # Hard negative InfoNCE
        elif loss == 'hard_negative':
            if 'temperature' not in loss_kwargs:
                loss_kwargs['temperature'] = 0.1
            loss_kwargs['hard_negative_weight'] = self.config['hard_negative_weight']
        
        # Triplet loss
        elif loss == 'triplet':
            if 'margin' not in loss_kwargs:
                loss_kwargs['margin'] = 0.2
        
        # Listwise loss
        elif loss == 'listwise':
            if 'temperature' not in loss_kwargs:
                loss_kwargs['temperature'] = 1.0
        
        # Update loss kwargs
        self.config['loss_kwargs'] = loss_kwargs
    
    def get_dataset_config(self):
        """Get configuration specific to dataset creation"""
        return {
            'dataset_strategy': self.config.get('dataset_strategy', 'standard'),
            'batch_transform': self.config.get('batch_transform', 'infonce'),
            'take_top': self.config.get('take_top', True),
            'pos_count': self.config.get('pos_count', 3),
            'neg_strategy': self.config.get('neg_strategy', 'hard_negative'),
            'max_answers': self.config.get('max_answers', 5),
        }
    
    def get_batch_transform_kwargs(self):
        """Get keyword arguments for batch transformation"""
        kwargs = {}
        
        # Add transform-specific parameters
        transform = self.config.get('batch_transform', 'infonce')
        
        if transform == 'infonce':
            kwargs['take_top'] = self.config.get('take_top', True)
        elif transform == 'multiple_positives':
            kwargs['pos_count'] = self.config.get('pos_count', 3)
        elif transform == 'triplet':
            kwargs['neg_strategy'] = self.config.get('neg_strategy', 'hard_negative')
        elif transform == 'listwise':
            kwargs['max_answers'] = self.config.get('max_answers', 5)
        
        return kwargs


# Enhanced default configurations for different dataset strategies
flexible_configs = {
    # Standard InfoNCE with in-batch negatives
    'infonce_standard': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='infonce',
        take_top=True,
        loss='infonce',
        loss_kwargs={'temperature': 0.1}
    ),
    
    # InfoNCE with random answer selection
    'infonce_random': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='infonce',
        take_top=False,  # Choose random answer instead of top
        loss='infonce',
        loss_kwargs={'temperature': 0.1}
    ),
    
    # Multiple positives with rank weighting
    'multiple_positives_rank': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='multiple_positives',
        pos_count=3,
        use_ranks=True,
        use_scores=False,
        rank_weight=0.1,
        loss_kwargs={'temperature': 0.1}
    ),
    
    # Multiple positives with score weighting
    'multiple_positives_score': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='multiple_positives',
        pos_count=3,
        use_ranks=False,
        use_scores=True,
        score_weight=0.05,
        loss_kwargs={'temperature': 0.1}
    ),
    
    # Multiple positives with both rank and score
    'multiple_positives_combined': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='multiple_positives',
        pos_count=3,
        use_ranks=True,
        use_scores=True,
        rank_weight=0.1,
        score_weight=0.05,
        loss_kwargs={'temperature': 0.1}
    ),
    
    # Hard negative approach
    'hard_negative': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='hard_negative',
        loss='hard_negative',
        loss_kwargs={
            'temperature': 0.1,
            'hard_negative_weight': 1.0
        }
    ),
    
    # Triplet loss with hard negatives
    'triplet_hard_neg': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='triplet',
        neg_strategy='hard_negative',
        loss='triplet',
        loss_kwargs={'margin': 0.2}
    ),
    
    # Triplet loss with in-batch negatives
    'triplet_in_batch': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='triplet',
        neg_strategy='in_batch',
        loss='triplet',
        loss_kwargs={'margin': 0.2}
    ),
    
    # Triplet loss with mixed negative strategy
    'triplet_mixed': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='triplet',
        neg_strategy='mixed',
        loss='triplet',
        loss_kwargs={'margin': 0.2}
    ),
    
    # Listwise ranking
    'listwise': FlexibleConfig(
        dataset_strategy='flexible',
        batch_transform='listwise',
        max_answers=5,
        loss='listwise',
        loss_kwargs={'temperature': 1.0}
    )
}
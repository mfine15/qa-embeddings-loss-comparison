#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration module for experiments"""

import os
import yaml
import json

class ExperimentConfig:
    """Configuration class for experiments"""
    
    def __init__(self, **kwargs):
        """
        Initialize experiment configuration
        
        Args:
            **kwargs: Configuration parameters
        """
        # Default configuration
        self.config = {
            # Data settings
            'data_path': 'data/ranked_qa.json',
            'limit': None,
            'test_size': 0.2,
            'seed': 42,
            
            # Model settings
            'embed_dim': 768,
            'projection_dim': 128,
            
            # Training settings
            'batch_size': 16,
            'epochs': 5,
            'learning_rate': 2e-5,
            'loss': 'infonce',
            'loss_kwargs': {},
            'checkpoint_interval': 1,
            
            # Output settings
            'output_dir': 'models',
            'log_to_wandb': True,
            'wandb_project': 'qa-embeddings-comparison',
            
            # Debug settings
            'debug': False,
        }
        
        # Update with provided arguments
        self.config.update(kwargs)
        
        # Handle loss-specific settings
        self._set_loss_defaults()
    
    def _set_loss_defaults(self):
        """Set default loss parameters based on loss type"""
        loss = self.config['loss']
        loss_kwargs = self.config.get('loss_kwargs', {})
        
        # InfoNCE defaults
        if loss.startswith('infonce'):
            if 'temperature' not in loss_kwargs:
                loss_kwargs['temperature'] = 0.1
                
            if loss == 'infonce_no_batch_neg':
                loss_kwargs['in_batch_negatives'] = False
                
            if loss == 'infonce_no_hard_neg':
                loss_kwargs['hard_negatives'] = False
        
        # Triplet loss defaults
        elif loss == 'triplet':
            if 'margin' not in loss_kwargs:
                loss_kwargs['margin'] = 0.2
        
        # Listwise loss defaults
        elif loss.startswith('listwise'):
            if 'temperature' not in loss_kwargs:
                loss_kwargs['temperature'] = 1.0
                
            if loss == 'listwise_no_batch_neg':
                loss_kwargs['in_batch_negatives'] = False
                
            # Set answers per question for batched dataset
            if 'answers_per_question' not in self.config:
                self.config['answers_per_question'] = 5
        
        # MSE loss defaults
        elif loss == 'mse':
            if 'normalize' not in loss_kwargs:
                loss_kwargs['normalize'] = True
        
        # Set loss kwargs
        self.config['loss_kwargs'] = loss_kwargs
    
    def get(self, key, default=None):
        """Get configuration parameter with optional default"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """Dictionary-style access to configuration"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Dictionary-style setting of configuration"""
        self.config[key] = value
        
        # If loss is set, update loss defaults
        if key == 'loss':
            self._set_loss_defaults()
    
    def save(self, path):
        """Save configuration to file"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Save based on file extension
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    @classmethod
    def load(cls, path):
        """Load configuration from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load based on file extension
        if path.endswith('.json'):
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        return cls(**config)
    
    def get_batch_size(self):
        """Get batch size, with smaller size if in debug mode"""
        if self.config.get('debug', False):
            return min(self.config.get('batch_size', 16), 4)
        return self.config.get('batch_size', 16)
    
    def get_limit(self):
        """Get sample limit, with smaller limit if in debug mode"""
        if self.config.get('debug', False):
            return min(self.config.get('limit', 1000) or 1000, 50)
        return self.config.get('limit')
    
    def get_epochs(self):
        """Get epochs, with fewer epochs if in debug mode"""
        if self.config.get('debug', False):
            return min(self.config.get('epochs', 5), 2)
        return self.config.get('epochs', 5)
    
    def as_dict(self):
        """Return configuration as dictionary"""
        return self.config.copy()

# Default experiment configurations for different losses
default_configs = {
    'infonce': ExperimentConfig(
        loss='infonce',
        loss_kwargs={'temperature': 0.1},
        batch_size=16,
        epochs=5
    ),
    
    'infonce_no_batch_neg': ExperimentConfig(
        loss='infonce_no_batch_neg',
        loss_kwargs={'temperature': 0.1},
        batch_size=16,
        epochs=5
    ),
    
    'infonce_no_hard_neg': ExperimentConfig(
        loss='infonce_no_hard_neg',
        loss_kwargs={'temperature': 0.1},
        batch_size=16,
        epochs=5
    ),
    
    'mse': ExperimentConfig(
        loss='mse',
        loss_kwargs={'normalize': True},
        batch_size=16,
        epochs=5
    ),
    
    'triplet': ExperimentConfig(
        loss='triplet',
        loss_kwargs={'margin': 0.2},
        batch_size=16,
        epochs=5
    ),
    
    'listwise': ExperimentConfig(
        loss='listwise',
        loss_kwargs={'temperature': 1.0},
        batch_size=16,
        epochs=5,
        answers_per_question=5
    ),
    
    'listwise_no_batch_neg': ExperimentConfig(
        loss='listwise_no_batch_neg',
        loss_kwargs={'temperature': 1.0},
        batch_size=16,
        epochs=5,
        answers_per_question=5
    )
}
"""
Configuration settings for training.
Override with environment variables.
"""

import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingConfig:
    # Run configuration
    RUN_NAME: str = "default_run"
    RANDOM_SEED: int = 42
    OUTPUT_DIR: str = None
    
    # Data configuration
    N_COMPONENTS: int = 200
    
    # Training hyperparameters
    EPOCHS: int = 600
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    PATIENCE: int = 200
    L2_REGULARIZATION: float = 2e-4
    
    # Model architecture defaults
    DEFAULT_WIDTH: tuple = (384, 256, 128)
    DEFAULT_DROPOUT: tuple = (0.15, 0.10, 0.0)
    
    # Parameter-specific architectures
    MODEL_WIDTHS: dict = None
    DROPOUT_RATES: dict = None
    
    # Loss function configurations
    # 'mse' for Mean Squared Error, 'huber' for Huber loss
    LOSS_TYPES: dict = None
    LOSS_DELTAS: dict = None  # Only used for Huber loss
    
    def __post_init__(self):
        # Set output directory
        if self.OUTPUT_DIR is None:
            self.OUTPUT_DIR = f"./runs/{self.RUN_NAME}"
        
        # Ensure output directory exists
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        for subdir in ['models', 'logs', 'plots']:
            os.makedirs(f"{self.OUTPUT_DIR}/{subdir}", exist_ok=True)
        
        # Initialize parameter-specific settings with defaults
        if self.MODEL_WIDTHS is None:
            self.MODEL_WIDTHS = {
                'tau': (384, 256, 128),
                'soc': (256, 256, 128, 128),
                'emin': (256, 128),
                'emax': (256, 128)
            }
        
        if self.DROPOUT_RATES is None:
            self.DROPOUT_RATES = {
                'tau': (0.15, 0.10, 0.0),
                'soc': (0.0, 0.15, 0.15, 0.0),
                'emin': (0.15, 0.0),
                'emax': (0.15, 0.0)
            }
        
        # Loss configurations - SOC uses MSE, others use Huber
        if self.LOSS_TYPES is None:
            self.LOSS_TYPES = {
                'tau': 'huber',
                'soc': 'mse',      # SOC uses MSE instead of Huber
                'emin': 'huber',
                'emax': 'huber'
            }
        
        if self.LOSS_DELTAS is None:
            self.LOSS_DELTAS = {
                'tau': 0.03,       # Delta for Huber loss (τ)
                'soc': None,       # Not used for MSE
                'emin': 0.07,      # Delta for Huber loss (ε_min)
                'emax': 0.07       # Delta for Huber loss (ε_max)
            }
        
        # Add timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_config():
    """Create configuration from environment variables."""
    
    # Environment variables for loss types (optional overrides)
    loss_types = {
        'tau': os.getenv('TAU_LOSS', 'huber').lower(),
        'soc': os.getenv('SOC_LOSS', 'mse').lower(),
        'emin': os.getenv('EMIN_LOSS', 'huber').lower(),
        'emax': os.getenv('EMAX_LOSS', 'huber').lower()
    }
    
    return TrainingConfig(
        RUN_NAME=os.getenv('RUN_NAME', 'default_run'),
        RANDOM_SEED=int(os.getenv('RANDOM_SEED', '42')),
        N_COMPONENTS=int(os.getenv('NCOMP', '200')),
        EPOCHS=int(os.getenv('EPOCHS', '600')),
        BATCH_SIZE=int(os.getenv('BATCH', '64')),
        LEARNING_RATE=float(os.getenv('LR', '1e-4')),
        PATIENCE=int(os.getenv('PATIENCE', '200')),
        LOSS_TYPES=loss_types,
        LOSS_DELTAS={
            'tau': float(os.getenv('TAU_DELTA', '0.03')),
            'soc': None,  # MSE doesn't use delta
            'emin': float(os.getenv('EMIN_DELTA', '0.07')),
            'emax': float(os.getenv('EMAX_DELTA', '0.07'))
        }
    )
"""
Fake News Detection Package

This package contains modules for fake news detection including
text preprocessing, model training, and prediction capabilities.
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"
__description__ = "AI-powered fake news detection system"

from .preprocessing import TextPreprocessor
from .model_training import FakeNewsModel
from .prediction import FakeNewsPredictor

__all__ = [
    'TextPreprocessor',
    'FakeNewsModel',
    'FakeNewsPredictor'
]
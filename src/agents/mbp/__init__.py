"""
Fixed MBP (Multi-Byte Prediction) models for MarioBrosAgent.

This module provides fixed-size MBP models without pruning or mutation,
designed for stable convergence testing in the Mario environment.
"""

from .config import FixedMBPConfig
from .network import FixedMBPNetwork
from .trainer import FixedMBPTrainer
from .agent import FixedMBPAgent

__all__ = [
    "FixedMBPConfig",
    "FixedMBPNetwork", 
    "FixedMBPTrainer",
    "FixedMBPAgent"
]

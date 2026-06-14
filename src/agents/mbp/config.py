"""
Fixed MBP Configuration for MarioBrosAgent.

This configuration defines hyperparameters for fixed-size MBP networks
without pruning or metabolism, designed for stable convergence testing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FixedMBPConfig:
    """Hyperparameters for Fixed-Size MBP network.
    
    This configuration removes all dynamic growth/pruning mechanisms and
    focuses on stable, fixed-size network architecture for convergence testing.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the latent state vector in StateNeurons.
    embedding_dim : int
        Dimension for byte embeddings.
    num_state_neurons : int
        Fixed number of state neurons (no growth/pruning).
    num_prediction_neurons : int
        Fixed number of prediction neurons per layer.
    num_layers : int
        Number of processing layers.
    use_bits : bool
        If True, explode each byte into 8 binary features.
    batch_size : int
        Batch size for training.
    use_gpu : bool
        Whether to use GPU acceleration.
    eta : float
        Learning rate for weight updates.
    alpha_ema : float
        EMA decay for synapse utility tracking.
    max_synapses_per_neuron : int
        Maximum incoming synapses per neuron (fixed).
    """
    
    # ── Network Architecture ─────────────────────────────────────────────────
    state_dim: int = 128
    embedding_dim: int = 256
    num_state_neurons: int = 256  # Fixed number of state neurons
    num_prediction_neurons: int = 512  # Fixed number of prediction neurons
    num_layers: int = 3  # Number of processing layers
    
    # ── Input Representation ────────────────────────────────────────────────
    use_bits: bool = False
    batch_size: int = 32
    use_gpu: bool = False
    
    # ── Learning Parameters ─────────────────────────────────────────────────
    eta: float = 5e-3
    alpha_ema: float = 0.01
    weight_decay: float = 1e-4
    
    # ── Fixed Topology ──────────────────────────────────────────────────────
    max_synapses_per_neuron: int = 8  # Fixed max synapses per neuron
    
    # ── Training Parameters ─────────────────────────────────────────────────
    log_interval: int = 50
    checkpoint_interval: int = 100
    save_interval: int = 500
    gamma: float = 0.9  # Discount factor for TD learning
    
    # ── Mario Environment ───────────────────────────────────────────────────
    input_channels: int = 4  # From frame stacking
    input_height: int = 84
    input_width: int = 84
    action_dim: int = 6  # Mario action space
    
    # ── Derived Parameters ─────────────────────────────────────────────────
    input_width_bytes: int = field(init=False)
    total_neurons: int = field(init=False)
    
    def __post_init__(self) -> None:
        """Validate and compute derived parameters."""
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.num_state_neurons <= 0:
            raise ValueError(f"num_state_neurons must be positive, got {self.num_state_neurons}")
        if self.num_prediction_neurons <= 0:
            raise ValueError(f"num_prediction_neurons must be positive, got {self.num_prediction_neurons}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        
        # Compute derived parameters
        self.input_width_bytes = 8 if self.use_bits else 256
        self.total_neurons = self.num_state_neurons + (self.num_prediction_neurons * self.num_layers)
    
    @classmethod
    def get_small_config(cls) -> "FixedMBPConfig":
        """Small configuration for quick testing."""
        return cls(
            state_dim=64,
            embedding_dim=128,
            num_state_neurons=128,
            num_prediction_neurons=256,
            num_layers=2,
            batch_size=16,
            max_synapses_per_neuron=4
        )
    
    @classmethod
    def get_medium_config(cls) -> "FixedMBPConfig":
        """Medium configuration for balanced performance."""
        return cls(
            state_dim=128,
            embedding_dim=256,
            num_state_neurons=256,
            num_prediction_neurons=512,
            num_layers=3,
            batch_size=32,
            max_synapses_per_neuron=8
        )
    
    @classmethod
    def get_large_config(cls) -> "FixedMBPConfig":
        """Large configuration for maximum capacity."""
        return cls(
            state_dim=256,
            embedding_dim=512,
            num_state_neurons=512,
            num_prediction_neurons=1024,
            num_layers=4,
            batch_size=64,
            max_synapses_per_neuron=12
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "state_dim": self.state_dim,
            "embedding_dim": self.embedding_dim,
            "num_state_neurons": self.num_state_neurons,
            "num_prediction_neurons": self.num_prediction_neurons,
            "num_layers": self.num_layers,
            "use_bits": self.use_bits,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "eta": self.eta,
            "alpha_ema": self.alpha_ema,
            "weight_decay": self.weight_decay,
            "max_synapses_per_neuron": self.max_synapses_per_neuron,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "save_interval": self.save_interval,
            "input_channels": self.input_channels,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "action_dim": self.action_dim,
        }

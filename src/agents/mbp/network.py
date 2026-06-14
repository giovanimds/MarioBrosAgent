"""
Fixed MBP Network for MarioBrosAgent.

This module implements a fixed-size MBP (Multi-Byte Prediction) network
without pruning, metabolism, or any form of dynamic growth. The network
has a stable topology designed for convergence testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from .config import FixedMBPConfig


class FixedStateNeuron(nn.Module):
    """Fixed State Neuron with stable topology.
    
    Unlike the organic MBP, this neuron has fixed connections and no
    dynamic growth or pruning mechanisms.
    """
    
    def __init__(self, neuron_id: int, state_dim: int, max_inputs: int, config: FixedMBPConfig):
        super().__init__()
        self.neuron_id = neuron_id
        self.state_dim = state_dim
        self.max_inputs = max_inputs
        self.config = config
        
        # Fixed state vector
        self.C = nn.Parameter(torch.randn(state_dim) * 0.1)
        
        # Fixed input weights (no growth/pruning)
        self.W_delta = nn.Parameter(torch.randn(max_inputs, state_dim) * 0.01)
        self.W_B = nn.Parameter(torch.randn(max_inputs, state_dim) * 0.01)
        self.W_x = nn.Parameter(torch.randn(max_inputs, state_dim) * 0.01)
        
        # State matrix (learned transformation)
        self.A = nn.Parameter(torch.eye(state_dim) + torch.randn(state_dim, state_dim) * 0.01)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(state_dim))
        
        # Utility tracking (for monitoring only, no pruning)
        self.utility = nn.Parameter(torch.ones(max_inputs) * 0.5, requires_grad=False)
        
        # EMA for utility tracking
        self.alpha_ema = config.alpha_ema
    
    def forward(self, x: torch.Tensor, delta: torch.Tensor, 
                active_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the state neuron.
        
        Args:
            x: Input tensor [batch, max_inputs]
            delta: Prediction error [batch, max_inputs]
            active_mask: Mask for active inputs [batch, max_inputs]
            
        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]
        
        # Expand state to batch
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1)  # [batch, state_dim]
        
        # Compute input contributions
        if active_mask is not None:
            # Weighted sum of inputs
            W_delta_contrib = torch.einsum('bi,id->bd', x * active_mask, self.W_delta)
            W_B_contrib = torch.einsum('bi,id->bd', delta * active_mask, self.W_B)
            W_x_contrib = torch.einsum('bi,ix->bx', x * active_mask, self.W_x)
        else:
            W_delta_contrib = torch.einsum('bi,id->bd', x, self.W_delta)
            W_B_contrib = torch.einsum('bi,id->bd', delta, self.W_B)
            W_x_contrib = torch.einsum('bi,ix->bx', x, self.W_x)
        
        # State update
        delta_C = W_delta_contrib + W_B_contrib + self.bias.unsqueeze(0)
        new_C = torch.matmul(C_expanded, self.A) + delta_C
        
        # Output
        output = torch.tanh(new_C + W_x_contrib)
        
        return output, new_C
    
    def update_utility(self, surprise: torch.Tensor, active_mask: Optional[torch.Tensor] = None):
        """Update utility tracking (monitoring only, no pruning effect)."""
        if active_mask is not None:
            masked_surprise = surprise * active_mask
        else:
            masked_surprise = surprise
        
        # EMA update
        self.utility.data = (1 - self.alpha_ema) * self.utility.data + \
                           self.alpha_ema * masked_surprise.mean(dim=0).detach()


class FixedPredictionNeuron(nn.Module):
    """Fixed Prediction Neuron with stable topology.
    
    This neuron makes predictions and computes surprise, but has
    no dynamic growth or pruning mechanisms.
    """
    
    def __init__(self, neuron_id: int, input_dim: int, output_dim: int, 
                 max_inputs: int, config: FixedMBPConfig):
        super().__init__()
        self.neuron_id = neuron_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_inputs = max_inputs
        self.config = config
        
        # Fixed weights (no growth/pruning)
        self.weights = nn.Parameter(torch.randn(max_inputs, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Utility tracking (monitoring only)
        self.utility = nn.Parameter(torch.ones(max_inputs) * 0.5, requires_grad=False)
        self.alpha_ema = config.alpha_ema
        
        # Statistics tracking
        self.rolling_surprise = 0.0
        self.stagnation_counter = 0
    
    def forward(self, x: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through prediction neuron.
        
        Args:
            x: Input tensor [batch, max_inputs]
            active_mask: Mask for active inputs [batch, max_inputs]
            
        Returns:
            Prediction output [batch, output_dim]
        """
        if active_mask is not None:
            x_masked = x * active_mask
        else:
            x_masked = x
        
        # Weighted sum
        output = torch.einsum('bi,io->bo', x_masked, self.weights) + self.bias.unsqueeze(0)
        return torch.tanh(output)
    
    def compute_surprise(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute surprise (prediction error)."""
        error = prediction - target
        surprise = 0.5 * error.pow(2)  # MSE-based surprise
        return surprise
    
    def update_utility(self, surprise: torch.Tensor, active_mask: Optional[torch.Tensor] = None):
        """Update utility tracking (monitoring only)."""
        if active_mask is not None:
            masked_surprise = surprise * active_mask
        else:
            masked_surprise = surprise
        
        # EMA update
        self.utility.data = (1 - self.alpha_ema) * self.utility.data + \
                           self.alpha_ema * masked_surprise.mean(dim=0).detach()


class FixedMBPNetwork(nn.Module):
    """Fixed-size MBP Network without pruning or metabolism.
    
    This network maintains a stable topology with:
    - Fixed number of state neurons
    - Fixed number of prediction neurons
    - Fixed connectivity (no growth/pruning)
    - Standard backpropagation for learning
    
    The architecture is designed for stable convergence testing in
    reinforcement learning environments like Mario.
    """
    
    def __init__(self, config: FixedMBPConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Input embedding layer
        self.embedding_dim = config.embedding_dim
        if config.use_bits:
            self.input_width = 8  # Binary features
            self.byte_embedding = nn.Linear(8, config.embedding_dim)
        else:
            self.input_width = 256  # Byte values
            self.byte_embedding = nn.Linear(256, config.embedding_dim)
        
        # For Mario, we need to handle visual input
        # CNN feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, config.input_channels, config.input_height, config.input_width)
            feature_dim = self.feature_extractor(dummy_input).shape[1]
        
        # Fixed state neurons
        self.state_neurons = nn.ModuleList()
        for i in range(config.num_state_neurons):
            neuron = FixedStateNeuron(
                neuron_id=i,
                state_dim=config.state_dim,
                max_inputs=feature_dim,  # Connect to feature extractor output
                config=config
            )
            self.state_neurons.append(neuron)
        
        # Fixed prediction neurons (for action values)
        self.prediction_neurons = nn.ModuleList()
        for i in range(config.num_prediction_neurons):
            neuron = FixedPredictionNeuron(
                neuron_id=i,
                input_dim=config.state_dim,
                output_dim=config.action_dim,
                max_inputs=config.num_state_neurons,
                config=config
            )
            self.prediction_neurons.append(neuron)
        
        # Output layer for action values
        self.output_layer = nn.Linear(
            config.num_prediction_neurons * config.action_dim,
            config.action_dim
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(config.num_prediction_neurons * config.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Training statistics
        self.step_count = 0
        self.total_surprise = 0.0
        self.average_surprise = 0.0
    
    def _build_feature_extractor(self) -> nn.Sequential:
        """Build CNN feature extractor for Mario visual input."""
        return nn.Sequential(
            # Input: [batch, channels, height, width]
            nn.Conv2d(self.config.input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),  # 7x7 from conv calculations
            nn.ReLU(),
            nn.Linear(512, self.config.embedding_dim),
            nn.ReLU()
        )
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, model: str = "online") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            model: "online" or "target" (for DQN-style training)
            
        Returns:
            Tuple of (action_values, state_value)
        """
        # Extract features
        features = self.feature_extractor(x)  # [batch, embedding_dim]
        
        # Process through state neurons
        state_outputs = []
        for neuron in self.state_neurons:
            # For simplicity, use the same input for all state neurons
            # In a full implementation, each would have specific connections
            state_out, _ = neuron(features, features, None)  # Simplified
            state_outputs.append(state_out)
        
        # Stack state outputs
        state_tensor = torch.stack(state_outputs, dim=1)  # [batch, num_states, state_dim]
        
        # Process through prediction neurons
        prediction_outputs = []
        for neuron in self.prediction_neurons:
            # Average state as input to prediction neurons
            avg_state = state_tensor.mean(dim=1)  # [batch, state_dim]
            pred_out = neuron(avg_state.unsqueeze(1).expand(-1, self.config.num_state_neurons, -1), None)
            prediction_outputs.append(pred_out)
        
        # Stack and reshape prediction outputs
        pred_tensor = torch.stack(prediction_outputs, dim=1)  # [batch, num_preds, action_dim]
        pred_flat = pred_tensor.reshape(pred_tensor.shape[0], -1)  # [batch, num_preds * action_dim]
        
        # Get action values
        action_values = self.output_layer(pred_flat)
        
        # Get state value
        state_value = self.value_head(pred_flat)
        
        return action_values, state_value.squeeze(-1)
    
    def forward_step(self, byte_vals: np.ndarray, learning_mode: np.ndarray, 
                     active_mask: np.ndarray) -> Dict[str, Any]:
        """
        Forward step for byte-level processing (compatibility with original MBP).
        
        This method provides compatibility with the original MBP training loop
        while maintaining fixed topology.
        """
        # Convert numpy to torch
        byte_tensor = torch.from_numpy(byte_vals).float().to(self.device)
        learning_mask = torch.from_numpy(learning_mode).float().to(self.device)
        active_mask_tensor = torch.from_numpy(active_mask).float().to(self.device)
        
        # Embed bytes
        if self.config.use_bits:
            # Convert bytes to bits
            byte_tensor = byte_tensor.long()
            bit_tensor = torch.zeros(byte_tensor.shape[0], 8, device=self.device)
            for i in range(8):
                bit_tensor[:, i] = (byte_tensor >> i) & 1
            embedded = self.byte_embedding(bit_tensor.float())
        else:
            # Normalize byte values
            byte_tensor = byte_tensor / 255.0
            embedded = self.byte_embedding(byte_tensor)
        
        # Process through network
        # Simplified: use first state neuron for demonstration
        if len(self.state_neurons) > 0:
            state_out, new_state = self.state_neurons[0](embedded, embedded, active_mask_tensor)
        else:
            state_out = embedded
            new_state = embedded
        
        # Compute predictions and surprise
        total_surprise = 0.0
        num_learning = learning_mask.sum().item()
        
        if num_learning > 0:
            # Use first prediction neuron
            if len(self.prediction_neurons) > 0:
                predictions = self.prediction_neurons[0](state_out, active_mask_tensor)
                surprise = self.prediction_neurons[0].compute_surprise(predictions, embedded)
                total_surprise = surprise.sum().item()
        
        # Update step count
        self.step_count += 1
        
        return {
            "output": state_out.detach().cpu().numpy(),
            "new_state": new_state.detach().cpu().numpy(),
            "total_surprise": total_surprise,
            "num_learning": num_learning
        }
    
    def increment_step(self) -> Optional[Dict[str, Any]]:
        """Increment step counter and return metadata."""
        self.step_count += 1
        return {
            "step": self.step_count,
            "nodes": self.config.num_state_neurons + self.config.num_prediction_neurons,
            "synapses": (self.config.num_state_neurons + self.config.num_prediction_neurons) * 
                       self.config.max_synapses_per_neuron
        }
    
    def metabolize(self):
        """No-op for fixed network (no metabolism)."""
        pass
    
    def save_state_dict(self) -> Dict[str, Any]:
        """Save network state as dictionary."""
        return {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "step_count": self.step_count,
            "total_surprise": self.total_surprise,
            "average_surprise": self.average_surprise
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load network state from dictionary."""
        nn.Module.load_state_dict(self, state_dict["state_dict"])
        self.step_count = state_dict.get("step_count", 0)
        self.total_surprise = state_dict.get("total_surprise", 0.0)
        self.average_surprise = state_dict.get("average_surprise", 0.0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current network metrics."""
        return {
            "step": self.step_count,
            "total_surprise": self.total_surprise,
            "average_surprise": self.average_surprise,
            "num_state_neurons": self.config.num_state_neurons,
            "num_prediction_neurons": self.config.num_prediction_neurons,
            "total_neurons": self.config.total_neurons,
            "max_synapses_per_neuron": self.config.max_synapses_per_neuron
        }
    
    def to(self, device: torch.device):
        """Move network to device."""
        super().to(device)
        self.device = device
        return self

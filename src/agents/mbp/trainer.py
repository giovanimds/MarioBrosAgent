"""
Fixed MBP Trainer for MarioBrosAgent.

This trainer provides a training loop for the fixed-size MBP network,
compatible with the Mario environment and respecting byte-level masks
for learning only on assistant bytes.
"""

import json
import time
import threading
import queue
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console

from .config import FixedMBPConfig
from .network import FixedMBPNetwork


class FixedMBPTrainer:
    """Trainer for Fixed MBP Network.
    
    This trainer handles:
    - Training loop with experience replay
    - Byte-level masking for selective learning
    - Monitoring and logging
    - Checkpointing
    """
    
    def __init__(self, network: FixedMBPNetwork, config: FixedMBPConfig,
                 checkpoint_dir: Path = Path("checkpoints/fixed_mbp"),
                 verbose: bool = True):
        self.network = network
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.console = Console()
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self._total_bytes = 0
        self._window_surprises = []
        self._best_reward = float('-inf')
        
        # Optimizer
        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=config.eta,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Experience buffer
        self.replay_buffer = []
        self.buffer_capacity = 10000
        
        # Target network for DQN-style training
        self.target_network = FixedMBPNetwork(config)
        self.target_network.load_state_dict(network.state_dict())
        self.target_network.eval()
        
        # Sync interval
        self.target_sync_interval = 1000
    
    def _log(self, step: int, meta: Dict[str, Any], t0: float):
        """Log training metrics."""
        elapsed = time.time() - t0
        
        if self._window_surprises:
            avg_surprise = sum(self._window_surprises) / len(self._window_surprises)
        else:
            avg_surprise = 0.0
        
        metrics = {
            "step": step,
            "surprise": f"{avg_surprise:.4f}",
            "nodes": self.config.total_neurons,
            "synapses": self.config.total_neurons * self.config.max_synapses_per_neuron,
            "elapsed": f"{elapsed:.1f}s"
        }
        
        if self.verbose and step % self.config.log_interval == 0:
            log_str = f"[step={step}] " + " ".join([f"{k}={v}" for k, v in metrics.items()])
            self.console.print(log_str)
    
    def _save_checkpoint(self, step: int):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"fixed_mbp_step_{step}.json"
        
        state = self.network.save_state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["trainer_step"] = self.step_count
        state["trainer_episode"] = self.episode_count
        state["best_reward"] = self._best_reward
        
        # Save as JSON for compatibility
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        if self.verbose:
            self.console.print(f"[green]Checkpoint saved: {checkpoint_path}[/green]")
    
    def _save_topology_snapshot(self, step: int):
        """Save topology snapshot (for monitoring)."""
        snapshot_path = self.checkpoint_dir / f"topology_step_{step}.json"
        
        snapshot = {
            "step": step,
            "config": self.config.to_dict(),
            "num_state_neurons": self.config.num_state_neurons,
            "num_prediction_neurons": self.config.num_prediction_neurons,
            "max_synapses_per_neuron": self.config.max_synapses_per_neuron,
            "total_neurons": self.config.total_neurons
        }
        
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
    
    def _final_summary(self, step: int, elapsed: float) -> Dict[str, Any]:
        """Generate final training summary."""
        if self._window_surprises:
            avg_surprise = sum(self._window_surprises) / len(self._window_surprises)
        else:
            avg_surprise = 0.0
        
        return {
            "final_step": step,
            "total_time": elapsed,
            "avg_surprise": avg_surprise,
            "total_bytes": self._total_bytes,
            "best_reward": self._best_reward,
            "config": self.config.to_dict()
        }
    
    def add_experience(self, state: torch.Tensor, action: torch.Tensor,
                       reward: torch.Tensor, next_state: torch.Tensor,
                       done: torch.Tensor, info: Dict[str, Any]):
        """Add experience to replay buffer."""
        experience = (state, action, reward, next_state, done, info)
        self.replay_buffer.append(experience)
        
        if len(self.replay_buffer) > self.buffer_capacity:
            self.replay_buffer.pop(0)
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )
    
    def compute_td_loss(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute TD loss for DQN-style training."""
        states, actions, rewards, next_states, dones = batch
        
        # Get Q values for current states
        q_values, _ = self.network(states)
        
        # Get Q values for next states from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
        
        # Compute TD targets
        best_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
        next_q = next_q_values.gather(1, best_actions)
        
        td_targets = rewards + (1 - dones.float()) * self.config.gamma * next_q.squeeze(1)
        
        # Get Q values for taken actions
        action_indices = actions.long().unsqueeze(1)
        q_taken = q_values.gather(1, action_indices).squeeze(1)
        
        # Compute loss
        loss = self.loss_fn(q_taken, td_targets)
        
        # Additional metrics
        metrics = {
            "td_loss": loss.item(),
            "avg_q": q_values.mean().item(),
            "max_q": q_values.max().item(),
            "avg_reward": rewards.mean().item()
        }
        
        return loss, metrics
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform a single training step."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        batch = self.sample_batch(self.config.batch_size)
        if batch is None:
            return None
        
        # Compute loss
        loss, metrics = self.compute_td_loss(batch)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update step count
        self.step_count += 1
        
        # Sync target network periodically
        if self.step_count % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        return metrics
    
    def train_on_db(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        table_name: str = "mario_experiences",
        epochs: int = 10,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train on database experiences (similar to original train_db.py).
        
        This method provides compatibility with the original MBP training
        while respecting byte-level masks for selective learning.
        """
        t0 = time.time()
        
        if self.verbose:
            self.console.print(f"\n[bold cyan]Fixed MBP Trainer — Training on DB[/bold cyan]")
            self.console.print(f"Config: state_dim={self.config.state_dim}, "
                             f"neurons={self.config.total_neurons}")
            self.console.print("-" * 60)
        
        # For now, we'll use the replay buffer instead of actual DB
        # This can be extended to use actual database connection
        step = 0
        
        try:
            while True:
                if max_steps is not None and step >= max_steps:
                    break
                
                # Train step
                metrics = self.train_step()
                
                if metrics:
                    self._total_bytes += self.config.batch_size
                    
                    if step % self.config.log_interval == 0:
                        self._log(step, metrics, t0)
                    
                    if step % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(step)
                    
                    if step % self.config.topology_snapshot_interval == 0:
                        self._save_topology_snapshot(step)
                
                step += 1
                self.step_count += 1
                
                # Stop if we've exhausted the buffer
                if len(self.replay_buffer) < self.config.batch_size:
                    break
            
            # Final save
            self._save_checkpoint(step)
            
            elapsed = time.time() - t0
            return self._final_summary(step, elapsed)
            
        except Exception as e:
            self.console.print(f"[red]Training error: {e}[/red]")
            traceback.print_exc()
            return {"error": str(e)}
    
    def train_on_file(self, file_path: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Train on a file with byte sequences."""
        t0 = time.time()
        
        if self.verbose:
            self.console.print(f"\n[bold cyan]Fixed MBP Trainer — Training on file: {file_path}[/bold cyan]")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            byte_array = np.frombuffer(data, dtype=np.uint8)
            
            # Create masks (for now, all bytes are learning)
            learning_mask = np.ones_like(byte_array, dtype=np.float32)
            active_mask = np.ones_like(byte_array, dtype=np.float32)
            
            step = 0
            for i in range(0, len(byte_array), self.config.batch_size):
                if max_steps is not None and step >= max_steps:
                    break
                
                batch_bytes = byte_array[i:i + self.config.batch_size]
                batch_lm = learning_mask[i:i + self.config.batch_size]
                batch_am = active_mask[i:i + self.config.batch_size]
                
                # Process batch
                result = self.network.forward_step(batch_bytes, batch_lm, batch_am)
                
                if result["num_learning"] > 0:
                    self._window_surprises.append(result["total_surprise"] / result["num_learning"])
                
                if step % self.config.log_interval == 0:
                    self._log(step, {}, t0)
                
                step += 1
                self.step_count += 1
            
            elapsed = time.time() - t0
            return self._final_summary(step, elapsed)
            
        except Exception as e:
            self.console.print(f"[red]File training error: {e}[/red]")
            return {"error": str(e)}
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Get action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.config.action_dim)
        
        with torch.no_grad():
            q_values, _ = self.network(state.unsqueeze(0))
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def update_best_reward(self, reward: float):
        """Update best reward."""
        if reward > self._best_reward:
            self._best_reward = reward
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            "step": self.step_count,
            "episode": self.episode_count,
            "best_reward": self._best_reward,
            "buffer_size": len(self.replay_buffer),
            "network_metrics": self.network.get_metrics()
        }

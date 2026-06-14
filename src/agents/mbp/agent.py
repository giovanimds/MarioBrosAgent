"""
Fixed MBP Agent for MarioBrosAgent.

This agent implements a DQN-style agent using the fixed-size MBP network,
designed for stable convergence testing in the Mario environment.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from rich.console import Console

from .config import FixedMBPConfig
from .network import FixedMBPNetwork
from .trainer import FixedMBPTrainer


class FixedMBPAgent:
    """Agent using Fixed MBP Network for Mario.
    
    This agent:
    - Uses a fixed-size MBP network without pruning or metabolism
    - Implements DQN-style learning with experience replay
    - Respects byte-level masks for selective learning
    - Provides monitoring and checkpointing
    """
    
    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int,
                 save_dir: Path, config: Optional[FixedMBPConfig] = None):
        """
        Initialize the Fixed MBP Agent.
        
        Args:
            state_dim: Dimensions of the observation space (C, H, W)
            action_dim: Number of possible actions
            save_dir: Directory to save checkpoints
            config: Configuration for the MBP network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with environment dimensions
        if config is None:
            config = FixedMBPConfig.get_medium_config()
        
        config.input_channels = state_dim[0]
        config.input_height = state_dim[1]
        config.input_width = state_dim[2]
        config.action_dim = action_dim
        
        self.config = config
        self.console = Console()
        
        # Device
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Network
        self.net = FixedMBPNetwork(config).to(self.device)
        
        # Trainer
        self.trainer = FixedMBPTrainer(
            network=self.net,
            config=config,
            checkpoint_dir=self.save_dir / "trainer",
            verbose=True
        )
        
        # Training parameters
        self.exploration_rate = 0.6
        self.exploration_rate_decay = 0.99999
        self.exploration_rate_min = 0.2
        self.curr_step = 0
        
        # DQN parameters
        self.gamma = 0.9
        self.burnin = 100
        self.learn_every = 6
        self.sync_every = 24
        self.save_every = 5000
        
        # Statistics
        self.episode_reward = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # Checkpoint path
        self.checkpoint_path = self.save_dir / "fixed_mbp_agent.chkpt"
        
        # Load if checkpoint exists
        if self.checkpoint_path.exists():
            self.load()
    
    def act(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current observation state
            
        Returns:
            Action index
        """
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Epsilon-greedy
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values, _ = self.net(state_tensor.unsqueeze(0))
                action = torch.argmax(q_values, dim=1).item()
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # Increment step
        self.curr_step += 1
        
        return action
    
    def cache(self, state: np.ndarray, next_state: np.ndarray, action: int,
              reward: float, done: bool, info: Dict[str, Any]):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            next_state: Next state
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            info: Additional info
        """
        # Convert to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor([action], dtype=torch.int64, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([done], dtype=torch.float32, device=self.device)
        
        # Add to trainer's replay buffer
        self.trainer.add_experience(
            state=state_tensor,
            action=action_tensor,
            reward=reward_tensor,
            next_state=next_state_tensor,
            done=done_tensor,
            info=info
        )
        
        # Update statistics
        self.episode_reward += reward
        
        if done:
            self.episode_count += 1
            self.trainer.update_best_reward(self.episode_reward)
            self.episode_reward = 0
    
    def learn(self) -> Optional[Tuple[float, float]]:
        """
        Perform learning step.
        
        Returns:
            Tuple of (q_value, loss) or None
        """
        # Sync target network
        if self.curr_step % self.sync_every == 0:
            self.trainer.target_network.load_state_dict(self.net.state_dict())
        
        # Save checkpoint
        if self.curr_step % self.save_every == 0:
            self.save()
        
        # Burnin period
        if self.curr_step < self.burnin:
            return None, None
        
        # Learn every N steps
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Perform training step
        metrics = self.trainer.train_step()
        
        if metrics:
            q_value = metrics.get('avg_q', 0)
            loss = metrics.get('td_loss', 0)
            return q_value, loss
        
        return None, None
    
    def calculate_reward(self, reward: float, done: bool, info: Dict[str, Any]) -> float:
        """
        Calculate custom reward for Mario.
        
        Args:
            reward: Base reward
            done: Whether episode is done
            info: Additional info from environment
            
        Returns:
            Custom reward
        """
        progress_reward = 0
        life_reward = 0
        coin_reward = 0
        score_reward = 0
        time_penalty = -0.01
        
        # Progress reward
        if hasattr(self, 'last_position') and self.last_position is not None:
            progress = (info.get("x_pos", 0) - self.last_position) / 10
            if progress > 1:
                progress_reward = progress
        else:
            self.last_position = info.get("x_pos", 0)
        
        # Level completion reward
        if info.get("flag_get", False):
            reward += 50
        
        # Life reward
        if 'life' in info:
            life_change = float(int(info["life"]) - int(getattr(self, 'last_life', 2)))
            if life_change > 0:
                life_reward = 10
            elif life_change < 0:
                life_reward = -5
            self.last_life = info["life"]
        
        # Coin reward
        if hasattr(self, 'last_coins'):
            coin_reward = info.get("coins", 0) - self.last_coins
            self.last_coins = info.get("coins", 0)
        else:
            self.last_coins = info.get("coins", 0)
        
        # Score reward
        if hasattr(self, 'last_score'):
            score_reward = float(info.get("score", 0) - self.last_score) / 2
            self.last_score = float(info.get("score", 0))
        else:
            self.last_score = float(info.get("score", 0))
        
        # Total reward
        total_reward = reward + life_reward + coin_reward + score_reward + time_penalty + progress_reward
        
        return total_reward
    
    def save(self):
        """Save agent state."""
        checkpoint = {
            "model": self.net.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
            "config": self.config.to_dict(),
            "trainer_state": self.trainer.get_metrics()
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        self.console.print(f"[green]Agent saved to {self.checkpoint_path}[/green]")
    
    def load(self) -> bool:
        """
        Load agent state from checkpoint.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.console.print(f"[yellow]Loading checkpoint from {self.checkpoint_path}...[/yellow]")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Load network
            self.net.load_state_dict(checkpoint["model"])
            
            # Load optimizer
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            
            # Load training state
            self.exploration_rate = checkpoint.get("exploration_rate", 0.6)
            self.curr_step = checkpoint.get("curr_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            self.best_reward = checkpoint.get("best_reward", float('-inf'))
            
            # Load trainer state
            trainer_state = checkpoint.get("trainer_state", {})
            self.trainer.step_count = trainer_state.get("step", 0)
            self.trainer.episode_count = trainer_state.get("episode", 0)
            self.trainer._best_reward = trainer_state.get("best_reward", float('-inf'))
            
            self.console.print(f"[green]Checkpoint loaded successfully! Step: {self.curr_step}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error loading checkpoint: {e}[/red]")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        return {
            "step": self.curr_step,
            "episode": self.episode_count,
            "exploration_rate": self.exploration_rate,
            "best_reward": self.best_reward,
            "network_metrics": self.net.get_metrics(),
            "trainer_metrics": self.trainer.get_metrics()
        }
    
    def print_stats(self, episode: int):
        """Print agent statistics."""
        metrics = self.get_metrics()
        
        self.console.print(f"\n[bold cyan]Fixed MBP Agent Stats — Episode {episode}[/bold cyan]")
        self.console.print(f"[yellow]Step:[/yellow] {metrics['step']}")
        self.console.print(f"[yellow]Exploration Rate:[/yellow] {metrics['exploration_rate']:.4f}")
        self.console.print(f"[yellow]Best Reward:[/yellow] {metrics['best_reward']:.2f}")
        
        # Network metrics
        net_metrics = metrics['network_metrics']
        self.console.print(f"[yellow]State Neurons:[/yellow] {net_metrics['num_state_neurons']}")
        self.console.print(f"[yellow]Prediction Neurons:[/yellow] {net_metrics['num_prediction_neurons']}")
        self.console.print(f"[yellow]Total Neurons:[/yellow] {net_metrics['total_neurons']}")
        
        # Trainer metrics
        trainer_metrics = metrics['trainer_metrics']
        self.console.print(f"[yellow]Buffer Size:[/yellow] {trainer_metrics['buffer_size']}")
        
        print()

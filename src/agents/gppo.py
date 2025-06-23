import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class TrajectoryBuffer:
    """
    Buffer for storing trajectories collected during GPPO training.
    Stores states, actions, rewards, values, log probabilities, and dones.
    Calculates returns and advantages using GAE.
    """
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def add(self, state, action, reward, value, log_prob, done):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Clear the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def compute_returns_and_advantages(self, last_value=0):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)

        Args:
            last_value: Value estimate for the state after the last state in the trajectory
        """
        # Convert lists to numpy arrays for faster computation
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        # Initialize arrays
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # Initialize gae
        gae = 0

        # Compute returns and advantages in reverse order
        for t in reversed(range(len(rewards))):
            # If done, next value is 0
            next_non_terminal = 1.0 - dones[t]

            # Delta = r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]

            # GAE(s_t) = delta_t + gamma * lambda * GAE(s_{t+1})
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            # Store advantage and return
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Store computed values
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()

    def get_batch(self, device):
        """
        Get all data as tensors on the specified device

        Returns:
            Dict containing all trajectory data as tensors
        """
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        returns = torch.FloatTensor(np.array(self.returns)).to(device)
        advantages = torch.FloatTensor(np.array(self.advantages)).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'returns': returns,
            'advantages': advantages
        }

class GPPO:
    """
    Generalized Proximal Policy Optimization algorithm.

    Implements the core GPPO algorithm with clipping objective and value function loss.
    """
    def __init__(
        self,
        actor_critic,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_clipped_value=True,
        gamma=0.99,
        gae_lambda=0.95,
        adaptive_clip=False,
        kl_target=0.01
    ):
        """
        Initialize GPPO algorithm.

        Args:
            actor_critic: Actor-critic model that outputs action distributions and value estimates
            clip_param: Clipping parameter for PPO
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum norm for gradient clipping
            use_clipped_value: Whether to use clipped value function loss
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            adaptive_clip: Whether to use adaptive clipping parameter
            kl_target: Target KL divergence for adaptive clipping
        """
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value = use_clipped_value
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.adaptive_clip = adaptive_clip
        self.kl_target = kl_target

        # Initialize trajectory buffer
        self.buffer = TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda)

    def update(self, optimizer, epochs=10, batch_size=None, device='cpu', last_value=0):
        # Skip update if buffer is empty to avoid invalid input
        if not self.buffer.states:
            return {}
        """
        Update policy and value function using collected trajectories.

        Args:
            optimizer: Optimizer for actor-critic model
            epochs: Number of epochs to train on the collected data
            batch_size: Batch size for training (if None, use all data)
            device: Device to use for computation
            last_value: Value estimate for the state after the last state in the trajectory

        Returns:
            Dict containing training metrics
        """
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value=last_value)

        # Get batch data
        batch_data = self.buffer.get_batch(device)
        states = batch_data['states']
        actions = batch_data['actions']
        old_log_probs = batch_data['old_log_probs']
        returns = batch_data['returns']
        advantages = batch_data['advantages']

        # Training metrics
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'approx_kl': [],
            'total_loss': []
        }

        # Training loop
        for epoch in range(epochs):
            # If batch_size is None, use all data
            if batch_size is None:
                # Forward pass
                dist, values = self.actor_critic(states)

                # Get log probabilities of actions
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Compute ratio between old and new policy
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                if self.use_clipped_value:
                    # Get old values from buffer
                    old_values = torch.FloatTensor(np.array(self.buffer.values)).to(device)

                    # Clipped value function loss
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -self.clip_param, self.clip_param
                    )
                    value_loss1 = F.mse_loss(values, returns)
                    value_loss2 = F.mse_loss(values_clipped, returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    # Standard value function loss
                    value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Compute approximate KL divergence for early stopping
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean().item()

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                optimizer.step()

                # Store metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.item())
                metrics['approx_kl'].append(approx_kl)
                metrics['clip_fraction'].append(clip_fraction)
                metrics['total_loss'].append(loss.item())

                # Early stopping based on KL divergence
                if self.adaptive_clip and approx_kl > 1.5 * self.kl_target:
                    break
            else:
                # TODO: Implement mini-batch training if needed
                pass

        # Compute mean metrics
        for k, v in metrics.items():
            metrics[k] = np.mean(v)

        # Adjust clipping parameter if using adaptive clipping
        if self.adaptive_clip:
            if metrics['approx_kl'] < self.kl_target / 1.5:
                self.clip_param *= 0.8
            elif metrics['approx_kl'] > self.kl_target * 1.5:
                self.clip_param *= 1.2

            # Clip to reasonable range
            self.clip_param = max(0.05, min(0.5, self.clip_param))

        # Clear buffer after update
        self.buffer.clear()

        return metrics

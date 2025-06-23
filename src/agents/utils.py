import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_reward_shaping(reward, done, info, last_position, max_pos, score_inicial, coins_inicial, vida_inicial):
    """
    Calculate shaped reward based on various game metrics
    
    Args:
        reward: Original reward from environment
        done: Whether the episode is done
        info: Info dictionary from environment
        last_position: Last known x position
        max_pos: Maximum x position reached
        score_inicial: Initial score
        coins_inicial: Initial coins
        vida_inicial: Initial lives
        
    Returns:
        shaped_reward: The shaped reward
        max_pos: Updated maximum position
    """
    progress_reward = 0
    life_reward = 0
    coin_reward = 0
    score_reward = 0
    time_penalty = -0.01  # Small penalty for time

    # Reward based on progress
    if last_position is not None:
        progress = (info["x_pos"] - last_position)/10

        if progress > 1:
            progress_reward = progress
            max_pos = max(max_pos, info["x_pos"])

    # Reward for completing the level
    if info.get("flag_get", False):
        reward += 50

    # Reward/Penalty for Life
    if 'life' in info:
        life_change = float(int(info["life"]) - int(vida_inicial))
        if life_change > 0:
            life_reward = 10
        elif life_change < 0:
            life_reward = -5

    # Reward for collecting coins
    coin_reward = info["coins"] - coins_inicial

    # Reward for eliminating enemies
    score_increase = float(info["score"] - score_inicial)/2
    score_reward = score_increase

    # Sum the rewards
    shaped_reward = reward + progress_reward + life_reward + coin_reward + score_reward + time_penalty

    return shaped_reward, max_pos

def preprocess_state(state, device='cpu'):
    """
    Preprocess state for neural network input
    
    Args:
        state: State from environment
        device: Device to put tensor on
        
    Returns:
        processed_state: Processed state tensor
    """
    # Handle tuple states (from frame stacking)
    if isinstance(state, tuple):
        state = state[0]
    
    # Convert to numpy array if it's not already
    if hasattr(state, '__array__'):
        state = state.__array__()
    
    # Convert to tensor and add batch dimension
    state = torch.tensor(state, device=device).unsqueeze(0)
    
    return state

def visualize_attention(attention_weights, save_path=None):
    """
    Visualize attention weights
    
    Args:
        attention_weights: Attention weights tensor
        save_path: Optional path to save visualization
    """
    if attention_weights is None:
        return
    
    # Convert to numpy if it's a tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def epsilon_greedy_action(q_values, exploration_rate, action_dim):
    """
    Choose action using epsilon-greedy policy
    
    Args:
        q_values: Q-values from neural network
        exploration_rate: Current exploration rate
        action_dim: Number of possible actions
        
    Returns:
        action: Selected action
    """
    # Explore
    if np.random.rand() < exploration_rate:
        return np.random.randint(action_dim)
    
    # Exploit
    return torch.argmax(q_values, dim=1).item()

def decay_exploration_rate(exploration_rate, decay_rate, min_rate):
    """
    Decay exploration rate
    
    Args:
        exploration_rate: Current exploration rate
        decay_rate: Rate of decay
        min_rate: Minimum exploration rate
        
    Returns:
        new_rate: Updated exploration rate
    """
    return max(min_rate, exploration_rate * decay_rate)

def calculate_td_error(td_estimate, td_target):
    """
    Calculate TD error
    
    Args:
        td_estimate: TD estimate from neural network
        td_target: TD target
        
    Returns:
        td_error: TD error
    """
    return torch.abs(td_estimate - td_target).detach().mean().item()
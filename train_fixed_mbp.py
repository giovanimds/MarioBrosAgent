"""
Fixed MBP Training Script for MarioBrosAgent.

This script trains a fixed-size MBP network on the Mario environment,
testing different sizes to find the ideal MVP configuration.

Usage:
    python train_fixed_mbp.py --size small|medium|large
    python train_fixed_mbp.py --state-dim 64 --num-state-neurons 128 --num-prediction-neurons 256
"""

import argparse
import torch
from pathlib import Path
from rich.console import Console

from src.helpers.config import (
    GAME_ID, RENDER_MODE, SKIP_FRAMES, FRAME_SHAPE, NUM_STACK,
    SAVE_DIR
)
from src.env_manager.environment import create_env
from src.agents.mbp import FixedMBPConfig, FixedMBPAgent
from src.helpers.logger import MetricLogger


def main():
    console = Console()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Fixed MBP on Mario")
    parser.add_argument("--size", type=str, default="medium", 
                        choices=["small", "medium", "large"],
                        help="Network size configuration")
    parser.add_argument("--state-dim", type=int, default=None,
                        help="State dimension (overrides size preset)")
    parser.add_argument("--num-state-neurons", type=int, default=None,
                        help="Number of state neurons (overrides size preset)")
    parser.add_argument("--num-prediction-neurons", type=int, default=None,
                        help="Number of prediction neurons (overrides size preset)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of layers (overrides size preset)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU if available")
    parser.add_argument("--experiment", type=str, default="fixed_mbp_test",
                        help="Experiment name for checkpoint directory")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the game")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device:[/bold cyan] {device}")
    
    # Select configuration based on size
    if args.size == "small":
        config = FixedMBPConfig.get_small_config()
    elif args.size == "medium":
        config = FixedMBPConfig.get_medium_config()
    else:  # large
        config = FixedMBPConfig.get_large_config()
    
    # Override with command-line arguments
    if args.state_dim is not None:
        config.state_dim = args.state_dim
    if args.num_state_neurons is not None:
        config.num_state_neurons = args.num_state_neurons
    if args.num_prediction_neurons is not None:
        config.num_prediction_neurons = args.num_prediction_neurons
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    config.use_gpu = args.use_gpu
    
    # Create save directory
    save_dir = Path(SAVE_DIR) / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]Configuration:[/bold cyan]")
    console.print(f"  Size: {args.size}")
    console.print(f"  State Dim: {config.state_dim}")
    console.print(f"  State Neurons: {config.num_state_neurons}")
    console.print(f"  Prediction Neurons: {config.num_prediction_neurons}")
    console.print(f"  Layers: {config.num_layers}")
    console.print(f"  Batch Size: {config.batch_size}")
    console.print(f"  Total Neurons: {config.total_neurons}")
    console.print(f"  Save Dir: {save_dir}")
    
    # Create environment
    render_mode = "human" if args.render else RENDER_MODE
    env = create_env(
        game_id=GAME_ID,
        render_mode=render_mode,
        skip_frames=SKIP_FRAMES,
        shape=FRAME_SHAPE,
        num_stack=NUM_STACK
    )
    
    # Get environment dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    console.print(f"[bold cyan]Environment:[/bold cyan]")
    console.print(f"  State Dim: {state_dim}")
    console.print(f"  Action Dim: {action_dim}")
    
    # Create agent
    agent = FixedMBPAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        save_dir=save_dir,
        config=config
    )
    
    console.print(f"[bold cyan]Agent created with {config.total_neurons} total neurons[/bold cyan]")
    
    # Initialize logger
    logger = MetricLogger()
    logger.start_live_display()
    
    try:
        for e in range(args.episodes):
            # Reset environment
            state = env.reset()
            
            # Reset episode-specific variables
            if hasattr(agent, 'last_position'):
                delattr(agent, 'last_position')
            if hasattr(agent, 'last_coins'):
                delattr(agent, 'last_coins')
            if hasattr(agent, 'last_score'):
                delattr(agent, 'last_score')
            if hasattr(agent, 'last_life'):
                delattr(agent, 'last_life')
            
            # Reset metrics
            logger.init_episode()
            
            episode_reward = 0
            step_count = 0
            
            # Play one episode
            done = False
            while not done:
                # Get action
                action = agent.act(state)
                
                # Execute action
                next_state, reward, done, trunc, info = env.step(action)
                
                # Calculate custom reward
                custom_reward = agent.calculate_reward(reward, done, info)
                
                # Cache experience
                agent.cache(state, next_state, action, custom_reward, done, info)
                
                # Learn
                q, loss = agent.learn()
                
                # Log metrics
                if q is not None and loss is not None:
                    logger.log_step(custom_reward, loss, q, agent.get_metrics())
                else:
                    logger.log_step(custom_reward, 0, 0, agent.get_metrics())
                
                # Update state
                state = next_state
                episode_reward += custom_reward
                step_count += 1
                
                # Check if done
                if done or trunc:
                    break
            
            # Log episode metrics
            logger.log_episode()
            
            # Update live display
            logger.update_live_display(
                episode=e,
                moe_metrics=agent.get_metrics(),
                mario_net=agent.net,
                epsilon=agent.exploration_rate,
                step=agent.curr_step
            )
            
            # Print stats every 20 episodes
            if e % 20 == 0:
                agent.print_stats(e)
                console.print(f"[bold yellow]Episode {e}: Reward = {episode_reward:.2f}, Steps = {step_count}[/bold yellow]")
            
            # Save checkpoint every 100 episodes
            if e % 100 == 0:
                agent.save()
                console.print(f"[green]Checkpoint saved at episode {e}[/green]")
            
            # Record metrics every 100 episodes
            if e % 100 == 0:
                logger.record(
                    episode=e,
                    epsilon=agent.exploration_rate,
                    step=agent.curr_step,
                    moe_metrics=agent.get_metrics(),
                    mario_net=agent.net
                )
    
    except KeyboardInterrupt:
        console.print("[bold red]Training interrupted by user[/bold red]")
    finally:
        # Stop live display
        logger.stop_live_display()
        
        # Save final state
        agent.save()
        console.print(f"[green]Final state saved. Total episodes: {agent.episode_count}[/green]")
        
        # Close environment
        env.close()
        
        console.print("[bold green]Training completed[/bold green]")


if __name__ == "__main__":
    main()

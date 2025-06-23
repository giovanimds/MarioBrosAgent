import torch
from pathlib import Path
import time
from rich.console import Console
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from src.helpers.config import (
    GAME_ID, RENDER_MODE, SKIP_FRAMES, FRAME_SHAPE, NUM_STACK,
    SAVE_DIR, NUM_AGENTS, TRAJECTORY_LENGTH
)
from src.env_manager.environment import create_env
from src.agents.gppo_agent import GPPOMario
from src.helpers.logger import MetricLogger
from src.helpers.io import save_checkpoint, load_checkpoint, save_agent_state, load_agent_state

def run_agent_episode_parallel(agent_data):
    """Run a single episode for an agent in its environment (parallel version)"""
    agent, env, agent_idx = agent_data
    
    # Reset environment
    state, _ = env.reset()
    
    # Reset episode-specific variables
    if hasattr(agent, 'last_x_pos'):
        delattr(agent, 'last_x_pos')
    if hasattr(agent, 'last_coins'):
        delattr(agent, 'last_coins')
    
    episode_reward = 0
    step_count = 0
    metrics_collected = []
    moe_metrics = None  # Initialize to avoid unbound variable
    
    # Play one episode
    done = False
    while not done:
        # Get action, log probability, and value
        action, log_prob, value = agent.act(state)
        
        # Execute action
        next_state, reward, done, trunc, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Store in trajectory buffer
        agent.cache(state, next_state, action, reward, done or trunc, info, log_prob, value)
        
        # Learn if trajectory is complete or episode ends
        if agent.trajectory_step >= TRAJECTORY_LENGTH or done or trunc:
            metrics = agent.learn()
            if metrics:
                metrics_collected.append(metrics)
        
        # Collect MoE metrics
        moe_metrics = agent.net.get_moe_metrics()
        
        # Update state
        state = next_state
        
        # Check if done
        if done or trunc:
            break
    
    # Ensure we have MoE metrics even if the loop didn't run
    if moe_metrics is None:
        moe_metrics = agent.net.get_moe_metrics()
    
    return {
        'agent_idx': agent_idx,
        'episode_reward': episode_reward,
        'step_count': step_count,
        'metrics': metrics_collected,
        'moe_metrics': moe_metrics,
        'curr_step': agent.curr_step
    }

def main():
    console = Console()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device:[/bold cyan] {device}")

    # Create environments - one per agent for parallel experience collection
    console.print(f"[bold cyan]Creating {NUM_AGENTS} environment(s)...[/bold cyan]")
    
    envs = []
    for i in range(NUM_AGENTS):
        # For multi-agent setup, only render the first environment to avoid performance issues
        render_mode = RENDER_MODE if i == 0 else "rgb_array"
        env = create_env(
            game_id=GAME_ID,
            render_mode=render_mode,
            skip_frames=SKIP_FRAMES,
            shape=FRAME_SHAPE,
            num_stack=NUM_STACK
        )
        envs.append(env)
    
    # Get environment dimensions from the first environment
    state_dim = envs[0].observation_space.shape
    action_dim = envs[0].action_space.n

    # Create save directory
    save_dir = Path(SAVE_DIR) / "gppo"
    save_dir.mkdir(exist_ok=True)

    # Initialize agents based on NUM_AGENTS configuration
    console.print(f"[bold cyan]Creating {NUM_AGENTS} GPPO agent(s)...[/bold cyan]")

    agents = []
    for i in range(NUM_AGENTS):
        agent_save_dir = save_dir / f"agent_{i}"
        agent_save_dir.mkdir(exist_ok=True)
        agents.append(GPPOMario(state_dim, action_dim, agent_save_dir))

    # For backward compatibility
    mario = agents[0]

    # Initialize logger
    logger = MetricLogger()

    # Start training
    episodes = 40000

    # Start live display
    logger.start_live_display()

    try:
        with ThreadPoolExecutor(max_workers=NUM_AGENTS) as executor:
            for e in range(episodes):
                # Reset metrics for this episode batch
                logger.init_episode()
                
                # Prepare agent data for parallel execution
                agent_data = [(agents[i], envs[i], i) for i in range(NUM_AGENTS)]
                
                # Submit all agent episodes to run in parallel
                future_to_agent = {
                    executor.submit(run_agent_episode_parallel, data): data[2] 
                    for data in agent_data
                }
                
                # Collect results as they complete
                episode_results = []
                for future in as_completed(future_to_agent):
                    result = future.result()
                    episode_results.append(result)
                    
                    # Log metrics for this agent
                    for metrics in result['metrics']:
                        policy_loss = metrics.get('policy_loss', 0)
                        value_loss = metrics.get('value_loss', 0)
                        logger.log_step(0, policy_loss, value_loss, result['moe_metrics'])
                    
                    # Log step metrics
                    logger.log_step(result['episode_reward'], 0, 0, result['moe_metrics'])

                # Sort results by agent index for consistent reporting
                episode_results.sort(key=lambda x: x['agent_idx'])

                # Log episode metrics
                logger.log_episode()

                # Update live display using the first agent's metrics
                logger.update_live_display(
                    episode=e, 
                    moe_metrics=episode_results[0]['moe_metrics'], 
                    mario_net=agents[0].net,
                    epsilon=0,  # No epsilon for GPPO
                    step=episode_results[0]['curr_step']
                )

                # Print GPPO stats every 20 episodes
                if e % 20 == 0:
                    console.print(f"[bold yellow]Episode {e} Statistics (Parallel):[/bold yellow]")
                    
                    episode_rewards = [r['episode_reward'] for r in episode_results]
                    avg_reward = sum(episode_rewards) / len(episode_rewards)
                    console.print(f"  Average Episode Reward: {avg_reward:.2f}")
                    
                    for result in episode_results:
                        i = result['agent_idx']
                        console.print(f"  Agent {i}: Step {result['curr_step']}, Reward: {result['episode_reward']:.2f}")
                        
                    # Print detailed stats for first agent
                    agents[0].print_gppo_stats(e)

                # Record metrics every 100 episodes
                if e % 100 == 0:
                    logger.record(
                        episode=e,
                        epsilon=0,  # No epsilon for GPPO
                        step=episode_results[0]['curr_step'],
                        moe_metrics=episode_results[0]['moe_metrics'],
                        mario_net=agents[0].net
                    )

                    # Save all agents' states
                    for i, agent in enumerate(agents):
                        console.print(f"[green]Saving Agent {i}...[/green]")
                        agent.save()

    except KeyboardInterrupt:
        console.print("[bold red]Training interrupted by user[/bold red]")
    finally:
        # Stop live display
        logger.stop_live_display()

        # Save final state for all agents
        for i, agent in enumerate(agents):
            console.print(f"[green]Saving final state for Agent {i}...[/green]")
            agent.save()

        # Close all environments
        for i, env in enumerate(envs):
            console.print(f"[cyan]Closing environment {i}...[/cyan]")
            env.close()

        console.print("[bold green]Parallel training completed[/bold green]")

if __name__ == "__main__":
    main()

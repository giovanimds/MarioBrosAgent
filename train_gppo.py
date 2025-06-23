import torch
from pathlib import Path
import time
from rich.console import Console

from src.helpers.config import (
    GAME_ID, RENDER_MODE, SKIP_FRAMES, FRAME_SHAPE, NUM_STACK,
    SAVE_DIR, NUM_AGENTS, TRAJECTORY_LENGTH
)
from src.env_manager.environment import create_env
from src.agents.gppo_agent import GPPOMario
from src.helpers.logger import MetricLogger
from src.helpers.io import save_checkpoint, load_checkpoint, save_agent_state, load_agent_state

def main():
    console = Console()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device:[/bold cyan] {device}")

    # Create environment
    env = create_env(
        game_id=GAME_ID,
        render_mode=RENDER_MODE,
        skip_frames=SKIP_FRAMES,
        shape=FRAME_SHAPE,
        num_stack=NUM_STACK
    )

    # Get environment dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # Create save directory
    save_dir = Path(SAVE_DIR) / "gppo"
    save_dir.mkdir(exist_ok=True)

    # Initialize agents based on NUM_AGENTS configuration
    console.print(f"[bold cyan]Creating {NUM_AGENTS} GPPO agent(s)...[/bold cyan]")

    if NUM_AGENTS == 1:
        # Single agent mode
        agents = [GPPOMario(state_dim, action_dim, save_dir)]
        mario = agents[0]  # For backward compatibility
    else:
        # Multi-agent mode
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
        for e in range(episodes):
            # Reset environment
            state, _ = env.reset()

            # Reset metrics
            logger.init_episode()

            # Select agent for this episode (round-robin for simplicity)
            current_agent = agents[e % NUM_AGENTS]

            # Reset episode-specific variables
            if hasattr(current_agent, 'last_x_pos'):
                delattr(current_agent, 'last_x_pos')
            if hasattr(current_agent, 'last_coins'):
                delattr(current_agent, 'last_coins')

            # Play one episode
            done = False
            while not done:
                # Get action, log probability, and value
                action, log_prob, value = current_agent.act(state)

                # Execute action
                next_state, reward, done, trunc, info = env.step(action)

                # Store in trajectory buffer
                current_agent.cache(state, next_state, action, reward, done or trunc, info, log_prob, value)

                # Learn if trajectory is complete or episode ends
                if current_agent.trajectory_step >= TRAJECTORY_LENGTH or done or trunc:
                    metrics = current_agent.learn()

                    # Log metrics if available (using existing log_step method)
                    if metrics:
                        # Use policy_loss as loss and value_loss as q for logging
                        policy_loss = metrics.get('policy_loss', 0)
                        value_loss = metrics.get('value_loss', 0)
                        logger.log_step(0, policy_loss, value_loss, current_agent.net.get_moe_metrics())

                # Log metrics
                moe_metrics = current_agent.net.get_moe_metrics()
                logger.log_step(reward, 0, 0, moe_metrics)  # No loss or Q values for GPPO

                # Update state
                state = next_state

                # Check if done
                if done or trunc:
                    break

            # Log episode metrics
            logger.log_episode()

            # Update live display
            logger.update_live_display(
                episode=e, 
                moe_metrics=current_agent.net.get_moe_metrics(), 
                mario_net=current_agent.net,
                epsilon=0,  # No epsilon for GPPO
                step=current_agent.curr_step
            )

            # Print GPPO stats every 20 episodes
            if e % 20 == 0:
                current_agent.print_gppo_stats(e)

                # Print agent information if using multiple agents
                if NUM_AGENTS > 1:
                    console.print(f"[bold yellow]Episode {e} completed by Agent {e % NUM_AGENTS}[/bold yellow]")

            # Record metrics every 100 episodes
            if e % 100 == 0:
                logger.record(
                    episode=e,
                    epsilon=0,  # No epsilon for GPPO
                    step=current_agent.curr_step,
                    moe_metrics=current_agent.net.get_moe_metrics(),
                    mario_net=current_agent.net
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

        # Close environment
        env.close()

        console.print("[bold green]Training completed[/bold green]")

if __name__ == "__main__":
    main()

import torch
from pathlib import Path
import time
from rich.console import Console

from src.helpers.config import (
    GAME_ID, RENDER_MODE, SKIP_FRAMES, FRAME_SHAPE, NUM_STACK,
    SAVE_DIR, NUM_AGENTS
)
from src.env_manager.environment import create_env
from src.agents.agent import Mario
from src.agents.manager import AgentManager
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
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(exist_ok=True)

    # Initialize agents based on NUM_AGENTS configuration
    console.print(f"[bold cyan]Creating {NUM_AGENTS} agent(s)...[/bold cyan]")

    if NUM_AGENTS == 1:
        # Single agent mode
        agents = [Mario(state_dim, action_dim, save_dir)]
        mario = agents[0]  # For backward compatibility
    else:
        # Multi-agent mode
        agents = []
        for i in range(NUM_AGENTS):
            agent_save_dir = save_dir / f"agent_{i}"
            agent_save_dir.mkdir(exist_ok=True)
            agents.append(Mario(state_dim, action_dim, agent_save_dir))

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
            state = env.reset()

            # Reset metrics
            logger.init_episode()

            # Select agent for this episode (round-robin for simplicity)
            current_agent = agents[e % NUM_AGENTS]

            # Play one episode
            while True:
                # Get action
                action = current_agent.act(state)

                # Execute action
                next_state, reward, done, trunc, info = env.step(action)

                # Remember
                current_agent.cache(state, next_state, action, reward, done, info)

                # Learn
                q, loss = current_agent.learn()

                # Log metrics
                logger.log_step(reward, loss, q, current_agent.net.get_moe_metrics())

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
                epsilon=current_agent.exploration_rate,
                step=current_agent.curr_step
            )

            # Print MoE stats every 20 episodes
            if e % 20 == 0:
                current_agent.print_moe_stats(e)

                # Print agent information if using multiple agents
                if NUM_AGENTS > 1:
                    console.print(f"[bold yellow]Episode {e} completed by Agent {e % NUM_AGENTS}[/bold yellow]")

            # Record metrics every 100 episodes
            if e % 100 == 0:
                logger.record(
                    episode=e,
                    epsilon=current_agent.exploration_rate,
                    step=current_agent.curr_step,
                    moe_metrics=current_agent.net.get_moe_metrics(),
                    mario_net=current_agent.net
                )

                # Save all agents' states
                for i, agent in enumerate(agents):
                    console.print(f"[green]Saving Agent {i}...[/green]")
                    save_agent_state(agent)

    except KeyboardInterrupt:
        console.print("[bold red]Training interrupted by user[/bold red]")
    finally:
        # Stop live display
        logger.stop_live_display()

        # Save final state for all agents
        for i, agent in enumerate(agents):
            console.print(f"[green]Saving final state for Agent {i}...[/green]")
            save_agent_state(agent)

        # Close environment
        env.close()

        console.print("[bold green]Training completed[/bold green]")

if __name__ == "__main__":
    main()

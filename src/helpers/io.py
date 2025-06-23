import pickle
import torch
from pathlib import Path
from src.helpers.config import SAVE_DIR

def save_checkpoint(model, exploration_rate, path=None):
    """
    Save model checkpoint and exploration rate

    Args:
        model: The model to save
        exploration_rate: The current exploration rate
        path: Optional path to save to (defaults to SAVE_DIR/mario_net.chkpt)
    """
    if path is None:
        path = SAVE_DIR / "mario_net.chkpt"

    torch.save(
        dict(model=model.state_dict(), exploration_rate=exploration_rate),
        path
    )
    print(f"Model saved to {path}")

def load_checkpoint(model, path=None, device='cpu'):
    """
    Load model checkpoint and return exploration rate

    Args:
        model: The model to load into
        path: Optional path to load from (defaults to SAVE_DIR/mario_net.chkpt)
        device: Device to load the model to

    Returns:
        exploration_rate: The saved exploration rate or None if loading failed
    """
    if path is None:
        path = SAVE_DIR / "mario_net.chkpt"

    if not path.exists():
        print(f"No checkpoint found at {path}")
        return None

    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        exploration_rate = checkpoint["exploration_rate"]
        print(f"Checkpoint loaded from {path}")
        return exploration_rate
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def save_agent_state(agent, path=None):
    """
    Save agent state to file
    
    Args:
        agent: The agent to save
        path: Optional path to save to (defaults to SAVE_DIR/agent_state.pkl)
    """
    if path is None:
        path = SAVE_DIR / "agent_state.pkl"
    
    state = agent.get_state() if hasattr(agent, 'get_state') else {
        'exploration_rate': agent.exploration_rate,
        'curr_step': agent.curr_step,
        'last_position': agent.last_position,
        'score_inicial': getattr(agent, 'score_inicial', 0),
        'coins_inicial': getattr(agent, 'coins_inicial', 0),
        'vida_inicial': getattr(agent, 'vida_inicial', 2),
    }
    
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    print(f"Agent state saved to {path}")

def load_agent_state(agent, path=None):
    """
    Load agent state from file
    
    Args:
        agent: The agent to load into
        path: Optional path to load from (defaults to SAVE_DIR/agent_state.pkl)
        
    Returns:
        success: True if loading succeeded, False otherwise
    """
    if path is None:
        path = SAVE_DIR / "agent_state.pkl"
    
    if not path.exists():
        print(f"No agent state found at {path}")
        return False
    
    try:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        if hasattr(agent, 'set_state'):
            agent.set_state(state)
        else:
            # Set attributes directly
            for key, value in state.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)

        print(f"Agent state loaded from {path}")
        return True
    except Exception as e:
        print(f"Error loading agent state: {e}")
        return False

def save_multiple_agents(agents, path=None):
    """
    Save multiple agents' states to file

    Args:
        agents: List of agents to save
        path: Optional path to save to (defaults to SAVE_DIR/agents_state.pkl)
    """
    if path is None:
        path = SAVE_DIR / "agents_state.pkl"

    states = [
        agent.get_state() if hasattr(agent, 'get_state') else {
            'exploration_rate': agent.exploration_rate,
            'curr_step': agent.curr_step
        }
        for agent in agents
    ]

    with open(path, 'wb') as f:
        pickle.dump(states, f)
    print(f"Multiple agent states saved to {path}")

def load_multiple_agents(agents, path=None):
    """
    Load multiple agents' states from file

    Args:
        agents: List of agents to load into
        path: Optional path to load from (defaults to SAVE_DIR/agents_state.pkl)

    Returns:
        success: True if loading succeeded, False otherwise
    """
    if path is None:
        path = SAVE_DIR / "agents_state.pkl"

    if not path.exists():
        print(f"No multiple agent states found at {path}")
        return False

    try:
        with open(path, 'rb') as f:
            states = pickle.load(f)

        if len(states) != len(agents):
            print(f"Warning: Number of saved states ({len(states)}) doesn't match number of agents ({len(agents)})")

        for i, (agent, state) in enumerate(zip(agents, states)):
            if hasattr(agent, 'set_state'):
                agent.set_state(state)
            else:
                # Set attributes directly
                for key, value in state.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)

        print(f"Multiple agent states loaded from {path}")
        return True
    except Exception as e:
        print(f"Error loading multiple agent states: {e}")
        return False

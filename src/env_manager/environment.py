import os
import gym
from gym.wrappers.frame_stack import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from src.env_manager.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from src.env_manager.reward_wrapper import MarioRewardWrapper, RewardConfig, create_reward_wrapper

# Fix for AMD GPU issue with PyTorch
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Configuração padrão de recompensas
DEFAULT_REWARD_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.1,
    coin_reward=1.0,
    flag_reward=50.0,
    
    # Recompensas de estado
    grow_reward=5.0,
    fire_reward=10.0,
    death_penalty=-10.0,
    
    # Recompensas de tempo
    time_penalty=-0.01,
    time_bonus=1.0,
    
    # Recompensas de exploração
    jump_reward=0.05,
    run_reward=0.02,
    
    # Penalidades
    backward_penalty=-0.05,
    stuck_penalty=-0.1,
    
    # Normalização
    max_reward=10.0,
    stuck_threshold=20
)

# Ações disponíveis no Mario
MARIO_ACTIONS = [
    ["right"],      # 0: Andar para direita
    ['up'],         # 1: Olhar para cima
    ['down'],       # 2: Agachar/descer cano
    ["left"],       # 3: Andar para esquerda
    ["A"],          # 4: Pular
    ["B"],          # 5: Atirar (se for fireball)
    [],             # 6: Não fazer nada
    ['A', 'A', 'A', 'right'],  # 7: Pular alto para direita (para subir canos)
    ['B', 'right']   # 8: Correr para direita
]

# Mapeamento de ações para nomes
ACTION_NAMES = {
    0: "right",
    1: "up",
    2: "down", 
    3: "left",
    4: "A (jump)",
    5: "B (shoot)",
    6: "no-op",
    7: "A,A,A,right (climb)",
    8: "B,right (run)"
}

def create_env(
    game_id="SuperMarioBros-v0", 
    render_mode='human', 
    skip_frames=4, 
    shape=(84, 84), 
    num_stack=4,
    use_custom_rewards=True,
    reward_config=None
):
    """
    Create and configure the Super Mario Bros environment with appropriate wrappers.
    
    Args:
        game_id (str): The game ID to use (e.g., "SuperMarioBros-v0", "SuperMarioBrosRandomStages-v0")
        render_mode (str): The render mode ('human' for visualization, 'rgb_array' for headless)
        skip_frames (int): Number of frames to skip in the SkipFrame wrapper
        shape (tuple): Shape to resize observations to
        num_stack (int): Number of frames to stack
        use_custom_rewards (bool): Whether to use custom reward system
        reward_config (RewardConfig): Configuration for custom rewards
        
    Returns:
        env: The configured environment
    """
    # Initialize Super Mario environment
    env = gym_super_mario_bros.make(game_id, render_mode=render_mode, apply_api_compatibility=True)
    
    # Define action space
    env = JoypadSpace(env, MARIO_ACTIONS)
    
    # Apply custom reward wrapper if enabled
    if use_custom_rewards:
        config = reward_config or DEFAULT_REWARD_CONFIG
        env = MarioRewardWrapper(env, config=config)
    
    # Apply observation wrappers
    env = SkipFrame(env, skip=skip_frames)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = FrameStack(env, num_stack=num_stack)
    
    return env

def get_action_meaning(action_idx: int) -> str:
    """Retorna o nome da ação com base no índice"""
    return ACTION_NAMES.get(action_idx, f"unknown_{action_idx}")

def get_available_games() -> list:
    """Retorna a lista de jogos do Mario disponíveis"""
    return [
        "SuperMarioBros-v0",
        "SuperMarioBros-v1",
        "SuperMarioBros-v2",
        "SuperMarioBros-v3",
        "SuperMarioBrosRandomStages-v0",
        "SuperMarioBrosRandomStages-v1",
        "SuperMarioBrosRandomStages-v2",
        "SuperMarioBrosRandomStages-v3",
        "SuperMarioBros2-v0",
        "LostLevels-v0",
        "LostLevels-v1",
    ]
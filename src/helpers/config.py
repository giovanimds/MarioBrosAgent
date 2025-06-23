from pathlib import Path

# Agent configuration
NUM_AGENTS = 4  # Number of agents to run in parallel

# Environment configuration
GAME_ID = "SuperMarioBros-v0"  # Game ID for gym_super_mario_bros
RENDER_MODE = None  # "human" for visualization, "rgb_array" for headless
SKIP_FRAMES = 4  # Number of frames to skip in the SkipFrame wrapper
FRAME_SHAPE = (84, 84)  # Shape to resize observations to
NUM_STACK = 4  # Number of frames to stack

# Training configuration
BATCH_SIZE = 32
GAMMA = 0.9  # Discount factor
EXPLORATION_RATE = 0.6  # Initial exploration rate
EXPLORATION_RATE_DECAY = 0.99999
EXPLORATION_RATE_MIN = 0.2
BURNIN = 1e2  # Minimum experiences before training
LEARN_EVERY = 6  # Number of experiences between updates to Q_online
SYNC_EVERY = 24  # Number of experiences between Q_target & Q_online sync
SAVE_EVERY = 5e5  # Number of experiences between saving Mario Net

# Optimizer configuration
LEARNING_RATE = 0.00075
WEIGHT_DECAY = 0.005

# MoE configuration
LOAD_BALANCING_SCHEDULER = {
    'base_coef': 0.05,  # Base value
    'max_coef': 0.35,   # Maximum value
    'min_coef': 0.02,   # Minimum value
    'imbalance_threshold': 0.25,  # Threshold for imbalance
    'adjustment_factor': 1.15,   # Adjustment factor
    'decay_factor': 0.985,       # Decay factor
    'severe_imbalance_threshold': 0.35,  # Threshold for severe imbalance
    'check_interval': 500  # Check interval
}

# GPPO configuration
GAE_LAMBDA = 0.95  # Lambda parameter for Generalized Advantage Estimation
CLIP_PARAM = 0.2  # Clipping parameter for PPO
VALUE_COEF = 0.5  # Coefficient for value function loss
ENTROPY_COEF = 0.01  # Coefficient for entropy bonus
MAX_GRAD_NORM = 0.5  # Maximum norm for gradient clipping
PPO_EPOCHS = 10  # Number of epochs to train on each batch of trajectories
ADAPTIVE_CLIP = True  # Whether to use adaptive clipping parameter
KL_TARGET = 0.01  # Target KL divergence for adaptive clipping
TRAJECTORY_LENGTH = 2048  # Length of trajectory before updating

# Paths
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

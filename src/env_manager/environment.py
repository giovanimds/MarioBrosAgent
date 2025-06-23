import os
import gym
from gym.wrappers.frame_stack import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from src.env_manager.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

# Fix for AMD GPU issue with PyTorch
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

def create_env(game_id="SuperMarioBros-v0", render_mode='human', skip_frames=4, shape=(84, 84), num_stack=4):
    """
    Create and configure the Super Mario Bros environment with appropriate wrappers.
    
    Args:
        game_id (str): The game ID to use (e.g., "SuperMarioBros-v0", "SuperMarioBrosRandomStages-v0")
        render_mode (str): The render mode ('human' for visualization, 'rgb_array' for headless)
        skip_frames (int): Number of frames to skip in the SkipFrame wrapper
        shape (tuple): Shape to resize observations to
        num_stack (int): Number of frames to stack
        
    Returns:
        env: The configured environment
    """
    # Initialize Super Mario environment
    env = gym_super_mario_bros.make(game_id, render_mode=render_mode, apply_api_compatibility=True)
    
    # Define action space
    env = JoypadSpace(env, [
        ["right"], ['up'], ['down'], ["left"], 
        ["A"], ["B"], [],
        ['A', 'A', 'A', 'right'],  # For climbing pipes
        ['B', 'right']  # For running
    ])
    
    # Apply wrappers
    env = SkipFrame(env, skip=skip_frames)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)
    env = FrameStack(env, num_stack=num_stack)
    
    return env
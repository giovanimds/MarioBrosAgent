import torch
from torch import nn
import torch_directml
import math
from torchvision import transforms as T
import pytorch_optimizer as opts
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os
import pickle  # Adicionado para salvar o estado dos agentes

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"  # Fixes AMD GPU issue with PyTorch

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

import retro
env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1')

# Imprimir as ações disponíveis
print(env.action_space)

# Imprimir o espaço de observação
print(env.observation_space)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [['B'], ['A'], ['Y'], ['X'], ['L'], ['R'], ['select'], ['start'], ['up'], ['down'], ['left'], ['right']])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        obs = None  # Inicializa obs
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(0)  # Adiciona uma dimensão para o batch
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Remove a dimensão do batch
        return x

class MarioNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = self.__build_cnn(c, h, w, output_dim)

        self.target = self.__build_cnn(c, h, w, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, h, w, output_dim):
        # Definição das camadas convolucionais
        conv1 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1)
        conv2 = nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=8, stride=4)

        # Função para calcular o tamanho da saída de uma camada convolucional
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Calcula o tamanho da saída após cada camada convolucional
        convw = conv2d_size_out(w, kernel_size=3, stride=1)  # Para conv1
        convw = conv2d_size_out(convw, kernel_size=4, stride=2)  # Para conv2
        convw = conv2d_size_out(convw, kernel_size=8, stride=4)  # Para conv3

        convh = conv2d_size_out(h, kernel_size=3, stride=1)  # Para conv1
        convh = conv2d_size_out(convh, kernel_size=4, stride=2)  # Para conv2
        convh = conv2d_size_out(convh, kernel_size=8, stride=4)  # Para conv3

        # Calcula o tamanho da entrada para a camada linear
        linear_input_size = convw * convh * c * 4

        # Define a sequência de camadas
        return nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(linear_input_size, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            SelfAttentionLayer(embed_dim=512, num_heads=8),
            nn.Dropout(0.5),
            nn.ReLU(),
            SelfAttentionLayer(embed_dim=512, num_heads=8),
            nn.Dropout(0.5),
            nn.ReLU(),
            SelfAttentionLayer(embed_dim=512, num_heads=8),
            nn.Dropout(0.5),
            nn.ReLU(),
            SelfAttentionLayer(embed_dim=512, num_heads=8),
            nn.Dropout(0.5),
            nn.ReLU(),
            SelfAttentionLayer(embed_dim=512, num_heads=8),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

import numpy as np
import time, datetime
import matplotlib.pyplot as plt

from rich.console import Console

class MetricLogger:
    def __init__(self):
        self.console = Console()

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        self.console.print(
            f":video_game: [bold green]Episode {episode}[/bold green] - "
            f"[cyan]Step {step}[/cyan] - "
            f":chart_with_upwards_trend: Epsilon [yellow]{epsilon:.3f}[/yellow] - "
            f":trophy: Mean Reward [magenta]{mean_ep_reward}[/magenta] - "
            f":stopwatch: Mean Length [blue]{mean_ep_length}[/blue] - "
            f":bar_chart: Mean Loss [red]{mean_ep_loss}[/red] - "
            f":brain: Mean Q Value [green]{mean_ep_q}[/green] - "
            f":hourglass: Time Delta [cyan]{time_since_last_record}[/cyan] - "
            f":calendar: Time [bold]{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}[/bold]"
        )

class MarioB:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = torch_directml.device()
        # self.device = 'cpu'
        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 0.6
        self.exploration_rate_decay = 0.99995
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net
        self.last_position = None  # Adiciona um atributo para rastrear a última posição

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action_idx

class Mario(MarioB):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.checkpoint_path = save_dir / "mario_net.chkpt"
        self.gamma = 0.7
        self.console = Console()
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 1  # no. of experiences between updates to Q_online
        self.sync_every = 1e2  # no. of experiences between Q_target & Q_online sync
        from torchrl.data import ListStorage
        self.memory = TensorDictReplayBuffer(storage=ListStorage(5000))
        self.batch_size = 32

        self.max_pos = 0
        self.score_inicial = 0
        self.coins_inicial = 0
        self.vida_inicial = 2

        self.optimizer = opts.Adan(self.net.parameters(), lr=0.001, weight_decay=0.03)
        self.loss_fn = torch.nn.SmoothL1Loss().to(torch_directml.device())
        if self.checkpoint_path.exists():
            self.load()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def calculate_reward(self, reward, done, info):

        progress_reward = 0
        # Recompensa baseada no progresso
        if self.last_position is not None:
            if (info["x_pos"] - self.max_pos) > 5:
                self.max_pos = max(self.max_pos, info["x_pos"])
                progress_reward = min((self.max_pos - self.last_position) / 10.0, 0.1)
                self.last_position = info["x_pos"]
            else:
                progress_reward = -0.2

        else:
            progress_reward = 0
            self.last_position = info["x_pos"]
            self.max_pos = info["x_pos"]

        # Recompensa por completar o nível
        if info.get("flag_get", False):
            reward += 100
    

        if 'life' in info:
            if info["life"] > self.vida_inicial:
                reward += 1000
                self.vida_inicial = info["life"]

            if info["life"] < self.vida_inicial:
                reward -= 500
                self.vida_inicial = info["life"]

        # Recompensa por coletar moedas

        if info["coins"] > self.coins_inicial:
            reward += info["coins"]
            self.coins_inicial = info["coins"]

        if self.coins_inicial == 0:
            reward -= 2.0

        # Recompensa por eliminar inimigos

        if info["score"] > self.score_inicial:
            reward += float(info["score"] - self.score_inicial)
            self.score_inicial = float(info["score"])

        if self.score_inicial == 0:
            reward -= 5.0

        # Penalidade por tempo gasto
        reward -= 0.1

        # Soma a recompensa de progresso
        reward += progress_reward

        return reward

    def cache(self, state, next_state, action, reward, done, info):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        reward = self.calculate_reward(reward, done, info)
        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": torch.tensor([reward]),
            "done": torch.tensor([done])
        }, batch_size=[]))

        # Atualiza a última posição conhecida
        self.last_position = info['x_pos']

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), self.checkpoint_path)
        print(f"MarioNet salvo em {self.checkpoint_path} no passo {self.curr_step}")

    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint["exploration_rate"]
        print(f"Checkpoint carregado de {self.checkpoint_path}")

# Apply Wrappers to environment
env = SkipFrame(env, skip=2)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=64)
env = FrameStack(env, num_stack=4)

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

mario = Mario(state_dim=(4, 64, 64), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger()

episodes = 500

# Initialize rich progress bar
progress = Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)
training_task = progress.add_task("Training Mario Agent", total=episodes)

with progress:
    for e in range(episodes):
        state = env.reset()
        mario.last_position = None  # Reseta a última posição no início de cada episódio
        mario.score_inicial = 0
        mario.coins_inicial = 0
        mario.vida_inicial = 2
        mario.max_pos = 0
        # Play the game!
        while True:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done, info)

            # Learn
            q, loss = mario.learn()

            progress.update(training_task, description='Reward on turn: ' + str(reward))

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        # Update progress bar
        progress.update(training_task, advance=1)
        logger.log_episode()
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    mario.save()
        


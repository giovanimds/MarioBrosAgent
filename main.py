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

# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
# env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0", render_mode='human', apply_api_compatibility=True)
env = gym_super_mario_bros.make("SuperMarioBros-v3", render_mode='human', apply_api_compatibility=True)

print(gym.envs.registry.keys())

env = JoypadSpace(env, [
    ["right"], ['up'], ['down'], ["left"], 
    ["A"], ["B"], [],
    ['A', 'A', 'A', 'right'], # Pra subir no tubo
    ['B', 'right'] # Pra correr
])

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

import matplotlib.pyplot as plt
# plt.ion()  # Ativa modo interativo

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim, num_layers=5):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attn_weights = None  # Para armazenar os pesos de atenção
        self.input_logits = None  # Para armazenar os logits de entrada

        # self.lstm_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        for i in range(num_layers):
            
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    batch_first=True
                )
            )
            # self.lstm_layers.append(
            #     nn.LSTM(embed_dim, embed_dim, batch_first=True)
            # )

        self.conv1x1 = nn.Conv2d(4, 1, kernel_size=1)  # Preserva 2D
        self.out_project = nn.Linear(embed_dim, output_dim)  # Projeção linear
        

    def forward(self, x):
        batch, channels, height, width = x.shape
        original = x.view(batch, channels, -1)  # [batch, channels, height*width]
        self.input_logits = original.detach().cpu()
        # Corrigindo a forma de entrada
        if x.dim() == 2:  # [batch, features]
            x = x.unsqueeze(1)  # [batch, seq_len=1, features]
        if x.dim() == 4:
            x = x.view(x.size(0), x.size(1), -1)

        # for attn, lstm in zip(self.attention_layers, self.lstm_layers):
        #     x, attn_weights = attn(x, x, x)
        #     x, _ = lstm(x)

        for attn in self.attention_layers:
            x, attn_weights = attn(x, x, x)

        self.attn_weights = x.detach().cpu()
        out = self.conv1x1(x.view(batch, channels, height, width)).view(batch, -1)
        out = self.out_project(out)
        return out # Retorna para [batch, features]





class MarioNet(nn.Module):
    """CNN com Self-Attention aprimorada"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.device = torch_directml.device()
        # self.device = 'cpu'
        # Self-Attention
        self.attention = SelfAttentionLayer(
            embed_dim=h*w,  # Deve corresponder ao número de canais
            num_heads=c,
            output_dim=output_dim,
            num_layers=5
        )

        # Inicialização dos modelos
        self.online = self.attention
        self.target = self.attention
        self.target.load_state_dict(self.online.state_dict())
        
        # Congelar parâmetros do target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def get_attention_heatmap(self):
        """Retorna o heatmap de atenção para visualização"""
        if self.attention.attn_weights is None:
            return None
            
        # Processar os pesos de atenção
        attn = self.attention.attn_weights.mean(dim=-1)[0]  # Média entre heads
        size = int(np.sqrt(attn.shape[-1]))  # Assume formato quadrado
        return attn.view(size, size) if size*size == attn.shape[-1] else None
    
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
        self.exploration_rate_decay = 0.99999
        self.exploration_rate_min = 0.2
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
        self.gamma = 0.9
        self.console = Console()
        self.burnin = 1e2  # min. experiences before training
        self.learn_every = 6  # no. of experiences between updates to Q_online
        self.sync_every = 24  # no. of experiences between Q_target & Q_online sync
        from torchrl.data import ListStorage
        self.memory = TensorDictReplayBuffer(storage=ListStorage(10000))
        self.batch_size = 32
        self.plot_every = 1000
        # self.fig, self.ax = plt.subplots()
        self.heatmap = None
        self.max_pos = 40
        self.score_inicial = 0
        self.coins_inicial = 0
        self.vida_inicial = 2

        self.optimizer = opts.Adan(self.net.parameters(), lr=0.001, weight_decay=0.01)
        self.loss_fn = torch.nn.MSELoss()
        if self.checkpoint_path.exists():
            self.load()

    def _update_plot(self):
        """Atualiza o mapa de calor em tempo real"""
        heatmap_data = self.net.get_attention_heatmap()
        
        if heatmap_data is not None:
            if self.heatmap is None:
                self.heatmap = self.ax.imshow(
                    heatmap_data, 
                    cmap='viridis', 
                    interpolation='nearest'
                )
                plt.colorbar(self.heatmap, ax=self.ax)
            else:
                self.heatmap.set_data(heatmap_data)
                self.heatmap.autoscale()
            
            plt.pause(0.001)

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
        life_reward = 0
        coin_reward = 0
        score_reward = 0
        time_penalty = -0.01  # Reduzido para -0.01

        # Recompensa baseada no progresso
        if self.last_position is not None:
            progress = (info["x_pos"] - self.last_position)/3

            if progress > 1:
                progress_reward = progress
                self.max_pos = max(self.max_pos, info["x_pos"])

        else:
            self.last_position = info["x_pos"]
            self.max_pos = info["x_pos"]

        # Recompensa por completar o nível
        if info.get("flag_get", False):
            reward += 50  # Reduzido para 50

        # Recompensa/Punição por Vida
        if 'life' in info:
            life_change = float(int(info["life"]) - int(self.vida_inicial))
            if life_change > 0:
                life_reward = 100  # Reduzido para 10
            elif life_change < 0:
                life_reward = -150  # Reduzido para -5
            self.vida_inicial = info["life"]

        # Recompensa por coletar moedas
        coin_reward = info["coins"] - self.coins_inicial
        self.coins_inicial = info["coins"]

        # Recompensa por eliminar inimigos
        score_increase = float(info["score"] - self.score_inicial)/2
        score_reward = score_increase
        self.score_inicial = float(info["score"])

        # Soma as recompensas
        reward += progress_reward + life_reward + coin_reward + score_reward + time_penalty

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
        state = state.requires_grad_()  # Ensure state requires gradients
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, dim=-1)
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

    def act(self, state):
        action = super().act(state)
        
        # Atualizar visualização periodicamente
        if self.curr_step % self.plot_every == 0:
            pass
            # self._update_plot()
            
        return action

    def save(self):
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), self.checkpoint_path)
        print(f"MarioNet salvo em {self.checkpoint_path} no passo {self.curr_step}")

    def load(self):
        checkpoint = torch.load(self.checkpoint_path)
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

            reward = mario.calculate_reward(reward, done, info)

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
plt.close()  # Fechar a figura ao final do treinamento


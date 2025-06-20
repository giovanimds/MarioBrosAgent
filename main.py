import torch
from torch import nn
import torch.nn.functional as F
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
from gym.wrappers.frame_stack import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"  # Fixes AMD GPU issue with PyTorch

# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
# env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0", render_mode='human', apply_api_compatibility=True)
env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode='human', apply_api_compatibility=True)

# print(gym.envs.registry.keys())

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
        trunc = False
        info = {}
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


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
    """CNN com arquitetura Mixture of Experts (MoE) melhorada - 16 especialistas, top-4"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.device = torch.device('cpu')

        # Número de experts e configurações
        self.num_experts = 16  # Aumentado para 16 experts
        self.top_k = 8  # Mantido em 4 para favorecer maior competição

        # Extrair features com CNN melhorada
        self.feature_extractor = self._build_cnn_features(c, h, w)

        # Calcular dimensão das features após CNN
        with torch.no_grad():
            dummy_input = torch.randn(1, c, h, w)
            feature_dim = self.feature_extractor(dummy_input).shape[1]

        # Camada MoE com mais especialistas e melhor balanceamento
        self.moe_layer = MoELayer(
            input_dim=feature_dim,
            hidden_dim=192,  # Aumentado de 128 para 192
            output_dim=output_dim,
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        # Armazenar loss de balanceamento para treinamento
        self.last_load_balancing_loss = 0.0
        self.last_gate_probs = None

        # Atribuir IDs aos especialistas para inicialização diversificada
        for i, expert in enumerate(self.moe_layer.experts):
            expert.expert_id = i

        # Criar modelos online e target
        self.online = nn.Sequential(self.feature_extractor, self.moe_layer)

        # Target com configuração diferente para aumentar estabilidade
        # - Usa mais especialistas no top-k para ter previsões mais suaves
        self.target = nn.Sequential(
            self._build_cnn_features(c, h, w),
            MoELayer(
                input_dim=feature_dim,
                hidden_dim=192,
                output_dim=output_dim,
                num_experts=self.num_experts,
                top_k=self.top_k + 2  # Target usa mais experts no ensemble
            )
        )
        self.target.load_state_dict(self.online.state_dict())

        # Congelar parâmetros do target
        for p in self.target.parameters():
            p.requires_grad = False

        # Adicionar contador de passos para tracking
        self.step_counter = 0

    def _build_cnn_features(self, c, h, w):
        """Construir as camadas CNN para extração de features"""
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # Camadas convolucionais
        conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Calcular dimensões de saída
        convw = conv2d_size_out(w, kernel_size=8, stride=4)
        convw = conv2d_size_out(convw, kernel_size=4, stride=2)
        convw = conv2d_size_out(convw, kernel_size=3, stride=1)
        
        convh = conv2d_size_out(h, kernel_size=8, stride=4)
        convh = conv2d_size_out(convh, kernel_size=4, stride=2)
        convh = conv2d_size_out(convh, kernel_size=3, stride=1)
        
        linear_input_size = convw * convh * 64
        
        return nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU()
        )

    def forward(self, input, model):
        self.step_counter += 1

        if model == "online":
            # Extrair features
            features = self.feature_extractor(input)

            # Processar com MoE
            output, load_balancing_loss, gate_probs = self.moe_layer(features)

            # Armazenar métricas para monitoramento
            self.last_load_balancing_loss = load_balancing_loss.item()
            self.last_gate_probs = gate_probs.detach()

            # Ajustar coeficiente de balanceamento periodicamente
            if self.step_counter % 500 == 0:
                self.adjust_load_balancing_dynamically()

            return output
        elif model == "target":
            features = self.target[0](input)  # feature_extractor
            output, _, _ = self.target[1](features)  # moe_layer
            return output

    def adjust_load_balancing_dynamically(self):
        """Ajusta dinamicamente o coeficiente de balanceamento com base nas métricas atuais"""
        stats = self.moe_layer.get_expert_usage_stats()

        # Verificar indicadores de desbalanceamento
        max_usage = stats['max_usage']
        inactive_count = stats['inactive_count']
        gini = stats.get('gini_coefficient', 0.5)  # Índice de desigualdade
        current_coef = self.moe_layer.load_balancing_loss_coef

        # Regras para ajuste dinâmico
        if max_usage > 0.25 or inactive_count > self.num_experts // 4 or gini > 0.4:
            # Desbalanceamento detectado - aumentar coeficiente
            new_coef = min(current_coef * 1.2, 0.25)  # Máximo de 0.25
            self.moe_layer.load_balancing_loss_coef = new_coef
            print(f"🔼 Aumentando coeficiente de balanceamento: {current_coef:.4f} → {new_coef:.4f}")
        elif max_usage < 0.15 and inactive_count < 2 and gini < 0.3:
            # Bem balanceado - reduzir coeficiente gradualmente
            new_coef = max(current_coef * 0.95, 0.02)  # Mínimo de 0.02
            self.moe_layer.load_balancing_loss_coef = new_coef
            print(f"🔽 Reduzindo coeficiente de balanceamento: {current_coef:.4f} → {new_coef:.4f}")

    def get_moe_metrics(self):
        """Retorna métricas do MoE para monitoramento"""
        if self.last_gate_probs is not None:
            # Calcular distribuição de uso dos especialistas
            expert_usage = self.last_gate_probs.mean(dim=0).cpu().numpy()
            # Skip metrics if usage contains NaN
            if np.isnan(expert_usage).any():
                return None
            
            return {
                'load_balancing_loss': self.last_load_balancing_loss,
                'expert_usage': expert_usage,
                'expert_entropy': -np.sum(expert_usage * np.log(expert_usage + 1e-8))
            }
        return None

    def get_attention_heatmap(self):
        """Compatibilidade com código existente - retorna distribuição de especialistas"""
        if self.last_gate_probs is not None:
            # Retornar uso médio dos especialistas como "heatmap"
            return self.last_gate_probs.mean(dim=0).unsqueeze(0)
        return None
    
import numpy as np
import time, datetime
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich.live import Live

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
        
        # Rich Live Display
        self.live = None
        self.current_episode = 0

    def log_step(self, reward, loss, q, moe_metrics=None):
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

    def start_live_display(self):
        """Inicia o display live da tabela"""
        self.live = Live(self.generate_table(), refresh_per_second=4, screen=True)
        self.live.start()

    def stop_live_display(self):
        """Para o display live da tabela"""
        if self.live:
            self.live.stop()

    def generate_table(self, moe_metrics=None, mario_net=None):
        """Gera a tabela principal de monitoramento com informações dos experts"""
        table = Table(title="🎮 Mario Bros Agent Training Monitor", show_header=True, header_style="bold magenta")
        
        # Colunas principais
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Current", style="green")
        table.add_column("Mean (Last 100)", style="yellow")
        table.add_column("Best", style="bold green")
        
        # Calcular métricas
        current_reward = self.curr_ep_reward
        current_length = self.curr_ep_length
        
        mean_reward = np.mean(self.ep_rewards[-100:]) if self.ep_rewards else 0
        mean_length = np.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0
        mean_loss = np.mean(self.ep_avg_losses[-100:]) if self.ep_avg_losses else 0
        mean_q = np.mean(self.ep_avg_qs[-100:]) if self.ep_avg_qs else 0
        
        best_reward = max(self.ep_rewards) if self.ep_rewards else 0
        best_length = max(self.ep_lengths) if self.ep_lengths else 0
        
        # Adicionar linhas
        table.add_row("🏆 Episode", str(self.current_episode), "-", "-")
        table.add_row("💰 Reward", f"{current_reward:.2f}", f"{mean_reward:.2f}", f"{best_reward:.2f}")
        table.add_row("⏱️ Length", str(current_length), f"{mean_length:.1f}", str(best_length))
        table.add_row("📉 Loss", "-", f"{mean_loss:.5f}", "-")
        table.add_row("🧠 Q-Value", "-", f"{mean_q:.3f}", "-")
        
        # Adicionar informações adicionais
        table.add_section()
        table.add_row("🎯 Epsilon", f"{getattr(self, 'current_epsilon', 0):.4f}", "-", "-")
        table.add_row("📊 Total Steps", f"{getattr(self, 'current_step', 0)}", "-", "-")
        
        # Adicionar seção MoE
        table.add_section()
        
        # Obter informações dos experts
        if moe_metrics and 'expert_usage' in moe_metrics:
            expert_usage = moe_metrics.get('expert_usage', [])
            
            # Obter estatísticas detalhadas se disponível
            if mario_net and hasattr(mario_net, 'moe_layer'):
                detailed_stats = mario_net.moe_layer.get_expert_usage_stats()
                expert_usage = detailed_stats.get('usage', expert_usage)
            
            if len(expert_usage) > 0:
                # Criar string com percentuais separados por barras
                expert_percentages = " | ".join([f"{usage*100:.0f}%" for usage in expert_usage])
                table.add_row("🤖 Expert Usage", expert_percentages, "-", "-")
                
                # Adicionar métricas de balanceamento
                table.add_row("⚖️ Load Balance", f"{moe_metrics.get('load_balancing_loss', 0):.4f}", "-", "-")
                table.add_row("🎯 Expert Entropy", f"{moe_metrics.get('expert_entropy', 0):.3f}", "-", "Higher is better")
                
                # Status de balanceamento
                max_usage = max(expert_usage) if len(expert_usage) > 0 else 0
                if max_usage > 0.5:
                    balance_status = "[bold red]⚠️ SEVERE IMBALANCE[/bold red]"
                elif max_usage > 0.4:
                    balance_status = "[red]🔥 High imbalance[/red]"
                elif max_usage > 0.3:
                    balance_status = "[yellow]⚠️ Moderate imbalance[/yellow]"
                else:
                    balance_status = "[green]✅ Well balanced[/green]"
                
                table.add_row("📊 Balance Status", balance_status, "-", "-")
            else:
                table.add_row("🤖 Expert Usage", "⏳ Loading...", "-", "-")
        else:
            table.add_row("🤖 Expert Usage", "❌ No data", "-", "-")
        
        return table

    # Método generate_expert_table removido - informações dos experts agora estão na tabela principal

    def update_live_display(self, episode, moe_metrics=None, mario_net=None, epsilon=None, step=None):
        """Atualiza o display live com as informações atuais"""
        self.current_episode = episode
        self.current_epsilon = epsilon if epsilon is not None else getattr(self, 'current_epsilon', 0)
        self.current_step = step if step is not None else getattr(self, 'current_step', 0)
        
        if self.live:
            # Usar apenas uma tabela com todas as informações
            main_table = self.generate_table(moe_metrics, mario_net)
            self.live.update(main_table)

    def record(self, episode, epsilon, step, moe_metrics=None, mario_net=None):
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

        # Atualizar display live se estiver ativo
        if self.live:
            self.update_live_display(episode, moe_metrics, mario_net)


class MarioB:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = torch.device('cpu')
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
        self.heatmap = None
        self.max_pos = 40
        self.score_inicial = 0
        self.coins_inicial = 0
        self.vida_inicial = 2

        # Otimizador com parâmetros melhorados para MoE
        self.optimizer = opts.ASGD(
            self.net.parameters(), 
            lr=0.00075,  # Taxa de aprendizado reduzida para estabilidade
            weight_decay=0.005  # Maior regularização
        )
        self.loss_fn = torch.nn.SmoothL1Loss()  # Huber loss mais robusta

        # Scheduler dinâmico para balanceamento de carga
        self.load_balancing_scheduler = {
            'base_coef': 0.05,  # Valor base maior
            'max_coef': 0.35,   # Valor máximo um pouco menor
            'min_coef': 0.02,   # Valor mínimo
            'imbalance_threshold': 0.25,  # Threshold mais sensível
            'adjustment_factor': 1.15,   # Ajuste mais agressivo
            'decay_factor': 0.985,       # Decaimento mais lento
            'severe_imbalance_threshold': 0.35,  # Para ações emergenciais
            'check_interval': 500  # Verificar a cada 500 passos
        }

        # Contadores para monitoramento de balanceamento
        self.balance_checks = 0
        self.balance_adjustments = 0
        self.severe_imbalance_count = 0

        # Rastreamento de especialistas dominantes
        self.dominant_experts_history = []
        
        if self.checkpoint_path.exists():
            self.load()

    def _update_plot(self):
        """Atualiza o mapa de calor em tempo real"""
        heatmap_data = self.net.get_attention_heatmap()
        
        if heatmap_data is not None:
            # Implementação removida - usando Rich table em vez de matplotlib
            pass

    def update_Q_online(self, td_estimate, td_target):
        # Loss principal (Q-learning)
        q_loss = self.loss_fn(td_estimate, td_target)
        
        # Loss de balanceamento do MoE
        load_balancing_loss = self.net.last_load_balancing_loss
        
        # Verificar se estamos em situação de desbalanceamento grave
        moe_metrics = self.net.get_moe_metrics()
        if moe_metrics and 'expert_usage' in moe_metrics:
            expert_usage = moe_metrics['expert_usage']
            max_usage = expert_usage.max() if hasattr(expert_usage, 'max') else max(expert_usage)

            # Verificar condições de desbalanceamento e ajustar o peso do balanceamento
            if max_usage > self.load_balancing_scheduler['severe_imbalance_threshold']:
                self.severe_imbalance_count += 1
                # Aumentar peso do balanceamento temporariamente
                balance_weight = 2.0

                # Log da situação
                if self.severe_imbalance_count % 10 == 0:
                    self.console.print(f"[bold red]⚠️ Severo desbalanceamento detectado! Especialista dominante: {max_usage:.2%}[/bold red]")
            else:
                # Peso normal
                balance_weight = 1.0
                self.severe_imbalance_count = max(0, self.severe_imbalance_count - 1)
        else:
            balance_weight = 1.0

        # Loss total com peso adaptativo
        total_loss = q_loss + (load_balancing_loss * balance_weight)

        # Aplicar gradiente apenas para os experts usados na forward pass atual
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clipar gradientes para estabilidade
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        # Aplicar otimização
        self.optimizer.step()

        # Rastrear e ajustar especialistas dominantes periodicamente
        if self.curr_step % self.load_balancing_scheduler['check_interval'] == 0:
            self.check_expert_dominance()
        
        return total_loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def check_expert_dominance(self):
        """Monitora e gerencia especialistas dominantes ou inativos"""
        self.balance_checks += 1

        # Obter estatísticas de uso
        stats = self.net.moe_layer.get_expert_usage_stats()
        if not stats or 'usage' not in stats:
            return

        usage = stats['usage']
        inactive_experts = stats.get('inactive_experts', [])
        dominant_experts = stats.get('dominant_experts', [])

        # Rastrear especialistas dominantes
        if dominant_experts:
            self.dominant_experts_history.append(set(dominant_experts))
            if len(self.dominant_experts_history) > 10:
                self.dominant_experts_history.pop(0)

        # Verificar se há um padrão consistente de dominância
        consistent_dominants = set()
        if len(self.dominant_experts_history) >= 3:
            # Encontrar especialistas que dominam consistentemente
            for expert in range(self.net.moe_layer.num_experts):
                if all(expert in history for history in self.dominant_experts_history[-3:]):
                    consistent_dominants.add(expert)

        # Medidas corretivas para balanceamento
        if consistent_dominants or len(inactive_experts) > self.net.moe_layer.num_experts // 3:
            self.balance_adjustments += 1

            # Ajustar coeficiente de balanceamento
            current_coef = self.net.moe_layer.load_balancing_loss_coef
            new_coef = min(
                current_coef * self.load_balancing_scheduler['adjustment_factor'],
                self.load_balancing_scheduler['max_coef']
            )
            self.net.moe_layer.load_balancing_loss_coef = new_coef

            # Log das ações tomadas
            if consistent_dominants:
                expert_ids = ", ".join([str(e+1) for e in consistent_dominants])
                self.console.print(f"[yellow]⚠️ Experts {expert_ids} consistentemente dominantes. Ajustando balanceamento: {current_coef:.4f} → {new_coef:.4f}[/yellow]")

            if len(inactive_experts) > self.net.moe_layer.num_experts // 3:
                inactive_count = len(inactive_experts)
                self.console.print(f"[yellow]⚠️ {inactive_count} experts inativos detectados. Ajustando balanceamento.[/yellow]")

            # Ação mais drástica: Reinicializar pesos do gate quando há desbalanceamento severo
            if max(usage) > 0.4 and self.balance_adjustments % 3 == 0:
                self.reinitialize_gate_network()
                self.console.print("[bold red]🔄 Reinicializando rede de gate para melhorar balanceamento![/bold red]")

        # Se o balanceamento estiver bom, reduzir gradualmente o coeficiente
        elif max(usage) < 0.2 and len(inactive_experts) < 2:
            current_coef = self.net.moe_layer.load_balancing_loss_coef
            # Só reduzir se o coeficiente atual for alto
            if current_coef > self.load_balancing_scheduler['base_coef']:
                new_coef = max(
                    current_coef * self.load_balancing_scheduler['decay_factor'],
                    self.load_balancing_scheduler['min_coef']
                )
                self.net.moe_layer.load_balancing_loss_coef = new_coef
                self.console.print(f"[green]✓ Balanceamento estável. Ajustando coeficiente: {current_coef:.4f} → {new_coef:.4f}[/green]")

    def reinitialize_gate_network(self):
        """Reinicializa a rede de gate quando há desbalanceamento severo"""
        # Salvar estado atual do gating para referência
        old_gate_probs = self.net.moe_layer.last_gate_probs.detach() if self.net.moe_layer.last_gate_probs is not None else None

        # Reinicializar pesos do gate
        self.net.moe_layer.gate._initialize_weights()

        # Aumentar escala de ruído temporariamente
        if hasattr(self.net.moe_layer.gate, 'noise_scale'):
            self.net.moe_layer.gate.noise_scale = min(self.net.moe_layer.gate.noise_scale * 2, 0.5)
            self.console.print(f"[yellow]🔊 Ruído aumentado para {self.net.moe_layer.gate.noise_scale:.3f}[/yellow]")

    def calculate_reward(self, reward, done, info):
        progress_reward = 0
        life_reward = 0
        coin_reward = 0
        score_reward = 0
        time_penalty = -0.01  # Reduzido para -0.01

        # Recompensa baseada no progresso
        if self.last_position is not None:
            progress = (info["x_pos"] - self.last_position)/10

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
                life_reward = 10  # Reduzido para 10
            elif life_change < 0:
                life_reward = -5  # Reduzido para -5
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

        # Ajustar coeficiente de load balancing dinamicamente a cada 100 steps
        if self.curr_step % 100 == 0:
            self.adjust_load_balancing_coefficient()

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
        try:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        except:
            return
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint["exploration_rate"]
        print(f"Checkpoint carregado de {self.checkpoint_path}")

    def print_moe_stats(self, episode):
        """Imprime estatísticas detalhadas e avançadas do MoE"""
        moe_metrics = self.net.get_moe_metrics()
        if not moe_metrics:
            self.console.print("[yellow]Sem métricas MoE disponíveis ainda[/yellow]")
            return

        # Obter estatísticas detalhadas
        moe_layer = self.net.moe_layer
        stats = moe_layer.get_expert_usage_stats()

        # Cabeçalho com informações básicas
        self.console.print("\n[bold cyan]🧠 Estatísticas do Mixture of Experts (MoE)[/bold cyan]")
        self.console.print(f"[yellow]Episódio:[/yellow] {episode}")
        self.console.print(f"[yellow]Passo:[/yellow] {self.curr_step}")

        # Status geral de balanceamento
        if stats['max_usage'] > 0.4:
            status = "[bold red]SEVERO DESBALANCEAMENTO[/bold red]"
        elif stats['max_usage'] > 0.25:
            status = "[yellow]Desbalanceamento Moderado[/yellow]"
        elif stats['max_usage'] > 0.15:
            status = "[green]Balanceamento Razoável[/green]"
        else:
            status = "[bold green]Excelente Balanceamento[/bold green]"

        self.console.print(f"[bold]Status de Balanceamento:[/bold] {status}")

        # Métricas de configuração
        self.console.print("\n[bold cyan]⚙️ Configuração Atual:[/bold cyan]")
        self.console.print(f"[yellow]Número de Experts:[/yellow] {moe_layer.num_experts}")
        self.console.print(f"[yellow]Top-K Selecionados:[/yellow] {moe_layer.top_k}")
        self.console.print(f"[yellow]Coeficiente de Balanceamento:[/yellow] {moe_layer.load_balancing_loss_coef:.5f}")
        self.console.print(f"[yellow]Escala de Ruído:[/yellow] {moe_layer.gate.noise_scale if hasattr(moe_layer.gate, 'noise_scale') else 0.1:.5f}")

        # Métricas principais de desempenho
        self.console.print("\n[bold cyan]📊 Métricas Principais:[/bold cyan]")
        self.console.print(f"[yellow]Loss de Balanceamento:[/yellow] {moe_metrics['load_balancing_loss']:.6f}")
        self.console.print(f"[yellow]Entropia dos Experts:[/yellow] {moe_metrics['expert_entropy']:.4f}")
        self.console.print(f"[yellow]Uso Máximo de Expert:[/yellow] {stats['max_usage']:.3f} (Expert {np.argmax(stats['usage'])+1})")
        self.console.print(f"[yellow]Uso Mínimo de Expert:[/yellow] {stats['min_usage']:.3f} (Expert {np.argmin(stats['usage'])+1})")

        # Métricas avançadas
        self.console.print("\n[bold cyan]🔍 Métricas Avançadas:[/bold cyan]")
        self.console.print(f"[yellow]Coeficiente de Variação:[/yellow] {stats['coefficient_of_variation']:.3f}")
        if 'gini_coefficient' in stats:
            gini = stats['gini_coefficient']
            gini_status = "[red]Alta Desigualdade[/red]" if gini > 0.4 else "[yellow]Desigualdade Média[/yellow]" if gini > 0.2 else "[green]Baixa Desigualdade[/green]"
            self.console.print(f"[yellow]Coeficiente de Gini:[/yellow] {gini:.3f} {gini_status}")
        if 'normalized_entropy' in stats:
            norm_entropy = stats['normalized_entropy']
            entropy_status = "[green]Excelente[/green]" if norm_entropy > 0.9 else "[yellow]Média[/yellow]" if norm_entropy > 0.7 else "[red]Baixa[/red]"
            self.console.print(f"[yellow]Entropia Normalizada:[/yellow] {norm_entropy:.3f} {entropy_status}")

        # Contagens de especialistas
        inactive_count = stats.get('inactive_count', sum(1 for u in stats['usage'] if u < 0.01))
        dominant_count = stats.get('dominant_count', sum(1 for u in stats['usage'] if u > 0.2))
        self.console.print(f"[yellow]Experts Inativos (<1%):[/yellow] {inactive_count} de {moe_layer.num_experts}")
        self.console.print(f"[yellow]Experts Dominantes (>20%):[/yellow] {dominant_count} de {moe_layer.num_experts}")

        # Histórico de ajustes
        self.console.print("\n[bold cyan]🔧 Histórico de Ajustes:[/bold cyan]")
        self.console.print(f"[yellow]Verificações de Balanceamento:[/yellow] {self.balance_checks}")
        self.console.print(f"[yellow]Ajustes Realizados:[/yellow] {self.balance_adjustments}")
        self.console.print(f"[yellow]Episódios com Desbalanceamento Severo:[/yellow] {self.severe_imbalance_count}")

        # Distribuição de uso dos especialistas
        self.console.print("\n[bold cyan]📈 Distribuição de Uso dos Experts:[/bold cyan]")
        expert_usage = stats['usage']
        ideal_usage = 1.0 / moe_layer.num_experts

        # Criar tabela para melhor visualização
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Expert", style="cyan", no_wrap=True)
        table.add_column("Uso (%)", justify="right")
        table.add_column("Visualização", no_wrap=True)
        table.add_column("Status", no_wrap=True)

        # Ordenar por uso
        sorted_indices = np.argsort(expert_usage)[::-1]  # Ordem decrescente

        for i, idx in enumerate(sorted_indices):
            usage = expert_usage[idx]
            bar_length = int(usage * 50)  # Barra de 50 caracteres
            bar = "█" * bar_length + "░" * (50 - bar_length)

            # Definir cores baseadas no desvio do ideal
            ratio = usage / ideal_usage
            if ratio > 3.0:
                color = "[bold red]"
                status = "🔥 Superutilizado"
            elif ratio > 2.0:
                color = "[red]"
                status = "⚠️ Muito utilizado"
            elif ratio > 1.5:
                color = "[yellow]"
                status = "⚡ Acima da média"
            elif ratio < 0.2:
                color = "[dim]"
                status = "💤 Inativo"
            elif ratio < 0.5:
                color = "[blue]"
                status = "❄️ Subutilizado"
            else:
                color = "[green]"
                status = "✅ Balanceado"

            table.add_row(
                f"{idx+1}", 
                f"{usage*100:.1f}%",
                f"{color}[{bar}][/{color.strip('[]')}",
                f"{color}{status}[/{color.strip('[]')}"
            )

        self.console.print(table)

        # Análise de ações para melhorar balanceamento
        self.console.print("\n[bold cyan]💡 Análise e Recomendações:[/bold cyan]")

        if stats['max_usage'] > 0.35:
            self.console.print("[yellow]🔸 Desbalanceamento significativo detectado[/yellow]")
            self.console.print("[yellow]🔹 Ações automáticas implementadas:[/yellow]")
            self.console.print(f"  - Coeficiente de balanceamento atual: {moe_layer.load_balancing_loss_coef:.4f}")
            self.console.print(f"  - Escala de ruído atual: {moe_layer.gate.noise_scale if hasattr(moe_layer.gate, 'noise_scale') else 0.1:.4f}")

            # Recomendações específicas
            self.console.print("[yellow]🔹 Medidas recomendadas:[/yellow]")
            if inactive_count > 3:
                self.console.print("  - [red]Muitos experts inativos - reduzir número de experts ou aumentar top-k[/red]")
            if stats['coefficient_of_variation'] > 1.0:
                self.console.print("  - [red]Alta variabilidade - aumentar coeficiente de balanceamento[/red]")
            if dominant_count > 0:
                self.console.print("  - [red]Experts dominantes - aumentar ruído ou reinicializar rede de gate[/red]")
        elif stats['max_usage'] < 0.15 and inactive_count < 2:
            self.console.print("[green]✅ Balanceamento atual é bom![/green]")
            if moe_layer.load_balancing_loss_coef > 0.03:
                self.console.print("  - Pode-se reduzir gradualmente o coeficiente de balanceamento")
        else:
            self.console.print("[yellow]🔸 Balanceamento razoável - monitorando ajustes automáticos[/yellow]")

        print()

    def adjust_load_balancing_coefficient(self):
        """Ajusta dinamicamente o coeficiente de load balancing baseado no desbalanceamento"""
        moe_layer = self.net.moe_layer
        stats = moe_layer.get_expert_usage_stats()
        
        max_usage = stats['max_usage']
        current_coef = moe_layer.load_balancing_loss_coef
        
        # Se há desbalanceamento severo, aumenta o coeficiente
        if max_usage > self.load_balancing_scheduler['imbalance_threshold']:
            new_coef = min(
                current_coef * self.load_balancing_scheduler['adjustment_factor'],
                self.load_balancing_scheduler['max_coef']
            )
            moe_layer.load_balancing_loss_coef = new_coef
            self.console.print(f"[yellow]🔧 Increased load balancing coef to {new_coef:.4f}[/yellow]")
        
        # Se está bem balanceado, diminui gradualmente o coeficiente
        elif max_usage < 0.25 and current_coef > self.load_balancing_scheduler['base_coef']:
            new_coef = max(
                current_coef * self.load_balancing_scheduler['decay_factor'],
                self.load_balancing_scheduler['base_coef']
            )
            moe_layer.load_balancing_loss_coef = new_coef
            self.console.print(f"[green]🔧 Decreased load balancing coef to {new_coef:.4f}[/green]")
            
        # Logar coeficiente atual para monitoramento
        self.console.log(f"Current load balancing coef: {current_coef:.4f}")
        
        # Garantir que o coeficiente não fique abaixo de um limite mínimo
        min_coef = 0.01
        if moe_layer.load_balancing_loss_coef < min_coef:
            moe_layer.load_balancing_loss_coef = min_coef
            self.console.print(f"[red]⚠️ Load balancing coef adjusted to minimum value: {min_coef}[/red]")


# Adicionando as classes MoE que estão faltando
class ExpertNetwork(nn.Module):
    """Rede especialista melhorada com arquitetura mais diversificada"""
    def __init__(self, input_dim, hidden_dim, output_dim, expert_id=None):
        super().__init__()
        self.expert_id = expert_id  # Identificador do especialista
        self.hidden_dim = hidden_dim

        # Arquitetura base com skip connections
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Camadas variadas por especialista para forçar especialização
        # Usamos módulos diferentes para quebrar simetria entre experts
        self.hidden1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Segunda camada com arquitetura diferente
        self.hidden2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Usando SiLU (Swish) como ativação variada
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim//2)
        )

        # Camada de saída
        self.output_layer = nn.Linear(hidden_dim//2, output_dim)

        # Inicialização diversificada para cada especialista
        self._initialize_weights()

        # Inserir alguma não-linearidade específica por especialista
        if expert_id is not None:
            # Para quebrar simetria, adicione pequeno bias específico por expert
            with torch.no_grad():
                # Pequena perturbação determinística baseada no expert_id
                seed_val = (expert_id * 1337) % 10000
                torch.manual_seed(seed_val)
                # Adicionar bias pequeno mas diferente para cada expert
                self.input_layer.bias.add_(torch.randn_like(self.input_layer.bias) * 0.01)
                self.output_layer.bias.add_(torch.randn_like(self.output_layer.bias) * 0.01)

    def _initialize_weights(self):
        """Inicialização especializada para cada camada"""
        # Input layer - inicialização normal
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.constant_(self.input_layer.bias, 0.1)

        # Output layer - inicialização cuidadosa
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

        # Camadas escondidas
        for module in [self.hidden1, self.hidden2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        # Processamento com residual connections
        h0 = F.relu(self.input_layer(x))
        h1 = self.hidden1(h0) + h0  # Residual connection
        h2 = self.hidden2(h1)  # Sem residual aqui
        output = self.output_layer(h2)
        return output

class GatingNetwork(nn.Module):
    """Rede de gating melhorada com mecanismos para promover balanceamento"""
    def __init__(self, input_dim, num_experts, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

        # Rede de gating mais expressiva com skip-connections
        self.input_layer = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.hidden2 = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.output_layer = nn.Linear(128, num_experts)

        # Camada de temperatura para controlar a suavidade da distribuição
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # Inicialização de pesos para distribuição inicial mais uniforme
        self._initialize_weights()

        # Histórico de ativação para monitoramento
        self.activation_history = []
        self.noise_scale = 0.15  # Escala de ruído inicial
        self.noise_decay = 0.9995  # Decaimento do ruído

    def _initialize_weights(self):
        """Inicialização especializada para distribuição mais uniforme"""
        # Input layer - inicialização cuidadosa para evitar saturação
        nn.init.xavier_uniform_(self.input_layer.weight, gain=0.5)
        nn.init.constant_(self.input_layer.bias, 0.0)

        # Output layer - inicialização especial para começar quase uniforme
        nn.init.constant_(self.output_layer.weight, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

        # Camadas escondidas - inicialização normal
        for module in [self.hidden1, self.hidden2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        # Processamento com skip connections
        h0 = F.relu(self.input_layer(x))
        h1 = self.hidden1(h0) + h0  # Residual connection
        h2 = self.hidden2(h1)  # No residual aqui

        # Logits dos gates com temperatura adaptativa
        gate_logits = self.output_layer(h2)

        # Guardar ativações para análise
        if self.training:
            self.activation_history.append(gate_logits.detach().cpu())
            if len(self.activation_history) > 100:
                self.activation_history.pop(0)

        # Adicionar ruído para diversidade (apenas durante treinamento)
        # Ruído proporcional aos valores para evitar distorção total
        if self.training:
            # Decair o ruído ao longo do tempo
            self.noise_scale = max(self.noise_scale * self.noise_decay, 0.05)

            # Ruído gaussiano padrão
            standard_noise = torch.randn_like(gate_logits) * self.noise_scale

            # Ruído adicional para especialistas subutilizados
            if len(self.activation_history) > 10:
                # Calcular uso médio recente
                recent_activations = torch.cat([act.mean(dim=0, keepdim=True) for act in self.activation_history[-10:]], dim=0)
                recent_probs = F.softmax(recent_activations, dim=1).mean(dim=0)

                # Identificar especialistas menos usados
                boost_factor = 1.0 / (recent_probs + 1e-5)  # Inverse probability
                boost_factor = boost_factor / boost_factor.mean()  # Normalizar

                # Aplicar boost aos especialistas menos utilizados
                boost_noise = torch.randn_like(gate_logits) * boost_factor.unsqueeze(0) * self.noise_scale * 0.5

                # Combinar ruídos
                gate_logits = gate_logits + standard_noise + boost_noise
            else:
                gate_logits = gate_logits + standard_noise

        # Aplicar temperatura para controlar a suavidade da distribuição
        # Temperatura menor = distribuição mais acentuada
        # Temperatura maior = distribuição mais suave
        temperature = torch.clamp(self.temperature, 0.5, 5.0)
        gate_logits = gate_logits / temperature

        # Aplicar softmax para obter probabilidades
        gate_probs = F.softmax(gate_logits, dim=1)

        # Selecionar top-k especialistas
        top_k_probs, top_k_indices = torch.topk(gate_probs, min(self.top_k, self.num_experts), dim=1)
        
        # Renormalizar as probabilidades top-k
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        
        return top_k_probs, top_k_indices, gate_probs

class MoELayer(nn.Module):
    """Camada Mixture of Experts com balanceamento melhorado"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=12, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Criar os especialistas com inicialização diversificada
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])

        # Rede de gating melhorada
        self.gate = GatingNetwork(input_dim, num_experts, top_k)

        # Coeficiente de balanceamento mais alto para forçar melhor distribuição
        self.load_balancing_loss_coef = 0.05

        # Métricas para monitoramento
        self.last_gate_probs = None
        self.expert_usage_history = []

        # Variável para controlar noise dinâmico
        self.noise_scale = 0.2
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Adicionar ruído aos inputs para forçar especialização
        if self.training:
            # Aplicar noise dependendo do nível de desbalanceamento
            stats = self.get_expert_usage_stats()
            if stats['coefficient_of_variation'] > 0.8:  # Desbalanceamento alto
                self.noise_scale = min(self.noise_scale * 1.05, 0.3)  # Aumenta noise gradualmente
            else:
                self.noise_scale = max(self.noise_scale * 0.98, 0.05)  # Reduz noise gradualmente

            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        # Obter decisões do gate com tokens de routing
        top_k_probs, top_k_indices, gate_probs = self.gate(x)

        # Armazenar para monitoramento
        self.last_gate_probs = gate_probs.detach()

        # Calcular saídas dos especialistas selecionados com processamento em paralelo
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Processar expert por expert (mais eficiente em GPU)
        for expert_idx in range(self.num_experts):
            # Encontrar quais amostras usam este especialista
            batch_indices = []
            for i in range(batch_size):
                if expert_idx in top_k_indices[i]:
                    # Encontrar posição do especialista no top_k
                    pos = (top_k_indices[i] == expert_idx).nonzero(as_tuple=True)[0]
                    batch_indices.append((i, pos.item()))

            if batch_indices:  # Se este especialista está sendo usado
                # Extrair índices e pesos
                indices = [b[0] for b in batch_indices]
                positions = [b[1] for b in batch_indices]

                # Processar entrada apenas para este especialista
                expert_input = x[indices]
                expert_output = self.experts[expert_idx](expert_input)

                # Aplicar pesos e adicionar ao resultado final
                for idx, pos, out in zip(indices, positions, expert_output):
                    weight = top_k_probs[idx, pos]
                    final_output[idx] += weight * out

        # Calcular loss de balanceamento com fator adaptativo
        load_balancing_loss = self._calculate_load_balancing_loss(gate_probs)

        # Adicionar regularização para os especialistas menos utilizados
        expert_usage = gate_probs.mean(dim=0)  # [num_experts]
        if self.training and torch.max(expert_usage) > 0.3:
            # Identificar especialistas menos usados
            underused = expert_usage < 0.05
            if underused.any():
                # Adicionar penalty por especialistas subutilizados
                underused_penalty = (0.05 - expert_usage[underused]).sum() * 0.5
                load_balancing_loss = load_balancing_loss + underused_penalty
        
        return final_output, load_balancing_loss, gate_probs
    
    def _calculate_load_balancing_loss(self, gate_probs):
        """Calcular loss de balanceamento com múltiplas estratégias"""
        # Frequência de uso de cada especialista
        expert_usage = gate_probs.mean(dim=0)  # [num_experts]

        # Cálculo de estatísticas para análise adaptativa
        max_usage = expert_usage.max().item()
        min_usage = expert_usage.min().item()
        std_usage = expert_usage.std().item()
        ideal_usage = 1.0 / self.num_experts

        # 1. Loss de divergência KL da distribuição uniforme (mais forte)
        uniform_prob = torch.full_like(expert_usage, ideal_usage)
        kl_loss = F.kl_div(
            torch.log(expert_usage + 1e-8),
            uniform_prob,
            reduction='sum'
        )

        # 2. Loss de variância quadrática (penaliza especialistas muito usados)
        variance_loss = ((expert_usage - ideal_usage).pow(2)).mean()

        # 3. Loss de entropia (encoraja diversidade)
        entropy = -(expert_usage * torch.log(expert_usage + 1e-8)).sum()
        max_entropy = -math.log(ideal_usage) * self.num_experts
        entropy_loss = max_entropy - entropy

        # 4. Loss de desigualdade (penaliza diferenças extremas entre especialistas)
        inequality_loss = (max_usage - min_usage) * 2.0

        # 5. Loss de quantil (penaliza o uso excessivo dos top experts)
        top_usage, _ = torch.topk(expert_usage, k=self.num_experts//3)
        quantile_loss = top_usage.mean() * 2.0  # Penaliza uso excessivo do top terço

        # Pesos adaptativos baseados no nível de desbalanceamento
        # Ajuste pesos para priorizar diferentes tipos de loss dependendo do estado atual
        if max_usage > 0.3:  # Desbalanceamento severo
            # Priorizar equalização e desigualdade
            weights = [0.3, 0.25, 0.1, 0.25, 0.1]  # kl, var, entropy, inequality, quantile
        elif std_usage > 0.05:  # Desbalanceamento moderado
            # Balancear todos os tipos
            weights = [0.25, 0.2, 0.2, 0.2, 0.15]
        else:  # Balanceamento razoável
            # Manter balanceamento estável
            weights = [0.2, 0.15, 0.3, 0.15, 0.2]

        # Combinar os diferentes tipos de loss com pesos adaptativos
        total_loss = (
            weights[0] * kl_loss + 
            weights[1] * variance_loss + 
            weights[2] * entropy_loss +
            weights[3] * inequality_loss +
            weights[4] * quantile_loss
        )

        # Registrar métricas para monitoramento
        self.expert_usage_history.append(expert_usage.detach().cpu().numpy())
        if len(self.expert_usage_history) > 100:  # Manter apenas histórico recente
            self.expert_usage_history.pop(0)

        # Fator adaptativo baseado no desbalanceamento
        adaptive_factor = 1.0
        if max_usage > 0.4:  # Desbalanceamento extremo
            adaptive_factor = 2.0
        elif max_usage > 0.25:  # Desbalanceamento significativo
            adaptive_factor = 1.5

        return self.load_balancing_loss_coef * total_loss * adaptive_factor
    
    def get_expert_usage_stats(self):
        """Retorna estatísticas detalhadas sobre o uso dos especialistas com métricas avançadas"""
        if self.last_gate_probs is not None:
            usage = self.last_gate_probs.mean(dim=0).cpu().numpy()

            # Evitar divisão por zero
            if np.any(np.isnan(usage)) or np.any(usage <= 0):
                usage = np.full(self.num_experts, 1.0/self.num_experts)

            # Calcular estatísticas históricas se disponível
            historical_stats = {}
            if len(self.expert_usage_history) > 5:  # Se temos histórico suficiente
                historical_usage = np.stack(self.expert_usage_history[-10:], axis=0).mean(axis=0)
                historical_stats = {
                    'historical_usage': historical_usage,
                    'historical_max': float(np.max(historical_usage)),
                    'historical_min': float(np.min(historical_usage))
                }

            # Detectar especialistas inativos (< 1% de uso)
            inactive_count = np.sum(usage < 0.01)
            inactive_indices = np.where(usage < 0.01)[0].tolist()

            # Detectar especialistas dominantes (> 20% de uso)
            dominant_count = np.sum(usage > 0.2)
            dominant_indices = np.where(usage > 0.2)[0].tolist()

            # Índices de especialistas ordenados por uso
            sorted_indices = np.argsort(usage)[::-1].tolist()

            # Calcular índice de Gini para medir desigualdade
            sorted_usage = np.sort(usage)
            cumulative_usage = np.cumsum(sorted_usage)
            gini = 1 - 2 * np.sum(cumulative_usage) / (len(usage) * cumulative_usage[-1])

            # Métricas de balanceamento
            entropy = -np.sum(usage * np.log(usage + 1e-8))
            max_entropy = -np.log(1.0/self.num_experts) * self.num_experts
            normalized_entropy = entropy / max_entropy

            # Quartis de uso
            quartiles = np.percentile(usage, [25, 50, 75])

            stats = {
                'usage': usage,
                'max_usage': float(np.max(usage)),
                'min_usage': float(np.min(usage)),
                'median_usage': float(np.median(usage)),
                'std_usage': float(np.std(usage)),
                'coefficient_of_variation': float(np.std(usage) / (np.mean(usage) + 1e-8)),
                'inactive_count': int(inactive_count),
                'inactive_experts': inactive_indices,
                'dominant_count': int(dominant_count),
                'dominant_experts': dominant_indices,
                'top_experts': sorted_indices[:3],  # Top 3 experts
                'bottom_experts': sorted_indices[-3:],  # Bottom 3 experts
                'gini_coefficient': float(gini),  # Índice de desigualdade
                'normalized_entropy': float(normalized_entropy),  # 1.0 é perfeito
                'quartiles': quartiles.tolist(),  # 25%, 50%, 75%
                'noise_scale': self.noise_scale  # Nível atual de ruído
            }

            # Adicionar estatísticas históricas se disponíveis
            if historical_stats:
                stats.update(historical_stats)

            return stats

        # Valores padrão para o caso de não termos dados ainda
        default_usage = np.full(self.num_experts, 1.0/self.num_experts)
        return {
            'usage': default_usage,
            'max_usage': 1.0/self.num_experts,
            'min_usage': 1.0/self.num_experts,
            'std_usage': 0.0,
            'coefficient_of_variation': 0.0,
            'inactive_count': 0,
            'inactive_experts': [],
            'dominant_count': 0,
            'dominant_experts': [],
            'gini_coefficient': 0.0,
            'normalized_entropy': 1.0,
            'noise_scale': self.noise_scale
        }

# Apply Wrappers to environment
env = SkipFrame(env, skip=2)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=64)
env = FrameStack(env, num_stack=4)

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

mario = Mario(state_dim=(4, 64, 64), action_dim=9, save_dir=save_dir)

logger = MetricLogger()

episodes = 500

# Start live display for Rich table
logger.start_live_display()

try:
    for e in range(episodes):
        state = env.reset()
        mario.last_position = None  # Reseta a última posição no início de cada episódio
        mario.score_inicial = 0
        mario.coins_inicial = 0
        mario.vida_inicial = 2
        mario.max_pos = 0
        
        # Verificar balanceamento no início do episódio a cada 5 episódios
        if e > 0 and e % 5 == 0:
            moe_metrics = mario.net.get_moe_metrics()
            if moe_metrics and 'expert_usage' in moe_metrics:
                max_usage = max(moe_metrics['expert_usage'])
                # Verificação de balanceamento no início do episódio
                if max_usage > 0.3:
                    mario.console.print(f"[bold yellow]⚠️ Iniciando episódio {e} com desbalanceamento (max: {max_usage:.2f})[/bold yellow]")
                    mario.check_expert_dominance()  # Forçar verificação de dominância

                    # Se desbalanceamento for muito grave, reinicializar gate
                    if max_usage > 0.4 and e % 20 == 0:
                        mario.reinitialize_gate_network()

        # Play the game!
        episode_steps = 0
        while True:
            episode_steps += 1

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done, info)

            # Learn
            q, loss = mario.learn()

            # Logging
            moe_metrics = mario.net.get_moe_metrics()
            logger.log_step(reward, loss, q, moe_metrics)

            # Verificar balanceamento periodicamente durante episódios longos
            if episode_steps > 0 and episode_steps % 200 == 0 and mario.curr_step > 1000:
                if moe_metrics and 'expert_usage' in moe_metrics:
                    max_usage = max(moe_metrics['expert_usage'])
                    inactive_count = sum(1 for usage in moe_metrics['expert_usage'] if usage < 0.01)

                    # Log para o console
                    mario.console.print(f"[cyan]📊 Ep {e}, Step {episode_steps}: max usage={max_usage:.2f}, {inactive_count} experts inativos[/cyan]")

                    # Verificação de balanceamento durante o episódio
                    if max_usage > 0.35 or inactive_count > mario.net.moe_layer.num_experts // 3:
                        mario.check_expert_dominance()

            # Atualizar tabela a cada 50 steps para ver progresso em tempo real
            if mario.curr_step % 50 == 0:
                logger.update_live_display(e, moe_metrics, mario.net, mario.exploration_rate, mario.curr_step)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        # Coletar métricas MoE finais do episódio
        moe_metrics = mario.net.get_moe_metrics()

        # Atualizar display live com informações finais do episódio ANTES de resetar
        logger.update_live_display(e, moe_metrics, mario.net, mario.exploration_rate, mario.curr_step)

        # Imprimir estatísticas detalhadas do MoE a cada 10 episódios
        if e % 10 == 0:
            mario.print_moe_stats(e)

        # Pequena pausa para garantir que a atualização seja visível
        import time
        time.sleep(0.1)

        # Marcar fim do episódio (isso reseta as métricas do episódio atual)
        logger.log_episode()

        # Forçar uma segunda atualização do display após log_episode para mostrar o episódio completado
        logger.update_live_display(e+1, moe_metrics, mario.net, mario.exploration_rate, mario.curr_step)

        # Salvar a cada 20 episódios ou quando houver progresso significativo
        if e % 20 == 0 or (len(logger.ep_rewards) > 0 and logger.ep_rewards[-1] > max(logger.ep_rewards[:-1], default=0)):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step, moe_metrics=moe_metrics, mario_net=mario.net)
            mario.save()
            mario.console.print(f"[bold green]💾 Progresso salvo no episódio {e}[/bold green]")

finally:
    # Stop live display when training finishes
    logger.stop_live_display()


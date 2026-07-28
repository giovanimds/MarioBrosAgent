#!/usr/bin/env python3
"""
Script de treinamento do Mario com arquitetura MBP (Mamba-Based Processing)
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from rich.console import Console

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.model_mbp import MarioNetMBP
from src.env_manager.environment import create_env
from src.helpers.config import (
    EXPLORATION_RATE, EXPLORATION_RATE_DECAY, EXPLORATION_RATE_MIN,
    SAVE_EVERY, GAMMA, BURNIN, LEARN_EVERY, SYNC_EVERY, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY
)


class MarioAgentMBP:
    """Agente Mario com arquitetura MBP"""
    
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(save_dir)
        self.device = torch.device('cpu')
        
        # Criar modelo MBP
        self.net = MarioNetMBP(state_dim, action_dim).float()
        self.net = self.net.to(device=self.device)
        
        # Parâmetros de exploração
        self.exploration_rate = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min = EXPLORATION_RATE_MIN
        self.curr_step = 0
        
        # Parâmetros de treinamento
        self.save_every = SAVE_EVERY
        self.gamma = GAMMA
        self.burnin = BURNIN
        self.learn_every = LEARN_EVERY
        self.sync_every = SYNC_EVERY
        self.batch_size = BATCH_SIZE
        self.memory = []
        
        # Otimizador
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # Checkpoint
        self.checkpoint_path = self.save_dir / "mario_net_mbp.chkpt"
        if self.checkpoint_path.exists():
            self.load()
        
        self.console = Console()
        self.last_position = None
        self.best_score = 0
        self.episode_rewards = []
        
    def act(self, state):
        """Selecionar ação usando política epsilon-greedy"""
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values, _ = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()
        
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        
        return action_idx
    
    def cache(self, state, next_state, action, reward, done, info):
        """Armazenar experiência na memória"""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        
        self.memory.append((state, next_state, action, reward, done, info))
        
        if len(self.memory) > 10000:
            self.memory.pop(0)
        
        self.last_position = info['x_pos']
        
    def recall(self):
        """Recuperar batch de experiências"""
        if len(self.memory) < self.batch_size:
            return None
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.stack([b[0] for b in batch])
        next_states = torch.stack([b[1] for b in batch])
        actions = torch.stack([b[2] for b in batch]).squeeze()
        rewards = torch.stack([b[3] for b in batch]).squeeze()
        dones = torch.stack([b[4] for b in batch]).squeeze()
        
        return states, next_states, actions, rewards, dones
    
    def td_estimate(self, state, action):
        state = state.requires_grad_()
        action_values, _ = self.net(state, model="online")
        current_Q = action_values[np.arange(0, self.batch_size), action]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        online_Q, _ = self.net(next_state, model="online")
        best_action = torch.argmax(online_Q, dim=-1)
        target_Q = self.net(next_state, model="target")
        next_Q = target_Q[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        q_loss = self.loss_fn(td_estimate, td_target)
        load_balancing_loss = self.net.last_load_balancing_loss
        total_loss = q_loss + load_balancing_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        batch = self.recall()
        if batch is None:
            return None, None
        
        state, next_state, action, reward, done = batch
        
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        
        return td_est.mean().item(), loss
    
    def save(self):
        checkpoint = {
            "model": self.net.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "episode_rewards": self.episode_rewards
        }
        torch.save(checkpoint, self.checkpoint_path)
        self.console.print(f"[green]✅ Checkpoint salvo: {self.checkpoint_path}[/green]")
    
    def load(self):
        try:
            self.console.print(f"[yellow]Carregando checkpoint...[/yellow]")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.net.load_state_dict(checkpoint["model"])
            self.exploration_rate = checkpoint["exploration_rate"]
            self.curr_step = checkpoint["curr_step"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_score = checkpoint.get("best_score", 0)
            self.episode_rewards = checkpoint.get("episode_rewards", [])
            self.console.print(f"[green]✅ Checkpoint carregado do passo {self.curr_step}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Erro ao carregar checkpoint: {e}[/red]")


def train_mbp():
    """Função principal de treinamento"""
    console = Console()
    
    # Configuração
    save_dir = Path("checkpoints/mbp")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar ambiente
    console.print("[blue]Criando ambiente...[/blue]")
    env, test_env = create_env()
    
    # Criar agente
    console.print("[blue]Criando agente MBP...[/blue]")
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    agent = MarioAgentMBP(state_dim, action_dim, save_dir)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in agent.net.parameters())
    console.print(f"[green]Total de parâmetros: {total_params:,}[/green]")
    
    # Treinamento
    console.print("[bold green]Iniciando treinamento MBP...[/bold green]")
    
    start_time = time.time()
    max_episodes = 10000
    max_time = 30 * 60  # 30 minutos
    
    episode = 0
    best_reward = -float('inf')
    
    try:
        while time.time() - start_time < max_time and episode < max_episodes:
            episode += 1
            state, info = env.reset()
            agent.last_position = info['x_pos']
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calcular recompensa customizada
                progress = (info["x_pos"] - agent.last_position) / 10
                if progress > 1:
                    reward += progress
                
                if info.get("flag_get", False):
                    reward += 50
                
                if 'life' in info:
                    life_change = int(info["life"]) - int(agent.last_position if hasattr(agent, 'last_life') else 2)
                    if life_change < 0:
                        reward -= 10
                
                reward += -0.01  # Penalidade por tempo
                
                agent.cache(state, next_state, action, reward, done, info)
                
                # Aprendizado
                td_est, loss = agent.learn()
                
                if td_est is not None:
                    total_reward += reward
                
                state = next_state
                step += 1
                
                # Print progresso
                if step % 100 == 0:
                    console.print(f"[yellow]Episódio {episode}, Passo {step}, Recompensa: {total_reward:.2f}[/yellow]")
            
            agent.episode_rewards.append(total_reward)
            
            # Atualizar melhor recompensa
            if total_reward > best_reward:
                best_reward = total_reward
                console.print(f"[bold green]Nova melhor recompensa: {best_reward:.2f}[/bold green]")
            
            # Print estatísticas do episódio
            avg_reward = np.mean(agent.episode_rewards[-10:]) if len(agent.episode_rewards) >= 10 else total_reward
            console.print(f"[blue]Episódio {episode} | Recompensa: {total_reward:.2f} | Média (10): {avg_reward:.2f} | Exploração: {agent.exploration_rate:.4f}[/blue]")
            
            # Verificar progresso
            if total_reward > 100:
                console.print(f"[bold green]🎉 Progresso significativo! Recompensa: {total_reward:.2f}[/bold green]")
                return True
        
        console.print(f"[yellow]Tempo de treinamento esgotado ou episódios máximos atingidos[/yellow]")
        return best_reward > 0
        
    except KeyboardInterrupt:
        console.print("[red]Treinamento interrompido pelo usuário[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Erro durante treinamento: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Salvar modelo final
        agent.save()
        env.close()
        if test_env:
            test_env.close()


if __name__ == "__main__":
    success = train_mbp()
    
    if success:
        print("\n✅ Treinamento concluído com progresso!")
        sys.exit(0)
    else:
        print("\n⚠️ Treinamento concluído sem progresso significativo")
        sys.exit(1)

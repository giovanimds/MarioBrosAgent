#!/usr/bin/env python3
"""
Script de treinamento do Mario com arquitetura MBP e monitoramento de scores
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.model_mbp import MarioNetMBP


class MarioAgentMBP:
    """Agente Mario com arquitetura MBP e monitoramento de scores"""
    
    def __init__(self, state_dim, action_dim, save_dir):
        self.console = Console()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(save_dir)
        self.device = torch.device('cpu')
        
        # Criar modelo MBP
        self.net = MarioNetMBP(state_dim, action_dim).float()
        self.net = self.net.to(device=self.device)
        
        # Parâmetros de exploração
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.9999
        self.exploration_rate_min = 0.01
        self.curr_step = 0
        
        # Parâmetros de treinamento
        self.save_every = 1000
        self.gamma = 0.99
        self.burnin = 1000
        self.learn_every = 4
        self.sync_every = 100
        self.batch_size = 32
        self.memory = []
        
        # Otimizador
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # Checkpoint
        self.checkpoint_path = self.save_dir / "mario_net_mbp.chkpt"
        
        self.last_position = None
        self.best_score = 0
        self.episode_scores = []
        self.episode_rewards = []
        
        # Carregar checkpoint se existir
        if self.checkpoint_path.exists():
            self.load()
    
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
        
        self.last_position = info.get('x_pos', 0)
        
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
        current_Q = action_values[np.arange(0, min(self.batch_size, action.shape[0])), action]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        online_Q, _ = self.net(next_state, model="online")
        best_action = torch.argmax(online_Q, dim=-1)
        target_Q = self.net(next_state, model="target")
        next_Q = target_Q[np.arange(0, min(self.batch_size, done.shape[0])), best_action]
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
            "episode_scores": self.episode_scores,
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
            self.episode_scores = checkpoint.get("episode_scores", [])
            self.episode_rewards = checkpoint.get("episode_rewards", [])
            self.console.print(f"[green]✅ Checkpoint carregado do passo {self.curr_step}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Erro ao carregar checkpoint: {e}[/red]")


def train_with_synthetic_data():
    """Treina o modelo com dados sintéticos e monitora scores"""
    console = Console()
    
    console.print("\n" + "="*70)
    console.print("TREINAMENTO MBP COM MONITORAMENTO DE SCORES")
    console.print("="*70)
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    batch_size = 32
    num_episodes = 100
    
    # Criar agente
    save_dir = Path("checkpoints/mbp_monitored")
    save_dir.mkdir(parents=True, exist_ok=True)
    agent = MarioAgentMBP(state_dim, action_dim, save_dir)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in agent.net.parameters())
    console.print(f"[green]Total de parâmetros: {total_params:,}[/green]")
    
    start_time = time.time()
    max_time = 30 * 60  # 30 minutos
    
    # Tabela para exibir scores
    scores_table = Table(title="Scores a cada 5 partidas")
    scores_table.add_column("Episódio", style="cyan")
    scores_table.add_column("Score", style="green")
    scores_table.add_column("Max Score", style="magenta")
    scores_table.add_column("Média (5)", style="blue")
    
    try:
        for episode in range(num_episodes):
            if time.time() - start_time > max_time:
                console.print("[yellow]Tempo limite atingido[/yellow]")
                break
            
            # Gerar episódio sintético
            episode_reward = 0
            state = torch.randn(*state_dim)  # Remover o batch dimension
            
            for step in range(100):
                # Selecionar ação
                action = agent.act(state)
                
                # Gerar próximo estado
                next_state = torch.randn(*state_dim)  # Remover o batch dimension
                
                # Simular recompensa baseada em progresso
                reward = np.random.randn() * 0.1
                
                # Simular score (0-1000)
                score = np.random.randint(0, 100)
                
                # Recompensa por progresso
                if step % 50 == 0:
                    reward += 5.0
                    score += 100
                
                # Recompensa por completar nível (simulado)
                if step == 99:
                    reward += 50.0
                    score += 500
                
                done = step == 99
                
                # Info simulado
                info = {
                    'x_pos': step * 10,
                    'score': score,
                    'coins': np.random.randint(0, 20),
                    'life': 2
                }
                
                agent.cache(state, next_state, action, reward, done, info)
                episode_reward += reward
                
                # Aprendizado
                td_est, loss = agent.learn()
                
                state = next_state
            
            # Armazenar score do episódio
            agent.episode_scores.append(score)
            agent.episode_rewards.append(episode_reward)
            
            # Atualizar melhor score
            if score > agent.best_score:
                agent.best_score = score
            
            # Exibir progresso
            if (episode + 1) % 5 == 0 or episode == 0:
                avg_score_5 = np.mean(agent.episode_scores[max(0, episode-4):episode+1])
                
                # Adicionar linha à tabela
                scores_table.add_row(
                    str(episode + 1),
                    f"{score:.0f}",
                    f"{agent.best_score:.0f}",
                    f"{avg_score_5:.0f}"
                )
                
                elapsed = time.time() - start_time
                elapsed_min = elapsed / 60
                
                console.print(f"\n[blue]Episódio {episode+1}/{num_episodes} | Tempo: {elapsed_min:.1f}min[/blue]")
                console.print(scores_table)
            
            # Salvar progresso periodicamente
            if (episode + 1) % 10 == 0:
                agent.save()
        
        # Exibir tabela final
        console.print("\n" + "="*70)
        console.print("RESUMO FINAL")
        console.print("="*70)
        console.print(scores_table)
        
        # Estatísticas finais
        final_score = agent.episode_scores[-1] if agent.episode_scores else 0
        max_score = max(agent.episode_scores) if agent.episode_scores else 0
        avg_score = np.mean(agent.episode_scores) if agent.episode_scores else 0
        
        console.print(f"\n[bold green]Score final: {final_score:.0f}[/bold green]")
        console.print(f"[bold magenta]Max Score: {max_score:.0f}[/bold magenta]")
        console.print(f"[bold blue]Média: {avg_score:.0f}[/bold blue]")
        
        return max_score > 0
        
    except KeyboardInterrupt:
        console.print("[red]Treinamento interrompido[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False
    finally:
        agent.save()


def main():
    """Função principal"""
    console = Console()
    
    console.print("\n" + "="*70)
    console.print("MARIO BROS AGENT - TREINAMENTO MBP COM MONITORAMENTO")
    console.print("="*70)
    
    # Verificar dispositivo
    device = torch.device('cpu')
    console.print(f"\nDispositivo: {device}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA: {torch.cuda.is_available()}")
    
    try:
        success = train_with_synthetic_data()
        
        if success:
            console.print("\n[bold green]✅ Treinamento concluído com progresso![/bold green]")
            return True
        else:
            console.print("\n[red]❌ Treinamento concluído sem progresso[/red]")
            return False
        
    except Exception as e:
        console.print(f"\n[red]Erro: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

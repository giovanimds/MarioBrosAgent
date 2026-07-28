#!/usr/bin/env python3
"""
Script de treinamento simples do Mario com arquitetura MBP
Testa o modelo com dados sintéticos para verificar funcionamento
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


def test_model_with_random_data():
    """Testa o modelo com dados aleatórios para simular treinamento"""
    console = Console()
    
    console.print("[bold green]Testando MarioNetMBP com dados sintéticos[/bold green]")
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    
    # Criar modelo
    console.print("[blue]Criando modelo...[/blue]")
    model = MarioNetMBP(state_dim, action_dim)
    model.train()
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Total de parâmetros: {total_params:,}[/green]")
    console.print(f"[green]Parâmetros treináveis: {trainable_params:,}[/green]")
    
    # Otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()
    
    # Testar forward pass
    console.print("\n[blue]Testando forward pass...[/blue]")
    x = torch.randn(8, *state_dim)
    
    start_time = time.time()
    for i in range(100):
        output_online, loss_bal = model(x, model="online")
        output_target = model(x, model="target")
        
        # Simular loss de Q-learning
        with torch.no_grad():
            target_values = torch.randn_like(output_online)
        
        q_loss = loss_fn(output_online, target_values)
        total_loss = q_loss + loss_bal
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            console.print(f"[yellow]Step {i+1}/100 | Loss: {total_loss.item():.4f} | Tempo: {elapsed:.2f}s[/yellow]")
            start_time = time.time()
    
    console.print("\n[bold green]✅ Modelo testado com sucesso![/bold green]")
    
    # Testar inferência
    console.print("\n[blue]Testando inferência...[/blue]")
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            x_test = torch.randn(1, *state_dim)
            output, _ = model(x_test, model="online")
            action = torch.argmax(output, dim=1).item()
            console.print(f"[green]Ação selecionada: {action}[/green]")
    
    return True


def train_with_synthetic_data():
    """Treina o modelo com dados sintéticos"""
    console = Console()
    
    console.print("\n" + "="*60)
    console.print("TREINAMENTO COM DADOS SINTÉTICOS - MBP")
    console.print("="*60)
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    batch_size = 32
    num_episodes = 100
    
    # Criar modelo
    model = MarioNetMBP(state_dim, action_dim)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()
    
    # Buffer de replay
    memory = []
    max_memory = 10000
    
    start_time = time.time()
    max_time = 30 * 60  # 30 minutos
    
    try:
        for episode in range(num_episodes):
            if time.time() - start_time > max_time:
                console.print("[yellow]Tempo limite atingido[/yellow]")
                break
            
            # Gerar episódio sintético
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_dones = []
            
            # Simular 100 passos
            state = torch.randn(1, *state_dim)
            for step in range(100):
                # Selecionar ação aleatória
                action = torch.randint(0, action_dim, (1,))
                
                # Gerar próximo estado
                next_state = torch.randn(1, *state_dim)
                
                # Recompensa baseada em progresso simulado
                reward = np.random.randn() * 0.1
                if step % 50 == 0:
                    reward += 1.0  # Recompensa por "progresso"
                
                done = step == 99
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)
                episode_dones.append(done)
                
                state = next_state
            
            # Armazenar experiências
            for i in range(len(episode_states)):
                memory.append((
                    episode_states[i],
                    episode_next_states[i],
                    episode_actions[i],
                    episode_rewards[i],
                    episode_dones[i]
                ))
                
                if len(memory) > max_memory:
                    memory.pop(0)
            
            # Treinamento
            if len(memory) >= batch_size:
                indices = np.random.choice(len(memory), batch_size, replace=False)
                batch = [memory[i] for i in indices]
                
                states = torch.cat([b[0] for b in batch], dim=0)
                next_states = torch.cat([b[1] for b in batch], dim=0)
                actions = torch.cat([b[2] for b in batch], dim=0).squeeze()
                rewards = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.bool)
                
                # Forward pass
                with torch.no_grad():
                    online_Q, _ = model(next_states, model="online")
                    best_action = torch.argmax(online_Q, dim=-1)
                    target_Q = model(next_states, model="target")
                    next_Q = target_Q[np.arange(0, batch_size), best_action]
                    td_target = (rewards + (1 - dones.float()) * 0.99 * next_Q).float()
                
                # Estimativa
                state_requires_grad = states.requires_grad_(True)
                action_values, _ = model(states, model="online")
                td_estimate = action_values[np.arange(0, batch_size), actions]
                
                # Loss
                q_loss = loss_fn(td_estimate, td_target)
                load_bal_loss = model.last_load_balancing_loss
                total_loss = q_loss + load_bal_loss
                
                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Sync target
                if episode % 10 == 0:
                    model.target.load_state_dict(model.online.state_dict())
            
            # Print progresso
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60
            console.print(f"[blue]Episódio {episode+1}/{num_episodes} | Tempo: {elapsed_min:.1f}min | Loss: {total_loss.item():.4f}[/blue]")
            
            # Verificar progresso
            if episode % 10 == 0:
                console.print(f"[green]Progresso: {min(100, (episode+1)/num_episodes*100):.1f}%[/green]")
        
        console.print("\n[bold green]✅ Treinamento concluído![/bold green]")
        return True
        
    except KeyboardInterrupt:
        console.print("[red]Treinamento interrompido[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Função principal"""
    console = Console()
    
    console.print("\n" + "="*60)
    console.print("MARIO BROS AGENT - TREINAMENTO MBP")
    console.print("="*60)
    
    # Verificar dispositivo
    device = torch.device('cpu')
    console.print(f"\nDispositivo: {device}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA: {torch.cuda.is_available()}")
    
    try:
        # Testar modelo
        if not test_model_with_random_data():
            return False
        
        # Treinar com dados sintéticos
        if not train_with_synthetic_data():
            return False
        
        console.print("\n" + "="*60)
        console.print("✅ TODOS OS TESTES PASSARAM!")
        console.print("="*60)
        console.print("\nO modelo MBP está funcionando corretamente!")
        return True
        
    except Exception as e:
        console.print(f"\n[red]Erro: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

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
                
                # Adicionar estatísticas de balanceamento
                cv = detailed_stats.get('coefficient_of_variation', 0)
                entropy = detailed_stats.get('entropy', 0)
                
                # Determinar status de balanceamento
                if cv > 1.0:
                    balance_status = "[bold red]Severo Desbalanceamento[/bold red]"
                elif cv > 0.7:
                    balance_status = "[red]Desbalanceado[/red]"
                elif cv > 0.4:
                    balance_status = "[yellow]Moderado[/yellow]"
                else:
                    balance_status = "[green]Bom Balanceamento[/green]"
                
                table.add_row("🧠 MoE Balance", balance_status, f"CV: {cv:.2f}", f"Entropy: {entropy:.2f}")
                
                # Adicionar informações sobre coeficiente de balanceamento
                if 'load_balancing_coef' in moe_metrics:
                    table.add_row("⚖️ Balance Coef", f"{moe_metrics['load_balancing_coef']:.4f}", "-", "-")
                
                # Adicionar informações sobre experts dominantes/inativos
                max_usage = max(expert_usage) if len(expert_usage) > 0 else 0
                min_usage = min(expert_usage) if len(expert_usage) > 0 else 0
                
                dominant_count = sum(1 for u in expert_usage if u > 0.2)
                inactive_count = sum(1 for u in expert_usage if u < 0.01)
                
                table.add_row(
                    "👑 Expert Stats", 
                    f"Max: {max_usage:.2f}", 
                    f"Dom: {dominant_count}/{len(expert_usage)}", 
                    f"Inact: {inactive_count}/{len(expert_usage)}"
                )
        
        return table

    def update_live_display(self, episode, moe_metrics=None, mario_net=None, epsilon=None, step=None):
        """Atualiza o display live com novas informações"""
        if self.live:
            self.current_episode = episode
            if epsilon is not None:
                self.current_epsilon = epsilon
            if step is not None:
                self.current_step = step
            self.live.update(self.generate_table(moe_metrics, mario_net))

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
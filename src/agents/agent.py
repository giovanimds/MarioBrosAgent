import torch
import numpy as np
from pathlib import Path
import pytorch_optimizer as opts
from rich.console import Console
from rich.table import Table

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, ListStorage

from src.agents.model import MarioNet
from src.helpers.logger import MetricLogger
from src.helpers.config import (
    EXPLORATION_RATE, EXPLORATION_RATE_DECAY, EXPLORATION_RATE_MIN,
    SAVE_EVERY, GAMMA, BURNIN, LEARN_EVERY, SYNC_EVERY, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, LOAD_BALANCING_SCHEDULER
)

class MarioB:
    def __init__(self, state_dim, action_dim, save_dir):
        """
        Inicializa a classe base do agente com configurações básicas.

        Args:
            state_dim: Dimensões do estado de observação
            action_dim: Número de ações possíveis
            save_dir: Diretório para salvar checkpoints
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        # Usar parâmetros de configuração centralizados
        self.exploration_rate = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min = EXPLORATION_RATE_MIN
        self.curr_step = 0

        self.save_every = SAVE_EVERY  # no. of experiences between saving Mario Net
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
            action_values, _ = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action_idx


class Mario(MarioB):
    """
    Agente Mario avançado com capacidades de aprendizado por reforço.
    Estende a classe base MarioB com funcionalidades de aprendizado, memória e
    monitoramento de desempenho.
    """
    def __init__(self, state_dim, action_dim, save_dir):
        """
        Inicializa o agente Mario com configurações avançadas.

        Args:
            state_dim: Dimensões do estado de observação
            action_dim: Número de ações possíveis
            save_dir: Diretório para salvar checkpoints
        """
        super().__init__(state_dim, action_dim, save_dir)
        self.checkpoint_path = save_dir / "mario_net.chkpt"
        self.gamma = GAMMA
        self.console = Console()
        self.burnin = BURNIN  # min. experiences before training
        self.learn_every = LEARN_EVERY  # no. of experiences between updates to Q_online
        self.sync_every = SYNC_EVERY  # no. of experiences between Q_target & Q_online sync
        self.memory = TensorDictReplayBuffer(storage=ListStorage(10000))
        self.batch_size = BATCH_SIZE
        self.plot_every = 1000
        self.heatmap = None
        self.max_pos = 40
        self.score_inicial = 0
        self.coins_inicial = 0
        self.vida_inicial = 2

        # Otimizador com parâmetros melhorados para MoE
        self.optimizer = opts.ASGD(
            self.net.parameters(), 
            lr=LEARNING_RATE,  # Taxa de aprendizado da configuração centralizada
            weight_decay=WEIGHT_DECAY  # Regularização da configuração centralizada
        )
        self.loss_fn = torch.nn.SmoothL1Loss()  # Huber loss mais robusta

        # Scheduler dinâmico para balanceamento de carga da configuração centralizada
        self.load_balancing_scheduler = LOAD_BALANCING_SCHEDULER

        # Contadores para monitoramento de balanceamento
        self.balance_checks = 0
        self.balance_adjustments = 0
        self.severe_imbalance_count = 0

        # Rastreamento de especialistas dominantes
        self.dominant_experts_history = []

        if self.checkpoint_path.exists():
            self.load()

    def _update_plot(self):
        """
        Atualiza o mapa de calor em tempo real usando Rich table.
        Esta função obtém dados de atenção da rede neural e os exibe
        em formato de tabela no console.
        """
        heatmap_data = self.net.get_attention_heatmap()

        if heatmap_data is not None:
            # Criar tabela Rich para visualização no console
            table = Table(title="Mapa de Atenção")

            # Adicionar colunas
            for i in range(heatmap_data.shape[1]):
                table.add_column(f"E{i+1}", justify="center")

            # Adicionar linhas
            for i in range(heatmap_data.shape[0]):
                row_values = [f"{val:.2f}" for val in heatmap_data[i].tolist()]
                table.add_row(*row_values)

            # Exibir tabela
            self.console.print(table)

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
        max_usage = stats.get('max_usage', 0)
        # Immediate corrective action on severe imbalance
        if max_usage > self.load_balancing_scheduler['severe_imbalance_threshold']:
            self.console.print(f"[bold red]🔄 Reinicializando gate devido a desbalanceamento severo: {max_usage:.2%}[/bold red]")
            self.reinitialize_gate_network()

        if not stats or 'expert_usage' not in stats:
            return

        usage = stats['expert_usage']
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

        # Soma as recompensas (sem progress_reward para zerar recompensa por movimentação)
        reward += life_reward + coin_reward + score_reward + time_penalty

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
        # Ensure state requires gradients
        state = state.requires_grad_()
        # Unpack online outputs (action values, load balancing loss)
        action_values, _ = self.net(state, model="online")
        # Q_online(s,a)
        current_Q = action_values[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Compute next actions and values
        online_Q, _ = self.net(next_state, model="online")
        best_action = torch.argmax(online_Q, dim=-1)
        target_Q = self.net(next_state, model="target")
        next_Q = target_Q[np.arange(0, self.batch_size), best_action]
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
        """
        Seleciona uma ação baseada no estado atual e atualiza visualizações periodicamente.

        Args:
            state: O estado atual do ambiente

        Returns:
            int: Índice da ação selecionada
        """
        action = super().act(state)

        # Atualizar visualização periodicamente
        if self.curr_step % self.plot_every == 0:
            self._update_plot()

        return action

    def save(self):
        """
        Salva o estado completo do agente, incluindo rede neural, taxa de exploração,
        contador de passos e outros parâmetros importantes para continuar o treinamento.
        """
        checkpoint = {
            "model": self.net.state_dict(),
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step,
            "optimizer": self.optimizer.state_dict(),
            "balance_checks": self.balance_checks,
            "balance_adjustments": self.balance_adjustments,
            "severe_imbalance_count": self.severe_imbalance_count,
            "load_balancing_coef": self.net.moe_layer.load_balancing_loss_coef
        }
        torch.save(checkpoint, self.checkpoint_path)
        self.console.print(f"[green]✅ MarioNet salvo em {self.checkpoint_path} no passo {self.curr_step}[/green]")

    def load(self):
        """
        Carrega o estado completo do agente a partir de um checkpoint,
        permitindo continuar o treinamento de onde parou.
        """
        try:
            self.console.print(f"[yellow]Tentando carregar checkpoint de {self.checkpoint_path}...[/yellow]")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Carregar estado da rede neural
            self.net.load_state_dict(checkpoint["model"])

            # Carregar parâmetros de treinamento
            self.exploration_rate = checkpoint["exploration_rate"]

            # Carregar contador de passos se disponível
            if "curr_step" in checkpoint:
                self.curr_step = checkpoint["curr_step"]

            # Carregar estado do otimizador se disponível
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # Carregar contadores de balanceamento se disponíveis
            if "balance_checks" in checkpoint:
                self.balance_checks = checkpoint["balance_checks"]
            if "balance_adjustments" in checkpoint:
                self.balance_adjustments = checkpoint["balance_adjustments"]
            if "severe_imbalance_count" in checkpoint:
                self.severe_imbalance_count = checkpoint["severe_imbalance_count"]

            # Carregar coeficiente de balanceamento se disponível
            if "load_balancing_coef" in checkpoint:
                self.net.moe_layer.load_balancing_loss_coef = checkpoint["load_balancing_coef"]

            self.console.print(f"[bold green]✅ Checkpoint carregado com sucesso! Continuando do passo {self.curr_step}[/bold green]")
            return True
        except Exception as e:
            self.console.print(f"[bold red]❌ Erro ao carregar checkpoint: {str(e)}[/bold red]")
            return False

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
        self.console.print(f"[yellow]Entropia dos Experts:[/yellow] {moe_metrics.get('entropy', 0):.4f}")
        self.console.print(f"[yellow]Uso Máximo de Expert:[/yellow] {stats['max_usage']:.3f} (Expert {np.argmax(stats['expert_usage'])+1})")
        self.console.print(f"[yellow]Uso Mínimo de Expert:[/yellow] {stats['min_usage']:.3f} (Expert {np.argmin(stats['expert_usage'])+1})")

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
        inactive_count = sum(1 for u in stats['expert_usage'] if u < 0.01)
        dominant_count = sum(1 for u in stats['expert_usage'] if u > 0.2)
        self.console.print(f"[yellow]Experts Inativos (<1%):[/yellow] {inactive_count} de {moe_layer.num_experts}")
        self.console.print(f"[yellow]Experts Dominantes (>20%):[/yellow] {dominant_count} de {moe_layer.num_experts}")

        # Histórico de ajustes
        self.console.print("\n[bold cyan]🔧 Histórico de Ajustes:[/bold cyan]")
        self.console.print(f"[yellow]Verificações de Balanceamento:[/yellow] {self.balance_checks}")
        self.console.print(f"[yellow]Ajustes Realizados:[/yellow] {self.balance_adjustments}")
        self.console.print(f"[yellow]Episódios com Desbalanceamento Severo:[/yellow] {self.severe_imbalance_count}")

        # Distribuição de uso dos especialistas
        self.console.print("\n[bold cyan]📈 Distribuição de Uso dos Experts:[/bold cyan]")
        expert_usage = stats['expert_usage']
        ideal_usage = 1.0 / moe_layer.num_experts

        # Criar tabela para melhor visualização
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Expert", style="cyan", no_wrap=True)
        table.add_column("Uso (%)", justify="right")
        table.add_column("Visualização", no_wrap=True)
        table.add_column("Status", no_wrap=True)

        # Ordenar por uso
        # Use torch.argsort for descending order indices
        sorted_indices = torch.argsort(expert_usage, descending=True).cpu().numpy()

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

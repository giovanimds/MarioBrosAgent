import torch
import numpy as np
import pytorch_optimizer as opts
from rich.console import Console
from rich.table import Table
from src.agents.model import ActorCriticMarioNet
from src.agents.gppo import GPPO, TrajectoryBuffer
from src.helpers.config import (
    SAVE_EVERY, GAMMA, GAE_LAMBDA, CLIP_PARAM, VALUE_COEF, ENTROPY_COEF,
    MAX_GRAD_NORM, PPO_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, ADAPTIVE_CLIP,
    KL_TARGET, TRAJECTORY_LENGTH
)

class GPPOMario:
    """
    Agente Mario usando GPPO (Generalized Proximal Policy Optimization) com
    arquitetura Mixture of Experts (MoE).
    """
    def __init__(self, state_dim, action_dim, save_dir):
        """
        Inicializa o agente GPPO.
        
        Args:
            state_dim: Dimensões do estado (c, h, w)
            action_dim: Número de ações possíveis
            save_dir: Diretório para salvar checkpoints
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar rede actor-critic
        self.net = ActorCriticMarioNet(state_dim, action_dim).to(self.device)
        
        # Inicializar otimizador
        self.optimizer = opts.AdaBelief(
            self.net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=False
        )
        
        # Inicializar algoritmo GPPO
        self.gppo = GPPO(
            actor_critic=self.net,
            clip_param=CLIP_PARAM,
            value_coef=VALUE_COEF,
            entropy_coef=ENTROPY_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            use_clipped_value=True,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            adaptive_clip=ADAPTIVE_CLIP,
            kl_target=KL_TARGET
        )
        
        # Inicializar buffer de trajetórias
        self.buffer = TrajectoryBuffer(gamma=GAMMA, gae_lambda=GAE_LAMBDA)
        
        # Contadores e métricas
        self.curr_step = 0
        self.trajectory_step = 0
        self.episodes = 0
        self.console = Console()
        
        # Carregar modelo se existir
        self.load()
    
    def act(self, state):
        """
        Seleciona uma ação com base no estado atual.
        
        Args:
            state: Estado atual do ambiente
            
        Returns:
            action: Ação selecionada
        """
        # Converter estado para tensor
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        # Obter distribuição de política e valor
        with torch.no_grad():
            policy_dist, value = self.net(state)
        
        # Amostrar ação da distribuição
        action = policy_dist.sample().item()
        
        # Obter log probabilidade da ação
        log_prob = policy_dist.log_prob(torch.tensor(action).to(self.device)).item()
        
        # Incrementar contador
        self.curr_step += 1
        self.trajectory_step += 1
        
        return action, log_prob, value.item()
    
    def cache(self, state, next_state, action, reward, done, info, log_prob, value):
        """
        Armazena uma transição no buffer de trajetórias.
        
        Args:
            state: Estado atual
            next_state: Próximo estado
            action: Ação tomada
            reward: Recompensa recebida
            done: Flag indicando fim do episódio
            info: Informações adicionais do ambiente
            log_prob: Log probabilidade da ação
            value: Valor estimado do estado
        """
        # Calcular recompensa com base nas informações do ambiente
        reward = self.calculate_reward(reward, done, info)
        
        # Adicionar transição ao buffer
        self.buffer.add(state, action, reward, value, log_prob, done)
    
    def calculate_reward(self, reward, done, info):
        """
        Calcula a recompensa com base nas informações do ambiente.
        
        Args:
            reward: Recompensa original do ambiente
            done: Flag indicando fim do episódio
            info: Informações adicionais do ambiente
            
        Returns:
            reward: Recompensa modificada
        """
        # Recompensa base do ambiente
        modified_reward = float(reward)
        
        # Penalidade por morte
        if done and info.get('flag_get', False) == False and info.get('time', 0) > 0:
            modified_reward -= 5.0
        
        # Bônus por completar o nível
        if info.get('flag_get', False):
            modified_reward += 10.0
        
        # Bônus por progresso no nível (x_pos)
        if 'x_pos' in info:
            # Verificar se houve progresso desde o último passo
            if hasattr(self, 'last_x_pos'):
                progress = info['x_pos'] - self.last_x_pos
                if progress > 0:
                    modified_reward += progress * 0.01
            
            # Armazenar posição atual para o próximo passo
            self.last_x_pos = info['x_pos']
        else:
            self.last_x_pos = 0
        
        # Bônus por moedas coletadas
        if 'coins' in info and hasattr(self, 'last_coins'):
            coins_collected = info['coins'] - self.last_coins
            if coins_collected > 0:
                modified_reward += coins_collected * 0.5
            self.last_coins = info['coins']
        else:
            self.last_coins = 0
        
        # Pequena penalidade por tempo para incentivar progresso
        modified_reward -= 0.001
        
        return modified_reward
    
    def learn(self):
        """
        Atualiza a política e a função de valor usando GPPO.
        
        Returns:
            metrics: Métricas de treinamento
        """
        # Skip learning if no states have been collected to avoid empty tensors
        if not self.buffer.states:
            return None

        # Verificar se temos dados suficientes para aprender
        if self.trajectory_step < TRAJECTORY_LENGTH and not self.buffer.dones[-1]:
            return None
        
        # Obter último estado para calcular valor final (se não for terminal)
        if not self.buffer.dones[-1]:
            # Obter último estado
            last_state = self.buffer.states[-1]
            last_state = torch.FloatTensor(np.array(last_state)).unsqueeze(0).to(self.device)
            
            # Calcular valor do último estado
            with torch.no_grad():
                _, last_value = self.net(last_state)
                last_value = last_value.item()
        else:
            # Se o último estado for terminal, o valor é 0
            last_value = 0
        
        # Atualizar política e valor usando GPPO
        metrics = self.gppo.update(
            optimizer=self.optimizer,
            epochs=PPO_EPOCHS,
            device=self.device,
            last_value=last_value
        )
        
        # Ajustar balanceamento de carga dinamicamente
        self.net.adjust_load_balancing_dynamically()
        
        # Resetar contador de passos da trajetória
        self.trajectory_step = 0
        
        # Incrementar contador de episódios se o último estado for terminal
        if self.buffer.dones[-1]:
            self.episodes += 1
        
        # Salvar periodicamente
        if self.episodes > 0 and self.episodes % SAVE_EVERY == 0:
            self.save()
        
        return metrics
    
    def save(self):
        """Salva o modelo e o otimizador"""
        save_path = self.save_dir / f"gppo_mario_net_{self.episodes}.chkpt"
        
        torch.save(
            {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'gppo': {
                    'clip_param': self.gppo.clip_param,
                    'value_coef': self.gppo.value_coef,
                    'entropy_coef': self.gppo.entropy_coef
                },
                'episodes': self.episodes,
                'curr_step': self.curr_step
            },
            save_path
        )
        
        self.console.print(f"[green]Modelo salvo em {save_path}[/green]")
    
    def load(self):
        """Carrega o modelo mais recente se existir"""
        checkpoints = list(self.save_dir.glob("gppo_mario_net_*.chkpt"))
        
        if not checkpoints:
            self.console.print("[yellow]Nenhum checkpoint encontrado. Iniciando do zero.[/yellow]")
            return
        
        # Encontrar o checkpoint mais recente
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        
        self.console.print(f"[green]Carregando modelo de {latest_checkpoint}[/green]")
        
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        # Carregar modelo e otimizador
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Carregar parâmetros do GPPO
        self.gppo.clip_param = checkpoint['gppo']['clip_param']
        self.gppo.value_coef = checkpoint['gppo']['value_coef']
        self.gppo.entropy_coef = checkpoint['gppo']['entropy_coef']
        
        # Carregar contadores
        self.episodes = checkpoint['episodes']
        self.curr_step = checkpoint['curr_step']
        
        self.console.print(f"[green]Modelo carregado com sucesso. Episódios: {self.episodes}, Passos: {self.curr_step}[/green]")
    
    def print_gppo_stats(self, episode):
        """
        Imprime estatísticas do GPPO.
        
        Args:
            episode: Número do episódio atual
        """
        table = Table(title=f"GPPO Stats - Episódio {episode}")
        
        # Adicionar colunas
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")
        
        # Adicionar linhas
        table.add_row("Episódios", str(self.episodes))
        table.add_row("Passos", str(self.curr_step))
        table.add_row("Clip Param", f"{self.gppo.clip_param:.4f}")
        table.add_row("Value Coef", f"{self.gppo.value_coef:.4f}")
        table.add_row("Entropy Coef", f"{self.gppo.entropy_coef:.4f}")
        
        # Adicionar estatísticas de MoE
        moe_metrics = self.net.get_moe_metrics()
        
        # Estatísticas da rede de política
        policy_stats = moe_metrics['policy']
        table.add_row("Policy Max Usage", f"{policy_stats['max_usage']:.4f}")
        table.add_row("Policy Min Usage", f"{policy_stats['min_usage']:.4f}")
        table.add_row("Policy Entropy", f"{policy_stats['entropy']:.4f}")
        table.add_row("Policy CoV", f"{policy_stats['coefficient_of_variation']:.4f}")
        
        # Estatísticas da rede de valor
        value_stats = moe_metrics['value']
        table.add_row("Value Max Usage", f"{value_stats['max_usage']:.4f}")
        table.add_row("Value Min Usage", f"{value_stats['min_usage']:.4f}")
        table.add_row("Value Entropy", f"{value_stats['entropy']:.4f}")
        table.add_row("Value CoV", f"{value_stats['coefficient_of_variation']:.4f}")
        
        # Losses de balanceamento
        table.add_row("Policy Load Balancing Loss", f"{moe_metrics['load_balancing_loss']['policy']:.4f}")
        table.add_row("Value Load Balancing Loss", f"{moe_metrics['load_balancing_loss']['value']:.4f}")
        
        # Imprimir tabela
        self.console.print(table)
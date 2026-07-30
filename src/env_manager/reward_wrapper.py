"""
Reward Wrapper para Super Mario Bros

Este módulo implementa um sistema de recompensas customizado que aproveita
todas as informações disponíveis no ambiente para criar recompensas mais
significativas e direcionadas para o aprendizado do agente.

Informações disponíveis no ambiente:
- x_pos: Posição horizontal do Mario (0-3200+)
- y_pos: Posição vertical do Mario
- coins: Número de moedas coletadas
- score: Pontuação atual
- time: Tempo restante no nível
- life: Número de vidas restantes
- world: Mundo atual (1-8)
- stage: Fase atual (1-4)
- status: Estado do Mario ('small', 'tall', 'fireball')
- flag_get: Se pegou a bandeira (fim do nível)
"""

import gym
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class RewardConfig:
    """Configuração de pesos para o sistema de recompensas"""
    
    # Recompensas de progresso
    x_pos_reward: float = 0.1          # Recompensa por progresso horizontal
    coin_reward: float = 1.0           # Recompensa por moeda
    flag_reward: float = 50.0          # Recompensa por completar o nível
    
    # Recompensas de estado
    grow_reward: float = 5.0           # Recompensa por pegar cogumelo (small -> tall)
    fire_reward: float = 10.0          # Recompensa por pegar flor (tall -> fireball)
    death_penalty: float = -10.0       # Penalidade por morrer
    
    # Recompensas de tempo
    time_penalty: float = -0.01        # Penalidade por tempo (incentivar rapidez)
    time_bonus: float = 1.0            # Bônus por tempo restante ao completar nível
    
    # Recompensas de exploração
    jump_reward: float = 0.05          # Pequena recompensa por pular
    run_reward: float = 0.02           # Pequena recompensa por correr
    
    # Penalidades por ações
    backward_penalty: float = -0.05    # Penalidade por ir para trás
    stuck_penalty: float = -0.1        # Penalidade por ficar preso
    
    # Recompensas por inimigos
    enemy_kill_reward: float = 2.0     # Recompensa por matar inimigo
    
    # Normalização
    max_reward: float = 10.0           # Valor máximo para normalização
    
    # Parâmetros de detecção de progresso
    min_progress_threshold: float = 5.0  # Mínimo progresso para recompensa
    stuck_threshold: int = 20          # Passos sem progresso = preso
    
    # Curriculum learning
    use_curriculum: bool = False       # Ativar curriculum learning
    level_progression: Dict[int, float] = field(default_factory=lambda: {
        1: 1.0,  # Mundo 1 - peso normal
        2: 1.5,  # Mundo 2 - 50% mais recompensa
        3: 2.0,  # Mundo 3 - 100% mais recompensa
        4: 2.5,  # Mundo 4 - 150% mais recompensa
    })


@dataclass
class RewardState:
    """Estado interno para rastreamento de progresso"""
    last_x_pos: int = 0
    last_coins: int = 0
    last_life: int = 2
    last_status: str = "small"
    last_time: int = 400
    last_world: int = 1
    last_stage: int = 1
    last_score: int = 0
    
    # Para detecção de progresso
    x_pos_history: deque = field(default_factory=lambda: deque(maxlen=100))
    no_progress_count: int = 0
    
    # Para detecção de ações
    last_action: Optional[int] = None
    action_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Para curriculum learning
    current_level: int = 1
    levels_completed: set = field(default_factory=set)


class MarioRewardWrapper(gym.Wrapper):
    """
    Wrapper que implementa um sistema de recompensas customizado para Super Mario Bros.
    
    Este wrapper substitui as recompensas padrão do ambiente por um sistema mais
    sofisticado que considera:
    - Progresso horizontal (x_pos)
    - Coleta de moedas
    - Mudanças de estado (crescer, pegar fogo)
    - Tempo restante
    - Penalidades por ações indesejadas
    - Detecção de situações de risco
    """
    
    def __init__(self, env, config: Optional[RewardConfig] = None):
        """
        Inicializa o wrapper de recompensas.
        
        Args:
            env: Ambiente base do Mario
            config: Configuração de recompensas (opcional)
        """
        super().__init__(env)
        
        self.config = config or RewardConfig()
        self.state = RewardState()
        self.episode_rewards = []
        self.episode_info = {}
        
        # Contadores de episódio
        self.episode_step = 0
        self.episode_total_reward = 0.0
        
    def reset(self, **kwargs):
        """Reset do ambiente e do estado interno"""
        obs, info = self.env.reset(**kwargs)
        
        # Resetar estado interno
        self.state = RewardState()
        self.episode_rewards = []
        self.episode_info = {}
        self.episode_step = 0
        self.episode_total_reward = 0.0
        
        # Inicializar estado com informações do reset
        if info:
            self._update_state_from_info(info)
        
        return obs, info
    
    def step(self, action):
        """
        Executa uma ação e calcula a recompensa customizada.
        
        Args:
            action: Ação a ser executada
            
        Returns:
            obs, reward, done, trunc, info: Tupla padrão do Gym
        """
        # Armazenar ação atual
        self.state.last_action = action
        self.state.action_history.append(action)
        self.episode_step += 1
        
        # Executar ação no ambiente
        obs, base_reward, done, trunc, info = self.env.step(action)
        
        # Atualizar estado com novas informações
        self._update_state_from_info(info)
        
        # Calcular recompensa customizada
        custom_reward = self._calculate_custom_reward(info, done, action)
        
        # Armazenar recompensas para debugging
        self.episode_rewards.append({
            'step': self.episode_step,
            'base_reward': base_reward,
            'custom_reward': custom_reward,
            'info': info.copy()
        })
        self.episode_total_reward += custom_reward
        
        # Adicionar métricas ao info
        info['custom_reward'] = custom_reward
        info['base_reward'] = base_reward
        info['episode_step'] = self.episode_step
        info['episode_total_reward'] = self.episode_total_reward
        
        return obs, custom_reward, done, trunc, info
    
    def _update_state_from_info(self, info: Dict[str, Any]):
        """Atualiza o estado interno com informações do ambiente"""
        self.state.last_x_pos = info.get('x_pos', self.state.last_x_pos)
        self.state.last_coins = info.get('coins', self.state.last_coins)
        self.state.last_life = info.get('life', self.state.last_life)
        self.state.last_status = info.get('status', self.state.last_status)
        self.state.last_time = info.get('time', self.state.last_time)
        self.state.last_world = info.get('world', self.state.last_world)
        self.state.last_stage = info.get('stage', self.state.last_stage)
        self.state.last_score = info.get('score', self.state.last_score)
        
        # Atualizar histórico de posições
        self.state.x_pos_history.append(info.get('x_pos', self.state.last_x_pos))
        
        # Atualizar nível atual para curriculum
        world = info.get('world', 1)
        stage = info.get('stage', 1)
        self.state.current_level = (world - 1) * 4 + stage
    
    def _calculate_custom_reward(self, info: Dict[str, Any], done: bool, action: int) -> float:
        """
        Calcula a recompensa customizada com base em todas as informações disponíveis.
        
        Args:
            info: Dicionário de informações do ambiente
            done: Se o episódio terminou
            action: Ação executada
            
        Returns:
            Recompensa customizada
        """
        total_reward = 0.0
        
        # 1. Recompensas de progresso
        total_reward += self._calculate_progress_reward(info)
        
        # 2. Recompensas de estado
        total_reward += self._calculate_state_reward(info)
        
        # 3. Recompensas de tempo
        total_reward += self._calculate_time_reward(info, done)
        
        # 4. Recompensas por ações
        total_reward += self._calculate_action_reward(action, info)
        
        # 5. Penalidades por situações indesejadas
        total_reward += self._calculate_penalty_reward(info, action)
        
        # 6. Recompensa por completar o nível
        if info.get('flag_get', False) or (done and info.get('status', '') != 'small'):
            total_reward += self._calculate_completion_reward(info)
        
        # 7. Aplicar curriculum learning
        if self.config.use_curriculum:
            total_reward = self._apply_curriculum_scaling(total_reward)
        
        # 8. Normalizar recompensa
        total_reward = self._normalize_reward(total_reward)
        
        return total_reward
    
    def _calculate_progress_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por progresso horizontal e coleta de moedas"""
        reward = 0.0
        
        # Recompensa por progresso horizontal
        x_pos = info.get('x_pos', self.state.last_x_pos)
        progress = x_pos - self.state.last_x_pos
        
        if progress > 0:
            # Recompensa proporcional ao progresso
            reward += progress * self.config.x_pos_reward
            self.state.no_progress_count = 0
        elif progress < 0:
            # Penalidade por ir para trás
            reward += progress * self.config.backward_penalty
            self.state.no_progress_count += 1
        else:
            # Sem progresso
            self.state.no_progress_count += 1
        
        # Recompensa por moedas
        coins = info.get('coins', self.state.last_coins)
        coins_collected = coins - self.state.last_coins
        if coins_collected > 0:
            reward += coins_collected * self.config.coin_reward
        
        return reward
    
    def _calculate_state_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por mudanças de estado do Mario"""
        reward = 0.0
        
        # Verificar mudança de status
        current_status = info.get('status', self.state.last_status)
        
        if current_status != self.state.last_status:
            if current_status == 'tall' and self.state.last_status == 'small':
                # Cresceu (pegou cogumelo)
                reward += self.config.grow_reward
            elif current_status == 'fireball' and self.state.last_status in ['small', 'tall']:
                # Pegou fogo (pegou flor)
                reward += self.config.fire_reward
        
        # Verificar perda de vida
        current_life = info.get('life', self.state.last_life)
        if current_life < self.state.last_life:
            # Perdeu vida
            reward += self.config.death_penalty * (self.state.last_life - current_life)
        
        return reward
    
    def _calculate_time_reward(self, info: Dict[str, Any], done: bool) -> float:
        """Calcula recompensa/penalidade baseada no tempo"""
        reward = 0.0
        
        current_time = info.get('time', self.state.last_time)
        time_diff = self.state.last_time - current_time
        
        # Penalidade por tempo passando (incentivar rapidez)
        if time_diff > 0:
            reward += time_diff * self.config.time_penalty
        
        # Bônus por tempo restante ao completar o nível
        if done and info.get('flag_get', False):
            remaining_time = info.get('time', 0)
            reward += remaining_time * self.config.time_bonus / 400.0  # Normalizar por tempo máximo
        
        return reward
    
    def _calculate_action_reward(self, action: int, info: Dict[str, Any]) -> float:
        """Calcula recompensa por ações específicas (exploração)"""
        reward = 0.0
        
        # Recompensa por pular (A)
        # Ação 4 é ['A'] - pular
        if action == 4:
            reward += self.config.jump_reward
        
        # Recompensa por correr (B + right)
        # Ação 8 é ['B', 'right'] - correr para direita
        if action == 8:
            reward += self.config.run_reward
        
        return reward
    
    def _calculate_penalty_reward(self, info: Dict[str, Any], action: int) -> float:
        """Calcula penalidades por situações indesejadas"""
        penalty = 0.0
        
        # Penalidade por ir para trás
        x_pos = info.get('x_pos', self.state.last_x_pos)
        if x_pos < self.state.last_x_pos and action in [3, 6, 7]:  # left, left+A, left+B
            penalty += self.config.backward_penalty
        
        # Penalidade por ficar preso (sem progresso por muitos passos)
        if self.state.no_progress_count >= self.config.stuck_threshold:
            penalty += self.config.stuck_penalty
        
        # Penalidade por tempo esgotado
        if info.get('time', 0) <= 0:
            penalty += self.config.death_penalty * 0.5  # Metade da penalidade de morte
        
        return penalty
    
    def _calculate_completion_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por completar o nível"""
        reward = 0.0
        
        # Recompensa base por completar
        reward += self.config.flag_reward
        
        # Bônus por tempo restante
        remaining_time = info.get('time', 0)
        reward += remaining_time * self.config.time_bonus
        
        # Bônus por moedas coletadas
        coins = info.get('coins', 0)
        reward += coins * self.config.coin_reward * 0.5  # Metade do valor normal
        
        # Bônus por pontuação
        score = info.get('score', 0)
        reward += score * 0.01  # 1% da pontuação
        
        return reward
    
    def _apply_curriculum_scaling(self, reward: float) -> float:
        """Aplica scaling de recompensa com base no curriculum learning"""
        level = self.state.current_level
        
        # Encontrar o fator de scaling para este nível
        for max_level, factor in self.config.level_progression.items():
            if level <= max_level:
                return reward * factor
        
        # Se não encontrado, usar o maior fator
        max_factor = max(self.config.level_progression.values())
        return reward * max_factor
    
    def _normalize_reward(self, reward: float) -> float:
        """Normaliza a recompensa para um range razoável"""
        # Limitar recompensa máxima
        if abs(reward) > self.config.max_reward:
            reward = np.sign(reward) * self.config.max_reward
        
        return reward
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do episódio atual"""
        return {
            'total_reward': self.episode_total_reward,
            'step_count': self.episode_step,
            'rewards': self.episode_rewards,
            'final_info': self.episode_info
        }
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Retorna o breakdown das recompensas do último passo"""
        if not self.episode_rewards:
            return {}
        
        last_reward = self.episode_rewards[-1]
        return {
            'base_reward': last_reward.get('base_reward', 0),
            'custom_reward': last_reward.get('custom_reward', 0),
            'x_pos': last_reward.get('info', {}).get('x_pos', 0),
            'coins': last_reward.get('info', {}).get('coins', 0),
            'time': last_reward.get('info', {}).get('time', 0),
            'status': last_reward.get('info', {}).get('status', ''),
            'life': last_reward.get('info', {}).get('life', 0)
        }


class ProgressiveRewardWrapper(gym.Wrapper):
    """
    Wrapper que implementa recompensas progressivas com base em marcos (milestones).
    
    Este wrapper divide o nível em marcos e dá recompensas adicionais quando o agente
    atinge certos pontos no nível.
    """
    
    def __init__(self, env, milestones: Optional[Dict[int, float]] = None):
        """
        Inicializa o wrapper de recompensas progressivas.
        
        Args:
            env: Ambiente base
            milestones: Dicionário de posições x e recompensas (ex: {100: 5.0, 200: 10.0})
        """
        super().__init__(env)
        
        # Milestones padrão para o primeiro nível do Mario
        self.milestones = milestones or {
            100: 5.0,   # Primeiro obstáculo
            200: 10.0,  # Meio do nível
            300: 15.0,  # Próximo ao final
            400: 20.0   # Quase lá
        }
        
        self.reached_milestones = set()
        self.last_x_pos = 0
        
    def reset(self, **kwargs):
        """Reset do ambiente"""
        obs, info = self.env.reset(**kwargs)
        self.reached_milestones = set()
        self.last_x_pos = info.get('x_pos', 0)
        return obs, info
    
    def step(self, action):
        """Executa uma ação com recompensas progressivas"""
        obs, reward, done, trunc, info = self.env.step(action)
        
        current_x_pos = info.get('x_pos', self.last_x_pos)
        
        # Verificar se atingiu algum milestone
        milestone_reward = 0.0
        for pos, reward_val in self.milestones.items():
            if current_x_pos >= pos and pos not in self.reached_milestones:
                milestone_reward += reward_val
                self.reached_milestones.add(pos)
        
        # Adicionar recompensa do milestone à recompensa total
        info['milestone_reward'] = milestone_reward
        total_reward = reward + milestone_reward
        
        self.last_x_pos = current_x_pos
        
        return obs, total_reward, done, trunc, info


class CompositeRewardWrapper(gym.Wrapper):
    """
    Wrapper que combina múltiplos sistemas de recompensas.
    
    Este wrapper permite empilhar vários wrappers de recompensa e combinar
    suas saídas de forma flexível.
    """
    
    def __init__(self, env, wrappers: list):
        """
        Inicializa o wrapper composto.
        
        Args:
            env: Ambiente base
            wrappers: Lista de wrappers de recompensa a serem aplicados
        """
        super().__init__(env)
        self.wrappers = wrappers
        
    def reset(self, **kwargs):
        """Reset de todos os wrappers"""
        obs, info = self.env.reset(**kwargs)
        for wrapper in self.wrappers:
            if hasattr(wrapper, 'reset'):
                obs, info = wrapper.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Aplica todos os wrappers em sequência"""
        obs, reward, done, trunc, info = self.env.step(action)
        
        total_reward = reward
        for wrapper in self.wrappers:
            if hasattr(wrapper, 'step'):
                obs, wrapper_reward, done, trunc, info = wrapper.step(action)
                # Somar recompensas
                if 'custom_reward' in info:
                    total_reward += info['custom_reward']
                else:
                    total_reward += wrapper_reward
        
        info['total_custom_reward'] = total_reward
        
        return obs, total_reward, done, trunc, info


def create_reward_wrapper(
    env,
    use_mario_reward: bool = True,
    use_progressive: bool = True,
    config: Optional[RewardConfig] = None
) -> gym.Wrapper:
    """
    Função fábrica para criar o wrapper de recompensas.
    
    Args:
        env: Ambiente base
        use_mario_reward: Se usar o MarioRewardWrapper
        use_progressive: Se usar o ProgressiveRewardWrapper
        config: Configuração para o MarioRewardWrapper
        
    Returns:
        Ambiente com wrappers de recompensa aplicados
    """
    wrappers = []
    
    if use_mario_reward:
        wrappers.append(MarioRewardWrapper(env, config))
    
    if use_progressive:
        wrappers.append(ProgressiveRewardWrapper(env))
    
    if len(wrappers) == 1:
        return wrappers[0]
    elif len(wrappers) > 1:
        return CompositeRewardWrapper(env, wrappers)
    else:
        return env

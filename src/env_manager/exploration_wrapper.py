"""
Exploration Reward Wrapper para Super Mario Bros

Este módulo implementa um sistema de recompensas focado em EXPLORAÇÃO MÁXIMA.
O objetivo é incentivar o agente a:
1. Explorar áreas não visitadas
2. Descobrir rotas secretas
3. Encontrar segredos (moedas escondidas, power-ups, etc.)
4. Ter curiosidade intrínseca (novelty detection)

O sistema usa:
- Visitation Grid: Mapa de células visitadas
- Novelty Detection: Detecção de estados novos
- Secret Path Detection: Identificação de caminhos alternativos
- Intrinsic Motivation: Recompensas por curiosidade
"""

import gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque, defaultdict
import math
import hashlib


@dataclass
class ExplorationConfig:
    """Configuração para o sistema de recompensas por exploração"""
    
    # Recompensas por exploração
    new_cell_reward: float = 0.5        # Recompensa por visitar célula nova
    secret_area_reward: float = 5.0     # Recompensa por área secreta
    hidden_coin_reward: float = 3.0     # Recompensa por moeda escondida
    powerup_reward: float = 10.0        # Recompensa por power-up (cogumelo, flor, estrela)
    
    # Recompensas por novelty
    novelty_reward: float = 0.3         # Recompensa por estado novo
    novelty_decay: float = 0.99        # Decaimento da novelty
    
    # Recompensas por rotas secretas
    secret_path_reward: float = 15.0    # Recompensa por descobrir rota secreta
    path_diversity_reward: float = 2.0  # Recompensa por diversidade de caminhos
    
    # Parâmetros de detecção
    grid_cell_size: int = 16           # Tamanho da célula no grid de visitação (em pixels)
    min_secret_area_size: int = 3      # Tamanho mínimo de área secreta (em células)
    novelty_threshold: float = 0.1     # Threshold para considerar estado novo
    
    # Parâmetros de rotas secretas
    min_path_length: int = 10         # Comprimento mínimo para considerar um caminho
    path_similarity_threshold: float = 0.8  # Similaridade para considerar caminhos iguais
    
    # Recompensas por curiosidade intrínseca
    intrinsic_reward_scale: float = 0.1  # Escala da recompensa intrínseca
    prediction_error_reward: float = 0.2  # Recompensa por erro de previsão
    
    # Parâmetros de decaimento
    exploration_decay: float = 0.999    # Decaimento da recompensa de exploração
    min_exploration_reward: float = 0.01 # Recompensa mínima de exploração
    
    # Normalização
    max_exploration_reward: float = 20.0  # Limite máximo
    
    # Ativação/Desativação
    enable_visitation_grid: bool = True
    enable_novelty_detection: bool = True
    enable_secret_detection: bool = True
    enable_path_diversity: bool = True
    enable_intrinsic_motivation: bool = True


@dataclass
class ExplorationState:
    """Estado interno para rastreamento de exploração"""
    
    # Grid de visitação
    visited_cells: set = field(default_factory=set)  # Conjunto de células visitadas
    visit_count: Dict[Tuple[int, int], int] = field(default_factory=dict)  # Contador de visitas por célula
    
    # Novos estados
    state_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    novelty_scores: Dict[str, float] = field(default_factory=dict)  # Score de novelty por estado
    
    # Rota atual
    current_path: List[Tuple[int, int]] = field(default_factory=list)  # Caminho atual em células
    path_history: List[List[Tuple[int, int]]] = field(default_factory=list)  # Histórico de caminhos
    
    # Segredos descobertos
    discovered_secrets: set = field(default_factory=set)  # Conjunto de segredos descobertos
    secret_areas: List[Tuple[int, int, int, int]] = field(default_factory=list)  # Áreas secretas conhecidas
    
    # Power-ups
    powerups_collected: Dict[str, int] = field(default_factory=dict)  # Power-ups coletados
    
    # Estatísticas
    total_new_cells: int = 0
    total_secret_areas: int = 0
    total_hidden_coins: int = 0
    total_powerups: int = 0
    
    # Contadores
    episode_step: int = 0
    last_x_pos: int = 0
    last_y_pos: int = 0
    last_coins: int = 0
    last_status: str = "small"
    
    # Para detecção de progresso
    no_progress_count: int = 0


class ExplorationRewardWrapper(gym.Wrapper):
    """
    Wrapper que implementa recompensas por exploração.
    
    Este wrapper incentiva o agente a:
    - Visitar áreas não exploradas
    - Descobrir rotas secretas
    - Encontrar segredos (moedas escondidas, power-ups)
    - Ter curiosidade intrínseca
    """
    
    def __init__(self, env, config: Optional[ExplorationConfig] = None):
        """
        Inicializa o wrapper de recompensas por exploração.
        
        Args:
            env: Ambiente base do Mario
            config: Configuração de exploração (opcional)
        """
        super().__init__(env)
        
        self.config = config or ExplorationConfig()
        self.state = ExplorationState()
        self.episode_rewards = []
        
        # Inicializar grid de visitação
        self._init_visitation_grid()
        
        # Inicializar detector de segredos
        self._init_secret_detector()
        
        # Contadores
        self.episode_total_reward = 0.0
        self.episode_exploration_reward = 0.0
        
    def _init_visitation_grid(self):
        """Inicializa o grid de visitação"""
        # O grid será criado dinamicamente com base nas observações
        pass
    
    def _init_secret_detector(self):
        """Inicializa o detector de áreas secretas"""
        # Áreas secretas conhecidas no Mario 1-1 (exemplo)
        # Estas são áreas que normalmente não são visitadas em um path direto
        self.secret_areas = [
            # (x_min, x_max, y_min, y_max, nome)
            (50, 80, 50, 80, "primeiro_cano_secreto"),      # Primeiro cano secreto
            (150, 180, 30, 60, "bloco_secreto_1"),        # Bloco secreto após primeiro obstáculo
            (250, 280, 40, 70, "area_alta_1"),            # Área alta após segundo obstáculo
            (350, 380, 20, 50, "caminho_alternativo_1"), # Caminho alternativo
            (450, 480, 60, 90, "escada_secreta"),       # Escada para área secreta
        ]
        
        # Power-ups que indicam segredos
        self.powerup_types = ['mushroom', 'fire_flower', 'star', '1up_mushroom']
        
    def reset(self, **kwargs):
        """Reset do ambiente e do estado de exploração"""
        obs, info = self.env.reset(**kwargs)
        
        # Resetar estado
        self.state = ExplorationState()
        self.episode_rewards = []
        self.episode_total_reward = 0.0
        self.episode_exploration_reward = 0.0
        
        # Inicializar com informações do reset
        if info:
            self._update_state_from_info(info)
        
        return obs, info
    
    def step(self, action):
        """
        Executa uma ação e calcula recompensas por exploração.
        
        Args:
            action: Ação a ser executada
            
        Returns:
            obs, reward, done, trunc, info: Tupla padrão do Gym
        """
        # Executar ação no ambiente
        obs, base_reward, done, trunc, info = self.env.step(action)
        
        # Atualizar estado
        self._update_state_from_info(info)
        
        # Calcular recompensas por exploração
        exploration_reward = self._calculate_exploration_reward(info, action)
        
        # Recompensa total
        total_reward = base_reward + exploration_reward
        
        # Armazenar recompensas
        self.episode_rewards.append({
            'step': self.state.episode_step,
            'base_reward': base_reward,
            'exploration_reward': exploration_reward,
            'total_reward': total_reward,
            'info': info.copy()
        })
        self.episode_total_reward += total_reward
        self.episode_exploration_reward += exploration_reward
        
        # Adicionar métricas ao info
        info['exploration_reward'] = exploration_reward
        info['total_reward'] = total_reward
        info['new_cells_discovered'] = self.state.total_new_cells
        info['secret_areas_discovered'] = self.state.total_secret_areas
        
        return obs, total_reward, done, trunc, info
    
    def _update_state_from_info(self, info: Dict[str, Any]):
        """Atualiza o estado interno com informações do ambiente"""
        self.state.episode_step += 1
        self.state.last_x_pos = info.get('x_pos', self.state.last_x_pos)
        self.state.last_y_pos = info.get('y_pos', self.state.last_y_pos)
        self.state.last_coins = info.get('coins', self.state.last_coins)
        self.state.last_status = info.get('status', self.state.last_status)
        
        # Atualizar grid de visitação
        if self.config.enable_visitation_grid:
            self._update_visitation_grid(info)
        
        # Atualizar detector de novelty
        if self.config.enable_novelty_detection:
            self._update_novelty(info)
        
        # Atualizar detector de caminhos
        if self.config.enable_path_diversity:
            self._update_path_tracking(info)
        
        # Verificar segredos
        if self.config.enable_secret_detection:
            self._check_secrets(info)
    
    def _update_visitation_grid(self, info: Dict[str, Any]):
        """Atualiza o grid de visitação com a posição atual"""
        x_pos = info.get('x_pos', self.state.last_x_pos)
        y_pos = info.get('y_pos', self.state.last_y_pos)
        
        # Converter posição para coordenadas de célula
        cell_x = x_pos // self.config.grid_cell_size
        cell_y = y_pos // self.config.grid_cell_size
        cell = (cell_x, cell_y)
        
        # Adicionar ao conjunto de células visitadas
        if cell not in self.state.visited_cells:
            self.state.visited_cells.add(cell)
            self.state.total_new_cells += 1
            self.state.visit_count[cell] = 1
        else:
            self.state.visit_count[cell] = self.state.visit_count.get(cell, 0) + 1
        
        # Adicionar à rota atual
        self.state.current_path.append(cell)
    
    def _update_novelty(self, info: Dict[str, Any]):
        """Atualiza o detector de novelty"""
        # Criar uma representação do estado atual
        state_repr = self._create_state_representation(info)
        
        # Calcular hash do estado
        state_hash = hashlib.md5(state_repr.encode()).hexdigest()
        
        # Atualizar histórico de estados
        self.state.state_history.append(state_hash)
        
        # Calcular novelty score
        if state_hash not in self.state.novelty_scores:
            # Estado novo
            self.state.novelty_scores[state_hash] = 1.0
        else:
            # Estado já visto, decair novelty
            self.state.novelty_scores[state_hash] *= self.config.novelty_decay
    
    def _create_state_representation(self, info: Dict[str, Any]) -> str:
        """Cria uma representação do estado para detecção de novelty"""
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)
        status = info.get('status', 'small')
        coins = info.get('coins', 0)
        time = info.get('time', 0)
        
        # Discretizar valores
        x_bin = x_pos // self.config.grid_cell_size
        y_bin = y_pos // self.config.grid_cell_size
        
        return f"{x_bin},{y_bin},{status},{coins},{time}"
    
    def _update_path_tracking(self, info: Dict[str, Any]):
        """Atualiza o rastreamento de caminhos"""
        x_pos = info.get('x_pos', self.state.last_x_pos)
        
        # Verificar se mudou de posição significativamente
        if abs(x_pos - self.state.last_x_pos) > self.config.grid_cell_size:
            # Salvar caminho atual se for longo o suficiente
            if len(self.state.current_path) >= self.config.min_path_length:
                self.state.path_history.append(self.state.current_path.copy())
                
                # Verificar diversidade de caminhos
                self._check_path_diversity()
            
            # Reiniciar caminho atual
            self.state.current_path = []
    
    def _check_path_diversity(self):
        """Verifica diversidade de caminhos"""
        # Comparar caminho atual com caminhos anteriores
        if len(self.state.path_history) < 2:
            return
        
        current_path = self.state.current_path
        for prev_path in self.state.path_history[-5:]:  # Comparar com últimos 5 caminhos
            similarity = self._calculate_path_similarity(current_path, prev_path)
            
            # Se caminho é diferente o suficiente, é diversificado
            if similarity < self.config.path_similarity_threshold:
                # Recompensa por diversidade será aplicada no próximo passo
                pass
    
    def _calculate_path_similarity(self, path1: List, path2: List) -> float:
        """Calcula similaridade entre dois caminhos"""
        if not path1 or not path2:
            return 0.0
        
        # Converter para conjuntos
        set1 = set(path1)
        set2 = set(path2)
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _check_secrets(self, info: Dict[str, Any]):
        """Verifica se o agente descobriu algum segredo"""
        x_pos = info.get('x_pos', self.state.last_x_pos)
        y_pos = info.get('y_pos', self.state.last_y_pos)
        
        # Verificar áreas secretas conhecidas
        for area in self.secret_areas:
            x_min, x_max, y_min, y_max, name = area
            if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
                secret_key = f"{name}_{x_pos}_{y_pos}"
                if secret_key not in self.state.discovered_secrets:
                    self.state.discovered_secrets.add(secret_key)
                    self.state.total_secret_areas += 1
                    # Recompensa será aplicada no cálculo
        
        # Verificar power-ups
        status = info.get('status', self.state.last_status)
        if status != self.state.last_status:
            if status == 'tall' and self.state.last_status == 'small':
                self.state.powerups_collected['mushroom'] = self.state.powerups_collected.get('mushroom', 0) + 1
                self.state.total_powerups += 1
            elif status == 'fireball' and self.state.last_status in ['small', 'tall']:
                self.state.powerups_collected['fire_flower'] = self.state.powerups_collected.get('fire_flower', 0) + 1
                self.state.total_powerups += 1
        
        # Verificar moedas (possivelmente escondidas)
        coins = info.get('coins', self.state.last_coins)
        if coins > self.state.last_coins:
            coins_collected = coins - self.state.last_coins
            # Se coletou muitas moedas de uma vez, provavelmente é uma área secreta
            if coins_collected >= 3:
                self.state.total_hidden_coins += coins_collected
    
    def _calculate_exploration_reward(self, info: Dict[str, Any], action: int) -> float:
        """
        Calcula a recompensa por exploração.
        
        Args:
            info: Dicionário de informações do ambiente
            action: Ação executada
            
        Returns:
            Recompensa por exploração
        """
        total_reward = 0.0
        
        # 1. Recompensa por células novas
        if self.config.enable_visitation_grid:
            total_reward += self._calculate_new_cell_reward(info)
        
        # 2. Recompensa por novelty
        if self.config.enable_novelty_detection:
            total_reward += self._calculate_novelty_reward(info)
        
        # 3. Recompensa por áreas secretas
        if self.config.enable_secret_detection:
            total_reward += self._calculate_secret_reward(info)
        
        # 4. Recompensa por diversidade de caminhos
        if self.config.enable_path_diversity:
            total_reward += self._calculate_path_diversity_reward(info)
        
        # 5. Recompensa por curiosidade intrínseca
        if self.config.enable_intrinsic_motivation:
            total_reward += self._calculate_intrinsic_reward(info, action)
        
        # 6. Aplicar decaimento
        total_reward = self._apply_exploration_decay(total_reward)
        
        # 7. Normalizar
        total_reward = self._normalize_reward(total_reward)
        
        return total_reward
    
    def _calculate_new_cell_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por visitar células novas"""
        x_pos = info.get('x_pos', self.state.last_x_pos)
        y_pos = info.get('y_pos', self.state.last_y_pos)
        
        cell_x = x_pos // self.config.grid_cell_size
        cell_y = y_pos // self.config.grid_cell_size
        cell = (cell_x, cell_y)
        
        # Se é uma célula nova
        if cell in self.state.visited_cells and self.state.visit_count.get(cell, 0) == 1:
            # Primeira visita a esta célula
            return self.config.new_cell_reward
        
        return 0.0
    
    def _calculate_novelty_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por novelty"""
        state_repr = self._create_state_representation(info)
        state_hash = hashlib.md5(state_repr.encode()).hexdigest()
        
        # Se é um estado novo
        if state_hash in self.state.novelty_scores:
            novelty = self.state.novelty_scores[state_hash]
            if novelty > self.config.novelty_threshold:
                return self.config.novelty_reward * novelty
        
        return 0.0
    
    def _calculate_secret_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por descobrir segredos"""
        reward = 0.0
        x_pos = info.get('x_pos', self.state.last_x_pos)
        y_pos = info.get('y_pos', self.state.last_y_pos)
        
        # Verificar áreas secretas
        for area in self.secret_areas:
            x_min, x_max, y_min, y_max, name = area
            if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
                secret_key = f"{name}_{x_pos}_{y_pos}"
                if secret_key in self.state.discovered_secrets:
                    # Acabou de descobrir
                    reward += self.config.secret_area_reward
                    break  # Apenas uma recompensa por passo
        
        # Verificar power-ups
        status = info.get('status', self.state.last_status)
        if status != self.state.last_status:
            if status == 'tall' and self.state.last_status == 'small':
                reward += self.config.powerup_reward
            elif status == 'fireball' and self.state.last_status in ['small', 'tall']:
                reward += self.config.powerup_reward * 1.5  # Flor é mais valiosa
        
        # Verificar moedas escondidas
        coins = info.get('coins', self.state.last_coins)
        if coins > self.state.last_coins:
            coins_collected = coins - self.state.last_coins
            if coins_collected >= 3:
                reward += coins_collected * self.config.hidden_coin_reward
        
        return reward
    
    def _calculate_path_diversity_reward(self, info: Dict[str, Any]) -> float:
        """Calcula recompensa por diversidade de caminhos"""
        # Verificar se o caminho atual é diferente dos anteriores
        if len(self.state.path_history) >= 2:
            current_path = self.state.current_path
            for prev_path in self.state.path_history[-3:]:  # Últimos 3 caminhos
                similarity = self._calculate_path_similarity(current_path, prev_path)
                if similarity < self.config.path_similarity_threshold:
                    return self.config.path_diversity_reward
        
        return 0.0
    
    def _calculate_intrinsic_reward(self, info: Dict[str, Any], action: int) -> float:
        """Calcula recompensa por curiosidade intrínseca"""
        # Implementação simples: recompensa por ações de exploração
        # Ações que tipicamente levam à exploração
        exploration_actions = [1, 2, 4, 7]  # up, down, A (jump), A,A,A,right (climb)
        
        if action in exploration_actions:
            return self.config.intrinsic_reward_scale
        
        return 0.0
    
    def _apply_exploration_decay(self, reward: float) -> float:
        """Aplica decaimento à recompensa de exploração"""
        # Decaimento com base no progresso do episódio
        decay_factor = self.config.exploration_decay ** self.state.episode_step
        return reward * decay_factor
    
    def _normalize_reward(self, reward: float) -> float:
        """Normaliza a recompensa"""
        # Limitar recompensa máxima
        if abs(reward) > self.config.max_exploration_reward:
            reward = np.sign(reward) * self.config.max_exploration_reward
        
        # Garantir recompensa mínima
        if abs(reward) < self.config.min_exploration_reward and reward != 0:
            reward = np.sign(reward) * self.config.min_exploration_reward
        
        return reward
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de exploração"""
        return {
            'total_new_cells': self.state.total_new_cells,
            'total_secret_areas': self.state.total_secret_areas,
            'total_hidden_coins': self.state.total_hidden_coins,
            'total_powerups': self.state.total_powerups,
            'visited_cells_count': len(self.state.visited_cells),
            'discovered_secrets_count': len(self.state.discovered_secrets),
            'path_history_count': len(self.state.path_history),
            'episode_exploration_reward': self.episode_exploration_reward,
            'episode_total_reward': self.episode_total_reward
        }
    
    def get_exploration_breakdown(self) -> Dict[str, float]:
        """Retorna o breakdown das recompensas de exploração do último passo"""
        if not self.episode_rewards:
            return {}
        
        last_reward = self.episode_rewards[-1]
        return {
            'base_reward': last_reward.get('base_reward', 0),
            'exploration_reward': last_reward.get('exploration_reward', 0),
            'total_reward': last_reward.get('total_reward', 0),
            'new_cells': self.state.total_new_cells,
            'secret_areas': self.state.total_secret_areas
        }
    
    def visualize_visitation_grid(self, width: int = 50, height: int = 30) -> np.ndarray:
        """
        Gera uma visualização do grid de visitação.
        
        Args:
            width: Largura do grid em células
            height: Altura do grid em células
            
        Returns:
            Matriz numpy com o grid de visitação (0 = não visitado, 1 = visitado)
        """
        grid = np.zeros((height, width))
        
        for cell in self.state.visited_cells:
            x, y = cell
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = 1
        
        return grid


class SecretRewardWrapper(gym.Wrapper):
    """
    Wrapper especializado em recompensas por segredos descobertos.
    
    Este wrapper usa conhecimento prévio sobre os segredos do Mario
    para fornecer recompensas generosas quando o agente os descobre.
    """
    
    def __init__(self, env, config: Optional[ExplorationConfig] = None):
        """
        Inicializa o wrapper de recompensas por segredos.
        
        Args:
            env: Ambiente base
            config: Configuração de exploração
        """
        super().__init__(env)
        self.config = config or ExplorationConfig()
        self.state = ExplorationState()
        
        # Segredos conhecidos do Mario (por nível)
        self.known_secrets = self._load_known_secrets()
        self.discovered_secrets = set()
        
    def _load_known_secrets(self) -> Dict[str, List[Dict]]:
        """Carrega segredos conhecidos do Mario"""
        # Formato: {level: [{'type': 'hidden_block', 'x': 100, 'y': 50, 'reward': 10.0}, ...]}
        return {
            "1-1": [
                {'type': 'hidden_block', 'x': 64, 'y': 80, 'reward': 10.0, 'name': 'bloco_secreto_1'},
                {'type': 'pipe_secret', 'x': 160, 'y': 112, 'reward': 20.0, 'name': 'cano_secreto_1'},
                {'type': 'high_path', 'x': 250, 'y': 32, 'reward': 15.0, 'name': 'caminho_alto_1'},
                {'type': 'coin_heap', 'x': 350, 'y': 64, 'reward': 5.0, 'name': 'monte_moedas_1'},
            ],
            "1-2": [
                {'type': 'hidden_block', 'x': 80, 'y': 48, 'reward': 10.0, 'name': 'bloco_secreto_2'},
                {'type': 'pipe_secret', 'x': 200, 'y': 112, 'reward': 25.0, 'name': 'cano_secreto_2'},
            ],
            # Adicionar mais níveis conforme necessário
        }
    
    def reset(self, **kwargs):
        """Reset do ambiente"""
        obs, info = self.env.reset(**kwargs)
        self.state = ExplorationState()
        self.discovered_secrets = set()
        
        # Determinar nível atual
        world = info.get('world', 1)
        stage = info.get('stage', 1)
        self.current_level = f"{world}-{stage}"
        
        return obs, info
    
    def step(self, action):
        """Executa uma ação com recompensas por segredos"""
        obs, base_reward, done, trunc, info = self.env.step(action)
        
        # Verificar segredos
        secret_reward = self._check_secrets(info)
        
        total_reward = base_reward + secret_reward
        
        # Adicionar ao info
        info['secret_reward'] = secret_reward
        info['total_reward'] = total_reward
        
        return obs, total_reward, done, trunc, info
    
    def _check_secrets(self, info: Dict[str, Any]) -> float:
        """Verifica se o agente descobriu algum segredo"""
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)
        
        # Obter segredos do nível atual
        level_secrets = self.known_secrets.get(self.current_level, [])
        
        reward = 0.0
        for secret in level_secrets:
            secret_key = f"{secret['name']}_{self.current_level}"
            
            # Verificar se está na área do segredo
            if secret_key not in self.discovered_secrets:
                # Condições específicas para cada tipo de segredo
                if secret['type'] == 'hidden_block':
                    # Verificar se está na posição do bloco secreto
                    if (abs(x_pos - secret['x']) < 16 and 
                        abs(y_pos - secret['y']) < 16 and
                        info.get('status', '') != self.state.last_status):
                        # Provavelmente bateu no bloco
                        self.discovered_secrets.add(secret_key)
                        reward += secret['reward']
                        
                elif secret['type'] == 'pipe_secret':
                    # Verificar se está na posição do cano secreto
                    if abs(x_pos - secret['x']) < 24 and abs(y_pos - secret['y']) < 32:
                        self.discovered_secrets.add(secret_key)
                        reward += secret['reward']
                        
                elif secret['type'] == 'high_path':
                    # Verificar se está no caminho alto
                    if y_pos < 48 and abs(x_pos - secret['x']) < 32:
                        self.discovered_secrets.add(secret_key)
                        reward += secret['reward']
                        
                elif secret['type'] == 'coin_heap':
                    # Verificar se coletou moedas na área
                    coins = info.get('coins', self.state.last_coins)
                    if coins > self.state.last_coins and abs(x_pos - secret['x']) < 32:
                        self.discovered_secrets.add(secret_key)
                        reward += secret['reward']
        
        self.state.last_coins = info.get('coins', self.state.last_coins)
        self.state.last_status = info.get('status', self.state.last_status)
        
        return reward


class CompositeExplorationWrapper(gym.Wrapper):
    """
    Wrapper que combina exploração + segredos + recompensas padrão.
    """
    
    def __init__(self, env, exploration_config=None, secret_config=None):
        """
        Inicializa o wrapper composto.
        
        Args:
            env: Ambiente base
            exploration_config: Configuração de exploração
            secret_config: Configuração de segredos
        """
        super().__init__(env)
        
        # Criar wrappers
        self.exploration_wrapper = ExplorationRewardWrapper(env, exploration_config)
        self.secret_wrapper = SecretRewardWrapper(env, secret_config)
        
    def reset(self, **kwargs):
        """Reset de todos os wrappers"""
        obs, info = self.env.reset(**kwargs)
        self.exploration_wrapper.reset(**kwargs)
        self.secret_wrapper.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Aplica todos os wrappers"""
        # Primeiro aplicar exploração
        obs, exp_reward, done, trunc, info = self.exploration_wrapper.step(action)
        
        # Depois aplicar segredos
        _, secret_reward, done, trunc, info = self.secret_wrapper.step(action)
        
        # Combinar recompensas
        total_reward = exp_reward + secret_reward
        
        # Adicionar ao info
        info['exploration_reward'] = exp_reward
        info['secret_reward'] = secret_reward
        info['total_reward'] = total_reward
        
        return obs, total_reward, done, trunc, info


def create_exploration_wrapper(
    env,
    use_exploration: bool = True,
    use_secrets: bool = True,
    exploration_config: Optional[ExplorationConfig] = None,
    secret_config: Optional[ExplorationConfig] = None
) -> gym.Wrapper:
    """
    Função fábrica para criar o wrapper de exploração.
    
    Args:
        env: Ambiente base
        use_exploration: Se usar exploração
        use_secrets: Se usar detecção de segredos
        exploration_config: Configuração de exploração
        secret_config: Configuração de segredos
        
    Returns:
        Ambiente com wrappers de exploração aplicados
    """
    if use_exploration and use_secrets:
        return CompositeExplorationWrapper(env, exploration_config, secret_config)
    elif use_exploration:
        return ExplorationRewardWrapper(env, exploration_config)
    elif use_secrets:
        return SecretRewardWrapper(env, secret_config)
    else:
        return env

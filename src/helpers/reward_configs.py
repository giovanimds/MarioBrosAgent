"""
Configurações predefinidas para o sistema de recompensas do Mario.

Este módulo fornece configurações otimizadas para diferentes objetivos de treinamento.
"""

from dataclasses import dataclass
from src.env_manager.reward_wrapper import RewardConfig
from src.env_manager.exploration_wrapper import ExplorationConfig


@dataclass
class RewardPreset:
    """Preset de configuração de recompensas"""
    name: str
    description: str
    config: RewardConfig


# Configuração para aprendizado rápido (foco em progresso)
FAST_LEARNING_CONFIG = RewardConfig(
    # Recompensas de progresso - alto peso
    x_pos_reward=0.2,
    coin_reward=0.5,
    flag_reward=100.0,
    
    # Recompensas de estado
    grow_reward=3.0,
    fire_reward=5.0,
    death_penalty=-15.0,
    
    # Recompensas de tempo - penalidade forte por tempo
    time_penalty=-0.05,
    time_bonus=2.0,
    
    # Recompensas de exploração - baixo peso
    jump_reward=0.02,
    run_reward=0.01,
    
    # Penalidades
    backward_penalty=-0.1,
    stuck_penalty=-0.2,
    
    # Normalização
    max_reward=15.0,
    stuck_threshold=15
)

# Configuração para exploração (foco em descobrir o ambiente)
EXPLORATION_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.05,
    coin_reward=2.0,
    flag_reward=50.0,
    
    # Recompensas de estado
    grow_reward=8.0,
    fire_reward=15.0,
    death_penalty=-5.0,
    
    # Recompensas de tempo
    time_penalty=-0.005,
    time_bonus=0.5,
    
    # Recompensas de exploração - alto peso
    jump_reward=0.1,
    run_reward=0.05,
    
    # Penalidades - leves
    backward_penalty=-0.02,
    stuck_penalty=-0.05,
    
    # Normalização
    max_reward=10.0,
    stuck_threshold=30
)

# Configuração balanceada (padrão recomendado)
BALANCED_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.1,
    coin_reward=1.0,
    flag_reward=50.0,
    
    # Recompensas de estado
    grow_reward=5.0,
    fire_reward=10.0,
    death_penalty=-10.0,
    
    # Recompensas de tempo
    time_penalty=-0.01,
    time_bonus=1.0,
    
    # Recompensas de exploração
    jump_reward=0.05,
    run_reward=0.02,
    
    # Penalidades
    backward_penalty=-0.05,
    stuck_penalty=-0.1,
    
    # Normalização
    max_reward=10.0,
    stuck_threshold=20
)

# Configuração para sobrevivência (foco em não morrer)
SURVIVAL_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.05,
    coin_reward=0.5,
    flag_reward=50.0,
    
    # Recompensas de estado
    grow_reward=10.0,
    fire_reward=20.0,
    death_penalty=-50.0,  # Penalidade muito alta por morrer
    
    # Recompensas de tempo
    time_penalty=-0.001,
    time_bonus=0.5,
    
    # Recompensas de exploração
    jump_reward=0.03,
    run_reward=0.01,
    
    # Penalidades
    backward_penalty=-0.02,
    stuck_penalty=-0.05,
    
    # Normalização
    max_reward=20.0,
    stuck_threshold=25
)

# Configuração para coleta de moedas (foco em pontuação)
COIN_COLLECTOR_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.05,
    coin_reward=5.0,  # Alto peso para moedas
    flag_reward=50.0,
    
    # Recompensas de estado
    grow_reward=3.0,
    fire_reward=5.0,
    death_penalty=-10.0,
    
    # Recompensas de tempo
    time_penalty=-0.005,
    time_bonus=0.5,
    
    # Recompensas de exploração
    jump_reward=0.05,  # Pular para pegar moedas altas
    run_reward=0.02,
    
    # Penalidades
    backward_penalty=-0.03,
    stuck_penalty=-0.08,
    
    # Normalização
    max_reward=15.0,
    stuck_threshold=20
)

# Configuração para speedrun (completar rápido)
SPEEDRUN_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.3,  # Alto peso para progresso rápido
    coin_reward=0.1,   # Baixo peso para moedas
    flag_reward=100.0,
    
    # Recompensas de estado
    grow_reward=2.0,
    fire_reward=3.0,
    death_penalty=-20.0,
    
    # Recompensas de tempo - penalidade muito forte
    time_penalty=-0.1,
    time_bonus=5.0,
    
    # Recompensas de exploração
    jump_reward=0.02,
    run_reward=0.05,  # Alto peso para correr
    
    # Penalidades
    backward_penalty=-0.2,
    stuck_penalty=-0.3,
    
    # Normalização
    max_reward=20.0,
    stuck_threshold=10
)

# Configuração com curriculum learning ativo
CURRICULUM_CONFIG = RewardConfig(
    # Recompensas de progresso
    x_pos_reward=0.1,
    coin_reward=1.0,
    flag_reward=50.0,
    
    # Recompensas de estado
    grow_reward=5.0,
    fire_reward=10.0,
    death_penalty=-10.0,
    
    # Recompensas de tempo
    time_penalty=-0.01,
    time_bonus=1.0,
    
    # Recompensas de exploração
    jump_reward=0.05,
    run_reward=0.02,
    
    # Penalidades
    backward_penalty=-0.05,
    stuck_penalty=-0.1,
    
    # Normalização
    max_reward=10.0,
    stuck_threshold=20,
    
    # Curriculum learning
    use_curriculum=True,
    level_progression={
        1: 1.0,   # Mundo 1 - peso normal
        2: 1.5,   # Mundo 2 - 50% mais recompensa
        3: 2.0,   # Mundo 3 - 100% mais recompensa
        4: 2.5,   # Mundo 4 - 150% mais recompensa
        5: 3.0,   # Mundo 5 - 200% mais recompensa
        6: 3.5,   # Mundo 6 - 250% mais recompensa
        7: 4.0,   # Mundo 7 - 300% mais recompensa
        8: 5.0,   # Mundo 8 - 400% mais recompensa
    }
)

# Configurações para Exploração Máxima (ExplorationConfig)
# Estas configurações são para o ExplorationRewardWrapper

EXPLORATION_MAX_CONFIG = ExplorationConfig(
    # Recompensas por exploração - ALTÍSSIMAS
    new_cell_reward=1.0,          # Recompensa por visitar célula nova
    secret_area_reward=25.0,     # Recompensa por área secreta (MUITO ALTA!)
    hidden_coin_reward=5.0,      # Recompensa por moeda escondida
    powerup_reward=20.0,         # Recompensa por power-up
    
    # Recompensas por novelty
    novelty_reward=0.5,          # Recompensa por estado novo
    novelty_decay=0.995,        # Decaimento lento da novelty
    
    # Recompensas por rotas secretas
    secret_path_reward=50.0,     # Recompensa por descobrir rota secreta (MUITO ALTA!)
    path_diversity_reward=10.0,  # Recompensa por diversidade de caminhos
    
    # Parâmetros de detecção
    grid_cell_size=8,            # Grid mais fino para detecção precisa
    min_secret_area_size=2,      # Áreas secretas menores
    novelty_threshold=0.05,     # Threshold mais baixo para novelty
    
    # Parâmetros de rotas secretas
    min_path_length=5,           # Caminhos mais curtos também contam
    path_similarity_threshold=0.7,  # Similaridade mais baixa para considerar diferente
    
    # Recompensas por curiosidade intrínseca
    intrinsic_reward_scale=0.2,  # Mais recompensa por curiosidade
    prediction_error_reward=0.5, # Recompensa por erro de previsão
    
    # Parâmetros de decaimento
    exploration_decay=0.9995,    # Decaimento muito lento
    min_exploration_reward=0.05, # Recompensa mínima mais alta
    
    # Normalização
    max_exploration_reward=50.0,  # Limite máximo alto
    
    # Tudo ativo
    enable_visitation_grid=True,
    enable_novelty_detection=True,
    enable_secret_detection=True,
    enable_path_diversity=True,
    enable_intrinsic_motivation=True
)

# Configuração para descoberta de segredos
SECRET_HUNTER_CONFIG = ExplorationConfig(
    # Recompensas por exploração
    new_cell_reward=0.3,
    secret_area_reward=50.0,     # Recompensa ENORME por área secreta!
    hidden_coin_reward=10.0,     # Recompensa alta por moedas escondidas
    powerup_reward=30.0,         # Recompensa alta por power-ups
    
    # Recompensas por novelty
    novelty_reward=0.2,
    novelty_decay=0.99,
    
    # Recompensas por rotas secretas
    secret_path_reward=100.0,    # Recompensa EXTREMA por rota secreta!
    path_diversity_reward=5.0,
    
    # Parâmetros de detecção
    grid_cell_size=16,
    min_secret_area_size=3,
    novelty_threshold=0.1,
    
    # Parâmetros de rotas secretas
    min_path_length=10,
    path_similarity_threshold=0.8,
    
    # Recompensas por curiosidade intrínseca
    intrinsic_reward_scale=0.1,
    prediction_error_reward=0.2,
    
    # Parâmetros de decaimento
    exploration_decay=0.999,
    min_exploration_reward=0.01,
    
    # Normalização
    max_exploration_reward=100.0,  # Limite máximo muito alto
    
    # Tudo ativo
    enable_visitation_grid=True,
    enable_novelty_detection=True,
    enable_secret_detection=True,
    enable_path_diversity=True,
    enable_intrinsic_motivation=True
)

# Lista de todos os presets disponíveis
REWARD_PRESETS = {
    'fast_learning': RewardPreset(
        name='Fast Learning',
        description='Foco em progresso rápido, ideal para aprendizado inicial',
        config=FAST_LEARNING_CONFIG
    ),
    'exploration': RewardPreset(
        name='Exploration',
        description='Foco em explorar o ambiente e descobrir novas áreas',
        config=EXPLORATION_CONFIG
    ),
    'balanced': RewardPreset(
        name='Balanced',
        description='Configuração balanceada, recomendada para uso geral',
        config=BALANCED_CONFIG
    ),
    'survival': RewardPreset(
        name='Survival',
        description='Foco em sobreviver e evitar mortes',
        config=SURVIVAL_CONFIG
    ),
    'coin_collector': RewardPreset(
        name='Coin Collector',
        description='Foco em coletar moedas e maximizar pontuação',
        config=COIN_COLLECTOR_CONFIG
    ),
    'speedrun': RewardPreset(
        name='Speedrun',
        description='Foco em completar o nível o mais rápido possível',
        config=SPEEDRUN_CONFIG
    ),
    'curriculum': RewardPreset(
        name='Curriculum Learning',
        description='Recompensas escalonadas por nível de dificuldade',
        config=CURRICULUM_CONFIG
    )
}

# Presets para Exploração (ExplorationConfig)
EXPLORATION_PRESETS = {
    'exploration_max': {
        'name': 'Exploration Max',
        'description': 'Exploração MÁXIMA! Recompensas generosas por descobrir áreas novas, segredos e rotas alternativas',
        'config': EXPLORATION_MAX_CONFIG
    },
    'secret_hunter': {
        'name': 'Secret Hunter',
        'description': 'Caçador de segredos! Recompensas ENORMES por descobrir áreas secretas, moedas escondidas e power-ups',
        'config': SECRET_HUNTER_CONFIG
    }
}


def get_reward_preset(name: str) -> RewardConfig:
    """
    Retorna a configuração de recompensas com base no nome do preset.
    
    Args:
        name: Nome do preset ('fast_learning', 'exploration', 'balanced', etc.)
        
    Returns:
        Configuração de recompensas
    """
    preset = REWARD_PRESETS.get(name)
    if preset:
        return preset.config
    else:
        # Retornar configuração balanceada como padrão
        return BALANCED_CONFIG


def get_exploration_preset(name: str) -> ExplorationConfig:
    """
    Retorna a configuração de exploração com base no nome do preset.
    
    Args:
        name: Nome do preset ('exploration_max', 'secret_hunter')
        
    Returns:
        Configuração de exploração
    """
    preset = EXPLORATION_PRESETS.get(name)
    if preset:
        return preset['config']
    else:
        # Retornar configuração de exploração máxima como padrão
        return EXPLORATION_MAX_CONFIG


def list_reward_presets() -> list:
    """Retorna a lista de todos os presets disponíveis (RewardConfig)"""
    return list(REWARD_PRESETS.keys())


def list_exploration_presets() -> list:
    """Retorna a lista de todos os presets de exploração disponíveis"""
    return list(EXPLORATION_PRESETS.keys())


def get_preset_info(name: str) -> dict:
    """Retorna informações sobre um preset específico (RewardConfig)"""
    preset = REWARD_PRESETS.get(name)
    if preset:
        return {
            'name': preset.name,
            'description': preset.description,
            'config': preset.config,
            'type': 'RewardConfig'
        }
    return {}


def get_exploration_preset_info(name: str) -> dict:
    """Retorna informações sobre um preset de exploração específico"""
    preset = EXPLORATION_PRESETS.get(name)
    if preset:
        return {
            'name': preset['name'],
            'description': preset['description'],
            'config': preset['config'],
            'type': 'ExplorationConfig'
        }
    return {}


def list_all_presets() -> dict:
    """Retorna todos os presets disponíveis (RewardConfig e ExplorationConfig)"""
    return {
        'reward_presets': list_reward_presets(),
        'exploration_presets': list_exploration_presets()
    }

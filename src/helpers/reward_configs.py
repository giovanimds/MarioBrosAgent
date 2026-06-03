"""
Configurações predefinidas para o sistema de recompensas do Mario.

Este módulo fornece configurações otimizadas para diferentes objetivos de treinamento.
"""

from dataclasses import dataclass
from src.env_manager.reward_wrapper import RewardConfig


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


def list_reward_presets() -> list:
    """Retorna a lista de todos os presets disponíveis"""
    return list(REWARD_PRESETS.keys())


def get_preset_info(name: str) -> dict:
    """Retorna informações sobre um preset específico"""
    preset = REWARD_PRESETS.get(name)
    if preset:
        return {
            'name': preset.name,
            'description': preset.description,
            'config': preset.config
        }
    return {}

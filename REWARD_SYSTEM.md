# 🎮 Sistema de Recompensas Customizado para MarioBrosAgent

## Visão Geral

Este documento descreve o novo sistema de recompensas implementado para o MarioBrosAgent, que substitui as recompensas padrão do ambiente por um sistema mais sofisticado e direcionado.

## 🎯 Objetivo

O sistema de recompensas customizado tem como objetivo:
- **Acelerar o aprendizado** do agente
- **Incentivar comportamentos desejados** (progresso, coleta de moedas, exploração)
- **Penalizar comportamentos indesejados** (ir para trás, ficar preso)
- **Fornecer feedback mais rico** para o algoritmo de RL

## 📊 Informações Disponíveis do Ambiente

O ambiente `gym-super-mario-bros` fornece as seguintes informações em cada passo:

| Campo | Tipo | Descrição | Valores Possíveis |
|-------|------|-----------|------------------|
| `x_pos` | int | Posição horizontal do Mario | 0 - 3200+ |
| `y_pos` | int | Posição vertical do Mario | 0 - 240 |
| `coins` | int | Número de moedas coletadas | 0 - 100+ |
| `score` | int | Pontuação atual | 0 - 10000+ |
| `time` | int | Tempo restante no nível | 0 - 400 |
| `life` | uint8 | Número de vidas restantes | 0 - 5 |
| `world` | int | Mundo atual | 1 - 8 |
| `stage` | int | Fase atual | 1 - 4 |
| `status` | str | Estado do Mario | 'small', 'tall', 'fireball' |
| `flag_get` | bool | Se pegou a bandeira | True/False |

## 🏗️ Arquitetura do Sistema

### Componentes Principais

```
src/env_manager/
├── environment.py          # Função create_env() atualizada
├── wrappers.py            # Wrappers de observação
├── reward_wrapper.py      # NOVO: Wrappers de recompensa
└── __init__.py

src/helpers/
└── reward_configs.py      # NOVO: Configurações predefinidas
```

### Classes Principais

#### 1. `RewardConfig`
Configuração de pesos para o sistema de recompensas.

```python
from dataclasses import dataclass

@dataclass
class RewardConfig:
    # Recompensas de progresso
    x_pos_reward: float = 0.1      # Por progresso horizontal
    coin_reward: float = 1.0       # Por moeda coletada
    flag_reward: float = 50.0      # Por completar nível
    
    # Recompensas de estado
    grow_reward: float = 5.0       # Por pegar cogumelo
    fire_reward: float = 10.0      # Por pegar flor
    death_penalty: float = -10.0   # Por morrer
    
    # Recompensas de tempo
    time_penalty: float = -0.01    # Por tempo passando
    time_bonus: float = 1.0        # Por tempo restante ao completar
    
    # Recompensas de exploração
    jump_reward: float = 0.05      # Por pular
    run_reward: float = 0.02       # Por correr
    
    # Penalidades
    backward_penalty: float = -0.05 # Por ir para trás
    stuck_penalty: float = -0.1    # Por ficar preso
    
    # Normalização
    max_reward: float = 10.0       # Limite máximo
    stuck_threshold: int = 20      # Passos sem progresso = preso
    
    # Curriculum learning
    use_curriculum: bool = False
    level_progression: Dict[int, float] = {...}
```

#### 2. `MarioRewardWrapper`
Wrapper principal que implementa o sistema de recompensas customizado.

**Funcionalidades:**
- Calcula recompensas com base em progresso, estado, tempo e ações
- Detecta situações de risco (ficar preso, ir para trás)
- Suporta curriculum learning
- Normaliza recompensas
- Fornece estatísticas detalhadas

#### 3. `ProgressiveRewardWrapper`
Wrapper que adiciona recompensas por atingir marcos (milestones) no nível.

#### 4. `CompositeRewardWrapper`
Combina múltiplos wrappers de recompensa.

## 🎛️ Configurações Predefinidas

Fornecemos várias configurações otimizadas para diferentes objetivos:

### 1. **Fast Learning** (`fast_learning`)
- **Foco:** Progresso rápido
- **Ideal para:** Aprendizado inicial
- **Características:** Alto peso para progresso, penalidade forte por tempo

### 2. **Exploration** (`exploration`)
- **Foco:** Explorar o ambiente
- **Ideal para:** Descobrir novas áreas
- **Características:** Alto peso para pular e correr, penalidades leves

### 3. **Balanced** (`balanced`)
- **Foco:** Equilíbrio entre todos os objetivos
- **Ideal para:** Uso geral (recomendado)
- **Características:** Configuração padrão balanceada

### 4. **Survival** (`survival`)
- **Foco:** Evitar mortes
- **Ideal para:** Sobreviver em níveis difíceis
- **Características:** Penalidade muito alta por morrer, alto peso para crescer

### 5. **Coin Collector** (`coin_collector`)
- **Foco:** Coletar moedas
- **Ideal para:** Maximizar pontuação
- **Características:** Alto peso para moedas, recompensa por pular

### 6. **Speedrun** (`speedrun`)
- **Foco:** Completar rápido
- **Ideal para:** Speedruns
- **Características:** Alto peso para progresso, penalidade forte por tempo

### 7. **Curriculum Learning** (`curriculum`)
- **Foco:** Aprendizado progressivo
- **Ideal para:** Treinamento em múltiplos níveis
- **Características:** Recompensas escalonadas por dificuldade

## 🚀 Como Usar

### Uso Básico

```python
from src.env_manager.environment import create_env
from src.env_manager.reward_wrapper import MarioRewardWrapper, RewardConfig

# Criar ambiente com recompensas customizadas (padrão)
env = create_env(
    game_id="SuperMarioBros-v0",
    render_mode=None,
    use_custom_rewards=True  # Ativado por padrão
)
```

### Uso com Configuração Customizada

```python
from src.env_manager.reward_wrapper import RewardConfig

# Criar configuração personalizada
config = RewardConfig(
    x_pos_reward=0.2,
    coin_reward=1.5,
    death_penalty=-20.0,
    jump_reward=0.1
)

# Criar ambiente com configuração customizada
env = create_env(
    game_id="SuperMarioBros-v0",
    use_custom_rewards=True,
    reward_config=config
)
```

### Uso com Presets

```python
from src.helpers.reward_configs import get_reward_preset, list_reward_presets

# Listar presets disponíveis
print(list_reward_presets())
# Output: ['fast_learning', 'exploration', 'balanced', 'survival', 'coin_collector', 'speedrun', 'curriculum']

# Usar um preset
config = get_reward_preset('speedrun')
env = create_env(
    game_id="SuperMarioBros-v0",
    use_custom_rewards=True,
    reward_config=config
)
```

### Uso Avançado (Wrapper Direto)

```python
from src.env_manager.environment import create_env
from src.env_manager.reward_wrapper import MarioRewardWrapper, ProgressiveRewardWrapper

# Criar ambiente base
env = create_env(game_id="SuperMarioBros-v0", use_custom_rewards=False)

# Adicionar wrappers manualmente
env = MarioRewardWrapper(env)
env = ProgressiveRewardWrapper(env, milestones={100: 5.0, 200: 10.0, 300: 15.0})
```

## 📈 Cálculo de Recompensas

A recompensa total é calculada como a soma de vários componentes:

```
Total Reward = 
    Progress Reward (x_pos, coins) +
    State Reward (grow, fire, death) +
    Time Reward (time penalty, time bonus) +
    Action Reward (jump, run) +
    Penalty Reward (backward, stuck) +
    Completion Reward (flag_get) +
    Curriculum Scaling (optional)
```

### 1. Recompensas de Progresso

```python
def _calculate_progress_reward(self, info):
    reward = 0.0
    
    # Progresso horizontal
    progress = info['x_pos'] - self.state.last_x_pos
    if progress > 0:
        reward += progress * self.config.x_pos_reward
        self.state.no_progress_count = 0
    elif progress < 0:
        reward += progress * self.config.backward_penalty
        self.state.no_progress_count += 1
    else:
        self.state.no_progress_count += 1
    
    # Moedas
    coins_collected = info['coins'] - self.state.last_coins
    if coins_collected > 0:
        reward += coins_collected * self.config.coin_reward
    
    return reward
```

### 2. Recompensas de Estado

```python
def _calculate_state_reward(self, info):
    reward = 0.0
    
    # Mudança de status
    if info['status'] != self.state.last_status:
        if info['status'] == 'tall' and self.state.last_status == 'small':
            reward += self.config.grow_reward  # Pegou cogumelo
        elif info['status'] == 'fireball':
            reward += self.config.fire_reward  # Pegou flor
    
    # Perda de vida
    if info['life'] < self.state.last_life:
        reward += self.config.death_penalty * (self.state.last_life - info['life'])
    
    return reward
```

### 3. Recompensas de Tempo

```python
def _calculate_time_reward(self, info, done):
    reward = 0.0
    
    # Penalidade por tempo passando
    time_diff = self.state.last_time - info['time']
    if time_diff > 0:
        reward += time_diff * self.config.time_penalty
    
    # Bônus por tempo restante ao completar
    if done and info.get('flag_get', False):
        remaining_time = info['time']
        reward += remaining_time * self.config.time_bonus / 400.0
    
    return reward
```

### 4. Recompensas por Ações

```python
def _calculate_action_reward(self, action, info):
    reward = 0.0
    
    # Pular (A)
    if action == 4:
        reward += self.config.jump_reward
    
    # Correr (B + right)
    if action == 8:
        reward += self.config.run_reward
    
    return reward
```

### 5. Penalidades

```python
def _calculate_penalty_reward(self, info, action):
    penalty = 0.0
    
    # Ir para trás
    if info['x_pos'] < self.state.last_x_pos and action in [3, 6, 7]:
        penalty += self.config.backward_penalty
    
    # Ficar preso
    if self.state.no_progress_count >= self.config.stuck_threshold:
        penalty += self.config.stuck_penalty
    
    # Tempo esgotado
    if info.get('time', 0) <= 0:
        penalty += self.config.death_penalty * 0.5
    
    return penalty
```

### 6. Recompensa por Completar Nível

```python
def _calculate_completion_reward(self, info):
    reward = 0.0
    
    # Recompensa base
    reward += self.config.flag_reward
    
    # Bônus por tempo restante
    reward += info['time'] * self.config.time_bonus
    
    # Bônus por moedas
    reward += info['coins'] * self.config.coin_reward * 0.5
    
    # Bônus por pontuação
    reward += info['score'] * 0.01
    
    return reward
```

## 🎓 Curriculum Learning

O sistema suporta curriculum learning, onde as recompensas são escalonadas com base no nível atual:

```python
config = RewardConfig(
    use_curriculum=True,
    level_progression={
        1: 1.0,   # Mundo 1 - peso normal
        2: 1.5,   # Mundo 2 - 50% mais recompensa
        3: 2.0,   # Mundo 3 - 100% mais recompensa
        # ...
    }
)
```

O nível atual é calculado como: `level = (world - 1) * 4 + stage`

## 📊 Monitoramento e Debugging

### Obter Estatísticas do Episódio

```python
# Depois de um episódio
stats = reward_wrapper.get_episode_stats()
print(f"Total reward: {stats['total_reward']}")
print(f"Step count: {stats['step_count']}")
```

### Obter Breakdown da Recompensa

```python
# Em qualquer passo
breakdown = reward_wrapper.get_reward_breakdown()
print(breakdown)
# Output: {
#     'base_reward': 0.0,
#     'custom_reward': 0.5,
#     'x_pos': 100,
#     'coins': 2,
#     'time': 350,
#     'status': 'tall',
#     'life': 2
# }
```

## 🔧 Ações Disponíveis

O ambiente usa o seguinte mapeamento de ações:

| Índice | Ação | Descrição |
|--------|------|-----------|
| 0 | `right` | Andar para direita |
| 1 | `up` | Olhar para cima |
| 2 | `down` | Agachar/descer cano |
| 3 | `left` | Andar para esquerda |
| 4 | `A` | Pular |
| 5 | `B` | Atirar (se for fireball) |
| 6 | `[]` | Não fazer nada |
| 7 | `A,A,A,right` | Pular alto para direita (subir canos) |
| 8 | `B,right` | Correr para direita |

## 🎯 Melhores Práticas

### 1. Comece com a configuração balanceada
```python
config = get_reward_preset('balanced')
```

### 2. Ajuste os pesos gradualmente
- Aumente `x_pos_reward` se o agente não está progredindo
- Aumente `coin_reward` se você quer mais coleta de moedas
- Aumente `death_penalty` se o agente está morrendo muito

### 3. Use curriculum learning para múltiplos níveis
```python
config = get_reward_preset('curriculum')
```

### 4. Monitore as recompensas
- Verifique se as recompensas estão dentro de um range razoável
- Ajuste `max_reward` se necessário

### 5. Teste diferentes presets
- Experimente com `fast_learning` para aprendizado inicial
- Use `speedrun` para otimizar tempo
- Use `survival` para níveis difíceis

## 📈 Exemplo de Treinamento

```python
from src.env_manager.environment import create_env
from src.helpers.reward_configs import get_reward_preset
from src.agents.gppo_agent import GPPOMario

# Criar ambiente com recompensas customizadas
config = get_reward_preset('balanced')
env = create_env(
    game_id="SuperMarioBros-v0",
    render_mode=None,
    use_custom_rewards=True,
    reward_config=config
)

# Criar agente
action_dim = env.action_space.n
state_dim = env.observation_space.shape
agent = GPPOMario(state_dim, action_dim, save_dir="checkpoints/gppo")

# Treinamento
for episode in range(1000):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, trunc, info = env.step(action)
        
        # O reward já está customizado!
        episode_reward += reward
        
        agent.cache(obs, next_obs, action, reward, done, info, log_prob, value)
        
        if agent.trajectory_step >= 2048 or done:
            agent.learn()
        
        obs = next_obs
    
    print(f"Episode {episode}: Reward = {episode_reward}")
    
    # Salvar estatísticas
    if hasattr(env.env, 'get_episode_stats'):
        stats = env.env.get_episode_stats()
        print(f"Custom reward breakdown: {stats}")
```

## 🔍 Solução de Problemas

### O agente não está progredindo
- **Solução:** Aumente `x_pos_reward` e `run_reward`
- **Exemplo:** `config.x_pos_reward = 0.3`

### O agente está morrendo muito
- **Solução:** Aumente `death_penalty` e `grow_reward`
- **Exemplo:** `config.death_penalty = -20.0`

### O agente está indo para trás
- **Solução:** Aumente `backward_penalty`
- **Exemplo:** `config.backward_penalty = -0.2`

### O agente está ficando preso
- **Solução:** Aumente `stuck_penalty` e diminua `stuck_threshold`
- **Exemplo:** `config.stuck_penalty = -0.3`, `config.stuck_threshold = 15`

### As recompensas estão muito altas/baixas
- **Solução:** Ajuste `max_reward`
- **Exemplo:** `config.max_reward = 20.0`

## 📚 Referências

- [gym-super-mario-bros Documentation](https://github.com/KautilyaPrashant/gym-super-mario-bros)
- [NES-Py Documentation](https://github.com/KautilyaPrashant/nes-py)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## 🎉 Próximos Passos

1. **Testar diferentes presets** para encontrar o melhor para seu objetivo
2. **Ajustar os pesos** com base no comportamento do agente
3. **Implementar curriculum learning** para treinamento em múltiplos níveis
4. **Adicionar mais métricas** para monitoramento
5. **Integrar com TensorBoard** para visualização das recompensas

---

**Dica:** O sistema de recompensas é altamente configurável. Experimente com diferentes configurações para encontrar a combinação ideal para seu caso de uso!

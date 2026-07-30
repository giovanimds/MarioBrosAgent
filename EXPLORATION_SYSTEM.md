# 🗺️ Sistema de Exploração Máxima para MarioBrosAgent

## Visão Geral

Este documento descreve o **Sistema de Exploração Máxima** implementado para o MarioBrosAgent, projetado especificamente para incentivar o agente a:

1. **🗺️ Explorar áreas não visitadas** do nível
2. **🔍 Descobrir rotas secretas** e caminhos alternativos
3. **💎 Encontrar segredos** (moedas escondidas, power-ups, áreas secretas)
4. **🧠 Ter curiosidade intrínseca** (novelty detection)

## 🎯 Objetivo Principal

> **Fazer o agente AMAR explorar e descobrir segredos, assim como humanos fazem!**

Os humanos têm uma **recompensa intrínseca** enorme ao encontrar caminhos secretos no Mario. Este sistema replica isso com **recompensas generosas** para:
- Descobrir áreas nunca visitadas
- Encontrar rotas alternativas
- Coletar power-ups escondidos
- Achar moedas em locais inusitados

## 🏗️ Arquitetura

```
src/env_manager/
├── environment.py              # create_env() e create_exploration_env()
├── reward_wrapper.py          # Sistema de recompensas padrão
├── exploration_wrapper.py     # NOVO: Sistema de exploração
└── __init__.py

src/helpers/
└── reward_configs.py          # Configurações (inclui ExplorationConfig)
```

## 🎛️ Componentes do Sistema

### 1. **ExplorationRewardWrapper**
Wrapper principal que implementa **5 mecanismos de exploração**:

#### a) **Visitation Grid** (Grid de Visitação)
- Divide o nível em células (default: 16x16 pixels)
- Rastreia quais células foram visitadas
- **Recompensa:** `new_cell_reward` por célula nova (default: 0.5)

#### b) **Novelty Detection** (Detecção de Novidade)
- Detecta estados nunca vistos antes
- Usa hash de estado (posição + status + moedas + tempo)
- **Recompensa:** `novelty_reward` por estado novo (default: 0.3)
- **Decaimento:** `novelty_decay` para estados repetidos (default: 0.99)

#### c) **Secret Area Detection** (Detecção de Áreas Secretas)
- Áreas pré-definidas que não são visitadas em um path direto
- Exemplo: canos secretos, blocos escondidos, caminhos altos
- **Recompensa:** `secret_area_reward` por área secreta (default: 5.0)

#### d) **Path Diversity** (Diversidade de Caminhos)
- Detecta quando o agente toma caminhos diferentes
- Compara similaridade entre caminhos (Jaccard similarity)
- **Recompensa:** `path_diversity_reward` por caminho diferente (default: 2.0)

#### e) **Intrinsic Motivation** (Motivação Intrínseca)
- Recompensa por ações de exploração (pular, olhar para cima, etc.)
- **Recompensa:** `intrinsic_reward_scale` por ação exploratória (default: 0.1)

### 2. **SecretRewardWrapper**
Wrapper especializado em **segredos conhecidos**:

- Base de dados de segredos por nível
- Detecção precisa de quando o agente descobre um segredo
- **Recompensas generosas:** 10.0 - 100.0 por segredo (configurável)

### 3. **CompositeExplorationWrapper**
Combina ambos os wrappers para **exploração máxima**.

## 🎮 Configuração (ExplorationConfig)

```python
from dataclasses import dataclass

@dataclass
class ExplorationConfig:
    # === Recompensas por Exploração ===
    new_cell_reward: float = 0.5        # Por célula nova
    secret_area_reward: float = 5.0     # Por área secreta
    hidden_coin_reward: float = 3.0     # Por moeda escondida
    powerup_reward: float = 10.0        # Por power-up
    
    # === Recompensas por Novelty ===
    novelty_reward: float = 0.3         # Por estado novo
    novelty_decay: float = 0.99        # Decaimento da novelty
    
    # === Recompensas por Rotas Secretas ===
    secret_path_reward: float = 15.0    # Por rota secreta
    path_diversity_reward: float = 2.0  # Por diversidade de caminhos
    
    # === Parâmetros de Detecção ===
    grid_cell_size: int = 16           # Tamanho da célula (pixels)
    min_secret_area_size: int = 3      # Tamanho mínimo de área secreta
    novelty_threshold: float = 0.1     # Threshold para novelty
    min_path_length: int = 10         # Comprimento mínimo de caminho
    path_similarity_threshold: float = 0.8  # Similaridade para caminhos iguais
    
    # === Recompensas por Curiosidade ===
    intrinsic_reward_scale: float = 0.1
    prediction_error_reward: float = 0.2
    
    # === Parâmetros de Decaimento ===
    exploration_decay: float = 0.999    # Decaimento por passo
    min_exploration_reward: float = 0.01
    
    # === Normalização ===
    max_exploration_reward: float = 20.0
    
    # === Ativação ===
    enable_visitation_grid: bool = True
    enable_novelty_detection: bool = True
    enable_secret_detection: bool = True
    enable_path_diversity: bool = True
    enable_intrinsic_motivation: bool = True
```

## 🚀 Presets de Exploração

### 1. **Exploration Max** (`exploration_max`)
**O preset definitivo para exploração máxima!**

```python
EXPLORATION_MAX_CONFIG = ExplorationConfig(
    new_cell_reward=1.0,          # Recompensa DOBRADA por célula nova
    secret_area_reward=25.0,     # Recompensa 5x MAIOR por área secreta!
    hidden_coin_reward=5.0,      # Recompensa alta por moedas escondidas
    powerup_reward=20.0,         # Recompensa alta por power-ups
    
    novelty_reward=0.5,          # Mais recompensa por novelty
    novelty_decay=0.995,        # Decaimento mais lento
    
    secret_path_reward=50.0,     # Recompensa ENORME por rota secreta!
    path_diversity_reward=10.0,  # Recompensa alta por diversidade
    
    grid_cell_size=8,            # Grid mais fino (mais células = mais recompensas)
    min_secret_area_size=2,      # Detectar áreas secretas menores
    novelty_threshold=0.05,     # Mais sensível a novelty
    
    intrinsic_reward_scale=0.2,  # Mais recompensa por curiosidade
    
    exploration_decay=0.9995,    # Decaimento MUITO lento
    min_exploration_reward=0.05, # Recompensa mínima mais alta
    
    max_exploration_reward=50.0,  # Limite alto
    
    # TUDO ATIVADO
    enable_visitation_grid=True,
    enable_novelty_detection=True,
    enable_secret_detection=True,
    enable_path_diversity=True,
    enable_intrinsic_motivation=True
)
```

**🎯 Objetivo:** Fazer o agente **explorar cada centímetro** do nível!

### 2. **Secret Hunter** (`secret_hunter`)
**Para caçar segredos especificamente!**

```python
SECRET_HUNTER_CONFIG = ExplorationConfig(
    new_cell_reward=0.3,
    secret_area_reward=50.0,     # Recompensa ENORME por área secreta!
    hidden_coin_reward=10.0,     # Recompensa ALTA por moedas escondidas
    powerup_reward=30.0,         # Recompensa ALTA por power-ups
    
    secret_path_reward=100.0,    # Recompensa EXTREMA por rota secreta!
    path_diversity_reward=5.0,
    
    max_exploration_reward=100.0,  # Limite MUITO alto
    
    # TUDO ATIVADO
    enable_visitation_grid=True,
    enable_novelty_detection=True,
    enable_secret_detection=True,
    enable_path_diversity=True,
    enable_intrinsic_motivation=True
)
```

**🎯 Objetivo:** Fazer o agente **encontrar TODOS os segredos** do nível!

## 🚀 Como Usar

### **Opcão 1: Usar `create_exploration_env()` (Recomendado)**

```python
from src.env_manager.environment import create_exploration_env

# Criar ambiente otimizado para exploração
env = create_exploration_env(
    game_id="SuperMarioBros-v0",
    render_mode=None,
    use_custom_rewards=True,
    reward_preset='balanced',      # ou 'exploration', 'fast_learning', etc.
    use_exploration=True,
    exploration_preset='exploration_max'  # ou 'secret_hunter'
)
```

### **Opcão 2: Configuração Manual**

```python
from src.env_manager.environment import create_env
from src.env_manager.exploration_wrapper import ExplorationRewardWrapper, ExplorationConfig
from src.helpers.reward_configs import get_exploration_preset

# Obter configuração
config = get_exploration_preset('exploration_max')

# Criar ambiente base
env = create_env(
    game_id="SuperMarioBros-v0",
    render_mode=None,
    use_custom_rewards=True,
    use_exploration=False  # Desativar para adicionar manualmente
)

# Adicionar wrapper de exploração
env = ExplorationRewardWrapper(env, config=config)
```

### **Opcão 3: Usar com GPPO**

```python
from src.env_manager.environment import create_exploration_env
from src.agents.gppo_agent import GPPOMario

# Criar ambiente de exploração
env = create_exploration_env(
    game_id="SuperMarioBros-v0",
    render_mode=None,
    exploration_preset='exploration_max'
)

# Criar agente
state_dim = env.observation_space.shape
action_dim = env.action_space.n
agent = GPPOMario(state_dim, action_dim, save_dir="checkpoints/exploration")

# Treinar
for episode in range(10000):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, trunc, info = env.step(action)
        
        # A recompensa já inclui exploração!
        agent.cache(obs, next_obs, action, reward, done, info, log_prob, value)
        
        if agent.trajectory_step >= 2048 or done:
            agent.learn()
        
        obs = next_obs
    
    # Ver estatísticas de exploração
    if hasattr(env.env, 'get_exploration_stats'):
        stats = env.env.get_exploration_stats()
        print(f"Episódio {episode}: {stats['total_new_cells']} células novas, {stats['total_secret_areas']} segredos")
```

## 📊 Segredos Conhecidos do Mario

O sistema já vem com uma **base de dados de segredos** para os níveis do Mario:

### **Mario 1-1 (World 1, Stage 1)**
```python
{
    'bloco_secreto_1': {'type': 'hidden_block', 'x': 64, 'y': 80, 'reward': 10.0},
    'cano_secreto_1': {'type': 'pipe_secret', 'x': 160, 'y': 112, 'reward': 20.0},
    'caminho_alto_1': {'type': 'high_path', 'x': 250, 'y': 32, 'reward': 15.0},
    'monte_moedas_1': {'type': 'coin_heap', 'x': 350, 'y': 64, 'reward': 5.0},
}
```

### **Mario 1-2 (World 1, Stage 2)**
```python
{
    'bloco_secreto_2': {'type': 'hidden_block', 'x': 80, 'y': 48, 'reward': 10.0},
    'cano_secreto_2': {'type': 'pipe_secret', 'x': 200, 'y': 112, 'reward': 25.0},
}
```

**💡 Dica:** Você pode **adicionar mais segredos** editando o arquivo `exploration_wrapper.py`!

## 🎯 Como o Agente Aprende a Explorar

### **Fase 1: Exploração Inicial (0-100 episódios)**
- O agente recebe **recompensas altas** por visitar células novas
- Descobre que **ir para frente não é suficiente**
- Começa a **pular e explorar verticalmente**

### **Fase 2: Descoberta de Segredos (100-500 episódios)**
- O agente encontra **primeiros segredos** (blocos escondidos)
- Recebe **recompensas generosas** (10.0-25.0)
- Aprende que **explorar vale a pena**

### **Fase 3: Exploração Estratégica (500-2000 episódios)**
- O agente **prioriza áreas não visitadas**
- Descobre **rotas alternativas**
- Encontra **caminhos secretos** (recompensa: 50.0-100.0)

### **Fase 4: Master Explorer (2000+ episódios)**
- O agente **explora sistematicamente** todo o nível
- Encontra **todos os segredos**
- **Repete rotas secretas** porque são recompensadoras

## 📈 Exemplo de Recompensas

Com o preset **`exploration_max`**:

```
Step 0-50:   reward=0.5-1.0 por célula nova (explorando área inicial)
Step 51:     reward=10.0  (encontrou bloco secreto!)
Step 100:    reward=25.0  (entrou em área secreta!)
Step 150:    reward=5.0   (coletou moedas escondidas)
Step 200:    reward=20.0  (pegou cogumelo em bloco secreto)
Step 250:    reward=50.0  (descobriu rota secreta!)
Step 300:    reward=0.5-1.0 (explorando nova área)
Step 350:    reward=15.0  (encontrou caminho alto)
...
Total:       reward=200-500+ por episódio (com exploração)
```

## 🎓 Melhores Práticas

### **1. Comece com `exploration_max`**
```python
env = create_exploration_env(exploration_preset='exploration_max')
```

### **2. Ajuste as recompensas**
- **Aumente** `secret_area_reward` se o agente não está encontrando segredos
- **Aumente** `new_cell_reward` se o agente não está explorando
- **Diminua** `exploration_decay` para manter a exploração por mais tempo

### **3. Use com Recompensas Customizadas**
```python
env = create_exploration_env(
    use_custom_rewards=True,
    reward_preset='exploration',  # ou 'balanced'
    exploration_preset='exploration_max'
)
```

### **4. Monitore o Progresso**
```python
# Depois de um episódio
stats = env.env.get_exploration_stats()
print(f"Células novas: {stats['total_new_cells']}")
print(f"Segredos encontrados: {stats['total_secret_areas']}")
print(f"Power-ups: {stats['total_powerups']}")
print(f"Recompensa de exploração: {stats['episode_exploration_reward']:.2f}")
```

### **5. Visualize o Grid de Visitação**
```python
# Obter grid de visitação
grid = env.env.visualize_visitation_grid(width=50, height=30)

# Visualizar com matplotlib
import matplotlib.pyplot as plt
plt.imshow(grid, cmap='hot')
plt.title('Grid de Visitação')
plt.colorbar()
plt.show()
```

## 🔧 Solução de Problemas

### **O agente não está explorando**
- **Solução:** Aumente `new_cell_reward` e `intrinsic_reward_scale`
- **Exemplo:** `config.new_cell_reward = 2.0`

### **O agente não está encontrando segredos**
- **Solução:** Aumente `secret_area_reward` e `powerup_reward`
- **Exemplo:** `config.secret_area_reward = 50.0`

### **O agente está explorando, mas não está progredindo**
- **Solução:** Ajuste o balanceamento entre exploração e progresso
- **Exemplo:** Use `reward_preset='balanced'` + `exploration_preset='exploration_max'`

### **As recompensas estão muito altas**
- **Solução:** Diminua `max_exploration_reward` ou ajuste os valores individuais
- **Exemplo:** `config.max_exploration_reward = 30.0`

### **O agente está repetindo o mesmo caminho**
- **Solução:** Aumente `path_diversity_reward` e diminua `path_similarity_threshold`
- **Exemplo:** `config.path_diversity_reward = 15.0`, `config.path_similarity_threshold = 0.6`

## 📚 Adicionando Novos Segredos

Você pode **adicionar segredos personalizados** para qualquer nível:

```python
from src.env_manager.exploration_wrapper import SecretRewardWrapper

# Criar wrapper com segredos personalizados
class CustomSecretWrapper(SecretRewardWrapper):
    def _load_known_secrets(self):
        secrets = super()._load_known_secrets()
        
        # Adicionar segredos do Mario 1-3
        secrets["1-3"] = [
            {'type': 'hidden_block', 'x': 100, 'y': 64, 'reward': 15.0, 'name': 'meu_segredo_1'},
            {'type': 'pipe_secret', 'x': 250, 'y': 96, 'reward': 30.0, 'name': 'cano_escondido'},
        ]
        
        return secrets

# Usar
env = CustomSecretWrapper(env)
```

## 🎉 Exemplo Completo: Treinamento para Exploração

```python
from src.env_manager.environment import create_exploration_env
from src.agents.gppo_agent import GPPOMario
from src.helpers.logger import MetricLogger

# Criar ambiente de exploração máxima
env = create_exploration_env(
    game_id="SuperMarioBros-v0",
    render_mode=None,
    use_custom_rewards=True,
    reward_preset='exploration',
    use_exploration=True,
    exploration_preset='exploration_max'
)

# Criar agente
action_dim = env.action_space.n
state_dim = env.observation_space.shape
agent = GPPOMario(state_dim, action_dim, save_dir="checkpoints/explorer")

# Criar logger
logger = MetricLogger()
logger.start_live_display()

# Treinar
for episode in range(10000):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Agente age
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, trunc, info = env.step(action)
        
        episode_reward += reward
        
        # Armazenar experiência
        agent.cache(obs, next_obs, action, reward, done, info, log_prob, value)
        
        # Aprender
        if agent.trajectory_step >= 2048 or done:
            agent.learn()
        
        obs = next_obs
    
    # Logar progresso
    if episode % 10 == 0:
        stats = env.env.get_exploration_stats()
        print(f"Episódio {episode}: Reward={episode_reward:.2f}, "
              f"Células={stats['total_new_cells']}, "
              f"Segredos={stats['total_secret_areas']}, "
              f"Power-ups={stats['total_powerups']}")
    
    # Salvar agente
    if episode % 100 == 0:
        agent.save()

logger.stop_live_display()
env.close()
```

## 🎯 O que Esperar

Com o sistema de exploração ativo, você deve observar:

1. **🎮 Comportamento Curioso:** O agente **pula, olha para cima, vai para os lados**
2. **🗺️ Cobertura Completa:** O agente visita **quase todas as células** do nível
3. **🔍 Descoberta de Segredos:** O agente **encontra blocos escondidos, canos secretos, caminhos altos**
4. **💎 Coleta de Power-ups:** O agente **prioriza cogumelos, flores e estrelas**
5. **🔄 Rotas Alternativas:** O agente **tenta caminhos diferentes** em cada episódio

## 📈 Métricas de Sucesso

| Métrica | Valor Alvo | Descrição |
|---------|------------|-----------|
| `total_new_cells` | 100+ | Células novas descobertas por episódio |
| `total_secret_areas` | 3+ | Áreas secretas encontradas por episódio |
| `total_powerups` | 1+ | Power-ups coletados por episódio |
| `episode_exploration_reward` | 50+ | Recompensa de exploração por episódio |
| `visited_cells_count` | 500+ | Total de células únicas visitadas |

## 🎓 Por que Isso Funciona

### **1. Recompensas Intrínsecas**
- O agente recebe **recompensa por explorar**, não apenas por progresso
- Isso replica a **curiosidade humana**

### **2. Recompensas Generosas para Segredos**
- Encontrar um segredo dá **10-100x mais recompensa** do que progresso normal
- O agente **aprende que explorar vale a pena**

### **3. Decaimento Lento**
- As recompensas de exploração **decaiem lentamente**
- O agente continua explorando **mesmo após muitos episódios**

### **4. Diversidade de Caminhos**
- O agente é recompensado por **tentar caminhos diferentes**
- Isso evita que ele **fique preso em um único path**

## 🌟 Conclusão

Este sistema **transforma o agente de um simples "go-forward" para um verdadeiro explorador**!

Com as configurações certas, o agente:
- ✅ **Explora cada cantinho** do nível
- ✅ **Descobre segredos** que humanos encontrariam
- ✅ **Repete rotas secretas** porque são recompensadoras
- ✅ **Tem curiosidade intrínseca** como um jogador humano

**🎮 O auge do RL: Um agente que AMAR explorar!**

---

**Próximos passos:**
1. Treinar o agente com `exploration_max`
2. Observar o comportamento de exploração
3. Ajustar os pesos conforme necessário
4. Adicionar mais segredos para outros níveis
5. Integrar com visualização (TensorBoard, matplotlib)

# Guia de Reestruturação do Projeto MarioBrosAgent

Este documento descreve a nova estrutura de pastas e arquivos, indicando onde mover as classes e funções existentes para facilitar a organização, manutenção e evolução do código.

## 1. Estrutura Proposta

```
MarioBrosAgent/
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── train.py            # Loop de treino principal
└── src/
    ├── agents/         # Definição de agentes, modelos e gerenciadores
    │   ├── __init__.py
    │   ├── agent.py            # Classe Agent
    │   ├── manager.py          # AgentManager
    │   ├── model.py            # MarioNet, SelfAttentionLayer e camadas MoE
    │   └── utils.py            # Funções auxiliares específicas de agente
    ├── env_manager/     # Wrappers, configuração e inicialização do ambiente
    │   ├── __init__.py
    │   ├── wrappers.py        # SkipFrame, GrayScaleObservation, ResizeObservation
    │   └── environment.py     # Função de criação e configuração do gym_super_mario_bros / retro
    └── helpers/        # Funções utilitárias gerais
        ├── __init__.py
        ├── config.py           # Constantes (NUM_AGENTS, caminhos de checkpoint, etc.)
        ├── logger.py           # Configuração de logs (rich, matplotlib setups)
        └── io.py               # Funções de salvar/carregar estado (pickle, checkpoints)
```


## 2. Mapeamento de arquivos e classes existentes

- **agent.py** (origem: `agent_manager.py` e possível `agent.py` atual)
  - Mover classe `Agent` para `src/agents/agent.py`.

- **manager.py**
  - Daquele mesmo `agent_manager.py`, extrair `AgentManager` e migrar para `src/agents/manager.py`.

- **model.py**
  - No `main.py` e `mario_world.py`, as classes `MarioNet`, `SelfAttentionLayer` e camadas MoE devem ser consolidadas em `src/agents/model.py`.

- **wrappers.py**
  - Mover as classes `SkipFrame`, `GrayScaleObservation` e `ResizeObservation` presentes em `main.py` e `mario_world.py` para `src/env_manager/wrappers.py`.

- **environment.py**
  - Criar função `create_env()` que encapsula:
    - importação e configuração de `gym_super_mario_bros` ou `retro`;
    - aplicação de `JoypadSpace`;
    - aplicação dos wrappers.
  - Colocar em `src/env_manager/environment.py`.

- **config.py**
  - Definir constantes como `NUM_AGENTS`, caminhos de checkpoints, hiperparâmetros, em `src/helpers/config.py`.

- **logger.py** e **io.py**
  - Funções de logging com `rich` ou `matplotlib` para visualização de progresso em `src/helpers/logger.py`.
  - Funções de salvar/carregar estado e checkpoints em `src/helpers/io.py`.

- **train.py** (novo)
  - Criar na raiz: `train.py` será o script principal que:
    1. Carrega configurações de `helpers/config.py`.
    2. Cria ambiente com `env_manager`.
    3. Instancia `AgentManager` e `MarioNet` (via `agents/`).
    4. Executa loop de treino e avaliação.
    5. Salva checkpoints e logs via `helpers/io.py`.


## 3. Passos de migração

1. Criar pastas: `src/agents`, `src/env_manager`, `src/helpers`.
2. Mover e adaptar arquivos conforme mapeamento.
3. Atualizar imports relativos no código.
4. Garantir que `pyproject.toml` ou `setup.py` inclua `src/` como pacote.
5. Testar execução com `python train.py`.

---
*Este guia deve servir como referência para reorganizar o código, melhorar a escalabilidade e facilitar a manutenção.*

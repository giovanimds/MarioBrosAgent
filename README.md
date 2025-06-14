# 🎮 Mario Bros Agent

Um agente de IA para jogar Super Mario Bros usando Deep Reinforcement Learning com PyTorch.

## 🚀 Desenvolvimento no GitHub Codespaces

Este projeto está configurado para rodar perfeitamente no GitHub Codespaces com todos os ambientes pré-configurados!

### Início Rápido com Codespaces
1. Clique em "Code" > "Codespaces" > "Create codespace on main"
2. Aguarde o ambiente ser configurado automaticamente
3. Execute: `source activate_env.sh` para ativar o ambiente
4. Execute: `python main.py` para treinar o agente!

## 🛠️ Configuração Local com UV

### Pré-requisitos
- Python 3.8+
- UV package manager

### Instalação
```bash
# Instalar UV (se não estiver instalado)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar o repositório
git clone <seu-repo>
cd MarioBrosAgent

# Criar ambiente virtual
uv venv

# Ativar ambiente virtual
source .venv/bin/activate

# Instalar dependências
uv pip install -e .
```

## 📦 Dependências Principais

- **PyTorch** - Framework de deep learning
- **Gym** - Ambiente de reinforcement learning
- **gym-super-mario-bros** - Ambiente específico do Mario
- **nes-py** - Emulador NES
- **OpenCV** - Processamento de imagem
- **NumPy** - Computação numérica
- **Matplotlib** - Visualização

## 🎯 Como Usar

### Treinamento do Agente
```bash
python main.py
```

### Execução do Mundo Mario
```bash
python mario_world.py
```

### Gerenciamento de Agentes
```bash
python agent_manager.py
```

## 🏗️ Estrutura do Projeto

```
MarioBrosAgent/
├── main.py                 # Script principal para treinamento
├── mario_world.py           # Ambiente do Mario customizado
├── agent_manager.py         # Gerenciamento de agentes
├── pyproject.toml          # Configuração UV/Python
├── requirements.txt        # Dependências (compatibilidade)
├── checkpoints/            # Modelos salvos
├── .devcontainer/          # Configuração DevContainer
│   ├── devcontainer.json   # Configuração principal
│   ├── Dockerfile          # Imagem customizada
│   └── setup.sh            # Script de configuração
└── README.md               # Este arquivo
```

## 🧪 Testando o Ambiente

Para verificar se tudo está funcionando:

```bash
# No DevContainer
bash .devcontainer/test_environment.sh

# Localmente
python -c "
import torch
import gym_super_mario_bros
print('✅ Ambiente configurado corretamente!')
"
```

## 🎮 Recursos do Agente

- **Deep Q-Network (DQN)** para tomada de decisões
- **Experience Replay** para aprendizado eficiente
- **Target Network** para estabilidade de treinamento
- **Epsilon-greedy** para exploração/exploração
- **Checkpoints automáticos** para salvar progresso

## 📊 Monitoramento

O projeto inclui:
- Logs de treinamento detalhados
- Salvamento automático de checkpoints
- Métricas de desempenho
- Visualização do progresso

## 🤖 Configurações do Agente

Principais hiperparâmetros configuráveis:
- Taxa de aprendizado
- Batch size
- Epsilon decay
- Frequência de salvamento
- Arquitetura da rede neural

## 🐛 Solução de Problemas

### Problemas Comuns

1. **Erro de import**: Verifique se o ambiente virtual está ativado
2. **Problemas de display**: Configure `DISPLAY=:0` se necessário
3. **Falta de memória**: Reduza o batch size ou use CPU

### Logs e Debug

Os logs são salvos automaticamente e incluem:
- Progresso do treinamento
- Métricas de desempenho
- Erros e exceções

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🎯 Próximos Passos

- [ ] Implementar PPO (Proximal Policy Optimization)
- [ ] Adicionar support para múltiplos níveis
- [ ] Integrar TensorBoard para visualização
- [ ] Implementar treinamento distribuído
- [ ] Adicionar interface web para monitoramento

---

**Divirta-se treinando o Mario! 🍄👾**

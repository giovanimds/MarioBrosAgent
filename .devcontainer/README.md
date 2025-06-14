# Mario Bros Agent - DevContainer

Este projeto está configurado para rodar em GitHub Codespaces ou qualquer ambiente que suporte DevContainers.

## 🚀 Início Rápido

### GitHub Codespaces
1. Abra este repositório no GitHub
2. Clique em "Code" > "Codespaces" > "Create codespace on main"
3. Aguarde o container ser criado e configurado automaticamente
4. O ambiente estará pronto para uso!

### VS Code + Docker Local
1. Instale a extensão "Dev Containers" no VS Code
2. Abra o projeto no VS Code
3. Pressione `Ctrl+Shift+P` e selecione "Dev Containers: Reopen in Container"
4. Aguarde a construção do container

## 🛠️ Configuração do Ambiente

O DevContainer inclui:

- **Python 3.12** com ambiente virtual UV
- **PyTorch** para deep learning
- **Gym** e **gym-super-mario-bros** para reinforcement learning
- **OpenCV** para processamento de imagem
- **Jupyter** para experimentação
- **Extensions do VS Code** pré-configuradas

## 📦 Dependências Instaladas

- torch
- torch-directml
- torchvision
- pytorch-optimizer
- pillow
- numpy
- gym
- nes-py
- gym-super-mario-bros
- matplotlib
- tqdm
- opencv-python

## 🎮 Como Usar

1. **Ativar o ambiente virtual:**
   ```bash
   source activate_env.sh
   ```

2. **Executar o agente:**
   ```bash
   python main.py
   ```

3. **Executar o mundo do Mario:**
   ```bash
   python mario_world.py
   ```

## 🔧 Comandos Úteis

- `uv --version` - Verificar versão do UV
- `uv pip list` - Listar pacotes instalados
- `uv pip install <package>` - Instalar novo pacote
- `python -c "import torch; print(torch.__version__)"` - Testar PyTorch

## 🐛 Solução de Problemas

### Problema com Display (Para GUI)
```bash
export DISPLAY=:0
```

### Reinstalar Dependências
```bash
uv pip install -e . --force-reinstall
```

### Verificar Imports
```bash
python -c "
import torch
import numpy as np
import gym
import nes_py
print('✅ Todos os imports funcionando!')
"
```

## 📊 Portas Expostas

- **8888**: Jupyter Notebook
- **6006**: TensorBoard

## 🏗️ Estrutura do DevContainer

```
.devcontainer/
├── devcontainer.json    # Configuração principal
├── Dockerfile          # Imagem customizada
├── setup.sh            # Script de configuração
└── README.md           # Este arquivo
```

## 🤝 Contribuindo

O ambiente está configurado para desenvolvimento com:
- Formatação automática com Black
- Linting com Flake8
- Type checking com MyPy
- Extensões do VS Code pré-configuradas

Divirta-se treinando o Mario! 🎮🍄

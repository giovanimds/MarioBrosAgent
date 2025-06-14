#!/bin/bash

set -e

echo "🚀 Configurando ambiente Mario Bros Agent..."

# Adicionar uv ao PATH
export PATH="$HOME/.local/bin:$PATH"

# Verificar se uv está instalado
if ! command -v uv &> /dev/null; then
    echo "❌ UV não encontrado, instalando..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ UV instalado: $(uv --version)"

# Criar ambiente virtual com uv
echo "🔧 Criando ambiente virtual..."
uv venv

# Ativar ambiente virtual e instalar dependências
echo "📦 Instalando dependências..."
source .venv/bin/activate

# Instalar dependências do pyproject.toml
uv pip install -e .

# Instalar dependências adicionais para desenvolvimento
uv pip install matplotlib opencv-python tqdm

# Verificar instalação
echo "🧪 Testando imports..."
python -c "
import torch
import numpy as np
import gym
import nes_py
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
print('✅ Todas as dependências principais importadas com sucesso!')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Gym version: {gym.__version__}')
"

# Criar script de ativação rápida
cat > activate_env.sh << 'EOF'
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
echo "🎮 Ambiente Mario Bros Agent ativado!"
echo "Python: $(which python)"
echo "UV: $(uv --version)"
EOF

chmod +x activate_env.sh

echo "🎯 Setup completo!"
echo "💡 Para ativar o ambiente: source activate_env.sh"
echo "🎮 Pronto para treinar o Mario!"

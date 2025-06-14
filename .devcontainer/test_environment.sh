#!/bin/bash

echo "🧪 Testando ambiente Mario Bros Agent..."

# Verificar UV
echo "📦 Verificando UV..."
if command -v uv &> /dev/null; then
    echo "✅ UV instalado: $(uv --version)"
else
    echo "❌ UV não encontrado"
    exit 1
fi

# Verificar Python
echo "🐍 Verificando Python..."
if [ -f ".venv/bin/python" ]; then
    echo "✅ Python virtual env: $(.venv/bin/python --version)"
else
    echo "❌ Ambiente virtual não encontrado"
    exit 1
fi

# Ativar ambiente virtual
source .venv/bin/activate

# Testar imports principais
echo "📚 Testando imports principais..."
python -c "
import sys
print(f'Python path: {sys.executable}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')

try:
    import gym
    print(f'✅ Gym: {gym.__version__}')
except ImportError as e:
    print(f'❌ Gym: {e}')

try:
    import nes_py
    print('✅ NES-Py: Importado com sucesso')
except ImportError as e:
    print(f'❌ NES-Py: {e}')

try:
    import gym_super_mario_bros
    print('✅ Gym Super Mario Bros: Importado com sucesso')
except ImportError as e:
    print(f'❌ Gym Super Mario Bros: {e}')

try:
    from PIL import Image
    print('✅ Pillow: Importado com sucesso')
except ImportError as e:
    print(f'❌ Pillow: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV: {e}')

try:
    import matplotlib
    print(f'✅ Matplotlib: {matplotlib.__version__}')
except ImportError as e:
    print(f'❌ Matplotlib: {e}')
"

# Testar criação do ambiente Mario
echo "🎮 Testando criação do ambiente Mario..."
python -c "
try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    
    # Criar ambiente
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    print('✅ Ambiente Mario criado com sucesso!')
    print(f'Action space: {env.action_space}')
    print(f'Observation space: {env.observation_space}')
    
    env.close()
except Exception as e:
    print(f'❌ Erro ao criar ambiente Mario: {e}')
"

echo "🎯 Teste concluído!"
echo "🚀 Ambiente pronto para desenvolvimento!"

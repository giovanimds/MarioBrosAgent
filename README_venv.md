# Mario Bros Agent - Ambiente Virtual

Este projeto agora está configurado com um ambiente virtual Python usando **uv**.

## 🚀 Como usar o ambiente virtual

### 1. Ativar o ambiente virtual
```bash
# Opção 1: Usando o script .env
source .env

# Opção 2: Ativação manual
export PATH="$HOME/snap/code/194/.local/share/../bin:$PATH"
source .venv/bin/activate
```

### 2. Verificar se está ativo
Quando o ambiente virtual estiver ativo, você verá `(.venv)` no início do prompt do terminal.

### 3. Executar o projeto
```bash
# Com o ambiente virtual ativo
python main.py
# ou
python mario_world.py
```

### 4. Desativar o ambiente virtual
```bash
deactivate
```

## 📦 Dependências instaladas

O projeto inclui todas as dependências necessárias:
- **PyTorch** (torch, torchvision, torch-directml)
- **Gym** e **gym-super-mario-bros** para o ambiente de jogo
- **nes-py** para emulação do NES
- **pytorch-optimizer** para otimizadores avançados
- **numpy**, **pillow** para processamento de dados
- **tqdm** para barras de progresso

## 🔧 Gerenciamento de dependências

### Adicionar nova dependência
```bash
source .env
uv pip install nome-do-pacote
```

### Atualizar requirements.txt
```bash
source .env
uv pip freeze > requirements.txt
```

### Instalar dependências em outro ambiente
```bash
uv pip install -r requirements.txt
```

## 📁 Arquivos do ambiente virtual

- `.venv/` - Diretório do ambiente virtual
- `pyproject.toml` - Configuração do projeto com uv
- `requirements.txt` - Lista de dependências (compatibilidade)
- `.env` - Script para ativar o ambiente virtual

## 🎮 Executar o agente Mario

```bash
# Ativar ambiente
source .env

# Executar treinamento
python main.py

# Executar com checkpoint existente
python mario_world.py
```

Agora seu projeto Mario Bros Agent está pronto para usar com um ambiente virtual isolado! 🎉

# Simple DevContainer Dockerfile using UV and pyproject.toml
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Instalar UV (Astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | bash

# Adicionar UV ao PATH
ENV PATH="/root/.local/bin:${PATH}"

# Definir diretório de trabalho
WORKDIR /workspace

# Copiar arquivos do projeto
COPY . /workspace

# Comando padrão para manter o container ativo até o VS Code anexar
CMD ["sleep", "infinity"]

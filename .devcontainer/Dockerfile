# Use uma imagem base Python oficial
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Instalar dependências do sistema para pygame, OpenCV e outras bibliotecas
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libjpeg-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libgtk-3-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Definir variáveis de ambiente para display (para Codespaces)
ENV DISPLAY=:0
ENV SDL_VIDEODRIVER=x11

# Criar diretório de trabalho
WORKDIR /workspaces/MarioBrosAgent

# Copiar arquivos de configuração
COPY pyproject.toml ./

# Criar e ativar ambiente virtual, instalar dependências
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Configurar o ambiente virtual como padrão
ENV VIRTUAL_ENV=/workspaces/MarioBrosAgent/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expor portas para Jupyter e TensorBoard
EXPOSE 8888 6006

# Comando padrão
CMD ["/bin/bash"]

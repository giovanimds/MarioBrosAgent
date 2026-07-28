"""
State Space Model (SSM) Layer - Implementação simplificada

Esta camada substitui a SelfAttentionLayer por um modelo de espaço de estados
que é mais eficiente para processamento sequencial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleStateSpaceModel(nn.Module):
    """
    Implementação simplificada de State Space Model.
    
    Processa sequências usando:
    - Convolução 1D para extração de features locais
    - Evolução do estado oculto através de matrizes A, B, C
    - Saída linear do estado
    """
    def __init__(self, d_model, d_state=16, d_conv=3):
        """
        Inicializa o modelo de espaço de estados.
        
        Args:
            d_model: Dimensão do modelo (dimensão de entrada/saída)
            d_state: Dimensão do estado oculto
            d_conv: Tamanho do kernel de convolução
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Camada de convolução para processamento local
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=False
        )
        
        # Parâmetros do espaço de estados
        # A: matriz de evolução [d_state, d_state]
        # B: matriz de entrada [d_state, d_model]
        # C: matriz de saída [d_model, d_state]
        # D: skip connection [d_model]
        
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Normalização
        self.norm = nn.LayerNorm(d_model)
        
        # Inicialização dos parâmetros
        self._init_parameters()
        
    def _init_parameters(self):
        """Inicialização cuidadosa dos parâmetros."""
        # Inicialização para matrizes A, B, C
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.uniform_(self.D, -0.1, 0.1)
        
        # Garantir estabilidade
        with torch.no_grad():
            self.A.data = self.A.data * 0.5
            
        # Inicialização para convolução
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        
    def forward(self, x):
        """
        Forward pass do SSM.
        
        Args:
            x: Tensor de entrada [batch, seq_len, d_model]
            
        Returns:
            Tensor de saída [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # Normalizar
        x = self.norm(x)
        
        # Convolução 1D para processamento local
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, d_model]
        
        # Processamento sequencial com estado oculto
        # h: [batch, d_state]
        h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        y = []
        
        for i in range(seq_len):
            # Atualizar estado: h = A * h + B * x_i
            # A: [d_state, d_state], h: [batch, d_state]
            # B: [d_state, d_model], x_i: [batch, d_model]
            h = F.silu(self.A) @ h.T + F.silu(self.B) @ x[:, i, :].T
            h = h.T
            
            # Saída: y_i = x_i * C @ h + D * x_i
            # C: [d_model, d_state], h: [batch, d_state]
            y_i = x[:, i, :] * (F.silu(self.C) @ h.T).T + self.D.unsqueeze(0) * x[:, i, :]
            y.append(y_i)
            
        y = torch.stack(y, dim=1)  # [batch, seq_len, d_model]
        
        return y


class SSMLayer(nn.Module):
    """
    Camada SSM completa para substituição da SelfAttentionLayer.
    
    Esta camada processa tokens visuais usando State Space Model,
    que é mais eficiente que a atenção para sequências longas.
    """
    def __init__(self, input_channels, height, width, output_dim, d_state=16, num_layers=2):
        """
        Inicializa a camada SSM.
        
        Args:
            input_channels: Número de canais de entrada
            height: Altura da imagem
            width: Largura da imagem
            output_dim: Dimensão de saída
            d_state: Dimensão do estado oculto
            num_layers: Número de camadas SSM
        """
        super().__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.output_dim = output_dim
        self.d_state = d_state
        self.num_layers = num_layers
        
        # Extrair features com convoluções (mesma arquitetura do modelo original)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calcular dimensão real após CNN
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, height, width)
            feature_dim = self.feature_extractor(dummy).shape[1]
        
        # Camadas SSM
        self.ssm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.ssm_layers.append(
                SimpleStateSpaceModel(d_model=feature_dim, d_state=d_state, d_conv=3)
            )
            self.ssm_layers.append(nn.LayerNorm(feature_dim))
            self.ssm_layers.append(nn.SiLU())
        
        # Projeção final
        self.out_project = nn.Linear(feature_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass da camada SSM.
        
        Args:
            x: Tensor de entrada [batch, channels, height, width]
            
        Returns:
            Tensor de saída [batch, output_dim]
        """
        batch = x.shape[0]
        
        # Extrair features
        features = self.feature_extractor(x)  # [batch, feature_dim]
        
        # Adicionar dimensão de sequência
        features = features.unsqueeze(1)  # [batch, 1, feature_dim]
        
        # Processar através das camadas SSM
        for layer in self.ssm_layers:
            if isinstance(layer, SimpleStateSpaceModel):
                features = layer(features)
            else:
                # LayerNorm, SiLU
                features = layer(features)
        
        # Remover dimensão de sequência e projetar
        features = features.squeeze(1)  # [batch, feature_dim]
        out = self.out_project(features)
        
        return out


class EfficientSSMLayer(nn.Module):
    """
    Versão otimizada do SSM para processamento de imagens.
    """
    def __init__(self, input_channels, height, width, output_dim, d_state=8):
        """
        Inicializa a camada SSM eficiente.
        """
        super().__init__()
        
        # Extrair features com convoluções
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Calcular dimensão após convoluções
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, height, width)
            features = self.feature_extractor(dummy)
            feature_dim = features.view(1, -1).shape[1]
        
        # SSM para processamento
        self.ssm = SimpleStateSpaceModel(d_model=feature_dim, d_state=d_state, d_conv=3)
        
        # Projeção final
        self.out_project = nn.Linear(feature_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass.
        """
        batch = x.shape[0]
        
        # Extrair features
        features = self.feature_extractor(x)  # [batch, 64, h//4, w//4]
        
        # Aplanar features
        features = features.view(batch, -1)  # [batch, feature_dim]
        
        # Processar com SSM
        features = features.unsqueeze(1)  # [batch, 1, feature_dim]
        ssm_out = self.ssm(features)
        ssm_out = ssm_out.squeeze(1)  # [batch, feature_dim]
        
        # Projeção final
        out = self.out_project(ssm_out)
        
        return out


if __name__ == "__main__":
    # Teste básico da camada SSM
    print("Testando State Space Model Layer...")
    
    # Teste com entrada 2D (imagem)
    batch_size = 2
    channels = 4
    height = 84
    width = 84
    output_dim = 256
    
    # Criar camada SSM
    ssm_layer = SSMLayer(
        input_channels=channels,
        height=height,
        width=width,
        output_dim=output_dim,
        d_state=16,
        num_layers=2
    )
    
    # Criar entrada de teste
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    out = ssm_layer(x)
    
    print(f"Entrada: {x.shape}")
    print(f"Saída: {out.shape}")
    print(f"Saída esperada: [{batch_size}, {output_dim}]")
    
    assert out.shape == (batch_size, output_dim), f"Shape mismatch: {out.shape}"
    
    # Teste com EfficientSSMLayer
    print("\nTestando Efficient SSM Layer...")
    efficient_ssm = EfficientSSMLayer(
        input_channels=channels,
        height=height,
        width=width,
        output_dim=output_dim,
        d_state=8
    )
    
    out_efficient = efficient_ssm(x)
    print(f"Saída (Efficient): {out_efficient.shape}")
    
    assert out_efficient.shape == (batch_size, output_dim), f"Shape mismatch: {out_efficient.shape}"
    
    print("\n✅ Todos os testes passaram!")

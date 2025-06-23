import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Categorical

class SelfAttentionLayer(nn.Module):
    """
    Camada de auto-atenção para processamento de características.
    Utiliza múltiplas camadas de atenção para capturar relações entre diferentes
    partes da entrada.
    """
    def __init__(self, embed_dim, num_heads, output_dim, num_layers=5):
        """
        Inicializa a camada de auto-atenção.

        Args:
            embed_dim: Dimensão do embedding
            num_heads: Número de cabeças de atenção
            output_dim: Dimensão da saída
            num_layers: Número de camadas de atenção
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attn_weights = None  # Para armazenar os pesos de atenção
        self.input_logits = None  # Para armazenar os logits de entrada

        self.attention_layers = nn.ModuleList()

        for i in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=True
                )
            )

        self.conv1x1 = nn.Conv2d(4, 1, kernel_size=1)  # Preserva 2D
        self.out_project = nn.Linear(embed_dim, output_dim)  # Projeção linear


    def forward(self, x):
        """
        Processa a entrada através das camadas de atenção.

        Args:
            x: Tensor de entrada [batch, channels, height, width]

        Returns:
            Tensor de saída [batch, output_dim]
        """
        batch, channels, height, width = x.shape
        original = x.view(batch, channels, -1)  # [batch, channels, height*width]
        self.input_logits = original.detach().cpu()

        # Corrigindo a forma de entrada
        if x.dim() == 2:  # [batch, features]
            x = x.unsqueeze(1)  # [batch, seq_len=1, features]
        if x.dim() == 4:
            x = x.view(x.size(0), x.size(1), -1)

        for attn in self.attention_layers:
            x, attn_weights = attn(x, x, x)

        self.attn_weights = x.detach().cpu()
        out = self.conv1x1(x.view(batch, channels, height, width)).view(batch, -1)
        out = self.out_project(out)
        return out  # Retorna para [batch, features]


class ExpertNetwork(nn.Module):
    """Rede especialista melhorada com arquitetura mais diversificada"""
    def __init__(self, input_dim, hidden_dim, output_dim, expert_id=None):
        super().__init__()
        self.expert_id = expert_id  # Identificador do especialista
        self.hidden_dim = hidden_dim

        # Arquitetura base com skip connections
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Camadas variadas por especialista para forçar especialização
        # Usamos módulos diferentes para quebrar simetria entre experts
        self.hidden1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Segunda camada com arquitetura diferente
        self.hidden2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Usando SiLU (Swish) como ativação variada
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim//2)
        )

        # Camada de saída
        self.output_layer = nn.Linear(hidden_dim//2, output_dim)

        # Inicialização diversificada para cada especialista
        self._initialize_weights()

        # Inserir alguma não-linearidade específica por especialista
        if expert_id is not None:
            # Para quebrar simetria, adicione pequeno bias específico por expert
            with torch.no_grad():
                # Pequena perturbação determinística baseada no expert_id
                seed_val = (expert_id * 1337) % 10000
                torch.manual_seed(seed_val)
                # Adicionar bias pequeno mas diferente para cada expert
                self.input_layer.bias.add_(torch.randn_like(self.input_layer.bias) * 0.01)
                self.output_layer.bias.add_(torch.randn_like(self.output_layer.bias) * 0.01)

    def _initialize_weights(self):
        """Inicialização especializada para cada camada"""
        # Input layer - inicialização normal
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.constant_(self.input_layer.bias, 0.1)

        # Output layer - inicialização cuidadosa
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

        # Camadas escondidas
        for module in [self.hidden1, self.hidden2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        # Processamento com residual connections
        h0 = F.relu(self.input_layer(x))
        h1 = self.hidden1(h0) + h0  # Residual connection
        h2 = self.hidden2(h1)  # Sem residual aqui
        output = self.output_layer(h2)
        return output


class GatingNetwork(nn.Module):
    """Rede de gating melhorada com mecanismos para promover balanceamento"""
    def __init__(self, input_dim, num_experts, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

        # Rede de gating mais expressiva com skip-connections
        self.input_layer = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.hidden2 = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.output_layer = nn.Linear(128, num_experts)

        # Camada de temperatura para controlar a suavidade da distribuição
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # Inicialização de pesos para distribuição inicial mais uniforme
        self._initialize_weights()

        # Histórico de ativação para monitoramento
        self.activation_history = []
        self.noise_scale = 0.7  # Escala de ruído inicial aumentada
        self.noise_decay = 0.9995  # Decaimento do ruído

    def _initialize_weights(self):
        """Inicialização especializada para distribuição mais uniforme"""
        # Input layer - inicialização cuidadosa para evitar saturação
        nn.init.xavier_uniform_(self.input_layer.weight, gain=0.5)
        nn.init.constant_(self.input_layer.bias, 0.0)

        # Output layer - inicialização especial para começar quase uniforme
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

        # Camadas escondidas
        for module in [self.hidden1, self.hidden2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        # Processamento com skip connections
        h0 = F.relu(self.input_layer(x))
        h1 = self.hidden1(h0) + h0  # Residual connection
        h2 = self.hidden2(h1)

        # Logits de saída com temperatura adaptativa
        gate_logits = self.output_layer(h2) / self.temperature

        # Adicionar ruído durante treinamento para exploração
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_scale
            gate_logits = gate_logits + noise
            # Decair o ruído ao longo do tempo
            self.noise_scale = max(self.noise_scale * self.noise_decay, 0.01)

        # Converter para probabilidades
        gate_probs = F.softmax(gate_logits, dim=1)

        # Selecionar top-k especialistas
        top_k_probs, top_k_indices = torch.topk(gate_probs, min(self.top_k, self.num_experts), dim=1)

        # Renormalizar as probabilidades top-k
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)

        return top_k_probs, top_k_indices, gate_probs


class MoELayer(nn.Module):
    """Camada Mixture of Experts com balanceamento melhorado"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=12, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Criar os especialistas com inicialização diversificada
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])

        # Rede de gating melhorada
        self.gate = GatingNetwork(input_dim, num_experts, top_k)

        # Coeficiente de balanceamento mais alto para forçar melhor distribuição
        self.load_balancing_loss_coef = 0.05

        # Métricas para monitoramento
        self.last_gate_probs = None
        self.expert_usage_history = []

        # Variável para controlar noise dinâmico
        self.noise_scale = 0.2

    def forward(self, x):
        batch_size = x.shape[0]

        # Adicionar ruído aos inputs para forçar especialização
        if self.training:
            # Aplicar noise dependendo do nível de desbalanceamento
            stats = self.get_expert_usage_stats()
            if stats['coefficient_of_variation'] > 0.8:  # Desbalanceamento alto
                self.noise_scale = min(self.noise_scale * 1.05, 0.3)  # Aumenta noise gradualmente
            else:
                self.noise_scale = max(self.noise_scale * 0.98, 0.05)  # Reduz noise gradualmente

            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        # Obter decisões do gate com tokens de routing
        top_k_probs, top_k_indices, gate_probs = self.gate(x)

        # Armazenar para monitoramento
        self.last_gate_probs = gate_probs.detach()

        # Calcular saídas dos especialistas selecionados com processamento em paralelo
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Processar expert por expert (mais eficiente em GPU)
        for expert_idx in range(self.num_experts):
            # Encontrar quais amostras usam este especialista
            batch_indices = []
            for i in range(batch_size):
                if expert_idx in top_k_indices[i]:
                    # Encontrar posição do especialista no top_k
                    pos = (top_k_indices[i] == expert_idx).nonzero(as_tuple=True)[0]
                    batch_indices.append((i, pos.item()))

            if batch_indices:  # Se este especialista está sendo usado
                # Extrair índices e pesos
                indices = [b[0] for b in batch_indices]
                positions = [b[1] for b in batch_indices]

                # Processar entrada apenas para este especialista
                expert_input = x[indices]
                expert_output = self.experts[expert_idx](expert_input)

                # Aplicar pesos e adicionar ao resultado final
                for idx, pos, out in zip(indices, positions, expert_output):
                    weight = top_k_probs[idx, pos]
                    final_output[idx] += weight * out

        # Calcular loss de balanceamento com fator adaptativo
        load_balancing_loss = self._calculate_load_balancing_loss(gate_probs)

        # Adicionar regularização para os especialistas menos utilizados
        expert_usage = gate_probs.mean(dim=0)  # [num_experts]
        if self.training and torch.max(expert_usage) > 0.3:
            underused = expert_usage < 0.05
            if underused.any():
                underused_penalty = (0.05 - expert_usage[underused]).sum() * 0.5
                load_balancing_loss = load_balancing_loss + underused_penalty

        # Agendamento de ruído: reduzir noise_scale quando distribuição for estável
        stats = self.get_expert_usage_stats()
        # Se coeficiente de variação baixo, indica distribuição estável
        if stats['coefficient_of_variation'] < 0.3:
            self.gate.noise_scale = max(self.gate.noise_scale * self.gate.noise_decay, 0.01)

        return final_output, load_balancing_loss, gate_probs

    def _calculate_load_balancing_loss(self, gate_probs):
        """Calcular loss de balanceamento com múltiplas estratégias"""
        # Frequência de uso de cada especialista
        expert_usage = gate_probs.mean(dim=0)  # [num_experts]

        # Cálculo de estatísticas para análise adaptativa
        max_usage = expert_usage.max().item()
        min_usage = expert_usage.min().item()
        std_usage = expert_usage.std().item()
        ideal_usage = 1.0 / self.num_experts

        # 1. Loss de divergência KL da distribuição uniforme (mais forte)
        uniform_prob = torch.full_like(expert_usage, ideal_usage)
        kl_loss = F.kl_div(
            torch.log(expert_usage + 1e-8),
            uniform_prob,
            reduction='sum'
        )

        # 2. Loss de variância quadrática (penaliza especialistas muito usados)
        variance_loss = ((expert_usage - ideal_usage) ** 2).sum()

        # 3. Loss de entropia (incentiva distribuição mais uniforme)
        entropy_loss = -(expert_usage * torch.log(expert_usage + 1e-8)).sum()
        entropy_loss = 1.0 - entropy_loss / math.log(self.num_experts)  # Normalizado para [0,1]

        # Combinação ponderada das losses
        combined_loss = (
            0.5 * kl_loss +
            0.3 * variance_loss +
            0.2 * entropy_loss
        )

        # Ajustar coeficiente com base no desbalanceamento
        coefficient_of_variation = std_usage / (expert_usage.mean().item() + 1e-8)

        # Aplicar o coeficiente de balanceamento
        final_loss = combined_loss * self.load_balancing_loss_coef

        return final_loss

    def get_expert_usage_stats(self):
        """Retorna estatísticas de uso dos especialistas"""
        if self.last_gate_probs is None:
            return {
                'max_usage': 0.0,
                'min_usage': 0.0,
                'std_usage': 0.0,
                'entropy': 0.0,
                'coefficient_of_variation': 0.0,
                'expert_usage': torch.zeros(self.num_experts)
            }

        # Calcular estatísticas de uso
        expert_usage = self.last_gate_probs.mean(dim=0)  # [num_experts]
        max_usage = expert_usage.max().item()
        min_usage = expert_usage.min().item()
        mean_usage = expert_usage.mean().item()
        std_usage = expert_usage.std().item()

        # Entropia normalizada (1.0 = perfeitamente uniforme)
        entropy = -(expert_usage * torch.log(expert_usage + 1e-8)).sum().item()
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy

        # Coeficiente de variação (medida de dispersão relativa)
        coefficient_of_variation = std_usage / (mean_usage + 1e-8)

        return {
            'max_usage': max_usage,
            'min_usage': min_usage,
            'std_usage': std_usage,
            'entropy': normalized_entropy,
            'coefficient_of_variation': coefficient_of_variation,
            'expert_usage': expert_usage.detach().cpu()
        }


class MarioNet(nn.Module):
    """CNN com arquitetura Mixture of Experts (MoE) melhorada - 16 especialistas, top-4"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.device = torch.device('cpu')

        # Número de experts e configurações
        self.num_experts = 10
        self.top_k = 3

        # Extrair features com CNN melhorada
        self.feature_extractor = self._build_cnn_features(c, h, w)

        # Calcular dimensão das features após CNN
        with torch.no_grad():
            dummy_input = torch.randn(1, c, h, w)
            feature_dim = self.feature_extractor(dummy_input).shape[1]

        # Camada MoE com especialistas compactos
        self.moe_layer = MoELayer(
            input_dim=feature_dim,
            hidden_dim=32,
            output_dim=output_dim,
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        # Armazenar loss de balanceamento para treinamento
        self.last_load_balancing_loss = 0.0
        self.last_gate_probs = None

        # Atribuir IDs aos especialistas para inicialização diversificada
        for i, expert in enumerate(self.moe_layer.experts):
            expert.expert_id = i

        # Criar modelos online e target
        self.online = nn.Sequential(self.feature_extractor, self.moe_layer)

        # Rede target com configuração reduzida
        self.target = nn.Sequential(
            self._build_cnn_features(c, h, w),
            MoELayer(
                input_dim=feature_dim,
                hidden_dim=32,
                output_dim=output_dim,
                num_experts=self.num_experts,
                top_k=self.top_k + 1  # Top-3 para suavização leve
            )
        )
        self.target.load_state_dict(self.online.state_dict())

        # Congelar parâmetros do target
        for p in self.target.parameters():
            p.requires_grad = False

    def _build_cnn_features(self, c, h, w):
        """Constrói a rede CNN para extração de features"""
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Arquitetura CNN melhorada com mais canais e skip connections
        cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcular dimensão de saída
        h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)

        # Adicionar camadas finais
        return nn.Sequential(
            cnn,
            nn.Linear(128 * h_out * w_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, input, model):
        """Forward pass com suporte para online e target networks"""
        if model == "online":
            # Extrair features
            features = self.feature_extractor(input)

            # Passar pelo MoE e capturar métricas
            output, load_balancing_loss, gate_probs = self.moe_layer(features)

            # Armazenar para monitoramento
            self.last_load_balancing_loss = load_balancing_loss
            self.last_gate_probs = gate_probs

            return output, load_balancing_loss

        elif model == "target":
            # No modo target, não precisamos da loss de balanceamento
            features = self.target[0](input)  # feature_extractor
            output, _, _ = self.target[1](features)  # moe_layer
            return output

    def adjust_load_balancing_dynamically(self):
        """Ajusta dinamicamente o coeficiente de balanceamento com base no uso"""
        # Obter estatísticas de uso
        stats = self.moe_layer.get_expert_usage_stats()
        cv = stats['coefficient_of_variation']

        # Ajustar coeficiente com base no desbalanceamento
        moe_layer = self.moe_layer
        current_coef = moe_layer.load_balancing_loss_coef

        if cv > 1.2:  # Muito desbalanceado
            # Aumentar coeficiente significativamente
            moe_layer.load_balancing_loss_coef = min(current_coef * 1.5, 0.2)
        elif cv > 0.8:  # Moderadamente desbalanceado
            # Aumentar coeficiente levemente
            moe_layer.load_balancing_loss_coef = min(current_coef * 1.2, 0.1)
        elif cv < 0.3:  # Muito balanceado
            # Reduzir coeficiente para permitir especialização
            moe_layer.load_balancing_loss_coef = max(current_coef * 0.8, 0.01)

        # Garantir que o coeficiente não fique abaixo de um limite mínimo
        min_coef = 0.01
        if moe_layer.load_balancing_loss_coef < min_coef:
            moe_layer.load_balancing_loss_coef = min_coef

    def get_moe_metrics(self):
        """Retorna métricas do MoE para monitoramento"""
        if not hasattr(self.moe_layer, 'get_expert_usage_stats'):
            return None

        stats = self.moe_layer.get_expert_usage_stats()

        # Adicionar loss de balanceamento às métricas
        if hasattr(self, 'last_load_balancing_loss'):
            stats['load_balancing_loss'] = self.last_load_balancing_loss.item() if isinstance(self.last_load_balancing_loss, torch.Tensor) else self.last_load_balancing_loss
            stats['load_balancing_coef'] = self.moe_layer.load_balancing_loss_coef

        return stats

    def get_attention_heatmap(self):
        """Retorna heatmap de atenção para visualização"""
        if not hasattr(self.feature_extractor[0], 'attn_weights'):
            return None

        return self.feature_extractor[0].attn_weights


class ActorCriticMarioNet(nn.Module):
    """
    Rede Actor-Critic para GPPO com arquitetura Mixture of Experts (MoE).
    Compartilha o extrator de características entre as redes de política e valor.
    """
    def __init__(self, input_dim, action_dim):
        super().__init__()
        c, h, w = input_dim
        self.device = torch.device('cpu')
        self.action_dim = action_dim

        # Número de experts e configurações
        self.num_experts = 10
        self.top_k = 3

        # Extrair features com CNN melhorada (compartilhada entre policy e value)
        self.feature_extractor = self._build_cnn_features(c, h, w)

        # Calcular dimensão das features após CNN
        with torch.no_grad():
            dummy_input = torch.randn(1, c, h, w)
            feature_dim = self.feature_extractor(dummy_input).shape[1]

        # Camada MoE para a rede de política (actor)
        self.policy_moe = MoELayer(
            input_dim=feature_dim,
            hidden_dim=32,
            output_dim=action_dim,  # Saída para cada ação possível
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        # Camada MoE para a rede de valor (critic)
        self.value_moe = MoELayer(
            input_dim=feature_dim,
            hidden_dim=32,
            output_dim=1,  # Saída única para o valor do estado
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        # Armazenar loss de balanceamento para treinamento
        self.last_policy_load_balancing_loss = 0.0
        self.last_value_load_balancing_loss = 0.0
        self.last_policy_gate_probs = None
        self.last_value_gate_probs = None

        # Atribuir IDs aos especialistas para inicialização diversificada
        for i, expert in enumerate(self.policy_moe.experts):
            expert.expert_id = i
        for i, expert in enumerate(self.value_moe.experts):
            expert.expert_id = i + self.num_experts  # IDs diferentes para value experts

    def _build_cnn_features(self, c, h, w):
        """Constrói a rede CNN para extração de features"""
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Arquitetura CNN melhorada com mais canais e skip connections
        cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcular dimensão de saída
        h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)

        # Adicionar camadas finais
        return nn.Sequential(
            cnn,
            nn.Linear(128 * h_out * w_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass que retorna a distribuição de política e o valor do estado.

        Args:
            x: Tensor de entrada [batch, channels, height, width]

        Returns:
            policy_dist: Distribuição de probabilidade sobre ações
            value: Valor estimado do estado
        """
        # Extrair features compartilhadas
        features = self.feature_extractor(x)

        # Rede de política (actor)
        policy_logits, policy_load_balancing_loss, policy_gate_probs = self.policy_moe(features)

        # Rede de valor (critic)
        value, value_load_balancing_loss, value_gate_probs = self.value_moe(features)

        # Armazenar para monitoramento
        self.last_policy_load_balancing_loss = policy_load_balancing_loss
        self.last_value_load_balancing_loss = value_load_balancing_loss
        self.last_policy_gate_probs = policy_gate_probs
        self.last_value_gate_probs = value_gate_probs

        # Criar distribuição de probabilidade sobre ações
        policy_dist = Categorical(logits=policy_logits)

        return policy_dist, value.squeeze(-1)

    def get_moe_metrics(self):
        """Retorna métricas de ambas as redes MoE"""
        policy_stats = self.policy_moe.get_expert_usage_stats()
        value_stats = self.value_moe.get_expert_usage_stats()

        combined_stats = {
            'policy': policy_stats,
            'value': value_stats,
            'load_balancing_loss': {
                'policy': self.last_policy_load_balancing_loss.item() if isinstance(self.last_policy_load_balancing_loss, torch.Tensor) else self.last_policy_load_balancing_loss,
                'value': self.last_value_load_balancing_loss.item() if isinstance(self.last_value_load_balancing_loss, torch.Tensor) else self.last_value_load_balancing_loss
            }
        }

        return combined_stats

    def adjust_load_balancing_dynamically(self):
        """Ajusta dinamicamente o coeficiente de balanceamento com base no uso"""
        # Ajustar para a rede de política
        policy_stats = self.policy_moe.get_expert_usage_stats()
        if policy_stats['coefficient_of_variation'] > 0.8:
            # Aumentar coeficiente se desbalanceado
            self.policy_moe.load_balancing_loss_coef = min(
                self.policy_moe.load_balancing_loss_coef * 1.05,
                0.2  # Limite máximo
            )
        elif policy_stats['coefficient_of_variation'] < 0.3:
            # Diminuir coeficiente se bem balanceado
            self.policy_moe.load_balancing_loss_coef = max(
                self.policy_moe.load_balancing_loss_coef * 0.95,
                0.01  # Limite mínimo
            )

        # Ajustar para a rede de valor
        value_stats = self.value_moe.get_expert_usage_stats()
        if value_stats['coefficient_of_variation'] > 0.8:
            # Aumentar coeficiente se desbalanceado
            self.value_moe.load_balancing_loss_coef = min(
                self.value_moe.load_balancing_loss_coef * 1.05,
                0.2  # Limite máximo
            )
        elif value_stats['coefficient_of_variation'] < 0.3:
            # Diminuir coeficiente se bem balanceado
            self.value_moe.load_balancing_loss_coef = max(
                self.value_moe.load_balancing_loss_coef * 0.95,
                0.01  # Limite mínimo
            )

    def get_attention_heatmap(self):
        """Retorna o mapa de calor da atenção para visualização"""
        if hasattr(self.feature_extractor[0], 'attn_weights'):
            return self.feature_extractor[0].attn_weights
        return None

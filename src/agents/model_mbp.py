"""
Modelo MarioNet com arquitetura MBP (Mamba-Based Processing)

Este modelo substitui a camada Transformer/SSM pela arquitetura MBP,
que é mais eficiente para processamento sequencial.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical

from src.agents.mbp_layer import MBPModel, RMSNorm


class ExpertNetworkMBP(nn.Module):
    """Rede especialista para uso com MBP"""
    def __init__(self, input_dim, hidden_dim, output_dim, expert_id=None):
        super().__init__()
        self.expert_id = expert_id
        self.hidden_dim = hidden_dim

        # Arquitetura base
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hidden2 = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        self.output_layer = nn.Linear(hidden_dim//2, output_dim)

        # Inicialização diversificada
        self._initialize_weights()

        if expert_id is not None:
            with torch.no_grad():
                seed_val = (expert_id * 1337) % 10000
                torch.manual_seed(seed_val)
                self.input_layer.bias.add_(torch.randn_like(self.input_layer.bias) * 0.01)
                self.output_layer.bias.add_(torch.randn_like(self.output_layer.bias) * 0.01)

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity="relu")
        nn.init.constant_(self.input_layer.bias, 0.1)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
        for module in [self.hidden1, self.hidden2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        h0 = F.gelu(self.input_layer(x))
        h1 = self.hidden1(h0) + h0
        h2 = self.hidden2(h1)
        output = self.output_layer(h2)
        return output


class GatingNetworkMBP(nn.Module):
    """Rede de gating para MoE com MBP"""
    def __init__(self, input_dim, num_experts, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

        self.input_layer = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Sequential(
            RMSNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.hidden2 = nn.Sequential(
            RMSNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.output_layer = nn.Linear(128, num_experts)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.noise_scale = 0.7
        self.noise_decay = 0.9995
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_layer.weight, gain=0.5)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
        for module in [self.hidden1, self.hidden2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        h0 = F.gelu(self.input_layer(x))
        h1 = self.hidden1(h0) + h0
        h2 = self.hidden2(h1)
        gate_logits = self.output_layer(h2) / self.temperature

        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_scale
            gate_logits = gate_logits + noise
            self.noise_scale = max(self.noise_scale * self.noise_decay, 0.01)

        gate_probs = F.softmax(gate_logits, dim=1)
        top_k_probs, top_k_indices = torch.topk(gate_probs, min(self.top_k, self.num_experts), dim=1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        return top_k_probs, top_k_indices, gate_probs


class MoELayerMBP(nn.Module):
    """Camada Mixture of Experts com MBP"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=10, top_k=3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.experts = nn.ModuleList([
            ExpertNetworkMBP(input_dim, hidden_dim, output_dim, i)
            for i in range(num_experts)
        ])

        self.gate = GatingNetworkMBP(input_dim, num_experts, top_k)
        self.load_balancing_loss_coef = 0.05
        self.last_gate_probs = None
        self.noise_scale = 0.2

    def forward(self, x):
        batch_size = x.shape[0]

        if self.training:
            stats = self.get_expert_usage_stats()
            if stats['coefficient_of_variation'] > 0.8:
                self.noise_scale = min(self.noise_scale * 1.05, 0.3)
            else:
                self.noise_scale = max(self.noise_scale * 0.98, 0.05)

            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        top_k_probs, top_k_indices, gate_probs = self.gate(x)
        self.last_gate_probs = gate_probs.detach()

        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)

        for expert_idx in range(self.num_experts):
            batch_indices = []
            for i in range(batch_size):
                if expert_idx in top_k_indices[i]:
                    pos = (top_k_indices[i] == expert_idx).nonzero(as_tuple=True)[0]
                    batch_indices.append((i, pos.item()))

            if batch_indices:
                indices = [b[0] for b in batch_indices]
                positions = [b[1] for b in batch_indices]
                expert_input = x[indices]
                expert_output = self.experts[expert_idx](expert_input)
                for idx, pos, out in zip(indices, positions, expert_output):
                    weight = top_k_probs[idx, pos]
                    final_output[idx] += weight * out

        load_balancing_loss = self._calculate_load_balancing_loss(gate_probs)
        expert_usage = gate_probs.mean(dim=0)
        if self.training and torch.max(expert_usage) > 0.3:
            underused = expert_usage < 0.05
            if underused.any():
                underused_penalty = (0.05 - expert_usage[underused]).sum() * 0.5
                load_balancing_loss = load_balancing_loss + underused_penalty

        stats = self.get_expert_usage_stats()
        if stats['coefficient_of_variation'] < 0.3:
            self.gate.noise_scale = max(self.gate.noise_scale * self.gate.noise_decay, 0.01)

        return final_output, load_balancing_loss, gate_probs

    def _calculate_load_balancing_loss(self, gate_probs):
        expert_usage = gate_probs.mean(dim=0)
        max_usage = expert_usage.max().item()
        min_usage = expert_usage.min().item()
        std_usage = expert_usage.std().item()
        ideal_usage = 1.0 / self.num_experts

        uniform_prob = torch.full_like(expert_usage, ideal_usage)
        kl_loss = F.kl_div(
            torch.log(expert_usage + 1e-8),
            uniform_prob,
            reduction='sum'
        )
        variance_loss = ((expert_usage - ideal_usage) ** 2).sum()
        entropy_loss = -(expert_usage * torch.log(expert_usage + 1e-8)).sum()
        entropy_loss = 1.0 - entropy_loss / math.log(self.num_experts)

        combined_loss = (
            0.5 * kl_loss +
            0.3 * variance_loss +
            0.2 * entropy_loss
        )
        coefficient_of_variation = std_usage / (expert_usage.mean().item() + 1e-8)
        final_loss = combined_loss * self.load_balancing_loss_coef
        return final_loss

    def get_expert_usage_stats(self):
        if self.last_gate_probs is None:
            return {
                'max_usage': 0.0, 'min_usage': 0.0, 'std_usage': 0.0,
                'entropy': 0.0, 'coefficient_of_variation': 0.0,
                'expert_usage': torch.zeros(self.num_experts)
            }

        expert_usage = self.last_gate_probs.mean(dim=0)
        max_usage = expert_usage.max().item()
        min_usage = expert_usage.min().item()
        mean_usage = expert_usage.mean().item()
        std_usage = expert_usage.std().item()
        entropy = -(expert_usage * torch.log(expert_usage + 1e-8)).sum().item()
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy
        coefficient_of_variation = std_usage / (mean_usage + 1e-8)

        return {
            'max_usage': max_usage, 'min_usage': min_usage, 'std_usage': std_usage,
            'entropy': normalized_entropy, 'coefficient_of_variation': coefficient_of_variation,
            'expert_usage': expert_usage.detach().cpu()
        }


class MarioNetMBP(nn.Module):
    """MarioNet com arquitetura MBP + MoE"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.device = torch.device('cpu')
        self.num_experts = 10
        self.top_k = 3

        # Extrair features com CNN
        self.feature_extractor = self._build_cnn_features(c, h, w)

        # Calcular dimensão das features
        with torch.no_grad():
            dummy_input = torch.randn(1, c, h, w)
            feature_dim = self.feature_extractor(dummy_input).shape[1]

        # Camada MoE
        self.moe_layer = MoELayerMBP(
            input_dim=feature_dim,
            hidden_dim=32,
            output_dim=output_dim,
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        self.last_load_balancing_loss = 0.0
        self.last_gate_probs = None

        for i, expert in enumerate(self.moe_layer.experts):
            expert.expert_id = i

        # Criar modelos online e target
        self.online = nn.Sequential(self.feature_extractor, self.moe_layer)
        self.target = nn.Sequential(
            self._build_cnn_features(c, h, w),
            MoELayerMBP(
                input_dim=feature_dim,
                hidden_dim=32,
                output_dim=output_dim,
                num_experts=self.num_experts,
                top_k=self.top_k + 1
            )
        )
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def _build_cnn_features(self, c, h, w):
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Flatten()
        )

        h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)

        return nn.Sequential(
            cnn,
            nn.Linear(64 * h_out * w_out, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU()
        )

    def forward(self, input, model):
        if model == "online":
            features = self.feature_extractor(input)
            output, load_balancing_loss, gate_probs = self.moe_layer(features)
            self.last_load_balancing_loss = load_balancing_loss
            self.last_gate_probs = gate_probs
            return output, load_balancing_loss
        elif model == "target":
            features = self.target[0](input)
            output, _, _ = self.target[1](features)
            return output

    def adjust_load_balancing_dynamically(self):
        stats = self.moe_layer.get_expert_usage_stats()
        cv = stats['coefficient_of_variation']
        moe_layer = self.moe_layer
        current_coef = moe_layer.load_balancing_loss_coef

        if cv > 1.2:
            moe_layer.load_balancing_loss_coef = min(current_coef * 1.5, 0.2)
        elif cv > 0.8:
            moe_layer.load_balancing_loss_coef = min(current_coef * 1.2, 0.1)
        elif cv < 0.3:
            moe_layer.load_balancing_loss_coef = max(current_coef * 0.8, 0.01)

        min_coef = 0.01
        if moe_layer.load_balancing_loss_coef < min_coef:
            moe_layer.load_balancing_loss_coef = min_coef

    def get_moe_metrics(self):
        if not hasattr(self.moe_layer, 'get_expert_usage_stats'):
            return None
        stats = self.moe_layer.get_expert_usage_stats()
        if hasattr(self, 'last_load_balancing_loss'):
            stats['load_balancing_loss'] = self.last_load_balancing_loss.item() if isinstance(self.last_load_balancing_loss, torch.Tensor) else self.last_load_balancing_loss
            stats['load_balancing_coef'] = self.moe_layer.load_balancing_loss_coef
        return stats


class ActorCriticMarioNetMBP(nn.Module):
    """Rede Actor-Critic com MBP para GPPO"""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        c, h, w = input_dim
        self.device = torch.device('cpu')
        self.action_dim = action_dim
        self.num_experts = 10
        self.top_k = 3

        self.feature_extractor = self._build_cnn_features(c, h, w)

        with torch.no_grad():
            dummy_input = torch.randn(1, c, h, w)
            feature_dim = self.feature_extractor(dummy_input).shape[1]

        self.policy_moe = MoELayerMBP(
            input_dim=feature_dim,
            hidden_dim=32,
            output_dim=action_dim,
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        self.value_moe = MoELayerMBP(
            input_dim=feature_dim,
            hidden_dim=32,
            output_dim=1,
            num_experts=self.num_experts,
            top_k=self.top_k
        )

        self.last_policy_load_balancing_loss = 0.0
        self.last_value_load_balancing_loss = 0.0
        self.last_policy_gate_probs = None
        self.last_value_gate_probs = None

        for i, expert in enumerate(self.policy_moe.experts):
            expert.expert_id = i
        for i, expert in enumerate(self.value_moe.experts):
            expert.expert_id = i + self.num_experts

    def _build_cnn_features(self, c, h, w):
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Flatten()
        )

        h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)

        return nn.Sequential(
            cnn,
            nn.Linear(64 * h_out * w_out, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        policy_logits, policy_load_balancing_loss, policy_gate_probs = self.policy_moe(features)
        value, value_load_balancing_loss, value_gate_probs = self.value_moe(features)

        self.last_policy_load_balancing_loss = policy_load_balancing_loss
        self.last_value_load_balancing_loss = value_load_balancing_loss
        self.last_policy_gate_probs = policy_gate_probs
        self.last_value_gate_probs = value_gate_probs

        policy_dist = Categorical(logits=policy_logits)
        return policy_dist, value.squeeze(-1)

    def get_moe_metrics(self):
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
        policy_stats = self.policy_moe.get_expert_usage_stats()
        if policy_stats['coefficient_of_variation'] > 0.8:
            self.policy_moe.load_balancing_loss_coef = min(
                self.policy_moe.load_balancing_loss_coef * 1.05, 0.2
            )
        elif policy_stats['coefficient_of_variation'] < 0.3:
            self.policy_moe.load_balancing_loss_coef = max(
                self.policy_moe.load_balancing_loss_coef * 0.95, 0.01
            )

        value_stats = self.value_moe.get_expert_usage_stats()
        if value_stats['coefficient_of_variation'] > 0.8:
            self.value_moe.load_balancing_loss_coef = min(
                self.value_moe.load_balancing_loss_coef * 1.05, 0.2
            )
        elif value_stats['coefficient_of_variation'] < 0.3:
            self.value_moe.load_balancing_loss_coef = max(
                self.value_moe.load_balancing_loss_coef * 0.95, 0.01
            )


if __name__ == "__main__":
    print("Testando MarioNetMBP...")
    
    state_dim = (4, 84, 84)
    action_dim = 6
    
    model = MarioNetMBP(state_dim, action_dim)
    x = torch.randn(2, *state_dim)
    
    output_online, loss = model(x, model="online")
    output_target = model(x, model="target")
    
    print(f"Entrada: {x.shape}")
    print(f"Saída online: {output_online.shape}")
    print(f"Saída target: {output_target.shape}")
    print(f"Loss de balanceamento: {loss.item():.6f}")
    
    # Testar ActorCritic
    ac_model = ActorCriticMarioNetMBP(state_dim, action_dim)
    policy_dist, value = ac_model(x)
    print(f"\nActorCritic:")
    print(f"Distribuição de política: {policy_dist.probs.shape}")
    print(f"Valor: {value.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal de parâmetros: {total_params:,}")
    
    print("\n✅ Todos os testes passaram!")

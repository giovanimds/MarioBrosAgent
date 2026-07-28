"""
Mamba-Based Processing (MBP) Layer

Implementação da arquitetura MBP com:
- Associative Scan (Kogge-Stone Prefix Scan)
- ConvEmbedding para contexto local
- RMSNorm para normalização
- MBPLayer com processamento de estado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ── Parallel Associative Scan Helper (Kogge-Stone Prefix Scan) ──
def associative_scan(a, b):
    """
    Computes prefix scan of elements (a_t, b_t) under the associative operator:
    (a_i, b_i) o (a_j, b_j) = (a_j * a_i, a_j * b_i + b_j)
    where j > i.
    
    a shape: (B, T, D)
    b shape: (B, T, D)
    """
    # SSMs acumulam erro numérico rápido em baixa precisão: fazemos o scan em
    # fp32 e devolvemos no dtype original (bf16 sob autocast, por exemplo).
    orig_dtype = a.dtype
    if orig_dtype != torch.float32:
        a = a.float()
        b = b.float()

    T = a.size(1)
    step = 1
    while step < T:
        a_left = a[:, :-step]
        b_left = b[:, :-step]

        a_right = a[:, step:]
        b_right = b[:, step:]

        a_new = a_right * a_left
        b_new = a_right * b_left + b_right

        a = torch.cat([a[:, :step], a_new], dim=1)
        b = torch.cat([b[:, :step], b_new], dim=1)

        step *= 2

    if orig_dtype != torch.float32:
        a = a.to(orig_dtype)
        b = b.to(orig_dtype)
    return a, b


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ConvEmbedding(nn.Module):
    """
    1D convolution over embeddings to provide local context mixing.
    Replaces RoPE's absolute position signal with relative local patterns.
    Uses a stateful buffer that tracks the last `kernel_size` tokens.
    """
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # Depthwise convolution: each embedding channel processed independently
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=0,  # no padding: we handle buffering manually
            groups=dim,
            bias=False,
        )
        self.norm = RMSNorm(dim)
        self.gate = nn.Linear(dim, dim)
        self.activation = nn.GELU(approximate='tanh')

    def forward(self, x, buffer):
        """
        x: (Batch, Dim) or (Batch, Seq_Len, Dim)
        buffer: (Batch, Dim, kernel_size-1)
        """
        if len(x.shape) == 3:
            B, T, D = x.shape
            K = self.kernel_size
            x_transposed = x.transpose(1, 2)  # (B, D, T)
            full = torch.cat([buffer, x_transposed], dim=-1)  # (B, D, T + K - 1)
            
            conv_out = self.conv(full)  # (B, D, T)
            conv_out = conv_out.transpose(1, 2)  # (B, T, D)
            
            gate = torch.sigmoid(self.gate(x))
            out = gate * x + (1 - gate) * self.activation(self.norm(conv_out))
            new_buffer = full[:, :, -K+1:].detach() if K > 1 else buffer
            return out, new_buffer
        else:
            # 2D case
            full = torch.cat([buffer, x.unsqueeze(-1)], dim=-1)
            conv_out = self.conv(full).squeeze(-1)
            gate = torch.sigmoid(self.gate(x))
            out = gate * x + (1 - gate) * self.activation(self.norm(conv_out))
            new_buffer = full[:, :, 1:].detach()
            return out, new_buffer

    def init_buffer(self, batch_size, device):
        """Initialize a zero buffer for a new sequence."""
        return torch.zeros(batch_size, self.dim, self.kernel_size - 1, device=device)


class MBPLayer(nn.Module):
    """Mamba-Based Processing Layer"""
    def __init__(self, dim, state_dim, ffn_mult=2, dropout=0.0, use_ple=False, ple_dim=128):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.gradient_checkpointing = False
        
        # SSM-inspired projections
        self.proj_delta = nn.Linear(dim, state_dim)
        self.proj_B = nn.Linear(dim, state_dim)
        self.proj_x = nn.Linear(dim, state_dim)
        
        # Fixed A parameter (learned but fixed topology)
        self.A = nn.Parameter(-torch.exp(torch.linspace(0, 2, state_dim)))
        
        self.norm = RMSNorm(dim)
        
        # Gated feed-forward (SwiGLU) block
        self.ffn_norm = RMSNorm(state_dim)
        hidden_dim = ffn_mult * state_dim
        self.ffn_gate = nn.Linear(state_dim, hidden_dim)
        self.ffn_up = nn.Linear(state_dim, hidden_dim)
        self.ffn_down = nn.Linear(hidden_dim, state_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.use_ple = use_ple
        if use_ple:
            self.ple_proj = nn.Linear(ple_dim, state_dim, bias=False)
            nn.init.normal_(self.ple_proj.weight, std=0.02)

    def forward(self, x, state, ple_slice=None, eos_mask=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, state, ple_slice, eos_mask, use_reentrant=False)
        return self._forward(x, state, ple_slice, eos_mask)

    def _forward(self, x, state, ple_slice=None, eos_mask=None):
        # x: (Batch, Dim) or (Batch, Seq_Len, Dim)
        # state: (Batch, state_dim)
        # ple_slice: (Batch, ple_dim) or (Batch, Seq_Len, ple_dim)
        # eos_mask: (Batch, Seq_Len)
        
        is_sequence = len(x.shape) == 3
        
        if is_sequence:
            B, T, D = x.shape
            normed_x = self.norm(x)
            
            delta = F.softplus(self.proj_delta(normed_x))
            B_proj = self.proj_B(normed_x)
            X = self.proj_x(normed_x)
            
            dA = torch.exp(delta * self.A[None, None, :].clamp(max=-1e-4))
            dB = delta * B_proj
            
            # Boundary reset: zero out transition multiplier dA if previous step was an EOS
            if eos_mask is not None:
                reset_mask = torch.cat([torch.zeros_like(eos_mask[:, :1]), eos_mask[:, :-1]], dim=1)
                dA = dA * (~reset_mask).unsqueeze(-1).to(dA.dtype)
            
            # Kogge-Stone parallel scan setup
            a_scan = torch.cat([torch.ones(B, 1, self.state_dim, device=x.device, dtype=dA.dtype), dA], dim=1)
            b_scan = torch.cat([state.unsqueeze(1).to(dB.dtype), dB * X], dim=1)
            
            _, b_scanned = associative_scan(a_scan, b_scan)
            
            next_state = b_scanned[:, -1]      # Final state at end of chunk
            layer_states = b_scanned[:, 1:]    # States at each sequence step
            
            output = torch.tanh(layer_states)
            
            ff_in = self.ffn_norm(output)
            ff = self.ffn_down(F.silu(self.ffn_gate(ff_in)) * self.ffn_up(ff_in))
            output = output + self.dropout(ff)
            
            if self.use_ple and ple_slice is not None:
                output = output + self.ple_proj(ple_slice)
                
            return output, next_state
        else:
            # 2D case (sequential)
            normed_x = self.norm(x)
            delta = F.softplus(self.proj_delta(normed_x))
            B_proj = self.proj_B(normed_x)
            X = self.proj_x(normed_x)
            
            dA = torch.exp(delta * self.A[None, :].clamp(max=-1e-4))
            dB = delta * B_proj
            
            next_state = dA * state + dB * X
            output = torch.tanh(next_state)
            
            ff_in = self.ffn_norm(output)
            ff = self.ffn_down(F.silu(self.ffn_gate(ff_in)) * self.ffn_up(ff_in))
            output = output + self.dropout(ff)
            
            if self.use_ple and ple_slice is not None:
                output = output + self.ple_proj(ple_slice)
                
            return output, next_state


class MBPModel(nn.Module):
    """Mamba-Based Processing Model"""
    def __init__(self, input_dim, output_dim, state_dim=128, num_layers=4, 
                 use_conv_embedding=True, conv_kernel_size=4, gradient_checkpointing=False,
                 ffn_mult=2, dropout=0.0, use_ple=False, ple_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.use_conv_embedding = use_conv_embedding
        self.use_ple = use_ple
        self.ple_dim = ple_dim
        
        # Input projection to embedding dimension
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.embed_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if use_conv_embedding:
            self.conv_embed = ConvEmbedding(state_dim, kernel_size=conv_kernel_size)
            
        if use_ple:
            self.embed_tokens_per_layer = nn.Embedding(input_dim, num_layers * ple_dim)
            nn.init.normal_(self.embed_tokens_per_layer.weight, std=0.02)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = state_dim if i == 0 else state_dim
            self.layers.append(
                MBPLayer(
                    in_dim, state_dim, ffn_mult=ffn_mult, dropout=dropout,
                    use_ple=use_ple, ple_dim=ple_dim
                )
            )
        
        if gradient_checkpointing:
            self.set_gradient_checkpointing(True)
        
        self.final_norm = RMSNorm(state_dim)
        self.head = nn.Linear(state_dim, output_dim)

    def set_gradient_checkpointing(self, enable=True):
        for layer in self.layers:
            layer.gradient_checkpointing = enable

    def gradient_checkpointing_enable(self):
        self.set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self):
        self.set_gradient_checkpointing(False)

    def forward(self, x, states=None, conv_buffer=None, eos_mask=None):
        """
        x: (Batch, Input_Dim) or (Batch, Seq_Len, Input_Dim)
        states: (Num_Layers, Batch, state_dim)
        conv_buffer: (Batch, Dim, kernel_size-1)
        eos_mask: (Batch, Seq_Len)
        """
        is_sequence = len(x.shape) == 3
        
        # Project input to embedding dimension
        if is_sequence:
            B, T, D = x.shape
            x = self.input_proj(x.view(-1, D)).view(B, T, -1)
        else:
            x = self.input_proj(x)
        
        x = self.embed_dropout(x)
        
        if self.use_conv_embedding:
            if conv_buffer is None:
                conv_buffer = self.conv_embed.init_buffer(x.shape[0], x.device)
            x, new_conv_buffer = self.conv_embed(x, conv_buffer)
        else:
            new_conv_buffer = None
            
        if self.use_ple:
            if is_sequence:
                input_shape = x.shape
                ple_embeds = self.embed_tokens_per_layer(x.view(-1, self.input_dim))
                ple_embeds = ple_embeds.view(*input_shape, self.num_layers, self.ple_dim)
            else:
                ple_embeds = self.embed_tokens_per_layer(x)
                ple_embeds = ple_embeds.view(x.shape[0], self.num_layers, self.ple_dim)
        
        # Initialize states if not provided
        if states is None:
            states = self.init_state(x.shape[0], x.device)
        
        new_states = []
        current_input = x
        for i, layer in enumerate(self.layers):
            ple_slice = ple_embeds[..., i, :] if self.use_ple else None
            out, next_s = layer(current_input, states[i], ple_slice=ple_slice, eos_mask=eos_mask)
            new_states.append(next_s)
            
            if current_input.shape == out.shape:
                current_input = current_input + out
            else:
                current_input = out
            
        logits = self.head(self.final_norm(current_input))
        return logits, torch.stack(new_states), new_conv_buffer

    def init_state(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.state_dim, device=device)


if __name__ == "__main__":
    # Teste básico
    print("Testando MBP Layer...")
    
    # Configuração
    input_dim = 256
    output_dim = 6
    state_dim = 128
    num_layers = 2
    batch_size = 2
    seq_len = 10
    
    # Criar modelo
    model = MBPModel(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=state_dim,
        num_layers=num_layers,
        use_conv_embedding=True,
        conv_kernel_size=4,
        gradient_checkpointing=False,
        ffn_mult=2,
        dropout=0.0,
        use_ple=False
    )
    
    # Teste com entrada 2D
    x_2d = torch.randn(batch_size, input_dim)
    logits, states, conv_buffer = model(x_2d)
    print(f"Entrada 2D: {x_2d.shape} -> Logits: {logits.shape}")
    print(f"States: {states.shape}")
    print(f"Conv Buffer: {conv_buffer.shape if conv_buffer is not None else 'None'}")
    
    # Teste com entrada 3D
    x_3d = torch.randn(batch_size, seq_len, input_dim)
    logits, states, conv_buffer = model(x_3d)
    print(f"Entrada 3D: {x_3d.shape} -> Logits: {logits.shape}")
    print(f"States: {states.shape}")
    
    assert logits.shape == (batch_size, output_dim) if len(x_2d.shape) == 2 else (batch_size, seq_len, output_dim)
    
    print("\n✅ Todos os testes passaram!")

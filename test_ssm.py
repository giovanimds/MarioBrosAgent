#!/usr/bin/env python3
"""
Script de teste para verificar se o modelo SSM funciona corretamente na CPU.
"""

import torch
import sys
import os

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ssm_layer():
    """Testa a camada SSM isoladamente."""
    print("=" * 60)
    print("TESTE 1: Camada SSM isolada")
    print("=" * 60)
    
    from src.agents.ssm_layer import SimpleStateSpaceModel, SSMLayer, EfficientSSMLayer
    
    # Configurações
    batch_size = 2
    channels = 4
    height = 84
    width = 84
    output_dim = 256
    
    # Testar SimpleStateSpaceModel
    print("\n1.1 Testando SimpleStateSpaceModel...")
    ssm = SimpleStateSpaceModel(d_model=256, d_state=16, d_conv=3)
    x_3d = torch.randn(batch_size, 10, 256)  # [batch, seq_len, d_model]
    out_3d = ssm(x_3d)
    print(f"   Entrada: {x_3d.shape} -> Saída: {out_3d.shape}")
    assert out_3d.shape == (batch_size, 10, 256)
    print("   ✅ SimpleStateSpaceModel: OK")
    
    # Testar SSMLayer
    print("\n1.2 Testando SSMLayer...")
    ssm_layer = SSMLayer(
        input_channels=channels,
        height=height,
        width=width,
        output_dim=output_dim,
        d_state=16,
        num_layers=2
    )
    x = torch.randn(batch_size, channels, height, width)
    out = ssm_layer(x)
    print(f"   Entrada: {x.shape} -> Saída: {out.shape}")
    assert out.shape == (batch_size, output_dim)
    print("   ✅ SSMLayer: OK")
    
    # Testar EfficientSSMLayer
    print("\n1.3 Testando EfficientSSMLayer...")
    efficient_ssm = EfficientSSMLayer(
        input_channels=channels,
        height=height,
        width=width,
        output_dim=output_dim,
        d_state=8
    )
    out_efficient = efficient_ssm(x)
    print(f"   Entrada: {x.shape} -> Saída: {out_efficient.shape}")
    assert out_efficient.shape == (batch_size, output_dim)
    print("   ✅ EfficientSSMLayer: OK")
    
    print("\n✅ Todos os testes da camada SSM passaram!")
    return True


def test_mario_net_ssm():
    """Testa a MarioNetSSM."""
    print("\n" + "=" * 60)
    print("TESTE 2: MarioNetSSM")
    print("=" * 60)
    
    from src.agents.model_ssm import MarioNetSSM
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    batch_size = 2
    
    print("\n2.1 Criando modelo...")
    model = MarioNetSSM(state_dim, action_dim)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    printable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total de parâmetros: {total_params:,}")
    print(f"   Parâmetros treináveis: {printable_params:,}")
    
    print("\n2.2 Testando forward pass (online)...")
    x = torch.randn(batch_size, *state_dim)
    output_online, loss = model(x, model="online")
    print(f"   Entrada: {x.shape}")
    print(f"   Saída online: {output_online.shape}")
    print(f"   Loss de balanceamento: {loss.item():.6f}")
    assert output_online.shape == (batch_size, action_dim)
    print("   ✅ Forward pass online: OK")
    
    print("\n2.3 Testando forward pass (target)...")
    output_target = model(x, model="target")
    print(f"   Saída target: {output_target.shape}")
    assert output_target.shape == (batch_size, action_dim)
    print("   ✅ Forward pass target: OK")
    
    print("\n2.4 Testando métricas MoE...")
    metrics = model.get_moe_metrics()
    print(f"   Métricas disponíveis: {list(metrics.keys())}")
    assert metrics is not None
    assert 'load_balancing_loss' in metrics
    assert 'load_balancing_coef' in metrics
    print("   ✅ Métricas MoE: OK")
    
    print("\n✅ Todos os testes da MarioNetSSM passaram!")
    return True


def test_actor_critic_ssm():
    """Testa a ActorCriticMarioNetSSM."""
    print("\n" + "=" * 60)
    print("TESTE 3: ActorCriticMarioNetSSM")
    print("=" * 60)
    
    from src.agents.model_ssm import ActorCriticMarioNetSSM
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    batch_size = 2
    
    print("\n3.1 Criando modelo...")
    model = ActorCriticMarioNetSSM(state_dim, action_dim)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    printable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total de parâmetros: {total_params:,}")
    print(f"   Parâmetros treináveis: {printable_params:,}")
    
    print("\n3.2 Testando forward pass...")
    x = torch.randn(batch_size, *state_dim)
    policy_dist, value = model(x)
    print(f"   Entrada: {x.shape}")
    print(f"   Distribuição de política: {policy_dist.probs.shape}")
    print(f"   Valor: {value.shape}")
    assert policy_dist.probs.shape == (batch_size, action_dim)
    assert value.shape == (batch_size,)
    print("   ✅ Forward pass: OK")
    
    print("\n3.3 Testando métricas MoE...")
    metrics = model.get_moe_metrics()
    print(f"   Métricas disponíveis: {list(metrics.keys())}")
    assert metrics is not None
    assert 'policy' in metrics
    assert 'value' in metrics
    assert 'load_balancing_loss' in metrics
    print("   ✅ Métricas MoE: OK")
    
    print("\n3.4 Testando amostragem de ações...")
    # Amostrar ações da distribuição (usar batch_size=1 para simplificar)
    x_single = torch.randn(1, *state_dim)
    policy_dist_single, _ = model(x_single)
    actions = [policy_dist_single.sample().item() for _ in range(10)]
    print(f"   Ações amostradas: {actions[:5]}")
    assert all(0 <= a < action_dim for a in actions)
    print("   ✅ Amostragem de ações: OK")
    
    print("\n✅ Todos os testes da ActorCriticMarioNetSSM passaram!")
    return True


def test_cpu_performance():
    """Testa o desempenho na CPU."""
    print("\n" + "=" * 60)
    print("TESTE 4: Desempenho na CPU")
    print("=" * 60)
    
    from src.agents.model_ssm import MarioNetSSM
    import time
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    batch_size = 4
    num_iterations = 10
    
    print("\n4.1 Criando modelo...")
    model = MarioNetSSM(state_dim, action_dim)
    model.eval()  # Modo de avaliação
    
    print("\n4.2 Medindo tempo de inferência...")
    x = torch.randn(batch_size, *state_dim)
    
    # Warm-up
    for _ in range(3):
        _ = model(x, model="online")
    
    # Medir tempo
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            output, _ = model(x, model="online")
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"   Tempo médio por forward pass: {avg_time*1000:.2f}ms")
    print(f"   Throughput: {num_iterations/avg_time:.2f} iterações/segundo")
    
    print("\n4.3 Testando com diferentes tamanhos de batch...")
    for bs in [1, 2, 4, 8]:
        x_batch = torch.randn(bs, *state_dim)
        start = time.time()
        with torch.no_grad():
            output, _ = model(x_batch, model="online")
        elapsed = (time.time() - start) * 1000
        print(f"   Batch {bs}: {elapsed:.2f}ms")
    
    print("\n✅ Testes de desempenho concluídos!")
    return True


def test_memory_usage():
    """Testa o uso de memória."""
    print("\n" + "=" * 60)
    print("TESTE 5: Uso de Memória")
    print("=" * 60)
    
    from src.agents.model_ssm import MarioNetSSM
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    
    print("\n5.1 Estimando uso de memória...")
    model = MarioNetSSM(state_dim, action_dim)
    
    # Estimar memória dos parâmetros
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    print(f"   Tamanho dos parâmetros: {param_size / 1024 / 1024:.2f} MB")
    print(f"   Tamanho dos buffers: {buffer_size / 1024 / 1024:.2f} MB")
    print(f"   Total: {total_size / 1024 / 1024:.2f} MB")
    
    # Testar com entrada
    x = torch.randn(1, *state_dim)
    
    # Estimar memória de ativações
    with torch.no_grad():
        output, _ = model(x, model="online")
    
    # Contar parâmetros treináveis vs não treináveis
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"   Parâmetros treináveis: {trainable:,}")
    print(f"   Parâmetros congelados: {frozen:,}")
    
    print("\n✅ Testes de memória concluídos!")
    return True


def main():
    """Função principal de teste."""
    print("\n" + "=" * 60)
    print("TESTES DO MODELO SSM PARA MARIO BROS AGENT")
    print("=" * 60)
    
    # Verificar dispositivo
    device = torch.device('cpu')
    print(f"\nDispositivo: {device}")
    print(f"PyTorch versão: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    
    try:
        # Executar todos os testes
        test_ssm_layer()
        test_mario_net_ssm()
        test_actor_critic_ssm()
        test_cpu_performance()
        test_memory_usage()
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("=" * 60)
        print("\nO modelo SSM está funcionando corretamente na CPU!")
        print("Pode ser usado como substituto para a camada Transformer.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Script para comparar o desempenho das arquiteturas Transformer vs MBP
"""

import os
import sys
import time
import torch
import numpy as np
from rich.console import Console
from rich.table import Table

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importar modelos
from src.agents.model import MarioNet  # Transformer
from src.agents.model_mbp import MarioNetMBP  # MBP


def benchmark_model(model, name, state_dim, action_dim, num_iterations=100):
    """Benchmark de um modelo"""
    console = Console()
    
    # Aquecer
    x = torch.randn(1, *state_dim)
    for _ in range(5):
        _ = model(x, model="online")
    
    # Medir tempo
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x, model="online")
    elapsed = time.time() - start_time
    
    avg_time = elapsed / num_iterations
    throughput = num_iterations / elapsed
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimar memória
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_mb = param_size / 1024 / 1024
    
    return {
        'name': name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_mb': memory_mb,
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
        'num_iterations': num_iterations
    }


def compare_architectures():
    """Compara as arquiteturas Transformer vs MBP"""
    console = Console()
    
    console.print("\n" + "="*70)
    console.print("COMPARATIVO: Transformer vs MBP (Mamba-Based Processing)")
    console.print("="*70)
    
    # Configuração
    state_dim = (4, 84, 84)
    action_dim = 6
    num_iterations = 100
    
    console.print("\n[blue]Inicializando modelos...[/blue]")
    
    # Criar modelos
    try:
        transformer_model = MarioNet(state_dim, action_dim)
        transformer_model.eval()
        console.print("[green]✅ Modelo Transformer carregado[/green]")
    except Exception as e:
        console.print(f"[red]❌ Erro ao carregar Transformer: {e}[/red]")
        transformer_model = None
    
    try:
        mbp_model = MarioNetMBP(state_dim, action_dim)
        mbp_model.eval()
        console.print("[green]✅ Modelo MBP carregado[/green]")
    except Exception as e:
        console.print(f"[red]❌ Erro ao carregar MBP: {e}[/red]")
        mbp_model = None
    
    # Benchmark
    results = []
    
    if transformer_model:
        console.print("\n[blue]Benchmark Transformer...[/blue]")
        transformer_results = benchmark_model(
            transformer_model, "Transformer", state_dim, action_dim, num_iterations
        )
        results.append(transformer_results)
        console.print(f"[green]✅ Transformer: {transformer_results['throughput']:.0f} it/s[/green]")
    
    if mbp_model:
        console.print("\n[blue]Benchmark MBP...[/blue]")
        mbp_results = benchmark_model(
            mbp_model, "MBP", state_dim, action_dim, num_iterations
        )
        results.append(mbp_results)
        console.print(f"[green]✅ MBP: {mbp_results['throughput']:.0f} it/s[/green]")
    
    # Criar tabela de resultados
    console.print("\n" + "="*70)
    console.print("RESULTADOS DO BENCHMARK")
    console.print("="*70)
    
    table = Table(title="Comparativo de Desempenho")
    table.add_column("Arquitetura", style="cyan")
    table.add_column("Parâmetros Totais", justify="right")
    table.add_column("Parâmetros Treináveis", justify="right")
    table.add_column("Memória (MB)", justify="right")
    table.add_column("Tempo Médio (ms)", justify="right")
    table.add_column("Throughput (it/s)", justify="right")
    
    for result in results:
        table.add_row(
            result['name'],
            f"{result['total_params']:,}",
            f"{result['trainable_params']:,}",
            f"{result['memory_mb']:.2f}",
            f"{result['avg_time_ms']:.2f}",
            f"{result['throughput']:.0f}"
        )
    
    console.print(table)
    
    # Cálculo de melhorias
    if len(results) == 2:
        transformer = results[0]
        mbp = results[1]
        
        console.print("\n" + "="*70)
        console.print("MELHORIAS")
        console.print("="*70)
        
        # Parâmetros
        param_reduction = (1 - mbp['total_params'] / transformer['total_params']) * 100
        console.print(f"[green]Redução de parâmetros: {param_reduction:.1f}%[/green]")
        
        # Memória
        memory_reduction = (1 - mbp['memory_mb'] / transformer['memory_mb']) * 100
        console.print(f"[green]Redução de memória: {memory_reduction:.1f}%[/green]")
        
        # Velocidade
        speed_improvement = (mbp['throughput'] / transformer['throughput'] - 1) * 100
        console.print(f"[green]Melhoria de throughput: {speed_improvement:.1f}%[/green]")
        
        # Tempo
        time_reduction = (1 - mbp['avg_time_ms'] / transformer['avg_time_ms']) * 100
        console.print(f"[green]Redução de tempo: {time_reduction:.1f}%[/green]")
    
    # Teste de forward pass
    console.print("\n" + "="*70)
    console.print("TESTE DE FORWARD PASS")
    console.print("="*70)
    
    x = torch.randn(1, *state_dim)
    
    if transformer_model:
        with torch.no_grad():
            transformer_out, _ = transformer_model(x, model="online")
        console.print(f"[blue]Transformer output shape: {transformer_out.shape}[/blue]")
    
    if mbp_model:
        with torch.no_grad():
            mbp_out, _ = mbp_model(x, model="online")
        console.print(f"[blue]MBP output shape: {mbp_out.shape}[/blue]")
    
    # Verificar compatibilidade
    if transformer_model and mbp_model:
        if transformer_out.shape == mbp_out.shape:
            console.print("[green]✅ Saídas compatíveis[/green]")
        else:
            console.print("[red]❌ Saídas incompatíveis[/red]")
    
    console.print("\n" + "="*70)
    console.print("CONCLUSÃO")
    console.print("="*70)
    
    if len(results) == 2:
        if mbp['throughput'] > transformer['throughput']:
            console.print("[bold green]✅ MBP é mais rápido que Transformer![/bold green]")
        else:
            console.print("[bold red]❌ Transformer é mais rápido que MBP[/bold red]")
        
        if mbp['memory_mb'] < transformer['memory_mb']:
            console.print("[bold green]✅ MBP usa menos memória que Transformer![/bold green]")
        else:
            console.print("[bold red]❌ Transformer usa menos memória que MBP[/bold red]")
        
        if mbp['total_params'] < transformer['total_params']:
            console.print("[bold green]✅ MBP tem menos parâmetros que Transformer![/bold green]")
        else:
            console.print("[bold red]❌ Transformer tem menos parâmetros que MBP[/bold red]")
    
    return results


def main():
    """Função principal"""
    console = Console()
    
    # Verificar dispositivo
    device = torch.device('cpu')
    console.print(f"\nDispositivo: {device}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA: {torch.cuda.is_available()}")
    
    try:
        results = compare_architectures()
        
        # Salvar resultados
        with open("benchmark_results.txt", "w") as f:
            for result in results:
                f.write(f"{result['name']}: {result['throughput']:.0f} it/s, {result['memory_mb']:.2f} MB\n")
        
        console.print("\n[bold green]✅ Comparativo concluído![/bold green]")
        console.print("[yellow]Resultados salvos em benchmark_results.txt[/yellow]")
        return True
        
    except Exception as e:
        console.print(f"\n[red]Erro: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

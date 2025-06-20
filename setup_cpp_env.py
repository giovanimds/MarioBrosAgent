"""
Configura as variáveis de ambiente para o Microsoft C++ Build Tools de forma permanente.
Este script deve ser executado com privilégios administrativos para configurar as variáveis
de ambiente do sistema.
"""
import os
import subprocess
import sys
import winreg
from typing import Dict, List, Optional, Tuple, Union

def is_admin() -> bool:
    """Verifica se o script está sendo executado com privilégios administrativos."""
    try:
        return subprocess.run(
            ["net", "session"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).returncode == 0
    except Exception:
        return False

def find_vs_installation_path() -> Optional[str]:
    """
    Localiza o caminho de instalação do Visual Studio usando o vswhere.
    Retorna None se não encontrado.
    """
    # Caminho específico do vswhere mencionado pelo usuário
    user_specified_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

    # Lista de possíveis caminhos para o vswhere
    vswhere_paths = [
        user_specified_path,
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe")
    ]

    vswhere_path = None
    for path in vswhere_paths:
        if os.path.exists(path):
            vswhere_path = path
            print(f"vswhere.exe encontrado em: {path}")
            break

    if not vswhere_path:
        print("vswhere.exe não encontrado em nenhum dos caminhos conhecidos.")
        return None

    # Tenta encontrar o Visual Studio Build Tools ou Visual Studio com várias estratégias diferentes
    try:
        print("\nTentando localizar instalação do Visual Studio ou Build Tools...")

        # Estratégia 1: Procura por qualquer instalação do Visual Studio
        print("Estratégia 1: Buscando qualquer instalação do Visual Studio...")
        result = subprocess.run(
            [vswhere_path, "-latest", "-property", "installationPath"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"Resultado da busca: {result}")
        if result.stdout.strip():
            path = result.stdout.strip()
            print(f"Encontrada instalação do Visual Studio em: {path}")

            # Verifica se o diretório VC\Auxiliary\Build existe nessa instalação
            vcvarsall_path = os.path.join(path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
            if os.path.exists(vcvarsall_path):
                print(f"vcvarsall.bat encontrado em: {vcvarsall_path}")
                return path
            else:
                print(f"vcvarsall.bat não encontrado em: {vcvarsall_path}")

        # Estratégia 2: Procura especificamente pelo Build Tools
        print("\nEstratégia 2: Buscando especificamente o Build Tools...")
        result = subprocess.run(
            [vswhere_path, "-latest", "-products", "Microsoft.VisualStudio.Product.BuildTools", "-property", "installationPath"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"Resultado da busca por Build Tools: {result}")
        if result.stdout.strip():
            path = result.stdout.strip()
            print(f"Encontrada instalação do Build Tools em: {path}")

            # Verifica se o diretório VC\Auxiliary\Build existe nessa instalação
            vcvarsall_path = os.path.join(path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
            if os.path.exists(vcvarsall_path):
                print(f"vcvarsall.bat encontrado em: {vcvarsall_path}")
                return path
            else:
                print(f"vcvarsall.bat não encontrado em: {vcvarsall_path}")

        # Estratégia 3: Procura por caminhos típicos do Visual Studio
        print("\nEstratégia 3: Verificando caminhos típicos de instalação...")

        typical_paths: List[str] = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise"
        ]

        for path in typical_paths:
            vcvarsall_path = os.path.join(path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
            if os.path.exists(vcvarsall_path):
                print(f"Encontrado vcvarsall.bat em: {vcvarsall_path}")
                return path

        # Se chegou até aqui, não encontrou nada
        print("\nNenhuma instalação adequada do Visual Studio encontrada com os componentes C++ necessários.")
        print("Por favor, instale o Visual Studio Build Tools com a carga de trabalho 'Desenvolvimento para desktop com C++'")
        print("Você pode baixá-lo em: https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/")

    except Exception as e:
        print(f"Erro ao procurar instalação do Visual Studio: {e}")

    return None

def get_vs_environment_variables(vs_path: str) -> Dict[str, str]:
    """
    Obtém as variáveis de ambiente configuradas pelo Visual Studio.
    """
    # Caminho para o script que configura o ambiente do VS
    vcvarsall_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvarsall.bat")

    if not os.path.exists(vcvarsall_path):
        print(f"vcvarsall.bat não encontrado em {vcvarsall_path}")
        return {}

    # Executa o script e captura as variáveis de ambiente
    try:
        # Comando para executar vcvarsall.bat e depois listar as variáveis de ambiente
        cmd = f'"{vcvarsall_path}" x64 && set'
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True
        )

        # Processa a saída para extrair as variáveis de ambiente
        env_vars = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                name, value = line.split("=", 1)
                env_vars[name.strip()] = value.strip()

        # Filtra apenas as variáveis relevantes para o C++ Build Tools
        relevant_vars = {
            k: v for k, v in env_vars.items()
            if k in [
                "INCLUDE", "LIB", "LIBPATH", "Path",
                "VCINSTALLDIR", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH",
                "VSCMD_VER", "VSINSTALLDIR", "WindowsSDKLibVersion", "WindowsSDKVersion"
            ]
        }

        return relevant_vars

    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar vcvarsall.bat: {e}")
        return {}

def set_system_environment_variable(name: str, value: str) -> bool:
    """
    Define uma variável de ambiente do sistema de forma permanente.
    Requer privilégios de administrador.
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            0,
            winreg.KEY_ALL_ACCESS
        )

        winreg.SetValueEx(key, name, 0, winreg.REG_EXPAND_SZ, value)
        winreg.CloseKey(key)

        # Notifica o sistema sobre a mudança nas variáveis de ambiente
        subprocess.run(
            ["powershell", "-Command", "& {$env:TEMP = [System.Environment]::GetEnvironmentVariable('TEMP','Machine'); $env:TMP = [System.Environment]::GetEnvironmentVariable('TMP','Machine')}"],
            shell=True
        )

        return True
    except Exception as e:
        print(f"Erro ao definir variável {name}: {e}")
        return False

def update_path_variable(new_paths: List[str]) -> bool:
    """
    Atualiza a variável PATH do sistema adicionando novos caminhos.
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE
        )

        # Obtém o valor atual do PATH
        current_path, _ = winreg.QueryValueEx(key, "Path")

        # Divide a string PATH em uma lista de caminhos
        path_list = current_path.split(";")

        # Adiciona novos caminhos, evitando duplicatas
        for path in new_paths:
            if path and path not in path_list:
                path_list.append(path)

        # Reconstrói a string PATH
        new_path = ";".join([p for p in path_list if p])

        # Define o novo valor de PATH
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
        winreg.CloseKey(key)

        return True
    except Exception as e:
        print(f"Erro ao atualizar PATH: {e}")
        return False

def main() -> None:
    """Função principal do script."""
    if not is_admin():
        print("Este script precisa ser executado como administrador.")
        print("Por favor, execute novamente com privilégios administrativos.")
        sys.exit(1)

    print("Configurando variáveis de ambiente para Microsoft C++ Build Tools...")

    # Localiza a instalação do Visual Studio
    vs_path = find_vs_installation_path()
    if not vs_path:
        print("Não foi possível encontrar uma instalação do Visual Studio com C++ Build Tools.")
        print("Por favor, instale o Visual Studio com a carga de trabalho 'Desenvolvimento para desktop com C++'")
        print("ou o Visual Studio Build Tools com a mesma carga de trabalho.")
        sys.exit(1)

    print(f"Visual Studio encontrado em: {vs_path}")

    # Obtém as variáveis de ambiente do Visual Studio
    env_vars = get_vs_environment_variables(vs_path)
    if not env_vars:
        print("Não foi possível obter as variáveis de ambiente do Visual Studio.")
        sys.exit(1)

    # Define as variáveis de ambiente do sistema
    success = True

    # Lista de variáveis a serem configuradas permanentemente
    variables_to_set = ["INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "VSINSTALLDIR"]

    for name in variables_to_set:
        if name in env_vars:
            print(f"Configurando {name}...")
            if not set_system_environment_variable(name, env_vars[name]):
                success = False

    # Atualiza a variável PATH
    if "Path" in env_vars:
        print("Atualizando PATH...")

        # Extrai novos caminhos da variável Path
        new_paths = [
            path for path in env_vars["Path"].split(";")
            if "Visual Studio" in path or "Microsoft Visual" in path or "WindowsSDK" in path or "MSBuild" in path
        ]

        if not update_path_variable(new_paths):
            success = False

    if success:
        print("\nVariáveis de ambiente configuradas com sucesso!")
        print("As variáveis do Microsoft C++ Build Tools agora estão disponíveis permanentemente.")
        print("Você pode precisar reiniciar o computador para que todas as alterações tenham efeito.")
    else:
        print("\nOcorreram erros ao configurar algumas variáveis de ambiente.")
        print("Verifique as mensagens acima para mais detalhes.")

if __name__ == "__main__":
    main()

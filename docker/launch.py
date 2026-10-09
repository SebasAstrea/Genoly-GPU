#!/usr/bin/env python3
"""
Lanzador inteligente para Genoly-GPU en Docker.
Detecta automáticamente la GPU y selecciona el perfil adecuado.
"""
import os
import subprocess
import sys
from pathlib import Path

from detect_gpu import detect_nvidia_gpus, get_recommended_build_mode, select_best_gpu

# Añadir el directorio actual al path para importar detect_gpu
sys.path.insert(0, str(Path(__file__).parent))

def docker_compose_up(profile: str, gpu_id: int = None, port: int = 8000):
    """Lanza docker-compose con el perfil adecuado."""
    env = os.environ.copy()
    env["GENOLY_PORT"] = str(port)
    
    # Mapear perfiles a servicios
    profile_map = {
        "cpu": ("", "genoly-cpu"),
        "gpu": ("--profile gpu", "genoly-gpu"),
        "gpu-select": ("--profile gpu-select", "genoly-gpu-selected")
    }
    
    if profile not in profile_map:
        print(f" Perfil inválido: {profile}")
        return False
    
    profile_args, service = profile_map[profile]
    
    if profile == "gpu-select" and gpu_id is not None:
        env["GPU_ID"] = str(gpu_id)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Comando docker compose
    cmd = ["docker", "compose"]
    if profile_args:
        cmd.extend(profile_args.split())
    cmd.extend(["up", "-d", service])
    
    print(f"🚀 Lanzando Genoly-GPU en modo {profile.upper()}...")
    if gpu_id is not None:
        print(f"   GPU seleccionada: {gpu_id}")
    print(f"   Comando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"✅ Contenedor lanzado exitosamente en modo {profile.upper()}")
        
        # Mostrar URL
        print(f"\n🌐 API disponible en: http://localhost:{port}")
        print(f"📚 Documentación: http://localhost:{port}/docs")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error al lanzar: {e}")
        return False

def interactive_menu():
    """Menú interactivo para seleccionar configuración."""
    print("\n" + "=" * 60)
    print("   GENOLY-GPU DOCKER LAUNCHER")
    print("=" * 60)
    
    gpus = detect_nvidia_gpus()
    mode = get_recommended_build_mode()
    
    print(f"\n🔧 Modo recomendado: {mode.upper()}")
    
    if not gpus:
        print("\n⚠️  No se detectaron GPUs NVIDIA.")
        print("   El contenedor se ejecutará en modo CPU.")
        choice = input("\n¿Continuar en modo CPU? [S/n]: ").strip().lower()
        if choice in ("n", "no"):
            return False
        return docker_compose_up("cpu")
    
    print(f"\n🎮 GPUs detectadas ({len(gpus)}):")
    for gpu in gpus:
        free_gb = gpu['memory_free_mb'] / 1024
        total_gb = gpu['memory_total_mb'] / 1024
        print(f"   [{gpu['index']}] {gpu['name']} "
              f"({free_gb:.1f} GB libre / {total_gb:.1f} GB total)")
    
    print("\nOpciones:")
    print("   [a] Usar TODAS las GPUs (modo GPU)")
    print("   [b] Usar la GPU con más VRAM libre (automático)")
    print("   [0-{0}] Usar GPU específica".format(len(gpus)-1))
    print("   [c] Usar CPU (sin GPU)")
    print("   [q] Salir")
    
    choice = input("\nSelecciona: ").strip().lower()
    
    if choice == "q":
        return False
    elif choice == "c":
        return docker_compose_up("cpu")
    elif choice == "a":
        return docker_compose_up("gpu")
    elif choice == "b":
        best = select_best_gpu(gpus)
        if best is not None:
            return docker_compose_up("gpu-select", gpu_id=best)
        return docker_compose_up("gpu")
    elif choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(gpus):
            return docker_compose_up("gpu-select", gpu_id=idx)
        else:
            print(f" GPU {idx} no disponible")
            return False
    else:
        print("Opción inválida")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lanzador inteligente para Genoly-GPU")
    parser.add_argument("--cpu", action="store_true", help="Modo CPU")
    parser.add_argument("--gpu", action="store_true", help="Modo GPU (todas las GPUs)")
    parser.add_argument("--gpu-id", type=int, help="GPU específica (índice)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto (default: 8000)")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    
    args = parser.parse_args()
    
    if args.cpu:
        docker_compose_up("cpu", port=args.port)
    elif args.gpu:
        docker_compose_up("gpu", port=args.port)
    elif args.gpu_id is not None:
        gpus = detect_nvidia_gpus()
        if 0 <= args.gpu_id < len(gpus):
            docker_compose_up("gpu-select", gpu_id=args.gpu_id, port=args.port)
        else:
            print(f"GPU {args.gpu_id} no disponible. GPUs detectadas: {len(gpus)}")
            sys.exit(1)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
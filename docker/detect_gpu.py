#!/usr/bin/env python3
"""
Detector de GPU para Genoly-GPU.
Usado por el script de lanzamiento y por el sistema de build.
"""
import json
import re
import subprocess
from typing import Dict, List, Optional


def detect_nvidia_gpus() -> List[Dict]:
    """Detecta GPUs NVIDIA usando nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "compute_cap": parts[4],
                    "driver_version": parts[5]
                })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

def get_cuda_version() -> Optional[str]:
    """Obtiene la version de CUDA soportada por el driver (cabecera de nvidia-smi)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5, check=True
        )
        match = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*([0-9]+(?:\.[0-9]+)?)", result.stdout)
        if match:
            return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

def select_best_gpu(gpus: List[Dict]) -> Optional[int]:
    """Selecciona la GPU con más VRAM libre."""
    if not gpus:
        return None
    best = max(gpus, key=lambda g: g.get("memory_free_mb", 0))
    return best["index"]

def has_cuda_support() -> bool:
    """Verifica si el sistema tiene soporte CUDA."""
    cuda_ver = get_cuda_version()
    if not cuda_ver:
        return False
    # CUDA 11.0+ es suficiente para PyTorch 2.x
    try:
        major = int(cuda_ver.split(".")[0])
        return major >= 11
    except (ValueError, IndexError):
        return False

def get_recommended_build_mode() -> str:
    """Recomienda 'gpu' o 'cpu' según el sistema."""
    if has_cuda_support() and detect_nvidia_gpus():
        return "gpu"
    return "cpu"

def main():
    """Punto de entrada para línea de comandos."""
    import argparse
    parser = argparse.ArgumentParser(description="Detector de GPU para Genoly-GPU")
    parser.add_argument("--mode", action="store_true", help="Solo imprime 'cpu' o 'gpu'")
    parser.add_argument("--list", action="store_true", help="Lista GPUs disponibles")
    parser.add_argument("--best", action="store_true", help="Índice de la mejor GPU (más VRAM libre)")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")
    
    args = parser.parse_args()
    
    if args.mode:
        print(get_recommended_build_mode())
        return
    
    if args.list:
        gpus = detect_nvidia_gpus()
        if args.json:
            print(json.dumps(gpus, indent=2))
        else:
            if not gpus:
                print("No se detectaron GPUs NVIDIA.")
            else:
                print("GPUs NVIDIA detectadas:")
                for gpu in gpus:
                    print(f"  [{gpu['index']}] {gpu['name']} "
                          f"({gpu['memory_free_mb']} MB libre / {gpu['memory_total_mb']} MB total, "
                          f"Compute {gpu['compute_cap']}, Driver {gpu['driver_version']})")
        return
    
    if args.best:
        gpus = detect_nvidia_gpus()
        best = select_best_gpu(gpus)
        if best is None:
            print(-1)
        else:
            print(best)
        return
    
    # Por defecto: información completa en formato legible
    cuda_ver = get_cuda_version()
    gpus = detect_nvidia_gpus()
    mode = get_recommended_build_mode()
    
    print("=" * 60)
    print("  GENOLY-GPU DETECTOR DE GPU")
    print("=" * 60)
    print(f"Modo recomendado: {mode.upper()}")
    print(f"CUDA Version: {cuda_ver or 'No detectada'}")
    print(f"GPUs: {len(gpus)} detectadas")
    
    if gpus:
        print("\nDetalle:")
        for gpu in gpus:
            print(f"  [{gpu['index']}] {gpu['name']} "
                  f"({gpu['memory_free_mb']} MB libre)")
    print("=" * 60)

if __name__ == "__main__":
    main()
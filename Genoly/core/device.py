import os
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class GPUInfo:
    """Información del dispositivo GPU disponible."""
    name: str
    memory_total_gb: float
    memory_free_gb: Optional[float]
    compute_capability: Optional[str]
    cuda_version: Optional[str]


class DeviceManager:
    """
    Gestión del dispositivo de cómputo (CUDA/NVIDIA o CPU).

    Centraliza la detección del dispositivo y evita repetir la lógica de
    auto-detección en cada módulo del proyecto.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 'cuda', 'cuda:N', 'cpu' o None para auto-detectar.
        """
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is None:
            # Variable de entorno: fija el dispositivo por defecto para todo
            # el proceso (la usan Docker y docker-compose para elegir la
            # rama CPU/GPU). Vacía o ausente = auto-detección.
            env = os.environ.get("GENOLY_DEVICE", "").strip()
            if env:
                device = env
        if device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resolved = torch.device(device)
        if resolved.type != 'cuda':
            return resolved
        # Fallo temprano y explicable: sin esta comprobación, pedir
        # device='cuda' sin CUDA solo revienta en la primera operación
        # de tensor (mucho más lejos de la causa).
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Se pidió el dispositivo '{device}' (argumento o variable "
                "de entorno GENOLY_DEVICE) pero CUDA no está disponible en "
                "este entorno. Usa device='cpu' (o None / GENOLY_DEVICE "
                "vacío para auto-detección) o instala PyTorch con soporte "
                "CUDA.")
        index = resolved.index
        if index is not None and index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Se pidió el dispositivo '{device}' pero solo hay "
                f"{torch.cuda.device_count()} GPU(s) disponibles.")
        return resolved

    @property
    def is_cuda(self) -> bool:
        return self.device.type == 'cuda'

    @property
    def cuda_index(self) -> Optional[int]:
        if self.is_cuda and self.device.index is None:
            return torch.cuda.current_device()
        return self.device.index

    def get_gpu_info(self) -> Optional[GPUInfo]:
        """
        Devuelve información detallada de la GPU activa.

        Returns:
            GPUInfo si CUDA está disponible, None en caso contrario.
        """
        if not self.is_cuda:
            return None

        idx = self.cuda_index
        props = torch.cuda.get_device_properties(idx)
        memory_free = None
        try:
            memory_free = torch.cuda.mem_get_info(idx)[0] / 1e9
        except Exception:
            pass

        cap = props.major and f"{props.major}.{props.minor}"
        return GPUInfo(
            name=torch.cuda.get_device_name(idx),
            memory_total_gb=props.total_memory / 1e9,
            memory_free_gb=memory_free,
            compute_capability=cap,
            cuda_version=torch.version.cuda,
        )

    def print_info(self) -> None:
        """Imprime un resumen del dispositivo seleccionado."""
        print(f"Dispositivo seleccionado: {self.device}")
        if self.is_cuda:
            info = self.get_gpu_info()
            if info:
                print(f"  GPU: {info.name}")
                print(f"  Memoria total: {info.memory_total_gb:.1f} GB")
                if info.memory_free_gb is not None:
                    print(f"  Memoria libre: {info.memory_free_gb:.1f} GB")
                if info.compute_capability:
                    print(f"  Compute capability: {info.compute_capability}")
                print(f"  CUDA version: {info.cuda_version}")
        else:
            print("  Sin GPU NVIDIA/CUDA detectada. Usando CPU.")

    def to(self, *args, **kwargs) -> Any:
        """
        Atajo para mover tensores o módulos al dispositivo gestionado.

        Ejemplo:
            x = manager.to(torch.zeros(4))
        """
        return torch.as_tensor(*args, **kwargs).to(self.device)

    def empty_cache(self) -> None:
        """Libera memoria caché de CUDA si está disponible."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def synchronize(self) -> None:
        """Sincroniza CUDA para mediciones de tiempo fiables."""
        if self.is_cuda:
            torch.cuda.synchronize()


def get_device(device: Optional[str] = None) -> torch.device:
    """Función de conveniencia para obtener el dispositivo."""
    return DeviceManager(device).device
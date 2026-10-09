from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from Genoly.core.device import DeviceManager
from Genoly.core.gpu_setup import GpuSetup

router = APIRouter(prefix="/api/device", tags=["device"])


class DeviceStatus(BaseModel):
    device: str
    cuda_available: bool
    torch_version: Optional[str] = None
    gpu: Optional[dict] = None
    nvidia: Optional[dict] = None
    compute_capability: Optional[list] = None


@router.get("", response_model=DeviceStatus)
def get_device_status() -> DeviceStatus:
    """Estado del dispositivo: PyTorch/CUDA y GPU NVIDIA (nvidia-smi)."""
    manager = DeviceManager()
    gpu_info = manager.get_gpu_info()

    gpu = None
    if gpu_info:
        gpu = {
            "name": gpu_info.name,
            "memory_total_gb": round(gpu_info.memory_total_gb, 2),
            "memory_free_gb": round(gpu_info.memory_free_gb, 2)
            if gpu_info.memory_free_gb else None,
            "compute_capability": gpu_info.compute_capability,
            "cuda_version": gpu_info.cuda_version,
        }

    nvidia = None
    nv = GpuSetup.detect_nvidia()
    if nv.available:
        nvidia = {
            "available": True,
            "gpu_name": nv.gpu_name,
            "driver_version": nv.driver_version,
            "cuda_version": nv.cuda_version,
            "memory_total_gb": nv.memory_total_gb,
        }
    else:
        nvidia = {"available": False, "error": nv.error}

    import torch

    return DeviceStatus(
        device=str(manager.device),
        cuda_available=manager.is_cuda,
        torch_version=torch.__version__,
        gpu=gpu,
        nvidia=nvidia,
        compute_capability=torch.cuda.get_device_capability(0)
        if manager.is_cuda else None,
    )
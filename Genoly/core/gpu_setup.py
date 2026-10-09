"""
Detección de la GPU NVIDIA e instalación de la build de PyTorch con CUDA.

Uso desde la línea de comandos:

    python -m Genoly.core.gpu_setup [--install] [--dry-run] [--tag cuXXX]
"""

import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Mapeo de builds de PyTorch (etiqueta cuXXX) -> versión de CUDA mínima
# que exige el driver NVIDIA.
#
# El driver NVIDIA es retrocompatible: un wheel compilado con cuXXX funciona
# si el driver soporta CUDA >= XXX (lo que reporta nvidia-smi).
# --------------------------------------------------------------------------- #
PYTORCH_CUDA_BUILDS: List[Tuple[str, float]] = [
    ("cu130", 13.0),
    ("cu129", 12.9),
    ("cu128", 12.8),
    ("cu126", 12.6),
    ("cu124", 12.4),
    ("cu121", 12.1),
    ("cu118", 11.8),
]

INDEX_URL = "https://download.pytorch.org/whl/{tag}"

#: Etiqueta CUDA válida: 'cu' seguido de exactamente tres dígitos.
CUDA_TAG_RE = re.compile(r"^cu\d{3}$")


@dataclass
class NvidiaSystemInfo:
    """Información del sistema NVIDIA obtenida con nvidia-smi."""
    available: bool
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None          # CUDA máximo soportado por el driver
    gpu_name: Optional[str] = None
    memory_total_gb: Optional[float] = None
    error: Optional[str] = None


class GpuSetup:
    """
    Detección de la GPU NVIDIA con nvidia-smi e instalación automática de la
    build de PyTorch con CUDA más conveniente.

    Evita el problema más común: tener PyTorch compilado solo para CPU y que
    `torch.cuda.is_available()` devuelva False, enviando el pipeline a CPU.
    """

    # ---------------------------------------------------------------------- #
    # Detección del sistema
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _run_nvidia_smi() -> Tuple[int, str]:
        try:
            proc = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=30,
            )
            return proc.returncode, proc.stdout
        except FileNotFoundError:
            return 1, ""
        except subprocess.TimeoutExpired:
            return 1, ""

    @classmethod
    def detect_nvidia(cls) -> NvidiaSystemInfo:
        """
        Consulta nvidia-smi y extrae driver, versión de CUDA y GPU.

        Returns:
            NvidiaSystemInfo con los datos del sistema.
        """
        code, output = cls._run_nvidia_smi()
        if code != 0 or not output:
            return NvidiaSystemInfo(available=False,
                                    error="nvidia-smi no disponible o sin GPU NVIDIA")

        # Formato antiguo: "Driver Version: 591.59 | CUDA Version: 13.1"
        # Formato nuevo (driver 610+): "KMD Version: 610.57.04 | CUDA UMD Version: 13.3"
        driver = re.search(r"(?:Driver|KMD)\s+Version:\s*([\d.]+)", output)
        cuda = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*([\d.]+)", output)
        gpu = re.search(r"NVIDIA\s+([A-Za-z0-9 \-]+)", output)
        memory = re.search(r"([\d]+)MiB\s*/\s*([\d]+)MiB", output)

        return NvidiaSystemInfo(
            available=True,
            driver_version=driver.group(1) if driver else None,
            cuda_version=cuda.group(1) if cuda else None,
            gpu_name=gpu.group(1).strip() if gpu else None,
            memory_total_gb=(float(memory.group(2)) / 1024) if memory else None,
        )

    # ---------------------------------------------------------------------- #
    # Recomendación de build
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _parse_version(version: Optional[str]) -> Optional[float]:
        if not version:
            return None
        match = re.match(r"(\d+)\.(\d+)", version)
        if not match:
            return None
        return float(f"{match.group(1)}.{match.group(2)}")

    @classmethod
    def recommend_cuda_tag(cls, cuda_version: Optional[str]) -> Optional[str]:
        """
        Elige la build de PyTorch más conveniente dado el CUDA del driver.

        Se selecciona el tag cuXXX más reciente que el driver soporte.
        """
        if not cuda_version:
            return None
        smi_cuda = cls._parse_version(cuda_version)
        if smi_cuda is None:
            return None

        for tag, required in PYTORCH_CUDA_BUILDS:
            if smi_cuda >= required:
                return tag
        return PYTORCH_CUDA_BUILDS[-1][0]

    # ---------------------------------------------------------------------- #
    # Detección de PyTorch
    # ---------------------------------------------------------------------- #
    @classmethod
    def torch_status(cls) -> Dict[str, object]:
        """Estado del PyTorch instalado (versión, CUDA detectado, GPU)."""
        status: Dict[str, object] = {"installed": False}
        try:
            import torch  # noqa: WPS433
        except ImportError:
            status["reason"] = "PyTorch no está instalado"
            return status

        cuda_available = torch.cuda.is_available()
        torch_cuda_version = getattr(torch.version, "cuda", None)

        status.update({
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "torch_cuda_version": torch_cuda_version,
            "device": str(torch.device("cuda" if cuda_available else "cpu")),
        })

        if cuda_available:
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["compute_capability"] = torch.cuda.get_device_capability(0)
        else:
            status["reason"] = (
                "PyTorch no detecta CUDA (probablemente build de CPU). "
                "Instala una build con CUDA para usar la GPU NVIDIA."
            )
        return status

    # ---------------------------------------------------------------------- #
    # Comandos de instalación
    # ---------------------------------------------------------------------- #
    @staticmethod
    def validate_cuda_tag(cuda_tag: str) -> str:
        """
        Comprueba que la etiqueta CUDA tenga la forma ``cuXXX``.

        La etiqueta se interpola en la URL del índice de pip y viaja hasta
        el proceso hijo, así que se valida antes de construir la lista de
        argumentos (nunca se pasa por un shell).

        Args:
            cuda_tag: Etiqueta candidata, p. ej. 'cu126'.

        Returns:
            La etiqueta validada.

        Raises:
            ValueError: Si no cumple ``cu\\d{3}``.
        """
        if not isinstance(cuda_tag, str) or not CUDA_TAG_RE.match(cuda_tag):
            raise ValueError(
                f"Etiqueta CUDA inválida: {cuda_tag!r} "
                "(se espera 'cu' + 3 dígitos, p. ej. 'cu126')")
        return cuda_tag

    @classmethod
    def install_argv(cls, cuda_tag: str,
                     include_torchvision: bool = True) -> List[str]:
        """
        Argumentos (sin shell) para instalar PyTorch con la build CUDA dada.

        Args:
            cuda_tag: Etiqueta CUDA, por ejemplo 'cu126'.
            include_torchvision: Incluir torchvision en la instalación.

        Returns:
            Lista de argumentos lista para ``subprocess.run``.
        """
        cls.validate_cuda_tag(cuda_tag)
        packages = ["torch"] + (["torchvision"] if include_torchvision else [])
        return [sys.executable, "-m", "pip", "install", *packages,
                "--index-url", INDEX_URL.format(tag=cuda_tag)]

    @classmethod
    def install_command(cls, cuda_tag: str,
                        include_torchvision: bool = True) -> str:
        """
        Comando pip (solo para mostrar) de la instalación con CUDA dada.

        Args:
            cuda_tag: Etiqueta CUDA, por ejemplo 'cu126'.
            include_torchvision: Incluir torchvision en la instalación.

        Returns:
            Comando pip listo para copiar y pegar.
        """
        return shlex.join(cls.install_argv(cuda_tag, include_torchvision))

    @classmethod
    def install_hint(cls, cuda_tag: Optional[str]) -> str:
        """Mensaje con el comando de instalación (o un aviso si no hay tag)."""
        if not cuda_tag:
            return ("No se pudo determinar la build de CUDA recomendada; "
                    "instala a mano con --tag cuXXX.")
        return cls.install_command(cuda_tag)

    @classmethod
    def install_cuda_torch(cls, cuda_tag: Optional[str] = None,
                           include_torchvision: bool = True,
                           dry_run: bool = True) -> int:
        """
        Instala la build de PyTorch con CUDA más conveniente.

        Args:
            cuda_tag: Forzar un tag concreto (p. ej. 'cu126'). Si es None,
                      se recomienda automáticamente a partir de nvidia-smi.
            include_torchvision: Incluir torchvision.
            dry_run: Si True, solo muestra el comando sin ejecutarlo.

        Returns:
            Código de salida (0 = éxito).
        """
        info = cls.detect_nvidia()
        if not info.available:
            print(f"Error: {info.error}")
            return 1

        tag = cuda_tag or cls.recommend_cuda_tag(info.cuda_version)
        if not tag:
            print("No se pudo determinar la build de CUDA recomendada.")
            return 1

        command = cls.install_command(tag, include_torchvision)
        print(f"GPU detectada: {info.gpu_name}")
        print(f"Driver: {info.driver_version} | CUDA soportado: {info.cuda_version}")
        print(f"Build de PyTorch recomendada: {tag}")
        print(f"Comando: {command}")

        if dry_run:
            print("\n[DRY RUN] No se ejecutó la instalación. "
                  "Ejecuta el comando anterior o llama con dry_run=False.")
            return 0

        print("\nInstalando PyTorch con CUDA...")
        # Nunca por shell: los argumentos van como lista, y el tag ya está
        # validado por install_argv.
        code = subprocess.run(
            cls.install_argv(tag, include_torchvision)
        ).returncode
        if code == 0:
            print("Instalación completada.")
        else:
            print(f"Error durante la instalación (código {code}).")
        return code

    # ---------------------------------------------------------------------- #
    # Orquestador principal
    # ---------------------------------------------------------------------- #
    @classmethod
    def ensure_cuda_torch(cls, auto_install: bool = False,
                          dry_run: bool = False) -> int:
        """
        Verifica la GPU NVIDIA y que PyTorch use CUDA, instalando lo necesario.

        Flujo:
            1. Consulta nvidia-smi (driver y versión de CUDA).
            2. Comprueba si PyTorch detecta la GPU.
            3. Si PyTorch no usa CUDA, recomienda/instala la build adecuada.

        Args:
            auto_install: Si True, ejecuta la instalación automáticamente.
                          Si False, solo muestra el comando recomendado.
            dry_run: Solo imprimir información y comandos, sin instalar.

        Returns:
            Código de salida (0 = GPU lista o instalación correcta).
        """
        print("=" * 70)
        print("GENOLY-GPU | Comprobación de GPU NVIDIA")
        print("=" * 70)

        info = cls.detect_nvidia()
        if not info.available:
            print(f"ERROR: {info.error}")
            print("No hay GPU NVIDIA o falta el driver. Se usará CPU.")
            return 1

        print(f"GPU: {info.gpu_name}")
        print(f"Driver: {info.driver_version} | CUDA del driver: {info.cuda_version}")
        if info.memory_total_gb:
            print(f"Memoria: {info.memory_total_gb:.1f} GB")

        tag = cls.recommend_cuda_tag(info.cuda_version)
        print(f"Build de PyTorch compatible recomendada: {tag}")

        status = cls.torch_status()
        if not status["installed"]:
            print(f"PyTorch: {status.get('reason')}")
            if auto_install and not dry_run and tag:
                return cls.install_cuda_torch(tag, dry_run=False)
            print(f"\nInstala PyTorch con:\n  {cls.install_hint(tag)}")
            return 1

        print(f"PyTorch: {status['version']} | "
              f"torch.cuda.is_available()={status['cuda_available']}")

        if status["cuda_available"]:
            print(f"GPU activa: {status['gpu_name']} "
                  f"(compute {status['compute_capability']})")
            print("Estado OK: PyTorch usa CUDA.")
            return 0

        print(f"AVISO: {status.get('reason')}")
        print(f"  Versión instalada: {status['version']}")
        print(f"  CUDA de la build: {status.get('torch_cuda_version') or 'CPU-only'}")

        if auto_install and not dry_run and tag:
            print("\nReinstalando PyTorch con CUDA...")
            return cls.install_cuda_torch(tag, dry_run=False)

        print(f"\nPara activar la GPU, reinstala PyTorch con:\n  "
              f"{cls.install_hint(tag)}")
        return 1


def recommend_cuda_tag(cuda_version: Optional[str]) -> Optional[str]:
    """Función de conveniencia para elegir la build de PyTorch (GpuSetup)."""
    return GpuSetup.recommend_cuda_tag(cuda_version)


def main() -> int:
    """Interfaz de línea de comandos."""
    args = sys.argv[1:]

    if "-h" in args or "--help" in args:
        print(__doc__)
        print("Uso: python -m Genoly.core.gpu_setup [--install] [--dry-run] [--tag cuXXX]")
        return 0

    tag = None
    if "--tag" in args:
        idx = args.index("--tag")
        if idx + 1 < len(args):
            tag = args[idx + 1]
        if tag is not None:
            try:
                GpuSetup.validate_cuda_tag(tag)
            except ValueError as exc:
                print(f"Error: {exc}")
                return 2

    try:
        if "--install" in args:
            return GpuSetup.install_cuda_torch(tag, dry_run="--dry-run" in args)

        return GpuSetup.ensure_cuda_torch(auto_install="--auto" in args,
                                          dry_run="--dry-run" in args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
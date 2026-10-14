#!/usr/bin/env bash
# Script: setup_genoly.sh
# Equivalente Linux de setup_genoly.ps1: monta la estructura completa de Genoly-GPU.
# Guardar en UTF-8 (sin BOM). Ejecutar: ./setup_genoly.sh

set -euo pipefail

PROJECT_NAME="genoly_gpu"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { printf "  ${GREEN}OK${NC} %s\n" "$1"; }
info() { printf "\n${YELLOW}%s${NC}\n" "$1"; }

printf "${CYAN}=========================================${NC}\n"
printf "${CYAN}  Creando estructura de Genoly-GPU${NC}\n"
printf "${CYAN}=========================================${NC}\n"

# Directorio principal
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"
printf "\n${GREEN}OK${NC} Directorio principal creado: %s\n" "$PROJECT_NAME"

# ============================================================================
# 1. ARCHIVOS DE CONFIGURACIÓN DEL PROYECTO
# ============================================================================

info "1. Creando archivos de configuración..."

cat > .gitignore <<'GITIGNORE_EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Contexto proyecto
context/
comando.txt
contexto.md

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Datos genomicos grandes
*.fasta
*.fastq
*.fastq.gz
*.bam
*.bai
*.vcf
*.vcf.gz
*.bcf
*.bcf.csi
*.fa.gz
*.fna.gz
*.sam

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Windows
Desktop.ini

# Tunel ngrok (script local de despliegue)
start_tunnel.ps1

# Descargas NCBI / archivos temporales de secuencias
pantera

# Datos reales descargados para pruebas locales (descargar_datos_test.py)
datos_reales/

# UI (React/Vite)
node_modules/
ui/frontend/dist/
.vite/
GITIGNORE_EOF
ok ".gitignore"

cat > setup.py <<'SETUP_EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genoly-gpu",
    version="0.3.0",
    author="Genoly-GPU",
    author_email="sebasdeasturias@gmail.com",
    description="Software de aceleracion por GPU (NVIDIA/CUDA) para el analisis de grandes datos del genoma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beoasaver-boop/Genoly-GPU",
    packages=find_packages(exclude=["examples", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
)
SETUP_EOF
ok "setup.py"

cat > requirements.txt <<'REQUIREMENTS_EOF'
torch>=2.0.0
numpy>=1.20.0
biopython>=1.79
matplotlib>=3.5
requests>=2.25
openpyxl>=3.1
REQUIREMENTS_EOF
ok "requirements.txt"

cat > README.md <<'README_EOF'
# Genoly-GPU

Software de análisis de genómica acelerado por GPU (NVIDIA/CUDA) usando PyTorch.
Incluye un pipeline completo: I/O de FASTA/FASTQ, control de calidad, conteo de
k-mers, alineamiento, llamada de variantes, modelos cuantitativos (LMM/GBLUP) y
una interfaz web (React + FastAPI).

> [!NOTE]
> Requiere Python >= 3.8 y NumPy. Para aceleración por GPU se necesita una GPU
> NVIDIA con CUDA; sin ella el pipeline funciona en CPU de forma automática.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Comprobación de GPU

```bash
python -m Genoly.core.gpu_setup
```

> [!IMPORTANT]
> Si PyTorch no detecta la GPU, ejecutar el comando anterior: detecta
> nvidia-smi y recomienda o instala la build de PyTorch con CUDA adecuada.

## Uso rápido

```bash
python examples/pipeline_completo.py
python tests/test_smoke.py
```

## Interfaz web

```bash
python -m uvicorn ui.backend.main:app --host 0.0.0.0 --port 8000
cd ui/frontend
npm install
npm run dev
```

> [!TIP]
> En producción, compilar el frontend con `npm run build`: el backend sirve
> `ui/frontend/dist` en la raíz.
README_EOF
ok "README.md"

# ============================================================================
# 2. PAQUETE PRINCIPAL Genoly/
# ============================================================================

info "2. Creando paquete principal Genoly/..."

mkdir -p Genoly/core Genoly/io Genoly/encoding Genoly/qc Genoly/kmer \
         Genoly/variants Genoly/quantitative Genoly/alignment

modulo() {
    printf '"""%s"""\n' "$2" > "$1"
    ok "$1"
}

touch Genoly/core/__init__.py Genoly/io/__init__.py Genoly/encoding/__init__.py
touch Genoly/qc/__init__.py Genoly/kmer/__init__.py Genoly/variants/__init__.py
touch Genoly/alignment/__init__.py

modulo "Genoly/__init__.py" "Paquete principal de Genoly-GPU: análisis genómico acelerado por GPU (NVIDIA/CUDA)."
modulo "Genoly/core/device.py" "DeviceManager, GPUInfo y get_device: gestión de dispositivo (CUDA/CPU)."
modulo "Genoly/core/gpu_setup.py" "GpuSetup y recommend_cuda_tag: detección de nvidia-smi y auto-instalación de PyTorch con CUDA."
modulo "Genoly/io/fasta.py" "FastaRecord, FastaReader, read_fasta, write_fasta y fasta_to_batches: I/O FASTA en streaming."
modulo "Genoly/io/fastq.py" "FastqRecord, FastqReader, read_fastq, write_fastq y fastq_to_batches: I/O FASTQ con scores Phred."
modulo "Genoly/encoding/encoder.py" "SequenceEncoder: secuencias a tensores int/one-hot en GPU."
modulo "Genoly/qc/quality.py" "QualityAnalyzer y QualityReport: GC, composición, calidad Phred, trimming y filtrado."
modulo "Genoly/kmer/kmers.py" "KmerCounter: conteo y espectro de k-mers en GPU (Horner int64 base-4, exacto hasta k=31)."
modulo "Genoly/variants/caller.py" "VariantCaller, Variant, Read y PileupResult: pileup y llamada de SNV/DEL en GPU."
modulo "Genoly/quantitative/__init__.py" "API cuantitativa: LMM, GRM, REML, GBLUP y preprocesado de datos."
modulo "Genoly/quantitative/lmm.py" "LinearMixedModel y LMMResult: fachada LMM animal (REML/ML + BLUP); re-exporta build_kinship."
modulo "Genoly/quantitative/grm.py" "build_kinship: GRM VanRaden y GCTA desde dosis {0, 1, 2}."
modulo "Genoly/quantitative/reml.py" "estimate_variance_components y solve_variance_components: motor REML/ML por puntuación de Fisher."
modulo "Genoly/quantitative/gblup.py" "GenomicBLUP y GBLUPResult: GBLUP directo con fiabilidad vía PEV."
modulo "Genoly/quantitative/preprocess.py" "prepare_quantitative_data: carga CSV/Excel con limpieza e imputación."
modulo "Genoly/quantitative/utils.py" "prepare_model_inputs y cholesky_regularized: validación de entradas y Cholesky con jitter."
modulo "Genoly/alignment/alignment.py" "GPUSequenceAligner: Smith-Waterman en GPU con variantes, CIGAR y benchmark."
modulo "Genoly/alignment/alignment_wExa.py" "Versión LEGACY del alineador: conservar, no modernizar sin orden explícita."

# ============================================================================
# 3. INTERFAZ WEB ui/
# ============================================================================

info "3. Creando interfaz web ui/..."

mkdir -p ui/backend/routers \
         ui/frontend/src/components ui/frontend/src/pages

modulo "ui/backend/main.py" "Aplicación FastAPI de Genoly-GPU: monta el frontend compilado (ui/frontend/dist)."
touch ui/backend/requirements.txt

modulo "ui/backend/routers/device.py" "Rutas GET /api/device y /api/setup: estado de GPU e instalación."
modulo "ui/backend/routers/qc.py" "Ruta POST /api/qc/analyze: análisis de control de calidad."
modulo "ui/backend/routers/kmer.py" "Ruta POST /api/kmer/count: conteo de k-mers."
modulo "ui/backend/routers/variants.py" "Ruta POST /api/variants/call: llamada de variantes."
modulo "ui/backend/routers/quantitative.py" "Ruta POST /api/quantitative/fit: ajuste del LMM."
modulo "ui/backend/routers/gblup.py" "Ruta POST /api/gblup/predict: predicción GBLUP."
modulo "ui/backend/routers/upload.py" "Subida de FASTA grandes por streaming."
touch ui/backend/routers/__init__.py

touch ui/frontend/package.json ui/frontend/vite.config.js
touch ui/frontend/tailwind.config.js ui/frontend/postcss.config.js
touch ui/frontend/index.html
touch ui/frontend/src/main.jsx ui/frontend/src/App.jsx
touch ui/frontend/src/api.js ui/frontend/src/index.css
touch ui/frontend/src/themes.jsx ui/frontend/src/fasta.js
touch ui/frontend/src/quantgen.js ui/frontend/src/tabular.js
touch ui/frontend/src/components/Layout.jsx
touch ui/frontend/src/components/ui.jsx
touch ui/frontend/src/components/FastaPanel.jsx
touch ui/frontend/src/components/PreprocessReport.jsx
touch ui/frontend/src/pages/Dashboard.jsx ui/frontend/src/pages/Device.jsx
touch ui/frontend/src/pages/Qc.jsx ui/frontend/src/pages/Kmer.jsx
touch ui/frontend/src/pages/Variants.jsx ui/frontend/src/pages/Quantitative.jsx
touch ui/frontend/src/pages/Gblup.jsx
ok "ui/frontend (estructura React/Vite)"

# ============================================================================
# 4. TESTS Y EJEMPLOS
# ============================================================================

info "4. Creando tests/ y examples/..."

mkdir -p tests examples datos_reales context

modulo "tests/test_smoke.py" "44 tests de humo de Genoly-GPU (pytest o unittest)."
modulo "examples/pipeline_completo.py" "Demo del pipeline completo: I/O, QC, k-mers y variantes."
modulo "examples/examp1.py" "Demo del alineador GPUSequenceAligner."
modulo "examples/analisis1.py" "Análisis de FASTA con Biopython."
modulo "examples/analisis2.py" "Sitios de restricción con Biopython."

printf "\n${GREEN}=========================================${NC}\n"
printf "${GREEN}  Estructura creada en %s/${NC}\n" "$PROJECT_NAME"
printf "${GREEN}=========================================${NC}\n"
printf "\nSiguientes pasos:\n"
printf "  1. python -m venv .venv && source .venv/bin/activate\n"
printf "  2. pip install -e .\n"
printf "  3. python -m Genoly.core.gpu_setup   (verificar GPU)\n"
printf "  4. python tests/test_smoke.py        (44 tests)\n"
printf "  5. Copiar el código real en los módulos placeholders\n"
printf "\nNota: datos_reales/ y context/ están ignorados por git.\n"

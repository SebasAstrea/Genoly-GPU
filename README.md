# Genoly-GPU

Software de aceleración por tecnología de GPU (NVIDIA por ahora) para el análisis de grandes datos del genoma.

Genoly-GPU ofrece las herramientas de un pipeline de genómica estándar —I/O de FASTA/FASTQ, control de calidad, conteo de k-mers, alineamiento, codificación a tensores y llamada de variantes— todo acelerado por GPUs NVIDIA a través de PyTorch y CUDA.

> **Documentación extendida**: [docs/que-es-genoly.md](docs/que-es-genoly.md) (qué es, necesidad y diferenciación) y [docs/arquitectura.md](docs/arquitectura.md) (arquitectura técnica y decisiones de diseño).

## Caracteristicas

- **Pipeline de streaming para archivos multi-GB**: lectura por bloques de disco (64 KiB), RAM batching de registros, micro-lotes adaptativos de GPU según la VRAM libre y liberación de memoria tras cada micro-lote. Procesa genomas completos sin OOM (ver [Pipeline de streaming](#pipeline-de-streaming-archivos-multi-gb)).
- Lectura/escritura de FASTA y FASTQ en streaming (consumo de memoria reducido), con soporte de FASTQ multi-registro y multilínea Illumina.
- Codificación de secuencias a tensores enteros y one-hot sobre GPU.
- Control de calidad: contenido GC, composición de bases y distribución de calidad Phred, con trimming y filtrado de lecturas.
- Conteo de k-mers y espectro k-mer acelerado por GPU (codificación base-4 entera en int64, exacta hasta k=31), con estimación de tamaño de genoma (Lander-Waterman).
- Pileup y llamada de variantes (SNV y deleciones) sobre GPU mediante operaciones de dispersión.
- **Llamada de variantes estilo GATK HaplotypeCaller**: regiones activas sobre GPU, reensamblaje local (grafo de de Bruijn), PairHMM con tres backends (kernel C nativo, PyTorch GPU o NumPy) y genotipado completo PL/GQ/AD/DP/QUAL con exportación VCF 4.2.
- **BQSR (Base Quality Score Recalibration)**: covariables calidad × ciclo × hebra, tabla de recalibración y SAM con calidades corregidas al estilo GATK.
- **Detección de duplicados (MarkDuplicates)**: claves 5' sin clips, puntuación por calidades, flags 0x400 y estadísticas por biblioteca.
- Genética cuantitativa sobre GPU: modelos lineales mixtos (LMM) con estimación REML/ML, matriz de parentesco genómica (VanRaden o GCTA) y predicción de valores de cría (BLUP).
- Predicción genómica GBLUP en un paso con varianzas conocidas o estimadas por REML, incluyendo fiabilidad y precisión por individuo (PEV).
- Predicción por modelo animal PBLUP a partir del pedigrí (columnas sire/dam): matriz de parentesco A, endogamia F, REML o varianzas fijas, fiabilidades.
- Flujo obligatorio de limpieza: todo dataset de genética cuantitativa (GBLUP, PBLUP o modelos mixtos) se adjunta y limpia primero en la página Datos, con vista previa (dry-run), filtros por columna y tabla interactiva paginada.
- Carga de tablas CSV o Excel con preprocesamiento: detección de cabecera y delimitador, conversión de decimales con coma, limpieza de columnas/filas no válidas e imputación de dosis perdidas por media o moda.
- Detección automática de la GPU NVIDIA con nvidia-smi e instalación automática de la build de PyTorch con CUDA más conveniente.
- Alineamiento con el algoritmo Smith-Waterman implementado sobre PyTorch.
- Análisis completo de mutaciones contra una referencia y comparación con variantes conocidas.
- Generación de CIGAR strings y benchmark de rendimiento GPU vs CPU.
- Utilidades para descargar secuencias FASTA desde NCBI.

> [!NOTE]
> Actualmente el soporte de GPU está orientado a NVIDIA (CUDA). Si el dispositivo no dispone de GPU CUDA, el software funciona igualmente sobre CPU.

> [!IMPORTANT]
> El alineamiento Smith-Waterman es un algoritmo de programación dinámica con coste cuadrático O(n*m). En secuencias de gran tamaño o volúmenes de datos masivos, es fundamental disponer de una GPU con memoria suficiente y usar el procesamiento por lotes.

## Estructura del proyecto

```
Genoly-GPU/
├── .github/
│   └── workflows/ci.yml       # CI: ruff + pytest en 3.10 y 3.12
├── pyproject.toml             # Paquete (pip install .), pytest y ruff
├── setup.py                   # Shim para pip/setuptools antiguos
├── requirements.txt           # Mismas dependencias que pyproject.toml
├── Genoly/
│   ├── __init__.py
│   ├── core/
│   │   ├── device.py             # DeviceManager: detección y gestión de GPU/CUDA
│   │   ├── gpu_setup.py          # Auto-detección nvidia-smi e instalación de PyTorch CUDA
│   │   └── vram.py               # VRAMManager: micro-batching, VRAM dinámica y reintentos OOM
│   ├── io/
│   │   ├── fasta.py              # FASTA por bloques de disco, lotes de RAM y ventanas
│   │   ├── fastq.py              # Lectura/escritura FASTQ con calidad Phred
│   │   └── sam.py                # Escritura SAM/BAM y parser de alineaciones
│   ├── encoding/
│   │   └── encoder.py            # Codificación a tensores enteros y one-hot (+ encode_stream)
│   ├── qc/
│   │   └── quality.py            # GC content, composición, calidad y filtrado
│   ├── kmer/
│   │   └── kmers.py              # Conteo de k-mers en GPU (+ streaming count_stream/count_fasta)
│   ├── variants/
│   │   ├── caller.py             # Pileup y llamada de variantes en GPU
│   │   └── vcf.py                # Escritura VCF (SNV y deleciones ancladas con END)
│   ├── alignment/
│   │   ├── alignment.py          # GPUSequenceAligner (fallback a parasail cuando no hay GPU)
│   │   └── mapper.py             # Seed-and-extend: lecturas → SAM
│   ├── annotation/
│   │   └── annotator.py          # Anotación funcional GFF / Gene Ontology
│   ├── downstream/
│   │   └── reduction.py          # PCA, KMeans y t-SNE sobre tensores GPU
│   ├── report/
│   │   └── plots.py              # SVG de heatmap, volcano e informes Markdown
│   ├── quantitative/
│       ├── lmm.py                # LinearMixedModel: fachada LMM (REML/ML + BLUP)
│       ├── gblup.py              # GenomicBLUP con fiabilidad y precisión
│       ├── gwas.py               # Asociación GWAS
│       ├── grm.py                # Matriz de parentesco genómica (VanRaden, GCTA)
│       ├── reml.py               # Estimación REML/ML por puntuación de Fisher
│       ├── preprocess.py         # Carga CSV/Excel con limpieza e imputación
│       ├── crossval.py           # Validación cruzada
│       ├── popgen.py             # MAF / HWE y estadísticas de poblaciones
│       └── utils.py              # Validación de datos y Cholesky regularizada
│   └── gatk/                     # Port de GATK 4.x (HaplotypeCaller, BQSR, MarkDuplicates)
│       ├── reads.py              # GATKRead y parser SAM de texto (sin dependencias)
│       ├── pairhmm.py            # PairHMM Gotoh: backends nativo C / torch / numpy
│       ├── active_region.py      # Detección de regiones activas (pileup GPU)
│       ├── assembly.py           # Reensamblaje local (grafo de de Bruijn)
│       ├── likelihoods.py        # Likelihoods de haplotipo y genotipado PL/GQ/AD
│       ├── genotyper.py          # HaplotypeCaller: orquestación de variantes
│       ├── bqsr.py               # BaseRecalibrator: covariables y tabla JSON
│       ├── markdups.py           # Detección y marcado de duplicados
│       └── vcf.py                # Escritura VCF estilo GATK (GT:AD:DP:GQ:PL)
├── csrc/
│   └── pairhmm.c                 # Kernel C del PairHMM (extensión opcional, MSVC/gcc)
├── docker/                         # Despliegue en contenedor (API + frontend con CUDA)
│   ├── Dockerfile                # Multi-stage: frontend Node + backend CPU/GPU condicional
│   ├── docker-compose.yml        # Servicios genoly-cpu / genoly-gpu / gpu-select
│   ├── detect_gpu.py             # Detector de GPU NVIDIA
│   ├── launch.py                 # Lanzador simple (Python)
│   ├── launch.sh                 # Lanzador simple (bash)
│   ├── setup_docker.sh           # Montaje automático robusto del contenedor (Linux/macOS/WSL2)
│   └── setup_docker.ps1          # Montaje automático robusto del contenedor (Windows)
├── docs/                           # arquitectura y documentación de producto
├── examples/
│   ├── pipeline_completo.py      # Pipeline completo: I/O -> QC -> k-mers -> variantes
│   ├── haplotype_caller_gatk.py  # HaplotypeCaller + BQSR + MarkDuplicates (GATK)
│   ├── examp1.py                 # Ejemplo de uso del alineador
│   ├── analisis1.py              # Análisis FASTA: composición y contenido GC
│   ├── analisis2.py              # Sitios de restricción y motivos consenso
│   └── seqdump.txt               # Isoformas del gen BRCA1 (ejemplo)
├── tests/
│   ├── test_smoke.py             # Humo: todos los módulos (dispositivo, I/O, QC, k-mers, LMM...)
│   ├── test_streaming.py         # Streaming RAM/VRAM acotadas y equivalencia con fuerza bruta
│   ├── test_fastq_pipeline.py    # Lectura FASTQ + jobs aislados de QC/procesado
│   ├── test_mapper.py            # Alineamiento seed-and-extend y su job
│   ├── test_variant_pipeline.py  # SAM -> variantes -> VCF y su job
│   ├── test_downstream.py        # PCA / KMeans / t-SNE y su job
│   ├── test_annotation_report.py # GFF/GO, SVG de heatmap y volcano, jobs de anotación/informe
│   ├── test_quantgen_extra.py    # Validación cruzada, GWAS, GRM, MAF/HWE, pedigrí (A/F), PBLUP
│   ├── test_gatk_pipeline.py     # HaplotypeCaller (paridad C/torch/numpy), BQSR, MarkDuplicates, API
│   ├── test_api.py               # API REST: auth, límites de subida, LFI, SSE, PBLUP
│   ├── test_worker.py            # Aislamiento de procesos y limpieza de spills huérfanos
│   └── conftest.py               # sys.path de la raíz (una sola vez)
├── ui/
│   ├── backend/                  # API REST FastAPI
│   │   ├── main.py               # App, CORS, autenticación por API key, limpieza periódica
│   │   ├── jobs.py               # Cola de trabajos con progreso SSE multi-suscriptor
│   │   ├── worker.py             # Proceso hijo aislado (OOM/segfault no tumban la API)
│   │   ├── uploads.py            # Registro de subidas
│   │   ├── datasets.py           # Registro de datasets y extracción ZIP
│   │   ├── security.py           # API key, CORS y directorios de datos permitidos
│   │   ├── registry.py           # Registro JSON con candado entre procesos
│   │   ├── maintenance.py        # Límites de tamaño y caducidad (TTL) de archivos
│   │   └── routers/              # device, qc, kmer, variants, map, downstream, annotation,
│   │                             # report, quantitative, gblup, pblup, qdata, dataset, upload,
│   │                             # jobs, gatk (HaplotypeCaller/BQSR/MarkDuplicates)
│   └── frontend/                 # UI React + Vite + Tailwind
│       └── src/                  # páginas y componentes
├── fetchingfasta.py              # Descarga de FASTA desde NCBI por accession
├── clean_fetching.py             # Descarga de FASTA desde una URL directa
├── descargar_datos_test.py       # Descarga datos reales para pruebas (datos_reales/, ignorado)
├── setup_genoly.ps1              # Script de instalación para Windows
├── setup_genoly.sh               # Script de instalación para Linux/macOS/WSL2
├── .dockerignore                 # Reduce el contexto de construcción de la imagen
├── requirements.txt
└── README.md
```

## Interfaz gráfica (UI)

Genoly-GPU incluye una interfaz web moderna construida con **React + Vite + Tailwind** y un backend **FastAPI** que expone los módulos GPU como API REST.

| Ruta | Funcionalidad |
|---|---|
| `/` | Dashboard con estado del dispositivo y accesos. |
| `/device` | Detección de GPU NVIDIA (nvidia-smi), driver, CUDA y build de PyTorch recomendada. |
| `/qc` | Control de calidad FASTA: GC content, composición y calidad Phred. |
| `/fastq` | Control de calidad y filtrado de lecturas cortas (FASTQ). |
| `/kmer` | Conteo de k-mers, espectro y estimación de tamaño de genoma. |
| `/map` | Alineamiento de lecturas (seed-and-extend) y exportación SAM. |
| `/variants` | Pileup y llamada de variantes (SNV/deleciones) y exportación VCF. |
| `/downstream` | PCA, clustering (KMeans) y t-SNE de datos genómicos. |
| `/annotation` | Anotación funcional a partir de GFF y Gene Ontology. |
| `/report` | Informes con heatmap y volcano (SVG/Markdown). |
| `/datos` | Adjuntar y limpiar datos de genética cuantitativa (obligatorio para LMM/GBLUP/PBLUP): subida CSV/Excel, columnas de fenotipo y pedigrí (ID/padre/madre), dry-run, filtros y tabla interactiva. |
| `/quantitative` | Genética cuantitativa: LMM (REML/ML) y valores de cría BLUP (consume la matriz limpia de Datos). |
| `/gblup` | Predicción genómica GBLUP con fiabilidad y precisión por individuo (consume la matriz limpia de Datos). |
| `/pblup` | Predicción por modelo animal PBLUP con matriz A de pedigrí, endogamia F y fiabilidades (consume la matriz limpia de Datos). |
| `/gatk` | Llamada de variantes estilo GATK HaplotypeCaller (ensamblaje local + PairHMM), MarkDuplicates y BQSR sobre SAM/FASTA, con VCF/SAM descargables. |

Además, los análisis de larga duración sobre archivos subidos se ejecutan como **trabajos de fondo con progreso en tiempo real**:

| Endpoint | Funcionalidad |
|---|---|
| `POST /api/kmer/count-async` | Encola el conteo de k-mers y devuelve un `job_id`. |
| `POST /api/gatk/call-file` | Encola HaplotypeCaller (SAM + FASTA -> VCF con progreso SSE). |
| `POST /api/gatk/bqsr` | Encola BQSR (tabla de recalibración JSON + SAM recalibrado). |
| `POST /api/gatk/markdups` | Encola MarkDuplicates (SAM con flags 0x400 + estadísticas). |
| `GET /api/jobs/{id}` | Estado y resultado del trabajo (JSON, para polling). |
| `GET /api/jobs/{id}/events` | Progreso en tiempo real vía **Server-Sent Events** (registros/bases leídas, micro-lotes GPU). |

## Instalación

### Requisitos previos

- **Python 3.8+** con pip y venv
- **Node.js 20.19+** (22 LTS recomendada) para la interfaz web
- **NVIDIA GPU** con drivers CUDA (opcional, recomendado)
- **Docker** (opcional, para despliegue en contenedor)

### Windows

#### Instalación automática (PowerShell)

```powershell
# Ejecutar como Administrador
powershell -ExecutionPolicy Bypass -File .\setup_genoly.ps1
```

El script realiza las siguientes acciones:
1. Crea el entorno virtual `.venv` en la raíz del proyecto
2. Instala las dependencias del proyecto (torch, numpy, openpyxl)
3. Instala las dependencias de la API (fastapi, uvicorn, python-multipart)
4. Detecta GPU NVIDIA y recomienda la instalación de PyTorch CUDA
5. Opcionalmente, instala PyTorch con soporte CUDA

#### Instalación manual

```powershell
# 1. Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt -r ui\backend\requirements.txt

# 3. (Opcional) Instalar PyTorch con CUDA si hay GPU NVIDIA
python -m Genoly.core.gpu_setup --install

# 4. Verificar instalación
python -c "import torch; print(torch.cuda.is_available())"
```

### Linux / macOS / WSL2

#### Instalación automática (bash)

```bash
# Dar permisos de ejecución y ejecutar
chmod +x setup_genoly.sh
./setup_genoly.sh
```

#### Instalación manual

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt -r ui/backend/requirements.txt

# 3. (Opcional) Instalar PyTorch con CUDA si hay GPU NVIDIA
python -m Genoly.core.gpu_setup --install

# 4. Verificar instalación
python -c "import torch; print(torch.cuda.is_available())"
```

### Instalación de la interfaz web (UI)

La UI requiere Node.js 20.19+ (22 LTS recomendada). Verifica con `node -v`.

```bash
# Navegar al directorio del frontend
cd ui/frontend

# Instalar dependencias
npm install

# Compilar para producción
npm run build

# (Opcional) Modo desarrollo con hot-reload
npm run dev
```

> [!WARNING]
> Con Node.js 18 (o 20 anterior a 20.19) `npm install` y `npm run dev` fallan con `SyntaxError: The requested module "node:util" does not provide an export named styleText`. La UI usa Vite 8, que exige Node 20.19+/22.12+. Actualiza Node (nvm-windows/nvm o instalador oficial).

### Instalación de GPU (NVIDIA)

Si tienes una GPU NVIDIA, instala los drivers y el toolkit correspondiente:

**Windows:**
1. Descargar drivers: https://www.nvidia.com/download/index.aspx
2. Instalar CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Verificar: `nvidia-smi` en PowerShell

**Linux (Ubuntu/Debian):**

```bash
# Añadir repositorio NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Instalar CUDA Toolkit
sudo apt-get install -y cuda-toolkit-12-6

# Verificar
nvidia-smi
```

**WSL2:**

```bash
# Instalar drivers NVIDIA en Windows (no en WSL)
# Desde WSL2, verificar
nvidia-smi
```

> [!NOTE]
> Para usar GPU dentro de Docker en WSL2, instala el nvidia-container-toolkit siguiendo: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

## Puesta en marcha

Requisitos: **Node.js 20.19+ (se recomienda la 22 LTS)** para la UI y
**Python 3.8+** con las dependencias del proyecto para el backend.
Comprueba las versiones antes de empezar (`node -v` y `python --version`).

> [!WARNING]
> Con Node.js 18 (o 20 anterior a 20.19) `npm install` y `npm run dev`
> fallan con
> `SyntaxError: The requested module "node:util" does not provide an export named styleText`:
> la UI usa Vite 8, que exige Node 20.19+/22.12+. Actualiza Node
> (nvm-windows/nvm o instalador oficial) y vuelve a ejecutar `npm install`.

```bash
# 1. Backend (FastAPI + uvicorn) — entorno virtual recomendado
python -m venv .venv
.venv\Scripts\Activate.ps1              # Windows PowerShell
# source .venv/bin/activate             # Linux/macOS
# instala las dependencias de Genoly (torch, numpy, ...) Y las de la API
pip install -r requirements.txt -r ui/backend/requirements.txt
python -m uvicorn ui.backend.main:app --host 0.0.0.0 --port 8000
# Documentación de la API: http://localhost:8000/docs

# 1b. Alternativa desde ui/backend (con su propio .venv)
#     el bootstrap de main.py añade la raíz al sys.path automáticamente
cd ui/backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Frontend (dev con hot-reload) — Node.js 20.19+/22 LTS
cd ui/frontend
npm install
npm run dev        # http://localhost:5173

# 3. Producción: compilar y servir desde el backend
npm run build      # genera ui/frontend/dist
# el backend sirve el frontend compilado en http://localhost:8000
```

> [!IMPORTANT]
> **No uses `--workers` mayor que 1.** El estado de los trabajos de fondo y sus streams de progreso SSE viven en la memoria del proceso: con varios workers, el `POST` que crea el trabajo podría aterrizar en un proceso y el `GET` de eventos en otro (404). Además, cada proceso ejecutaría análisis GPU en paralelo compitiendo por la misma VRAM, anulando el micro-batching adaptativo. La concurrencia ya se gestiona dentro del proceso (los trabajos GPU se serializan y las peticiones HTTP no bloquean el event loop); ajusta el paralelismo con la variable de entorno `GENOLY_MAX_WORKERS` si lo necesitas.

> [!TIP]
> Durante desarrollo, Vite redirige `/api/*` a `http://127.0.0.1:8000` automáticamente, así que no necesitas configurar CORS manualmente.

## Variables de entorno

La API del backend (`ui/backend`) y el pipeline se configuran por variables de entorno:

| Variable | Por defecto | Descripción |
|---|---|---|
| `GENOLY_API_KEY` | *(vacía)* | Si se define, **toda ruta `/api/*` exige la clave**. Se acepta en la cabecera `X-API-Key`, en `Authorization: Bearer` y en `?api_key=` (este último lo usan `EventSource` y los enlaces de descarga, que no pueden mandar cabeceras). `/api/health` queda siempre pública porque el HEALTHCHECK de Docker no envía cabeceras. Vacía = modo local abierto (se registra un aviso al arrancar). |
| `GENOLY_CORS_ORIGINS` | orígenes de Vite (`localhost:5173`, `4173`) | Orígenes permitidos en CORS, separados por comas. En producción el frontend se sirve desde el propio FastAPI (mismo origen) y no hace falta CORS. |
| `GENOLY_ALLOWED_DATA_DIRS` | raíz del proyecto | Directorios desde los que se pueden registrar rutas de datos del servidor (`/api/upload/register` y `/api/dataset/register`). Fuera de ahí se responde 403. Hay además una lista negra de rutas sensibles (`/etc`, `/proc`, `~/.ssh`, `.env`, `id_rsa`, credenciales, ...) que se bloquea **siempre**, aunque esté dentro del allowlist. |
| `GENOLY_MAX_UPLOAD_BYTES` | `53687091200` (50 GiB) | Tamaño máximo por archivo, tanto en la carga multipart como en la carga por partes (tamaño declarado, offset y total). `0` = sin límite. |
| `GENOLY_UPLOAD_TTL_HOURS` | `168` (7 días) | Caducidad de las subidas gestionadas y de las extracciones de datasets; se borran junto con su entrada de registro. Los archivos **registrados** por el usuario nunca se tocan. `0` desactiva la limpieza. |
| `GENOLY_UPLOAD_DIR` | temporal del sistema | Directorio de subidas. Para archivos de decenas de GB apúntalo a un disco real (p. ej. `/var/tmp/genoly/uploads`). |
| `GENOLY_DATASET_DIR` | temporal del sistema | Directorio de datasets y de sus extracciones. |
| `GENOLY_SPILL_DIR` | temporal del sistema | Directorio de derrame del conteo de k-mers (un genoma completo puede ocupar decenas de GB). |
| `GENOLY_CLEANUP_INTERVAL` | `600` | Segundos entre pasadas de limpieza (spills huérfanos + TTL). |
| `GENOLY_DEVICE` | auto-detección | `cpu`, `cuda` o vacío. Pedir `cuda` sin CUDA disponible falla al arrancar con un mensaje claro. |
| `GENOLY_PORT` | `8000` | Puerto del servidor. En `docker/docker-compose.yml` cada perfil tiene el suyo: `GENOLY_GPU_PORT` (8001) y `GENOLY_GPU_SELECT_PORT` (8002). |

En Docker, todas ellas se pueden fijar en el bloque `environment:` de `docker/docker-compose.yml` o en un fichero `.env` junto al compose.

## Requisitos

- Python 3.8 o superior.
- Node.js 20.19 o superior (solo para la interfaz web: `npm install`/`npm run build`; se recomienda la 22 LTS).
- NVIDIA GPU con drivers CUDA (recomendado, no obligatorio).
- Dependencias:

| Paquete | Mínimo | Notas |
|---|---|---|
| torch | >= 2.0.0 | imprescindible |
| numpy | >= 1.20.0 | imprescindible |
| openpyxl | >= 3.1 | solo para leer Excel |
| scipy | >= 1.7 | GWAS (`Genoly.quantitative.gwas`) |
| pysam | >= 0.21 | lectura BAM/CRAM (opcional en código) |
| parasail | >= 1.3 | SIMD en Smith-Waterman (opcional en código) |
| biopython | >= 1.80 | ejemplos y anotación |
| matplotlib | >= 3.5 | gráficos del informe |
| requests | >= 2.25 | descarga de secuencias |

> [!NOTE]
> `openpyxl` solo se necesita para leer archivos de Excel (`.xlsx`/`.xls`) con el módulo de preprocesamiento; los CSV funcionan sin él.

Dependencias opcionales para los ejemplos:

- Biopython (para analisis1.py y analisis2.py)
- matplotlib (para los gráficos de analisis1.py)
- requests (para clean_fetching.py)

> [!TIP]
> Si ya tienes una GPU NVIDIA pero PyTorch fue instalado sin soporte CUDA, reinstálalo con el índice correcto:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

## Setup de GPU NVIDIA (auto-detección e instalación)

Genoly-GPU incluye un módulo que consulta `nvidia-smi` para detectar el driver y la versión de CUDA, y recomienda (o instala) automáticamente la build de PyTorch con CUDA más conveniente. Esto evita que PyTorch no detecte la GPU de NVIDIA y el pipeline caiga a CPU.

### Comprobación

```bash
python -m Genoly.core.gpu_setup
```

Salida de ejemplo:

```
GPU: GeForce RTX 3050
Driver: 591.59 | CUDA del driver: 13.1
Build de PyTorch compatible recomendada: cu130
PyTorch: 2.13.0+cu126 | torch.cuda.is_available()=True
GPU activa: NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute (8, 6))
Estado OK: PyTorch usa CUDA.
```

### Instalación automática

```bash
# Ver el comando pip recomendado sin ejecutarlo
python -m Genoly.core.gpu_setup --install --dry-run

# Ejecutar la instalación de la build recomendada
python -m Genoly.core.gpu_setup --install

# Forzar una build concreta
python -m Genoly.core.gpu_setup --install --tag cu126
```

> [!IMPORTANT]
> El driver NVIDIA es retrocompatible: una build `cuXXX` de PyTorch funciona si el driver soporta CUDA >= XXX (lo que reporta `nvidia-smi`). El módulo elige la build más reciente que tu driver soporte.

> [!CAUTION]
> `--install` modifica el entorno Python actual (reinstala `torch`/`torchvision`). Se recomienda ejecutarlo dentro de un entorno virtual. Usa `--dry-run` primero para revisar el comando.

### Uso programático

```python
from Genoly import GpuSetup

# Verificar y, si hace falta, instalar la build correcta
GpuSetup.ensure_cuda_torch(auto_install=True)

# Solo inspeccionar
info = GpuSetup.detect_nvidia()
print(info.gpu_name, info.driver_version, info.cuda_version)
```

## Docker

Genoly-GPU incluye `docker/Dockerfile` (multi-stage), `docker/docker-compose.yml` y `.dockerignore` para levantar la API y el frontend en un contenedor. Los scripts `docker/setup_docker.sh` (Linux/macOS/WSL2) y `docker/setup_docker.ps1` (Windows) montan el contenedor de forma automática y robusta: con acceso a las GPUs NVIDIA del sistema y verificación final de CUDA dentro del contenedor.

> [!TIP]
> La imagen construye el frontend React en una primera etapa y en la segunda instala Python con PyTorch. Soporta dos modos de construcción: **CPU** (imagen ligera) y **GPU** (con CUDA 12.6).

### Modos de construcción

| Modo | Imagen base | Tamaño aprox. | Uso |
|------|-------------|---------------|-----|
| CPU | `python:3.12-slim` | ~300 MB | Desarrollo, máquinas sin NVIDIA |
| GPU | `nvidia/cuda:12.6.0-runtime-ubuntu24.04` | ~2.5 GB | Producción con GPU NVIDIA |

> [!NOTE]
> El modo GPU requiere tener instalado el **nvidia-container-toolkit** en el host. Si no está disponible, el contenedor funcionará en CPU automáticamente.

### Montaje automático robusto (recomendado)

Los scripts `docker/setup_docker.sh` (Linux/macOS/WSL2) y `docker/setup_docker.ps1` (Windows PowerShell) montan el contenedor completo en un solo paso: comprueban Docker, el demonio y los permisos; liberan el puerto y retiran contenedores anteriores; validan la GPU del host y el NVIDIA Container Toolkit (en Linux ofrecen instalarlo y configurarlo); comprueban el espacio en disco; construyen la imagen (con Compose o con `docker build` si no está disponible); lanzan el contenedor con `--gpus all`; y verifican al final que PyTorch ve la GPU y que la API responde dentro del contenedor. Cada error se detalla con la acción correctiva concreta.

| bash | PowerShell | Descripción |
|---|---|---|
| `--gpu` | `-Gpu` | Todas las GPUs (por defecto si hay NVIDIA) |
| `--gpu-id N` | `-GpuId N` | Usar solo la GPU con índice N |
| `--cpu` | `-Cpu` | Modo CPU (sin GPU) |
| `--puerto N` | `-Puerto N` | Puerto del host y del contenedor (8000 por defecto) |
| `--sin-cache` | `-SinCache` | Reconstruir la imagen sin usar caché |
| `--solo-comprobar` | `-SoloComprobar` | Solo diagnóstico del entorno (no construye ni lanza) |
| `--list` | `-Listar` | Listar las GPUs detectadas y salir |
| `--ayuda` / `-h` | `-Ayuda` | Mostrar la ayuda |

```bash
# Linux/macOS/WSL2: GPU (todas las tarjetas)
./docker/setup_docker.sh

# Solo la GPU con índice 1, en el puerto 8010
./docker/setup_docker.sh --gpu-id 1 --puerto 8010

# Diagnóstico sin construir ni lanzar nada
./docker/setup_docker.sh --solo-comprobar
```

```powershell
# Windows (Docker Desktop con WSL2): GPU (todas las tarjetas)
powershell -ExecutionPolicy Bypass -File .\docker\setup_docker.ps1

# Solo la GPU con índice 1
powershell -ExecutionPolicy Bypass -File .\docker\setup_docker.ps1 -GpuId 1
```

Las imágenes resultantes se llaman `genoly-gpu:cpu-latest` y `genoly-gpu:gpu-latest`; los contenedores, `genoly-gpu-cpu`, `genoly-gpu-gpu` y `genoly-gpu-gpu-N` (según el modo).

> [!IMPORTANT]
> Usar GPU en Docker requiere **nvidia-container-toolkit** en el host. En Linux, el propio script lo detecta y ofrece instalarlo y configurarlo (`nvidia-ctk runtime configure --runtime=docker`). En Windows, Docker Desktop con backend WSL2 lo gestiona: instala el driver NVIDIA en Windows, ejecuta `wsl --update` y reinicia Docker Desktop. Guía: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

### Lanzador simple (opcional)

También existen lanzadores más sencillos (`docker/launch.sh` y `docker/launch.py`) que seleccionan el perfil de Compose según la GPU detectada, sin las comprobaciones ni la verificación del montaje robusto.

```bash
# Lanzamiento interactivo (detecta GPU automáticamente)
python docker/launch.py

# O con bash
./docker/launch.sh

# Modo CPU forzado
python docker/launch.py --cpu

# Modo GPU (todas las GPUs)
python docker/launch.py --gpu

# GPU específica (índice 1)
python docker/launch.py --gpu-id 1

# Puerto personalizado
python docker/launch.py --port 8080 --gpu

# Listar GPUs disponibles
python docker/launch.py --list
```

### Docker Compose manual

Si prefieres usar `docker compose` directamente:

```bash
# CPU (por defecto)
docker compose -f docker/docker-compose.yml up -d genoly-cpu

# GPU (todas)
docker compose -f docker/docker-compose.yml --profile gpu up -d genoly-gpu

# GPU específica (índice 1)
GPU_ID=1 docker compose -f docker/docker-compose.yml --profile gpu-select up -d genoly-gpu-selected

# Puerto personalizado (variable GENOLY_PORT)
GENOLY_PORT=8080 docker compose -f docker/docker-compose.yml --profile gpu up -d genoly-gpu
```

### Construcción manual con Dockerfile

```bash
# CPU (imagen ligera)
docker build -f docker/Dockerfile --build-arg GENOLY_GPU_MODE=cpu -t genoly-gpu:cpu .

# GPU (con CUDA)
docker build -f docker/Dockerfile --build-arg GENOLY_GPU_MODE=gpu -t genoly-gpu:gpu .

# Ejecutar con la GPU visible dentro del contenedor
docker run -d --name genoly --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -p 8000:8000 genoly-gpu:gpu
```

### Comprobaciones

```bash
# Estado del contenedor
docker compose -f docker/docker-compose.yml ps

# Healthcheck
curl http://localhost:8000/api/health
# {"status":"ok","version":"0.3.0"}

# Detección de GPU
curl http://localhost:8000/api/device
# {"cuda_available":true,"gpu_name":"NVIDIA GeForce RTX 3050",...}
```

### GPU dentro del contenedor

La API funciona sin GPU (PyTorch cae a CPU automáticamente gracias a la auto-detección de Genoly). Para exponer la GPU NVIDIA al contenedor, los scripts de montaje automático lanzan el contenedor con `--gpus all` y el `docker-compose.yml` declara la reserva de dispositivos:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

> [!IMPORTANT]
> Usar GPU en Docker requiere **nvidia-container-toolkit** en el host. En Windows con WSL2 hay que instalarlo dentro de la distro WSL2 y reiniciar Docker Desktop: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

> [!NOTE]
> En Docker Desktop con WSL2, si la distro ya tiene configurado el toolkit, la GPU puede estar disponible incluso sin el bloque `deploy`. Comprueba con `curl http://localhost:8000/api/device` (campo `cuda_available`).

### Solución de problemas en Docker

| Problema | Solución |
|----------|----------|
| `CUDA driver version is insufficient` | Actualizar drivers NVIDIA. Mínimo: 545.23.06 para CUDA 12.6 |
| `nvidia-smi: command not found` | Instalar drivers NVIDIA y `nvidia-utils` (Linux) |
| `Permission denied: /var/run/docker.sock` | Añadir usuario al grupo `docker`: `sudo usermod -aG docker $USER` |
| El contenedor no detecta la GPU | Verificar `nvidia-container-toolkit`: `nvidia-smi` dentro del contenedor |
| `could not select device driver "nvidia"` | Falta el toolkit: instalar `nvidia-container-toolkit` y `sudo systemctl restart docker` (los scripts `setup_docker.*` lo hacen en Linux; en Windows, Docker Desktop + WSL2) |
| `docker: 'compose' is not a docker command` | Usar `docker-compose` en sistemas antiguos, o actualizar Docker |

## Pipeline de streaming (archivos multi-GB)

Para genomas completos y datos crudos de varios GB, Genoly-GPU procesa bajo demanda con RAM y VRAM acotadas sea cual sea el tamaño de la fuente:

```
FASTA en disco ──bloques 64 KiB──> registros/ventanas ──> LOTE DE RAM (batch_size)
                                                                       │
                                              VRAMManager.plan_micro_batches()
                                                                       │
                                                            MICRO-LOTE GPU (n * L_max * b <= presupuesto)
                                                                       │
                                              gc.collect() + torch.cuda.empty_cache()
```

- **Archivo → RAM**: `FastaReader` lee por bloques de disco (64 KiB) y reconstruye los registros sin cargar el archivo. `iter_batches(batch_size)` agrupa registros (RAM batching); `iter_windows(w, overlap)` y `iter_windows_codes(w, overlap)` emiten ventanas (texto o códigos uint8 vectorizados) sin materializar ni siquiera un registro; `scan_stats` cuenta registros/bases vectorizadamente.
- **RAM → VRAM**: `VRAMManager` divide cada lote en micro-lotes que caben en una fracción segura (25 % por defecto) de la VRAM libre, y tras cada micro-lote libera memoria (`gc.collect()` + `torch.cuda.empty_cache()`).
- **Agregación**: en genomas reales casi todos los k-mers son únicos y el resultado exacto puede pesar varios GB. `count_fasta_aggregated` reparte los conteos en 256 particiones por bits bajos, derrama a `.npy` y deduplica partición a partición en la GPU: resultado **exacto** con pico de RAM `O(total_únicos / 256)` — nunca materializa la lista completa de k-mers.

Los detalles de implementación (aritmética int32 de dos palabras, coste por base por etapa, límites matemáticos del planificador) están documentados en los docstrings de `Genoly/core/vram.py` y `Genoly/kmer/kmers.py`.

### Uso programático

```python
from Genoly import KmerCounter, FastaReader

kc = KmerCounter()

# Archivo FASTA de cualquier tamaño: RAM ~ lote de ventanas, VRAM ~ presupuesto
values, counts = kc.count_fasta("genoma_completo.fasta", k=31, canonical=True)

# Genomas completos con millones de k-mers únicos: estadísticas exactas
# (total, espectro, top-k) con RAM acotada por derrame a disco
resumen = kc.count_fasta_aggregated("cromosoma.fasta", k=21, canonical=True, top=25)
print(resumen["total_unique"], resumen["total_kmers"], resumen["top_kmers"][0])

# Iterable perezoso de secuencias (streaming manual)
values, counts = kc.count_stream(FastaReader("datos.fasta").iter_sequences(), k=21)

# Registros con progreso (registros y bases leídos)
values, counts = kc.count_records(
    FastaReader("datos.fasta").records(), k=21,
    on_progress=lambda info: print(info))
# -> {"stage": "kmer", "records": 120, "bases": 9600, "units_done": ...,
#     "bases_done": 9600, "micro_batches": 7, "window_size": 1048576}

# Codificación en streaming con micro-lotes adaptativos
from Genoly import SequenceEncoder
enc = SequenceEncoder()
for chunk in enc.encode_stream(FastaReader("datos.fasta").iter_batches(10_000)):
    ...  # chunk.tensor en GPU, chunk.lengths, chunk.indices
```

Los conteos del pipeline de streaming son **idénticos** a los de `count()` (verificado en `tests/test_streaming.py`), incluido el modo canónico y k=31.

## Uso rápido

```python
from Genoly import (
    DeviceManager,
    SequenceEncoder,
    QualityAnalyzer,
    KmerCounter,
    VariantCaller,
    Read,
    LinearMixedModel,
    build_kinship,
)
from Genoly.alignment.alignment import GPUSequenceAligner

# 1. Dispositivo
manager = DeviceManager()          # auto-detecta CUDA
manager.print_info()

# 2. Codificación a tensores GPU
encoder = SequenceEncoder()
tensor, lengths = encoder.encode(["ACGTACGTACGT", "TTTTGGGGCCCC"])
one_hot = encoder.encode_one_hot(["ACGTACGTACGT"])
print(tensor.device, tuple(one_hot.shape))

# 3. Control de calidad
qa = QualityAnalyzer()
gc = qa.gc_content_percent(["GCGCATACGTACGT"]).item()
print(f"GC: {gc:.1f}%")

# 4. K-mers
kc = KmerCounter()
values, counts = kc.count(["ACGTACGTACGTACGTACGTACGT"], k=8)

# 5. Llamada de variantes
vc = VariantCaller()
reads = [Read(sequence="ACGTACGTTCGTACGT", start=0) for _ in range(20)]
variants = vc.call_variants("ACGTACGTACGTACGT", reads, min_depth=5)

# 6. Alineamiento
aligner = GPUSequenceAligner()
result = aligner.align_pair("ACGTACGT", "ACGTTCGT")
print(result.cigar_string)

# 7. Genética cuantitativa: LMM + BLUP
grm = build_kinship(genotipos)      # genotipos: matriz (individuos x marcadores), dosis {0,1,2}
lmm = LinearMixedModel()            # auto-detecta CUDA
resultado = lmm.fit(fenotipos, efectos_fijos, grm)   # REML por defecto
print(resultado.heritability)
valores_cria = lmm.blup()

# 8. PBLUP: modelo animal con pedigrí (sire/dam), sin genotipos
from Genoly import PedigreeBLUP
pblup = PedigreeBLUP()
r = pblup.fit_pedigree(fenotipos, X, individuos, padres, madres)
h2 = r.genetic_variance / (r.genetic_variance + r.residual_variance)
print(h2, pblup.inbreeding())

# 9. Llamada de variantes estilo GATK HaplotypeCaller (ensamblaje + PairHMM)
from Genoly import HaplotypeCaller
caller = HaplotypeCaller()          # PairHMM: nativo C -> torch GPU -> numpy
variantes = caller.call_from_sam("alineamientos.sam", "referencia.fasta")
for v in variantes["by_sequence"]["chr1"][:5]:
    print(v.position, v.ref, v.alt, v.gt, v.qual)
```

## Ejemplos

```bash
# Pipeline completo: I/O -> QC -> k-mers -> variantes
python examples/pipeline_completo.py

# Llamada de variantes estilo GATK: HaplotypeCaller + MarkDuplicates + BQSR
python examples/haplotype_caller_gatk.py

# Ejemplo completo del alineador
python examples/examp1.py

# Análisis de archivos FASTA (requiere Biopython y matplotlib)
python examples/analisis1.py

# Sitios de restricción y motivos consenso (requiere Biopython)
python examples/analisis2.py

# Descargar una secuencia FASTA desde NCBI por accession
python fetchingfasta.py
```

> [!NOTE]
> Los ejemplos `analisis1.py` y `analisis2.py` leen `examples/seqdump.txt` y `examples/ejemplo.fasta`, incluidos en el repositorio. Resuelven la ruta junto al propio script, así que funcionan desde cualquier directorio de trabajo.

## Tests

```bash
python -m pytest tests -v
# o un fichero suelto
python tests/test_smoke.py
# lint (también se ejecuta en CI)
ruff check .
```

`pytest`, `ruff` y `httpx` (necesario para los tests de la API) vienen en el extra de desarrollo: `pip install -e ".[dev]"`. El CI (`.github/workflows/ci.yml`) ejecuta `ruff check .` y `python -m pytest tests -q` en Python 3.10 y 3.12.

Hay 220 tests repartidos en 11 ficheros:

| Fichero | Qué cubre |
|---|---|
| `test_smoke.py` | dispositivo, I/O FASTA/FASTQ, codificación, QC, k-mers, `gpu_setup`, caller, LMM/GBLUP/kinship, preprocesamiento, alineador, registro de datasets |
| `test_streaming.py` | `FastaReader` por bloques, `VRAMManager`, `encode_stream`, conteo en streaming vs fuerza bruta |
| `test_fastq_pipeline.py` | `FastqReader`, QC en streaming, jobs aislados de FASTQ |
| `test_mapper.py` | `ReadMapper` (seed-and-extend) y su job |
| `test_variant_pipeline.py` | SAM → variantes → VCF y su job |
| `test_downstream.py` | PCA / KMeans / t-SNE y su job |
| `test_annotation_report.py` | GFF/GO, SVG de heatmap y volcano, jobs de anotación e informe |
| `test_quantgen_extra.py` | validación cruzada, GWAS, GRM, MAF/HWE, pedigrí (matriz A, F, A⁻¹), PBLUP (varianzas dadas/REML, equivalencia con GBLUP) |
| `test_gatk_pipeline.py` | PairHMM (paridad exacta native/torch/numpy), regiones activas, ensamblaje, genotipado SNV/indel, HaplotypeCaller end-to-end (SAM→VCF), BQSR, MarkDuplicates y sus endpoints `/api/gatk/*` |
| `test_api.py` | auth `X-API-Key`/`Bearer`/`?api_key=` y `/api/health` pública, 401 en toda la superficie, límites de subida (413/409), registro de rutas (allowlist y LFI), stream SSE de jobs y smoke de `/api/pblup/predict` |
| `test_worker.py` | aislamiento de procesos (raise / segfault / `os._exit`) y spills huérfanos |

`test_streaming.py` verifica la equivalencia exacta entre el pipeline de streaming y el conteo en memoria, la matemática del planificador de VRAM y el ventanado con solape `k-1`.

## Clases principales

### GpuSetup

Detección de la GPU NVIDIA con `nvidia-smi` e instalación automática de la build de PyTorch con CUDA adecuada.

| Método | Descripción |
|---|---|
| `detect_nvidia()` | Consulta nvidia-smi: driver, CUDA, GPU y memoria. |
| `recommend_cuda_tag(cuda_version)` | Elige la build `cuXXX` más conveniente para el driver. |
| `torch_status()` | Estado del PyTorch instalado (¿detecta CUDA?). |
| `install_command(cuda_tag)` | Genera el comando pip correspondiente. |
| `install_cuda_torch(cuda_tag, dry_run)` | Instala (o muestra) la build recomendada. |
| `ensure_cuda_torch(auto_install)` | Orquesta verificación + instalación automática. |

También existe la función de conveniencia `recommend_cuda_tag(cuda_version)`.

### DeviceManager

Gestión del dispositivo de cómputo (CUDA/NVIDIA o CPU). Centraliza la auto-detección.

| Método | Descripción |
|---|---|
| `print_info()` | Imprime GPU, memoria, compute capability y versión CUDA. |
| `get_gpu_info()` | Devuelve `GPUInfo` con los datos del dispositivo. |
| `synchronize()` | Sincroniza CUDA (para mediciones de tiempo). |
| `empty_cache()` | Libera memoria caché de CUDA. |

### VRAMManager

Gestión dinámica de VRAM: presupuesto por micro-lote, planificación adaptativa y liberación de memoria. En CPU usa la RAM libre del sistema como presupuesto. Ver [Pipeline de streaming](#pipeline-de-streaming-archivos-multi-gb) para los límites matemáticos.

| Método | Descripción |
|---|---|
| `free_bytes()` | Memoria libre del dispositivo (`torch.cuda.mem_get_info()` o RAM del sistema). |
| `budget_bytes()` | Presupuesto por micro-lote: `libre * safety_fraction` (por defecto 25 %). |
| `plan_micro_batches(lengths, bytes_per_base)` | Divide longitudes en grupos de índices que caben en el presupuesto. |
| `suggest_window_bases(bytes_per_base)` | Tamaño de ventana sugerido (objetivo 128 MiB, potencia de 2). |
| `release()` | Fuerza `gc.collect()` + `torch.cuda.empty_cache()`. |
| `stats()` | Instantánea de memoria para telemetría/logs. |

Las funciones `estimate_kmer_bytes_per_base(k, canonical)` y `estimate_encode_bytes_per_base(num_classes, one_hot)` calculan el coste `b` (bytes/base) de cada etapa para alimentar el planificador.

### SequenceEncoder

Codifica secuencias de ADN/ARN a tensores listos para GPU.

| Método | Descripción |
|---|---|
| `encode(sequences)` | Codifica un lote a tensor entero (B, L) + longitudes. |
| `encode_one_hot(sequences)` | Codifica un lote a one-hot (B, L, C). |
| `encode_stream(batches, one_hot)` | Micro-lotes adaptativos de GPU según la VRAM libre; libera memoria tras cada lote. |
| `decode(tensor, lengths)` | Convierte índices de vuelta a secuencias de texto. |

### QualityAnalyzer

Control de calidad sobre GPU.

| Método | Descripción |
|---|---|
| `gc_content(sequences)` | Contenido GC por secuencia (tensor). |
| `base_composition(sequences)` | Conteo total de A/C/G/T/N. |
| `quality_distribution(records)` | Media de calidad Phred por posición. |
| `trim_by_quality(records, ...)` | Recorte 3' por ventana deslizante. |
| `filter_by_quality(records, ...)` | Filtro por calidad, longitud y ratio de N. |
| `report(records)` / `summarize(records)` | Reporte completo / resumen impreso. |

### KmerCounter

Conteo de k-mers acelerado por GPU (codificación base-4 entera en int64, exacta hasta k=31).

| Método | Descripción |
|---|---|
| `count(sequences, k, canonical)` | Frecuencia de cada k-mer (canónico opcional). |
| `count_stream(sequences, k, ...)` | Conteo en streaming sobre un iterable perezoso, con RAM/VRAM acotadas. |
| `count_records(records, k, ...)` | Streaming sobre registros FASTA con progreso de la fuente. |
| `count_fasta(path, k, ...)` | Streaming directo del archivo (ventanas desde bloques de disco, sin materializar registros). |
| `count_fasta_aggregated(path, k, ...)` | Estadísticas exactas (total, espectro, top-k) con acumulador particionado y derrame a disco: procesa genomas completos en máquinas modestas. |
| `decode_kmer(code, k)` | Convierte un código entero a secuencia de texto. |
| `spectrum(sequences, k)` | Distribución de k-mers por multiplicidad. |
| `estimate_genome_size(sequences, k)` | Estimación de tamaño de genoma (Lander-Waterman). |

> [!TIP]
> Los k-mers se codifican como enteros en base 4 (A=0, C=1, G=2, T=3) mediante aritmética entera exacta, válida para todo k <= 31. Con `canonical=True` se cuenta cada k-mer junto a su reverse complement, lo que evita dobles conteos en datos de doble hebra.

### VariantCaller

Pileup y llamada de variantes acelerado por GPU.

| Método | Descripción |
|---|---|
| `pileup(reference, reads)` | Cobertura y conteos de bases por posición. |
| `call_variants(reference, reads, ...)` | Llama SNVs y deleciones con umbrales de profundidad y frecuencia. |

> [!IMPORTANT]
> El caller asume lecturas ya alineadas a la referencia: cada `Read` indica su secuencia, la posición 0-based de inicio y la hebra (`+`/`-`). Las posiciones reportadas en las variantes son 1-based.

### HaplotypeCaller (paquete `Genoly.gatk`)

Llamada de variantes con reensamblaje local al estilo GATK 4.x: detección de regiones activas (pileup GPU), ensamblaje de haplotipos (grafo de de Bruijn), alineamiento read↔haplotipo con PairHMM (Gotoh affine, tres backends) y genotipado con PL/GQ/AD/DP/QUAL. Incluye además BQSR y MarkDuplicates.

| Método / clase | Descripción |
|---|---|
| `HaplotypeCaller.call_region(reference, reads)` | Genotipa una región; devuelve `List[GATKVariant]` con POS 1-based. |
| `HaplotypeCaller.call_from_sam(sam, ref, ...)` | Pipeline completo SAM/FASTA → dict con variantes por secuencia, contadores y muestra. |
| `active_regions(reads, reference, ...)` | Ventanas con evidencia de variación (mismatches, huecos cortos). |
| `BaseRecalibrator().estimate(ref, reads)` | Tabla BQSR (calidad × ciclo × hebra); `apply_recalibration` corrige las calidades. |
| `mark_duplicates(reads)` / `apply_duplicate_flags(...)` | Detección por clave 5' y marcado con flag 0x400. |
| `write_gatk_vcf(path, by_sequence, ...)` | VCF 4.2 con el esquema de HaplotypeCaller (`GT:AD:DP:GQ:PL`). |

El PairHMM usa el kernel C nativo (`csrc/pairhmm.c`, compilado con `python setup.py build_ext --inplace`) si está disponible; si no, cae a PyTorch GPU y, en último extremo, a NumPy. Los tres backends son numéricamente equivalentes (verificado en `tests/test_gatk_pipeline.py`).

```python
from Genoly import HaplotypeCaller
from Genoly.gatk import BaseRecalibrator, mark_duplicates, write_gatk_vcf

caller = HaplotypeCaller(device="auto", backend="auto")
result = caller.call_from_sam("alineamientos.sam", "referencia.fasta",
                              sample_name="MUESTRA1")
write_gatk_vcf("salida.vcf", result["by_sequence"],
               result["sequences"], sample="MUESTRA1")

stats = mark_duplicates(lecturas)
tabla = BaseRecalibrator().estimate("referencia.fasta", lecturas)
```

> [!NOTE]
> El parser SAM es texto puro (sin dependencia de pysam), así que funciona en cualquier plataforma. Para BAM, convierte antes con `samtools view -h`.

### GPUSequenceAligner

Alineador de secuencias genómicas con aceleración GPU. Unifica funcionalidades de alineamiento y análisis básico.

| Método | Descripción |
|---|---|
| `align_pair(query, target)` | Alinea un par de secuencias con Smith-Waterman. |
| `align_batch(queries, targets)` | Alinea múltiples pares de secuencias. |
| `find_variants(aligned_q, aligned_t)` | Detecta SNVs, inserciones y deleciones. |
| `analyze_mutations(query, target, known_variants)` | Análisis completo de mutaciones. |
| `encode_sequence(sequence)` | Codifica una secuencia a tensor numérico. |
| `encode_batch(sequences)` | Codifica un lote de secuencias con padding. |
| `benchmark(sequence_length, num_pairs)` | Test de rendimiento GPU vs CPU. |

Los parámetros de scoring del alineador son configurables en el constructor:

```python
aligner = GPUSequenceAligner(
    match_score=2.0,
    mismatch_penalty=-1.0,
    gap_open=-2.0,
    gap_extend=-0.5,
    device="cuda"  # "cuda", "cpu" o None para auto-detectar
)
```

> [!IMPORTANT]
> Para secuencias de gran tamaño el procesamiento por lotes (`align_batch`) aún procesa en serie cuando el número de pares es reducido. La verdadera paralelización por lotes en GPU es un trabajo en curso.

### LinearMixedModel

Modelos lineales mixtos univariantes para genética cuantitativa sobre GPU. Ajusta el modelo animal `y = Xβ + Zu + ε` con u ~ N(0, σg²K), estima las componentes de varianza por REML o ML (puntuación de Fisher con búsqueda de línea) y predice los efectos aleatorios con BLUP.

| Método | Descripción |
|---|---|
| `fit(y, X, K, Z=None, method)` | Ajusta el modelo; devuelve `LMMResult` con varianzas genética/residual, heredabilidad, log-verosimilitud e iteraciones. |
| `blue()` | Estimadores BLUE de los efectos fijos (β). |
| `blup()` | Valores de cría predichos BLUP de los efectos aleatorios (u). |
| `predict()` | Valores ajustados del modelo (Xβ + Zu). |

La función `build_kinship(genotypes, method)` construye la matriz de relación genómica (GRM) a partir de dosis alélicas {0, 1, 2}, imputando los valores perdidos con la media del marcador. Admite dos métodos: `vanraden` (por defecto) y `gcta` (columnas estandarizadas, excluye marcadores monórficos):

```python
from Genoly import LinearMixedModel, build_kinship

grm = build_kinship(genotipos)             # (individuos x marcadores)
grm = build_kinship(genotipos, method="gcta")
modelo = LinearMixedModel()                # auto-detecta CUDA
resultado = modelo.fit(fenotipos, X, grm)  # method="reml" por defecto
print(resultado.genetic_variance, resultado.heritability)
valores_cria = modelo.blup()
```

> [!IMPORTANT]
> El ajuste interno se realiza en doble precisión (float64) para garantizar la estabilidad numérica de la optimización, aunque el resto del paquete trabaje en float32. En GPUs de consumo esto es más lento, pero el coste es asumible: las operaciones son O(n³) sobre matrices de tamaño n = número de individuos.

> [!TIP]
> Con `Z=None` se usa el modelo animal estándar (Z igual a la identidad). Puedes pasar tu propia matriz Z (n x q) para modelar efectos aleatorios arbitrarios: efectos de grupo, medidas repetidas, etc.

> [!CAUTION]
> La matriz K debe ser simétrica y el diseño X debe tener menos columnas que observaciones; en caso contrario `fit()` lanza `ValueError`. Los métodos `blue()`, `blup()` y `predict()` requieren haber llamado antes a `fit()`.

### GenomicBLUP

Predicción genómica GBLUP sobre GPU. A diferencia de `LinearMixedModel`, permite fijar directamente las componentes de varianza (solución en un paso, sin iterar) y calcula la fiabilidad y la precisión de cada valor de cría a partir del error de predicción (PEV).

| Método | Descripción |
|---|---|
| `fit(y, X, K, Z=None, genetic_variance=None, residual_variance=None)` | Predice los valores de cría; si no se indican las varianzas, las estima por REML. Devuelve `GBLUPResult`. |
| `blup()` / `blue()` | Valores de cría genómicos (GEBV) / efectos fijos. |
| `reliabilities()` | Fiabilidad de cada individuo: 1 − PEV/Var(u), en [0, 1]. |
| `accuracies()` | Precisión de cada individuo (√fiabilidad). |
| `predict()` | Valores ajustados del modelo (Xβ + Zu). |

> [!NOTE]
> La fiabilidad usa el PEV del BLUP con efectos fijos **estimados** (β̂ se obtiene por BLUE): equivale al bloque inferior derecho de la inversa de las ecuaciones del modelo mixto de Henderson, es decir, PEV(u) = diag(G − GZ'PZG) con P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹ y G = σg²K. |

```python
from Genoly import GenomicBLUP

gblup = GenomicBLUP()
resultado = gblup.fit(fenotipos, X, grm,
                      genetic_variance=0.5, residual_variance=0.9)
print(resultado.variance_source)     # "dadas" o "reml"
print(gblup.blup(), gblup.accuracies())
```

### PedigreeBLUP (PBLUP)

Predicción de valores de cría por modelo animal a partir del pedigrí (columnas sire/dam), sin necesidad de genotipos. Construye la matriz de parentesco aditiva A y la invierte; admite varianzas fijas (solución en un paso) o estimación REML, y devuelve fiabilidades, precisiones y el coeficiente de endogamia F de cada individuo.

| Método | Descripción |
|---|---|
| `fit(y, X, A, Z=None, genetic_variance=None, residual_variance=None, ...)` | Ajusta el modelo animal con una matriz A ya construida. Devuelve `PBLUPResult`. |
| `fit_pedigree(y, X, individuals, sires, dams, ...)` | Construye A desde el pedigrí crudo (reordena y/X topológicamente) y ajusta. |
| `blup()` / `blue()` | Valores de cría predichos / efectos fijos. |
| `reliabilities()` / `accuracies()` | Fiabilidad (1 − PEV/Var(u)) y precisión (√fiabilidad) por individuo. |
| `inbreeding()` | Coeficiente de endogamia F de cada individuo. |

Funciones auxiliares del pedigrí: `normalize_pedigree` (valida y ordena topológicamente), `build_a_matrix` (A y F) y `build_a_inverse` (A⁻¹). El pedigrí se recibe en el **mismo CSV/Excel** que los fenotipos mediante las columnas de padre (sire) y madre (dam), seleccionadas en la página Datos.

```python
from Genoly import PedigreeBLUP

pblup = PedigreeBLUP()
resultado = pblup.fit_pedigree(fenotipos, X, individuos, padres, madres,
                               genetic_variance=0.5, residual_variance=0.9)
print(resultado.variance_source, resultado.founders)
print(pblup.blup(), pblup.accuracies(), pblup.inbreeding())
```

### Preprocesamiento de CSV/Excel

`prepare_quantitative_data(path)` carga una tabla de fenotipos y genotipos (`.csv`, `.tsv`, `.txt`, `.xlsx` o `.xls`) aplicando la limpieza estándar antes del análisis: detección automática de cabecera y delimitador (coma, punto y coma o tabulador), conversión de decimales con coma, descarte de columnas no numéricas y de filas sin fenotipo, e imputación de dosis perdidas por media o moda.

```python
from Genoly import prepare_quantitative_data

fenotipos, genotipos, informe = prepare_quantitative_data("datos.csv", impute_method="media")
print(informe.final_rows, informe.imputed_cells, informe.dropped_columns)
```

> [!TIP]
> La primera columna numérica se interpreta como fenotipo y el resto como marcadores. Excel requiere `openpyxl` (incluido en requirements.txt); los CSV usan solo la librería estándar.

## Descarga de secuencias

`fetchingfasta.py` descarga una secuencia FASTA desde NCBI usando el número de acceso:

```bash
python fetchingfasta.py
# Descarga el gen BRCA1 humano (NM_007294) como BRCA1_humano.fasta
```

`clean_fetching.py` permite descargar desde una URL directa de un archivo FASTA:

```bash
python clean_fetching.py
```

`descargar_datos_test.py` descarga datos **reales** para probar todos los análisis: genomas completos de NCBI (*E. coli* K-12 MG1655 y SARS-CoV-2 Wuhan-Hu-1), lecturas FASTQ reales de Illumina (datasets de prueba de nf-core) y el panel cuantitativo de maíz de GAPIT (fenotipos, genotipos numéricos y matriz de parentesco publicada):

```bash
python descargar_datos_test.py
# Deja todo en datos_reales/ (ignorada por git)
```

> [!TIP]
> La carpeta `datos_reales/` está en `.gitignore`: los datos descargados (~10 MB) nunca se suben al repositorio.

> [!CAUTION]
> Al descargar secuencias grandes desde NCBI, respeta los límites de uso de la API de E-utilities de NCBI (como máximo 3 solicitudes por segundo).

## Hoja de ruta

- [x] Carga de archivos FASTQ/FASTA en streaming.
- [x] Análisis de calidad (scores Phred), trimming y filtrado.
- [x] Conteo de k-mers y espectro k-mer en GPU.
- [x] Llamada de variantes (SNV/deleciones) en GPU.
- [x] Llamada de variantes estilo GATK HaplotypeCaller: regiones activas, reensamblaje local, PairHMM (C/torch/numpy), genotipado PL/GQ/AD y VCF 4.2.
- [x] BQSR: tabla de recalibración (calidad × ciclo × hebra) y SAM con calidades corregidas.
- [x] Detección de duplicados (MarkDuplicates) con flags 0x400 y estadísticas por biblioteca.
- [x] Página UI `/gatk` con HaplotypeCaller, MarkDuplicates y BQSR sobre archivos subidos (jobs + SSE).
- [x] Genética cuantitativa: modelos lineales mixtos (REML/ML) y BLUP sobre GPU.
- [x] Predicción genómica GBLUP con fiabilidad y precisión por individuo.
- [x] Predicción PBLUP por modelo animal (matriz A de pedigrí, endogamia F, REML o varianzas fijas).
- [x] Flujo obligatorio de limpieza para LMM/GBLUP/PBLUP: dry-run, filtros por columna y tabla interactiva paginada en la página Datos.
- [x] Carga de tablas CSV/Excel con limpieza e imputación de valores perdidos.
- [x] Auto-detección de la GPU e instalación automática de PyTorch CUDA.
- [x] Despliegue en contenedor Docker (API + frontend, CUDA opcional).
- [x] Pipeline de streaming para datasets masivos: chunking de disco, RAM batching, micro-batching adaptativo de VRAM y progreso en tiempo real (SSE).
- [x] Lanzador inteligente de Docker con detección automática de GPU y selección de dispositivo.
- [x] Scripts de instalación automática para Windows (PowerShell) y Linux (bash).
- [x] Montaje automático y robusto del contenedor Docker con GPU y verificación de CUDA (setup_docker.sh / setup_docker.ps1).
- [ ] Soporte real de paralelización por lotes en GPU para el alineador.
- [ ] Llamada de inserciones mediante CIGAR (lecturas alineadas).
- [x] Exportación a formatos estándar (SAM/BAM, VCF).
- [x] Anotación funcional con GFF/GO sobre regiones del genoma.

## Licencia

MIT

> [!NOTE]
> Este proyecto está en fase alpha (Development Status 3 - Alpha). Las APIs pueden cambiar en futuras versiones.
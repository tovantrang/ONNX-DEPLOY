# ONNX-DEPLOY

## Overview

ONNX-DEPLOY is a local ML/CV inference deployment pipeline. It trains a small
PyTorch image classification model, exports it to ONNX, validates ONNX Runtime
outputs against PyTorch, and benchmarks inference latency.

## Why this project

The goal is to understand the engineering path from a trained model to an
inference artifact: PyTorch training, checkpointing, ONNX export, runtime
selection, correctness checks, and reproducible latency measurements.

## Current features

- PyTorch training entry point.
- ONNX export.
- ONNX Runtime inference.
- PyTorch vs ONNX Runtime correctness check.
- Latency benchmark.
- Configurable ONNX Runtime providers.
- CPU-first setup.
- Optional GPU requirements.
- Standard Python logging.

## Architecture

```text
PyTorch model
    -> training checkpoint
    -> ONNX export
    -> ONNX Runtime session
    -> correctness check
    -> latency benchmark
```

## Installation

The recommended setup uses Conda only to create an isolated Python 3.11
environment. Project dependencies are installed with pip from locked
requirements files.

### CPU environment

The default setup is CPU-first and does not require CUDA.

```bash
conda create -n onnx-deploy-cpu python=3.11 pip -y
conda activate onnx-deploy-cpu

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Equivalent explicit CPU install:

```bash
python -m pip install -r requirements-cpu.txt
```

### GPU environment

Optional GPU install, for NVIDIA/CUDA-compatible machines only:

```bash
conda create -n onnx-deploy-gpu python=3.11 pip -y
conda activate onnx-deploy-gpu

python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
```

### Development tools

```bash
python -m pip install -r requirements-dev.txt
```

### venv alternative

Conda is optional. If Python 3.11 is already available, a standard virtual
environment also works:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Do not install `requirements-cpu.txt` and `requirements-gpu.txt` in the same
environment. The CPU file installs `onnxruntime`; the GPU file installs
`onnxruntime-gpu`.

## Configuration

The main configuration file is `configs/default.yaml`.

Default ONNX Runtime provider:

```yaml
onnx:
    providers:
        - CPUExecutionProvider
```

On a compatible GPU environment, CUDA can be requested with CPU fallback:

```yaml
onnx:
    providers:
        - CUDAExecutionProvider
        - CPUExecutionProvider
```

Requirements files define what is installed in the Python environment. The YAML
configuration defines which ONNX Runtime providers are requested at execution
time.

## Usage

Train the TinyCNN model and save a PyTorch checkpoint:

```bash
python train.py train
```

Export the saved checkpoint to ONNX:

```bash
python train.py export
```

Compare PyTorch and ONNX Runtime outputs:

```bash
python train.py benchmark
```

Run the latency benchmark:

```bash
python train.py latency
```

## Example output

Example logs from a CPU run:

```text
[INFO] ONNX Runtime providers used: ['CPUExecutionProvider']
[INFO] Correctness max_abs_diff: 2.3841858e-07
[INFO] Latency ONNX Runtime: 0.04 ms
```

Latency numbers depend heavily on hardware, installed runtimes, provider
selection, and current system load. They should not be interpreted as universal
benchmark results.

## Project structure

- `src/`: Python package with model, data loading, export, runtime, benchmark,
  training, and utility modules.
- `configs/`: YAML configuration files.
- `artifacts/`: generated checkpoints and ONNX files, ignored by Git.
- `docs/`: engineering notes and short technical decisions.
- `requirements*.txt`: locked dependency files for CPU, GPU, and development
  environments.

## Engineering choices

- CPU-first by default to keep the project portable.
- ONNX Runtime providers are configurable instead of hard-coded.
- CPU and GPU dependency sets are separated to avoid mixing `onnxruntime` and
  `onnxruntime-gpu`.
- Standard Python logging is used for project messages around export,
  correctness checks, provider selection, and latency.

## Roadmap

- Persist evaluation metrics and benchmark outputs to JSON.
- Add correctness checks on real data batches.
- Add automated tests.
- Add a local FastAPI inference API.
- Add a CPU Docker image.
- Add local MLflow experiment tracking.
- Add optional GitHub Actions checks.

## Limitations

- Not production-ready.
- Latency benchmark is still basic.
- No API yet.
- No Docker image yet.
- No automated tests yet.
- GPU usage depends on local CUDA, PyTorch, and ONNX Runtime compatibility.
- The export command still initializes dataloaders.

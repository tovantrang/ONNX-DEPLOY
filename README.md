# ONNX-DEPLOY

## Overview

ONNX-DEPLOY is a local inference deployment benchmarking project. It takes a
PyTorch model, exports it to ONNX, compares PyTorch eager and ONNX Runtime
execution, verifies numerical correctness, measures runtime latency, and writes
structured JSON reports.

The current TinyCNN/CIFAR-10 setup is a benchmark support: it provides a small,
local model and representative inputs for testing deployment workflows. The
main focus is deployment comparison, not maximizing model accuracy.

## Why this project

The goal is to answer a practical deployment question:

```text
Given a model, representative inputs, and local hardware constraints,
which deployment configuration looks most promising?
```

The project compares deployment candidates through correctness, latency,
latency stability, provider availability, portability, and implementation
complexity. It is a local ML engineering project, not a production serving
platform or a complete MLOps system.

## Current features

- PyTorch checkpoint/export path.
- ONNX export.
- Configurable ONNX Runtime providers.
- PyTorch vs ONNX Runtime correctness check.
- Runtime latency measurement.
- Synthetic and dataset batch input modes.
- Structured JSON outputs.
- CPU-first setup.
- Optional GPU requirements.
- Standard Python logging.
- Unit and integration tests.

## Architecture

```text
PyTorch model + input source
    -> export candidate: ONNX
    -> runtime candidates: PyTorch eager / ONNX Runtime
    -> correctness check
    -> latency measurement
    -> structured JSON results
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

Train the TinyCNN demo model and save a PyTorch checkpoint:

```bash
python train.py train
```

This command exists to produce a local checkpoint for benchmarking. Training is
not the main contribution of the project.

Export the saved checkpoint to ONNX:

```bash
python train.py export
```

Compare PyTorch and ONNX Runtime outputs for numerical consistency:

```bash
python train.py verify
```

By default, this command also writes structured numerical-difference results to
`outputs/verify_results.json`.

By default, `verify` uses a synthetic input for a fast runtime correctness
check. A dataset batch mode can be enabled through configuration when a local
dataset-backed check is desired. This checks numerical consistency between
PyTorch and ONNX Runtime; it does not measure model quality.

Example dataset-backed verify input:

```yaml
verify:
    input:
        source: dataset_batch
        split: test
        batch_size: 1
```

This mode uses the existing CIFAR-10 dataloader setup and may download CIFAR-10
if it is not already available in the configured data directory.

Measure inference latency:

```bash
python train.py latency
```

By default, this command writes structured latency results to
`outputs/latency_results.json`. The report includes warmup runs, timed runs,
mean, median, p50, p95, p99, min, and max latency for both PyTorch eager and
ONNX Runtime.

The current latency command uses a configurable synthetic input by default.
This isolates model/runtime inference from dataset loading and preprocessing.
It is not an end-to-end data pipeline benchmark.

A `dataset_batch` input mode can be enabled through configuration to measure
runtime inference latency on a real prepared batch. Data loading and
preprocessing are not included in the timed section. The current latency modes
reuse the same prepared input across warmup and timed runs.

```yaml
latency:
    input:
        source: dataset_batch
        split: test
        batch_size: 1
        include_data_loading: false
```

## Example output

Example logs from a CPU run:

```text
[INFO] ONNX Runtime providers used: ['CPUExecutionProvider']
[INFO] Correctness max_abs_diff: 2.3841858e-07
[INFO] Verification passed: True
[INFO] Saved verify results to outputs/verify_results.json
[INFO] Latency input source: synthetic
[INFO] Latency input shape: [1, 3, 32, 32]
[INFO] Latency input fill: zeros
[INFO] Running latency measurement
[INFO] Warmup runs: 10
[INFO] Timed runs: 100
[INFO] PyTorch eager latency p50: 0.15 ms
[INFO] PyTorch eager latency p95: 0.20 ms
[INFO] ONNX Runtime latency p50: 0.04 ms
[INFO] ONNX Runtime latency p95: 0.06 ms
[INFO] Saved latency results to outputs/latency_results.json
```

Latency numbers depend heavily on hardware, installed runtimes, provider
selection, and current system load. The current latency command is a basic
measurement utility, not a rigorous benchmark suite.

## Results

Current commands write structured local reports:

- `outputs/verify_results.json`: numerical consistency between PyTorch eager
  and ONNX Runtime.
- `outputs/latency_results.json`: runtime latency metrics and input/runtime
  metadata.

These reports are the first step toward comparing deployment candidates. The
project does not yet produce a combined candidate-level report or automatic
ranking. The target schema for that future report is documented in
`docs/deployment_report_schema.md`.

A deployment report represents one benchmark run: candidates are comparable
only when they share the same model, input protocol, measurement scope, and
hardware context. Varying the input protocol is useful, but belongs to input
sensitivity analysis rather than deployment candidate comparison.

## Project structure

- `src/`: Python package with model, data loading, export, runtime, latency,
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
- Latency results are persisted as JSON for local comparison, but are not
  treated as universal performance thresholds.

## Roadmap

- Add a local check script.
- Generate a deployment report from verify and latency outputs.
- Combine verify and latency into a candidate-level report.
- Add dynamic batch ONNX export.
- Add a deployment configuration matrix.
- Add ranking rules for promising configurations.
- Add a local FastAPI inference API.
- Add a CPU Docker image.
- Add optional local MLflow tracking.
- Add optional GitHub Actions checks.
- Add optional model quality evaluation with accuracy/loss JSON.

## Limitations

- Not production-ready.
- Benchmark results are local and hardware-dependent.
- TinyCNN/CIFAR-10 is a benchmark support, not the core objective.
- No full deployment configuration matrix yet.
- No combined candidate-level report yet.
- No automatic ranking/recommendation yet.
- No API yet.
- No Docker image yet.
- No MLflow tracking yet.
- No model quality evaluation yet.
- Automated tests are still limited to foundational unit and integration checks.
- ONNX export currently uses a static batch dimension.
- GPU usage depends on local CUDA, PyTorch, and ONNX Runtime compatibility.

# Engineering Log

## Current objective

Transform the current ONNX inference prototype into a reproducible local ML/CV
pipeline suitable for a professional portfolio.

The project should cover model training, evaluation, ONNX export, runtime
comparison between PyTorch and ONNX Runtime, latency measurement, a local
inference API, Docker packaging, automated tests, and a professional README.
The scope should remain local, free, standard, and easy to explain.

## Decisions

### D001 — CPU-first ONNX Runtime

Decision: use `CPUExecutionProvider` as the default ONNX Runtime provider.

CUDA remains optional through configuration, for example by requesting
`CUDAExecutionProvider` before `CPUExecutionProvider` on compatible machines.

Justification: CPU-first execution improves portability, reproducibility, and
ease of setup on machines without a GPU. It also makes the default pipeline
simpler to run during review or demonstration.

Trade-off: CPU latency measurements are less representative of GPU deployment
performance.

Validation: ONNX export, correctness verification, and latency measurement run with
the CPU provider.

### D002 — Separate CPU and GPU dependency sets

Decision: keep CPU and GPU dependency sets separate.

The default CPU installation uses `onnxruntime`. The optional GPU installation
uses `onnxruntime-gpu` and should not be installed in the same environment as
`onnxruntime`.

Justification: separate dependency files improve portability and
reproducibility, avoid forcing CUDA on machines without a GPU, and make the
runtime target explicit.

Limit: users must choose the dependency file that matches their machine and
deployment target.

Conceptual link: requirements files define what is installed in the Python
environment, while the YAML configuration defines which ONNX Runtime providers
are requested at execution time.

### D003 — Command names distinguish correctness from performance

Decision: use `verify` for numerical consistency checks and `latency` for
inference-time measurement.

`verify` checks that PyTorch and ONNX Runtime produce numerically consistent
outputs. `latency` measures inference time with the current lightweight timing
utility. The `benchmark` command is intentionally removed for now.

Justification: correctness validation should not be presented as a performance
benchmark.

Future option: reintroduce `benchmark` later for a real benchmark suite with
clear metrics, outputs, and methodology.

## Roadmap

### MVP

Keep the existing train, export, verify, and latency commands working on
CPU with clear configuration and documentation.

### Clean version

Add structured metrics, JSON outputs, correctness checks on real batches, and a
small automated test suite.

### Portfolio version

Add a local FastAPI inference API, a CPU Docker image, a polished README, and
lightweight local experiment tracking.

### Optional extensions

Add GitHub Actions, optional GPU documentation, richer latency reports, and
small usability improvements for demos.

## Tickets

- T001 — Configurable ONNX Runtime providers — Done
- T002 — Dependency split CPU/GPU — Done
- T003 — Real evaluation metrics and JSON outputs — Planned
- T004 — Correctness check on real batch — Planned
- T005 — Basic tests — Planned
- T006 — FastAPI inference API — Later
- T007 — Docker CPU image — Later
- T008 — MLflow local tracking — Later
- T009 — Rename correctness command from benchmark to verify — Done

## Known limitations

- No API yet.
- No Docker image yet.
- No automated tests yet.
- Latency measurement still basic.
- Export command still touches dataloaders.
- Active Python environment must include project dependencies.

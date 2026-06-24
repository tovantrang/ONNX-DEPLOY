# Engineering Log

## Current objective

Transform the current ONNX inference prototype into a reproducible local ML/CV
pipeline suitable for a professional portfolio.

The project should cover model training, evaluation, ONNX export, runtime
comparison between PyTorch and ONNX Runtime, latency benchmarking, a local
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

Trade-off: CPU latency benchmarks are less representative of GPU deployment
performance.

Validation: ONNX export, correctness benchmark, and latency benchmark run with
the CPU provider.

## Roadmap

### MVP

Keep the existing train, export, correctness, and latency commands working on
CPU with clear configuration and documentation.

### Clean version

Add structured metrics, JSON outputs, correctness checks on real batches, and a
small automated test suite.

### Portfolio version

Add a local FastAPI inference API, a CPU Docker image, a polished README, and
lightweight local experiment tracking.

### Optional extensions

Add GitHub Actions, optional GPU documentation, richer benchmark reports, and
small usability improvements for demos.

## Tickets

- T001 — Configurable ONNX Runtime providers — Done
- T002 — Dependency split CPU/GPU — Planned
- T003 — Real evaluation metrics and JSON outputs — Planned
- T004 — Correctness check on real batch — Planned
- T005 — Basic tests — Planned
- T006 — FastAPI inference API — Later
- T007 — Docker CPU image — Later
- T008 — MLflow local tracking — Later

## Known limitations

- No API yet.
- No Docker image yet.
- No automated tests yet.
- Latency benchmark still basic.
- Export command still touches dataloaders.
- Active Python environment must include project dependencies.

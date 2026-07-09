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

### D004 — Minimal pyproject for tool configuration

Decision: add a minimal `pyproject.toml` for tool configuration only.

`pyproject.toml` centralizes Ruff, pytest, and mypy settings. Runtime
dependencies remain in the requirements files for now.

Justification: this avoids duplicate dependency sources while standardizing
tool configuration in a conventional Python location.

Future option: move toward editable package installation later if the project
needs package metadata, console scripts, or a fuller packaging workflow.

### D005 — Initial pytest coverage

Decision: add a small pytest suite for project foundations.

Unit tests cover config loading, TinyCNN forward shape, and ONNX Runtime
provider resolution. The integration test covers a minimal PyTorch to ONNX to
ONNX Runtime CPU correctness path.

Constraints: tests use synthetic tensors, do not download CIFAR-10, do not
write to the real `artifacts/` directory, do not require CUDA, and do not assert
fixed latency thresholds.

### D006 — Persist verify results to JSON

Decision: make `verify` return and persist structured numerical consistency
results.

The verify output contains difference metrics, tolerance, output shapes, ONNX
Runtime provider information, input shape, dtype, device, and the ONNX path.
These are correctness metrics for PyTorch vs ONNX Runtime output consistency,
not model evaluation metrics on a dataset.

### D007 — Persist latency results to JSON

Decision: make `latency` return and persist structured inference timing
results.

The latency output contains separate PyTorch eager and ONNX Runtime metrics,
warmup run count, timed run count, p50, p95, p99, min, max, mean, median, input
shape, device, dtype, ONNX path, and provider information.

This is separate from `verify`: latency measures inference timing, while
verify measures numerical consistency. No absolute performance threshold is
used because timing depends on hardware, runtime providers, and system load.

### D008 — Latency uses synthetic input by default

Decision: measure local runtime latency with a configurable synthetic input by
default.

The current `latency` command uses an input such as
`torch.zeros(1, 3, 32, 32)`, configured through `latency.input.source`,
`latency.input.shape`, and `latency.input.fill`. This input choice is separate
from the broader runtime-pure latency decision: the command measures
model/runtime execution without dataset loading, while the synthetic tensor
makes that measurement stable, fast, independent of CIFAR-10, and easier to
test locally.

Limit: this is not representative of a real end-to-end inference flow with
data loading, transforms, or batches sampled from a dataset.

Future option: support a prepared `dataloader_batch` or `real_batch` mode
measured outside the data-loading timer.

### D009 — Shared runners for verify and latency

Decision: use `PyTorchRunner` and `ONNXRunner` for both correctness
verification and latency measurement.

This reduces duplicated ONNX Runtime provider/session handling while preserving
the existing `verify` and `latency` JSON outputs. `verify` remains responsible
for numerical comparison and tolerance checks; the runners only encapsulate
runtime execution.

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
- T010 — Minimal pyproject for tool configuration — Done
- T011 — Initial pytest suite — Done
- T012 — Persist verify results to JSON — Done
- T013 — Structured latency results to JSON — Done
- T014 — Make latency input source explicit — Done
- T015 — Harmonize verify and latency runners — Done

## Known limitations

- No API yet.
- No Docker image yet.
- Automated tests are still limited to foundation smoke tests.
- Latency measurement still basic and should not be interpreted as a universal
  benchmark.
- Active Python environment must include project dependencies.

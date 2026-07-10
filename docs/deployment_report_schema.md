# Deployment Report Schema

This document defines the target structure for a future deployment benchmark
report. It is a schema proposal, not an automatically generated output yet.

The goal is to combine existing `verify` and `latency` JSON outputs into a
candidate-level report that can compare deployment configurations.

## Purpose

The deployment report should answer:

```text
Given a model, representative inputs, and local hardware constraints,
which deployment configuration looks most promising?
```

The report should keep correctness and latency separate, then summarize which
candidates are valid and promising.

## Target Output

Proposed output path:

```text
outputs/deployment_report.json
```

## Top-Level Shape

```json
{
  "schema_version": "0.1",
  "model": {
    "name": "TinyCNN",
    "checkpoint_path": "./artifacts/tinycnn_cifar10.pt",
    "onnx_path": "./artifacts/tinycnn_cifar10.onnx"
  },
  "input": {
    "source": "synthetic",
    "shape": [1, 3, 32, 32],
    "batch_size": 1
  },
  "environment": {
    "device": "cpu",
    "providers_available": ["AzureExecutionProvider", "CPUExecutionProvider"]
  },
  "candidates": [
    {
      "name": "pytorch_eager_cpu",
      "runtime": "pytorch_eager",
      "device": "cpu",
      "correctness": {
        "reference": true
      },
      "latency": {
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0
      }
    },
    {
      "name": "onnxruntime_cpu",
      "runtime": "onnxruntime",
      "provider": "CPUExecutionProvider",
      "device": "cpu",
      "correctness": {
        "passed": true,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "tolerance": {
          "max_abs_diff": 0.0001
        }
      },
      "latency": {
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0
      }
    }
  ],
  "recommendation": {
    "best_latency_p50": "onnxruntime_cpu",
    "valid_candidates": ["onnxruntime_cpu"]
  }
}
```

## Field Notes

- `schema_version`: version of this report shape.
- `model`: model and artifact metadata used for the comparison.
- `input`: input source used for correctness and latency checks.
- `environment`: local execution context and available ONNX Runtime providers.
- `candidates`: deployment configurations being compared.
- `correctness`: numerical consistency information relative to the PyTorch
  reference.
- `latency`: runtime-only latency metrics for each candidate.
- `recommendation`: simple summary derived from correctness and latency.

## Current Inputs

The current project already produces two intermediate reports:

- `outputs/verify_results.json`
- `outputs/latency_results.json`

These files contain the data needed for an initial candidate report, but the
combined deployment report is not generated automatically yet.

## Non-Goals For The First Version

- No full deployment matrix yet.
- No automatic multi-config execution yet.
- No complex ranking model.
- No end-to-end data pipeline benchmark.
- No production monitoring or serving assumptions.


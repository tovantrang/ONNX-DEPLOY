# Deployment Report Schema

This document defines the target structure for a deployment benchmark report.
It is a schema and data-model target; it is not a full deployment matrix yet.

The report is centered on one `benchmark_run`.

## Benchmark Run

A `benchmark_run` compares deployment candidates under shared conditions:

```text
same model
same checkpoint or ONNX artifact
same input protocol
same measurement scope
same hardware/environment context
different deployment candidates
```

This distinction matters. Comparing `onnxruntime_cpu` on synthetic zeros with
`pytorch_eager_cpu` on a dataset batch is not a valid deployment comparison.
That would mix deployment comparison with input sensitivity analysis.

Input sensitivity analysis is a separate future direction:

```text
same deployment candidate
different input protocols
```

Examples of different input protocols:

- `synthetic_fixed_zeros_batch1`
- `dataset_fixed_test_batch1`
- `dataset_stream_test_batch1`

Examples of different measurement scopes:

- `runtime_only`
- `end_to_end`

The current project mostly covers `runtime_only`.

## Target Output

Proposed future output path:

```text
outputs/deployment_report.json
```

## Top-Level Shape

```json
{
  "benchmark_run": {
    "schema_version": "0.1",
    "model": {
      "name": "TinyCNN",
      "checkpoint_path": "./artifacts/tinycnn_cifar10.pt",
      "onnx_path": "./artifacts/tinycnn_cifar10.onnx"
    },
    "input_protocol": {
      "name": "synthetic_fixed_zeros_batch1",
      "source": "synthetic",
      "shape": [1, 3, 32, 32],
      "batch_size": 1,
      "fill": "zeros",
      "reuse_policy": "same_prepared_input_repeated"
    },
    "measurement": {
      "scope": "runtime_only",
      "data_loading_included": false,
      "input_reuse": "same_prepared_input_repeated"
    },
    "environment": {
      "device": "cpu",
      "dtype": "torch.float32",
      "providers_requested": ["CPUExecutionProvider"],
      "providers_available": ["AzureExecutionProvider", "CPUExecutionProvider"],
      "providers_used": ["CPUExecutionProvider"]
    },
    "candidates": [
      {
        "name": "pytorch_eager_cpu",
        "runtime": "pytorch_eager",
        "device": "cpu",
        "role": "reference",
        "correctness": {
          "reference": true
        },
        "latency": {
          "p50_ms": 0.074,
          "p95_ms": 0.103,
          "p99_ms": 0.193
        }
      },
      {
        "name": "onnxruntime_cpu",
        "runtime": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "role": "deployment_candidate",
        "correctness": {
          "passed": true,
          "max_abs_diff": 2.38e-7,
          "mean_abs_diff": 9.38e-8,
          "tolerance": {
            "max_abs_diff": 0.0001
          }
        },
        "latency": {
          "p50_ms": 0.023,
          "p95_ms": 0.026,
          "p99_ms": 0.035
        }
      }
    ],
    "recommendation": {
      "status": "not_computed"
    }
  }
}
```

## Field Notes

- `benchmark_run`: one comparable experiment under shared conditions.
- `model`: model and artifact metadata used for the comparison.
- `input_protocol`: the shared input contract for all candidates.
- `measurement`: what is included in the measured section.
- `environment`: local execution context and ONNX Runtime providers.
- `candidates`: deployment configurations compared inside the run.
- `correctness`: numerical consistency information relative to the PyTorch
  reference.
- `latency`: runtime-only latency metrics for each candidate.
- `recommendation`: intentionally minimal for now.

## Current Inputs

The current project already produces two intermediate reports:

- `outputs/verify_results.json`
- `outputs/latency_results.json`

The first implementation of the deployment report combines these dictionaries
in memory. It does not run a configuration matrix and does not write
`outputs/deployment_report.json` automatically yet.

## Non-Goals For The First Version

- No full deployment matrix.
- No automatic multi-config execution.
- No sophisticated ranking model.
- No input sensitivity analysis.
- No end-to-end data pipeline benchmark.
- No production monitoring or serving assumptions.


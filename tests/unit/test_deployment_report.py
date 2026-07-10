import json

import pytest

from src.reporting.deployment_report import build_deployment_report


def _fake_verify_results(input_metadata=None):
    return {
        "passed": True,
        "max_abs_diff": 2.38e-7,
        "mean_abs_diff": 9.38e-8,
        "tolerance": {
            "max_abs_diff": 1.0e-4,
        },
        "onnx_path": "./artifacts/tinycnn_cifar10.onnx",
        "device": "cpu",
        "dtype": "torch.float32",
        "providers_requested": ["CPUExecutionProvider"],
        "providers_available": ["AzureExecutionProvider", "CPUExecutionProvider"],
        "providers_used": ["CPUExecutionProvider"],
        "input": input_metadata
        or {
            "source": "synthetic",
            "shape": [1, 3, 32, 32],
            "fill": "zeros",
        },
    }


def _fake_latency_results(input_metadata=None):
    return {
        "pytorch_eager": {
            "p50_ms": 0.074,
            "p95_ms": 0.103,
            "p99_ms": 0.193,
        },
        "onnxruntime": {
            "p50_ms": 0.023,
            "p95_ms": 0.026,
            "p99_ms": 0.035,
        },
        "onnx_path": "./artifacts/tinycnn_cifar10.onnx",
        "device": "cpu",
        "dtype": "torch.float32",
        "providers_requested": ["CPUExecutionProvider"],
        "providers_available": ["AzureExecutionProvider", "CPUExecutionProvider"],
        "providers_used": ["CPUExecutionProvider"],
        "measurement_scope": "runtime_only",
        "input_reuse": "same_prepared_input_repeated",
        "data_loading_included": False,
        "input": input_metadata
        or {
            "source": "synthetic",
            "shape": [1, 3, 32, 32],
            "fill": "zeros",
        },
    }


def test_deployment_report_is_json_serializable():
    report = build_deployment_report(_fake_verify_results(), _fake_latency_results())

    json.dumps(report)


def test_deployment_report_has_main_sections():
    report = build_deployment_report(_fake_verify_results(), _fake_latency_results())
    benchmark_run = report["benchmark_run"]

    assert "model" in benchmark_run
    assert "input_protocol" in benchmark_run
    assert "measurement" in benchmark_run
    assert "environment" in benchmark_run
    assert "candidates" in benchmark_run
    assert "recommendation" in benchmark_run


def test_deployment_report_candidates_use_verify_and_latency_results():
    report = build_deployment_report(_fake_verify_results(), _fake_latency_results())
    candidates = report["benchmark_run"]["candidates"]

    pytorch_candidate = next(
        candidate for candidate in candidates if candidate["runtime"] == "pytorch_eager"
    )
    onnx_candidate = next(
        candidate for candidate in candidates if candidate["runtime"] == "onnxruntime"
    )

    assert pytorch_candidate["role"] == "reference"
    assert pytorch_candidate["correctness"] == {"reference": True}
    assert pytorch_candidate["latency"]["p50_ms"] == 0.074

    assert onnx_candidate["role"] == "deployment_candidate"
    assert onnx_candidate["provider"] == "CPUExecutionProvider"
    assert onnx_candidate["correctness"]["passed"] is True
    assert onnx_candidate["correctness"]["max_abs_diff"] == 2.38e-7
    assert onnx_candidate["correctness"]["mean_abs_diff"] == 9.38e-8
    assert onnx_candidate["latency"]["p95_ms"] == 0.026


def test_deployment_report_rejects_input_protocol_mismatch():
    verify_results = _fake_verify_results(
        {
            "source": "synthetic",
            "shape": [1, 3, 32, 32],
            "fill": "zeros",
        }
    )
    latency_results = _fake_latency_results(
        {
            "source": "dataset_batch",
            "split": "test",
            "batch_size": 1,
            "shape": [1, 3, 32, 32],
        }
    )

    with pytest.raises(
        ValueError,
        match="verify and latency results use different input protocols",
    ):
        build_deployment_report(verify_results, latency_results)


def test_deployment_report_names_dataset_fixed_protocol():
    input_metadata = {
        "source": "dataset_batch",
        "split": "test",
        "batch_size": 1,
        "shape": [1, 3, 32, 32],
    }
    report = build_deployment_report(
        _fake_verify_results(input_metadata),
        _fake_latency_results(input_metadata),
    )

    input_protocol = report["benchmark_run"]["input_protocol"]

    assert input_protocol["name"] == "dataset_fixed_test_batch1"

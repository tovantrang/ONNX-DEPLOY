from typing import Any, cast


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _first(values: list[Any]) -> Any:
    return values[0] if values else None


def _batch_size(input_metadata: dict[str, Any]) -> int | None:
    if "batch_size" in input_metadata:
        return int(input_metadata["batch_size"])

    shape = _as_list(input_metadata.get("shape"))
    if shape:
        return int(shape[0])

    return None


def _input_protocol(input_metadata: dict[str, Any]) -> dict[str, Any]:
    protocol = {
        "source": input_metadata.get("source"),
        "shape": _as_list(input_metadata.get("shape")),
        "batch_size": _batch_size(input_metadata),
    }

    for optional_key in ("fill", "split"):
        if optional_key in input_metadata:
            protocol[optional_key] = input_metadata[optional_key]

    return protocol


def _protocol_signature(protocol: dict[str, Any]) -> tuple[object, ...]:
    return (
        protocol.get("source"),
        tuple(_as_list(protocol.get("shape"))),
        protocol.get("batch_size"),
        protocol.get("fill"),
        protocol.get("split"),
    )


def _protocol_name(protocol: dict[str, Any]) -> str:
    source = protocol.get("source")
    batch_size = protocol.get("batch_size")

    if source == "synthetic":
        fill = protocol.get("fill", "unknown")
        return f"synthetic_fixed_{fill}_batch{batch_size}"

    if source == "dataset_batch":
        split = protocol.get("split", "unknown")
        return f"dataset_fixed_{split}_batch{batch_size}"

    return f"{source}_batch{batch_size}"


def _latency_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "p50_ms": metrics.get("p50_ms"),
        "p95_ms": metrics.get("p95_ms"),
        "p99_ms": metrics.get("p99_ms"),
    }


def _device_label(device: object) -> str:
    device_text = str(device or "unknown")
    return device_text.replace(":", "_")


def build_deployment_report(
    verify_results: dict[str, object],
    latency_results: dict[str, object],
) -> dict[str, object]:
    verify_input = _input_protocol(_as_dict(verify_results.get("input")))
    latency_input = _input_protocol(_as_dict(latency_results.get("input")))

    if _protocol_signature(verify_input) != _protocol_signature(latency_input):
        raise ValueError("verify and latency results use different input protocols")

    input_protocol = {
        "name": _protocol_name(verify_input),
        **verify_input,
        "reuse_policy": latency_results.get("input_reuse"),
    }

    device = latency_results.get("device") or verify_results.get("device")
    dtype = latency_results.get("dtype") or verify_results.get("dtype")
    providers_requested = _as_list(
        latency_results.get("providers_requested")
        or verify_results.get("providers_requested")
    )
    providers_available = _as_list(
        latency_results.get("providers_available")
        or verify_results.get("providers_available")
    )
    providers_used = _as_list(
        latency_results.get("providers_used") or verify_results.get("providers_used")
    )
    onnx_provider = _first(providers_used)
    device_label = _device_label(device)

    pytorch_latency = _latency_summary(_as_dict(latency_results.get("pytorch_eager")))
    onnx_latency = _latency_summary(_as_dict(latency_results.get("onnxruntime")))

    report: dict[str, object] = {
        "benchmark_run": {
            "schema_version": "0.1",
            "model": {
                "name": verify_results.get("model_name", "TinyCNN"),
                "checkpoint_path": verify_results.get("checkpoint_path"),
                "onnx_path": verify_results.get("onnx_path")
                or latency_results.get("onnx_path"),
            },
            "input_protocol": input_protocol,
            "measurement": {
                "scope": latency_results.get("measurement_scope", "runtime_only"),
                "data_loading_included": latency_results.get(
                    "data_loading_included", False
                ),
                "input_reuse": latency_results.get("input_reuse"),
            },
            "environment": {
                "device": device,
                "dtype": dtype,
                "providers_requested": providers_requested,
                "providers_available": providers_available,
                "providers_used": providers_used,
            },
            "candidates": [
                {
                    "name": f"pytorch_eager_{device_label}",
                    "runtime": "pytorch_eager",
                    "device": device,
                    "role": "reference",
                    "correctness": {
                        "reference": True,
                    },
                    "latency": pytorch_latency,
                },
                {
                    "name": f"onnxruntime_{device_label}",
                    "runtime": "onnxruntime",
                    "provider": onnx_provider,
                    "device": device,
                    "role": "deployment_candidate",
                    "correctness": {
                        "passed": verify_results.get("passed"),
                        "max_abs_diff": verify_results.get("max_abs_diff"),
                        "mean_abs_diff": verify_results.get("mean_abs_diff"),
                        "tolerance": verify_results.get("tolerance"),
                    },
                    "latency": onnx_latency,
                },
            ],
            "recommendation": {
                "status": "not_computed",
            },
        }
    }

    return report

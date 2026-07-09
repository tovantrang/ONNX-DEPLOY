import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def create_latency_input(
    input_config=None,
    config=None,
    device="cpu",
    dataloader_factory=None,
):
    input_config = input_config or {}
    source = input_config.get("source", "synthetic")

    if source == "synthetic":
        shape = input_config.get("shape", [1, 3, 32, 32])
        fill = input_config.get("fill", "zeros")

        if fill != "zeros":
            raise ValueError(f"Unsupported latency input fill: {fill}")

        input_shape = [int(dim) for dim in shape]
        input_metadata = {
            "source": source,
            "shape": input_shape,
            "fill": fill,
        }

        logger.info("Latency input source: %s", input_metadata["source"])
        logger.info("Latency input shape: %s", input_metadata["shape"])
        logger.info("Latency input fill: %s", input_metadata["fill"])

        return torch.zeros(*input_shape, device=device), input_metadata

    if source == "dataset_batch":
        if config is None:
            raise ValueError("Latency input source 'dataset_batch' requires config")

        include_data_loading = bool(input_config.get("include_data_loading", False))
        if include_data_loading:
            raise ValueError(
                "Latency input include_data_loading=true is not supported yet"
            )

        split = input_config.get("split", "test")
        batch_size = int(input_config.get("batch_size", 1))

        if dataloader_factory is None:
            from src.data.dataloader import generate_dataloaders

            dataloader_factory = generate_dataloaders

        train_dl, val_dl, test_dl = dataloader_factory(config, batch_size=batch_size)
        dataloaders = {
            "train": train_dl,
            "val": val_dl,
            "test": test_dl,
        }
        if split not in dataloaders:
            raise ValueError(f"Unsupported latency input split: {split}")

        batch = next(iter(dataloaders[split]))
        input_tensor = batch[0] if isinstance(batch, tuple | list) else batch
        input_tensor = input_tensor.to(device)
        input_metadata = {
            "source": source,
            "split": split,
            "batch_size": batch_size,
            "shape": list(input_tensor.shape),
            "include_data_loading": False,
        }

        logger.info("Latency input source: %s", input_metadata["source"])
        logger.info("Latency input split: %s", input_metadata["split"])
        logger.info("Latency input batch size: %s", input_metadata["batch_size"])
        logger.info("Latency input shape: %s", input_metadata["shape"])
        logger.info("Latency measurement scope: runtime_only")

        return input_tensor, input_metadata

    raise ValueError(f"Unsupported latency input source: {source}")


def _synchronize_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _latency_summary(timings_ms, warmup_runs):
    timings_ms = np.array(timings_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "p99_ms": float(np.percentile(timings_ms, 99)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "num_runs": int(timings_ms.size),
        "warmup_runs": int(warmup_runs),
    }


def measure_runner_latency(
    runner, input_tensor, device, warmup_runs=10, timed_runs=100
):
    prepared_input = runner.prepare(input_tensor)

    for _ in range(warmup_runs):
        runner.run(prepared_input)

    timings_ms = []
    for _ in range(timed_runs):
        _synchronize_if_cuda(device)
        start_time = time.perf_counter()
        runner.run(prepared_input)
        _synchronize_if_cuda(device)
        end_time = time.perf_counter()
        timings_ms.append((end_time - start_time) * 1000)

    return _latency_summary(timings_ms, warmup_runs)


def measure_latency(
    pytorch_runner,
    onnx_runner,
    input_tensor,
    device,
    onnx_path,
    warmup_runs=10,
    timed_runs=100,
    output_path=None,
    input_metadata=None,
):
    logger.info("Running latency measurement")
    logger.info("Latency measurement scope: runtime_only")
    logger.info("Latency input reuse: same_prepared_input_repeated")
    logger.info("Latency data loading included: False")
    logger.info("Warmup runs: %s", warmup_runs)
    logger.info("Timed runs: %s", timed_runs)

    pytorch_stats = measure_runner_latency(
        pytorch_runner,
        input_tensor,
        device,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    onnx_stats = measure_runner_latency(
        onnx_runner,
        input_tensor,
        device,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )

    logger.info("PyTorch eager latency p50: %.4f ms", pytorch_stats["p50_ms"])
    logger.info("PyTorch eager latency p95: %.4f ms", pytorch_stats["p95_ms"])
    logger.info("ONNX Runtime latency p50: %.4f ms", onnx_stats["p50_ms"])
    logger.info("ONNX Runtime latency p95: %.4f ms", onnx_stats["p95_ms"])

    results = {
        "pytorch_eager": pytorch_stats,
        "onnxruntime": onnx_stats,
        "input": input_metadata
        or {
            "source": "synthetic",
            "shape": list(input_tensor.shape),
            "fill": "unknown",
        },
        "measurement_scope": "runtime_only",
        "input_reuse": "same_prepared_input_repeated",
        "data_loading_included": False,
        "input_shape": list(input_tensor.shape),
        "providers_requested": list(onnx_runner.providers_requested),
        "providers_available": list(onnx_runner.providers_available),
        "providers_used": list(onnx_runner.session_providers),
        "onnx_path": str(onnx_path),
        "device": str(device),
        "dtype": str(input_tensor.dtype),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved latency results to %s", output_path)

    return results

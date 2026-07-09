import json

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.benchmark.latency import create_latency_input, measure_latency
from src.export.onnx import create_verify_input, export_onnx, verify_onnxruntime
from src.models.tinycnn import TinyCNN
from src.runtimes.runner import ONNXRunner, PyTorchRunner


@pytest.mark.integration
def test_onnx_export_runtime_matches_pytorch_output(tmp_path):
    model = TinyCNN()
    model.eval()
    onnx_path = tmp_path / "tinycnn.onnx"

    export_onnx(model, "cpu", out_path=onnx_path)

    batch = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        pytorch_logits = model(batch).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_logits = session.run(["logits"], {"input": batch.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pytorch_logits - ort_logits)))
    assert ort_logits.shape == pytorch_logits.shape
    assert max_abs_diff < 1e-4


@pytest.mark.integration
def test_verify_onnxruntime_returns_json_serializable_results(tmp_path):
    model = TinyCNN()
    model.eval()
    onnx_path = tmp_path / "tinycnn.onnx"
    output_path = tmp_path / "verify_results.json"

    export_onnx(model, "cpu", out_path=onnx_path)

    results = verify_onnxruntime(
        model,
        "cpu",
        onnx_path=onnx_path,
        providers=["CPUExecutionProvider"],
        tolerance={"max_abs_diff": 1.0e-4},
        output_path=output_path,
        input_tensor=torch.zeros(1, 3, 32, 32),
        input_metadata={
            "source": "synthetic",
            "shape": [1, 3, 32, 32],
            "fill": "zeros",
        },
    )

    json.dumps(results)
    saved_results = json.loads(output_path.read_text())

    expected_keys = {
        "passed",
        "max_abs_diff",
        "mean_abs_diff",
        "tolerance",
        "pytorch_output_shape",
        "onnx_output_shape",
        "providers_requested",
        "providers_available",
        "providers_used",
        "onnx_path",
        "device",
        "dtype",
        "input",
        "input_shape",
    }
    assert expected_keys.issubset(results)
    assert saved_results == results
    assert results["input"] == {
        "source": "synthetic",
        "shape": [1, 3, 32, 32],
        "fill": "zeros",
    }
    assert isinstance(results["passed"], bool)
    assert isinstance(results["max_abs_diff"], float)
    assert isinstance(results["mean_abs_diff"], float)
    assert results["passed"] is True
    assert results["max_abs_diff"] <= results["tolerance"]["max_abs_diff"]
    assert results["providers_used"] == ["CPUExecutionProvider"]


def test_verify_input_rejects_unsupported_source():
    with pytest.raises(
        ValueError,
        match="Unsupported verify input source: definitely_missing",
    ):
        create_verify_input({"source": "definitely_missing"})


def test_verify_input_can_use_mocked_dataset_batch():
    def fake_dataloader_factory(config, batch_size=None):
        del config
        batch_size = batch_size or 1
        features = torch.zeros(batch_size, 3, 32, 32)
        labels = torch.zeros(batch_size, dtype=torch.long)
        dataloader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
        return dataloader, dataloader, dataloader

    input_tensor, input_metadata = create_verify_input(
        {
            "source": "dataset_batch",
            "split": "test",
            "batch_size": 2,
        },
        config={"paths": {"data": "./data"}},
        dataloader_factory=fake_dataloader_factory,
    )

    assert list(input_tensor.shape) == [2, 3, 32, 32]
    assert input_metadata == {
        "source": "dataset_batch",
        "split": "test",
        "batch_size": 2,
        "shape": [2, 3, 32, 32],
    }


@pytest.mark.integration
def test_latency_returns_json_serializable_results(tmp_path):
    model = TinyCNN()
    model.eval()
    onnx_path = tmp_path / "tinycnn.onnx"
    output_path = tmp_path / "latency_results.json"

    export_onnx(model, "cpu", out_path=onnx_path)

    pytorch_runner = PyTorchRunner(model, "cpu")
    onnx_runner = ONNXRunner(onnx_path, providers=["CPUExecutionProvider"])
    input_tensor, input_metadata = create_latency_input(
        {
            "source": "synthetic",
            "shape": [1, 3, 32, 32],
            "fill": "zeros",
        }
    )

    results = measure_latency(
        pytorch_runner,
        onnx_runner,
        input_tensor,
        "cpu",
        onnx_path,
        warmup_runs=1,
        timed_runs=3,
        output_path=output_path,
        input_metadata=input_metadata,
    )

    json.dumps(results)
    saved_results = json.loads(output_path.read_text())

    assert saved_results == results
    assert results["input"] == {
        "source": "synthetic",
        "shape": [1, 3, 32, 32],
        "fill": "zeros",
    }
    assert results["measurement_scope"] == "runtime_only"
    assert results["input_reuse"] == "same_prepared_input_repeated"
    assert results["data_loading_included"] is False
    assert results["input_shape"] == [1, 3, 32, 32]
    assert results["providers_used"] == ["CPUExecutionProvider"]
    assert results["onnx_path"] == str(onnx_path)
    assert results["device"] == "cpu"
    assert results["dtype"] == "torch.float32"

    for runtime_name in ("pytorch_eager", "onnxruntime"):
        metrics = results[runtime_name]
        assert metrics["num_runs"] == 3
        assert metrics["warmup_runs"] == 1
        assert metrics["p95_ms"] >= metrics["p50_ms"]

        for key in (
            "mean_ms",
            "median_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
        ):
            assert isinstance(metrics[key], float)
            assert metrics[key] >= 0.0


def test_latency_input_rejects_unsupported_source():
    with pytest.raises(
        ValueError,
        match="Unsupported latency input source: definitely_missing",
    ):
        create_latency_input(
            {
                "source": "definitely_missing",
                "shape": [1, 3, 32, 32],
                "fill": "zeros",
            }
        )


def test_latency_input_rejects_data_loading_in_timed_scope():
    with pytest.raises(
        ValueError,
        match="Latency input include_data_loading=true is not supported yet",
    ):
        create_latency_input(
            {
                "source": "dataset_batch",
                "split": "test",
                "batch_size": 1,
                "include_data_loading": True,
            },
            config={"paths": {"data": "./data"}},
        )


def test_latency_can_measure_mocked_dataset_batch(tmp_path):
    def fake_dataloader_factory(config, batch_size=None):
        del config
        batch_size = batch_size or 1
        features = torch.zeros(batch_size, 3, 32, 32)
        labels = torch.zeros(batch_size, dtype=torch.long)
        dataloader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
        return dataloader, dataloader, dataloader

    model = TinyCNN()
    model.eval()
    onnx_path = tmp_path / "tinycnn.onnx"
    output_path = tmp_path / "latency_results.json"

    export_onnx(model, "cpu", out_path=onnx_path)

    input_tensor, input_metadata = create_latency_input(
        {
            "source": "dataset_batch",
            "split": "test",
            "batch_size": 1,
            "include_data_loading": False,
        },
        config={"paths": {"data": "./data"}},
        dataloader_factory=fake_dataloader_factory,
    )
    pytorch_runner = PyTorchRunner(model, "cpu")
    onnx_runner = ONNXRunner(onnx_path, providers=["CPUExecutionProvider"])

    results = measure_latency(
        pytorch_runner,
        onnx_runner,
        input_tensor,
        "cpu",
        onnx_path,
        warmup_runs=1,
        timed_runs=3,
        output_path=output_path,
        input_metadata=input_metadata,
    )

    saved_results = json.loads(output_path.read_text())
    assert saved_results == results
    assert results["measurement_scope"] == "runtime_only"
    assert results["input_reuse"] == "same_prepared_input_repeated"
    assert results["data_loading_included"] is False
    assert results["input"] == {
        "source": "dataset_batch",
        "split": "test",
        "batch_size": 1,
        "shape": [1, 3, 32, 32],
        "include_data_loading": False,
    }
    assert results["input_shape"] == [1, 3, 32, 32]
    assert results["pytorch_eager"]["num_runs"] == 3
    assert results["onnxruntime"]["num_runs"] == 3

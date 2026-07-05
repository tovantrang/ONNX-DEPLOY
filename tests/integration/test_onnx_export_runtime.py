import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.export.onnx import export_onnx, verify_onnxruntime
from src.models.tinycnn import TinyCNN


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
        input_tensor=torch.randn(1, 3, 32, 32),
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
        "input_shape",
    }
    assert expected_keys.issubset(results)
    assert saved_results == results
    assert isinstance(results["passed"], bool)
    assert isinstance(results["max_abs_diff"], float)
    assert isinstance(results["mean_abs_diff"], float)
    assert results["passed"] is True
    assert results["max_abs_diff"] <= results["tolerance"]["max_abs_diff"]
    assert results["providers_used"] == ["CPUExecutionProvider"]

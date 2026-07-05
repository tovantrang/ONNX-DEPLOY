import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.export.onnx import export_onnx
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

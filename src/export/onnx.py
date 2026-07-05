import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.runtimes.runner import resolve_onnx_providers

logger = logging.getLogger(__name__)


def export_sanity_forward(model, device):
    model.eval()
    x = torch.zeros(1, 3, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)
    logger.info(
        "Export sanity forward: input=%s output=%s",
        tuple(x.shape),
        tuple(y.shape),
    )
    assert y.shape == (1, 10), f"Unexpected output shape: {tuple(y.shape)}"


def export_onnx(model, device, out_path="./artifacts/tinycnn_cifar10.onnx"):
    """
    Minimal ONNX export.
    TODO later: add dynamic axes + onnxruntime verification.
    """
    model.eval()
    dummy = torch.zeros(1, 3, 32, 32, device=device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
    )
    logger.info("Exported ONNX model to %s", out_path)
    assert out_path.exists(), "ONNX export failed: file not found"


def verify_onnxruntime(
    model,
    device,
    onnx_path="./artifacts/tinycnn_cifar10.onnx",
    providers=None,
    tolerance=None,
    output_path=None,
    input_tensor=None,
):
    """
    Minimal ONNXRuntime check: PyTorch logits vs ONNXRuntime logits on the same input.
    TODO later: test with a real CIFAR batch (not just zeros).
    """

    logger.info("Verifying PyTorch vs ONNX Runtime output consistency")
    tolerance = tolerance or {"max_abs_diff": 1.0e-4}
    max_abs_diff_tolerance = float(tolerance["max_abs_diff"])
    providers_requested = providers or ["CPUExecutionProvider"]

    model.eval()
    x = (
        input_tensor.to(device)
        if input_tensor is not None
        else torch.zeros(1, 3, 32, 32, device=device)
    )

    with torch.no_grad():
        pt_logits = model(x).cpu().numpy()

    providers_used, providers_available = resolve_onnx_providers(providers_requested)
    sess = ort.InferenceSession(onnx_path, providers=providers_used)
    session_providers = sess.get_providers()
    logger.info("ONNX Runtime session providers: %s", session_providers)
    ort_out = sess.run(["logits"], {"input": x.cpu().numpy()})[0]

    abs_diff = np.abs(pt_logits - ort_out)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    passed = bool(
        ort_out.shape == pt_logits.shape and max_abs_diff <= max_abs_diff_tolerance
    )
    logger.info(
        "Correctness shapes: pytorch=%s onnxruntime=%s",
        pt_logits.shape,
        ort_out.shape,
    )
    logger.info("Correctness max_abs_diff: %.8g", max_abs_diff)
    logger.info("Correctness mean_abs_diff: %.8g", mean_abs_diff)
    log_passed = logger.info if passed else logger.warning
    log_passed("Verification passed: %s", passed)

    results = {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "tolerance": {
            "max_abs_diff": max_abs_diff_tolerance,
        },
        "pytorch_output_shape": list(pt_logits.shape),
        "onnx_output_shape": list(ort_out.shape),
        "providers_requested": list(providers_requested),
        "providers_available": list(providers_available),
        "providers_used": list(session_providers),
        "onnx_path": str(onnx_path),
        "device": str(device),
        "dtype": str(x.dtype),
        "input_shape": list(x.shape),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved verify results to %s", output_path)

    return results

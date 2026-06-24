from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.runtimes.runner import resolve_onnx_providers


def export_sanity_forward(model, device):
    model.eval()
    x = torch.zeros(1, 3, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"SANITY forward: in {tuple(x.shape)}, out {tuple(y.shape)}")
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
    print(f"Exported ONNX to: {out_path.resolve()}")
    assert out_path.exists(), "ONNX export failed: file not found"


def verify_onnxruntime(
    model,
    device,
    onnx_path="./artifacts/tinycnn_cifar10.onnx",
    providers=None,
):
    """
    Minimal ONNXRuntime check: PyTorch logits vs ONNXRuntime logits on the same input.
    TODO later: test with a real CIFAR batch (not just zeros).
    """

    model.eval()
    x = torch.zeros(1, 3, 32, 32, device=device)

    with torch.no_grad():
        pt_logits = model(x).cpu().numpy()

    providers, _ = resolve_onnx_providers(providers)
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print("ONNX Runtime session providers:", sess.get_providers())
    ort_out = sess.run(["logits"], {"input": x.cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pt_logits - ort_out)))
    print(
        "ONNXRuntime verify:",
        "pt",
        pt_logits.shape,
        "ort",
        ort_out.shape,
        "max_abs_diff",
        max_abs_diff,
    )

    assert (
        ort_out.shape == pt_logits.shape
    ), "Shape mismatch between PyTorch and ONNXRuntime"
    assert max_abs_diff < 1e-4, f"Outputs differ too much (max_abs_diff={max_abs_diff})"

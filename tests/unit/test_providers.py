from src.runtimes.runner import resolve_onnx_providers


def test_resolve_onnx_providers_uses_cpu_provider():
    providers, available = resolve_onnx_providers(["CPUExecutionProvider"])

    assert "CPUExecutionProvider" in available
    assert providers == ["CPUExecutionProvider"]


def test_resolve_onnx_providers_falls_back_to_cpu_provider():
    providers, available = resolve_onnx_providers(["DefinitelyMissingProvider"])

    assert "CPUExecutionProvider" in available
    assert providers == ["CPUExecutionProvider"]

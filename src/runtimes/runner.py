import logging

import torch


logger = logging.getLogger(__name__)


def resolve_onnx_providers(requested_providers=None):
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    requested_providers = requested_providers or ["CPUExecutionProvider"]
    providers = [
        provider for provider in requested_providers if provider in available_providers
    ]
    missing_providers = [
        provider
        for provider in requested_providers
        if provider not in available_providers
    ]

    if not providers and "CPUExecutionProvider" in available_providers:
        providers = ["CPUExecutionProvider"]

    if not providers:
        raise RuntimeError(
            "No usable ONNX Runtime provider found. "
            f"Requested: {requested_providers}. "
            f"Available: {available_providers}."
        )

    if missing_providers:
        logger.warning(
            "ONNX Runtime provider fallback: requested=%s missing=%s using=%s",
            requested_providers,
            missing_providers,
            providers,
        )

    logger.info("ONNX Runtime providers requested: %s", requested_providers)
    logger.info("ONNX Runtime providers available: %s", available_providers)
    logger.info("ONNX Runtime providers used: %s", providers)

    return providers, available_providers


class Runner:
    name: str

    def prepare(self, x):
        raise NotImplementedError

    def run(self, x):
        raise NotImplementedError


class PyTorchRunner(Runner):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.name = "pytorch_eager"

    def prepare(self, x):
        return x.to(self.device)

    def run(self, x):
        with torch.no_grad():
            y = self.model(x)

        return y.cpu().numpy()


class ONNXRunner(Runner):
    def __init__(self, onnx_path, providers):
        import onnxruntime as ort

        providers, _ = resolve_onnx_providers(providers)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.name = "onnxruntime"
        logger.info("ONNX Runtime session providers: %s", self.session.get_providers())

    def prepare(self, x):
        return x.cpu().numpy()

    def run(self, x):
        y = self.session.run(["logits"], {"input": x})[0]

        return y

import torch


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

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.name = "onnxruntime"

    def prepare(self, x):
        return x.cpu().numpy()

    def run(self, x):
        y = self.session.run(["logits"], {"input": x})[0]

        return y

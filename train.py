import argparse
import logging
import sys

import torch
import torch.nn as nn

from src.benchmark.latency import measure_runner_latency
from src.data.dataloader import generate_dataloaders
from src.export.eager import load_model_eager
from src.export.eager import save_model_eager
from src.export.onnx import export_onnx
from src.export.onnx import export_sanity_forward
from src.export.onnx import verify_onnxruntime
from src.models.tinycnn import TinyCNN
from src.runtimes.runner import ONNXRunner
from src.runtimes.runner import PyTorchRunner
from src.training.training_loop import evaluate
from src.training.training_loop import train
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


logger = logging.getLogger(__name__)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        default="verify",
        choices=["train", "export", "verify", "latency"],
        nargs="?",
        help=(
            "Command to run: 'train' to train the model, 'export' to export, "
            "'verify' to compare PyTorch and ONNX Runtime outputs, "
            "'latency' to measure inference latency"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yaml",
        help="Path to config file (YAML)",
    )
    return parser.parse_args(argv)


def main(args):
    C = load_config(args.config)
    setup_logging(C.get("logging", {}).get("level", "INFO"))
    set_seed(C["seed"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    onnx_providers = C.get("onnx", {}).get("providers")

    train_dl, val_dl, test_dl = generate_dataloaders(C)

    if args.command == "train":
        model = TinyCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=C["training"]["lr"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(C["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{C['training']['epochs']}")
            train(model, train_dl, opt, criterion, device)
            evaluate(model, val_dl, criterion, device)

        evaluate(model, test_dl, criterion, device, name="Test")

        print("Training and evaluation complete.")

        save_model_eager(model, C["export"]["pt_path"])

    elif args.command == "export":
        model_eager = TinyCNN().to(device)
        load_model_eager(model_eager, path=C["export"]["pt_path"], device=device)

        export_sanity_forward(model_eager, device)

        export_onnx(model_eager, device, out_path=C["export"]["onnx_path"])

    elif args.command == "verify":
        model_eager = TinyCNN().to(device)
        load_model_eager(model_eager, path=C["export"]["pt_path"], device=device)

        verify_onnxruntime(
            model_eager,
            device,
            onnx_path=C["export"]["onnx_path"],
            providers=onnx_providers,
        )

    elif args.command == "latency":
        model_eager = TinyCNN().to(device)
        _, _, test_dl = generate_dataloaders(
            C, batch_size=1
        )  # use batch size 1 for latency measurement
        load_model_eager(model_eager, path=C["export"]["pt_path"], device=device)
        eager_runner = PyTorchRunner(model_eager, device)
        onnx_runner = ONNXRunner(C["export"]["onnx_path"], providers=onnx_providers)

        eager_stats = measure_runner_latency(eager_runner, test_dl, device)
        onnx_stats = measure_runner_latency(onnx_runner, test_dl, device)
        logger.info(
            "Latency PyTorch eager: %.2f ms",
            eager_stats["mean_latency_ms"],
        )
        logger.info(
            "Latency ONNX Runtime: %.2f ms",
            onnx_stats["mean_latency_ms"],
        )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

import torch

from src.models.tinycnn import TinyCNN


def test_tinycnn_forward_shape():
    model = TinyCNN()
    model.eval()
    batch = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, 10)

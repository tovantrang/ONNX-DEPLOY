from src.utils.config import load_config


def test_default_config_loads_expected_keys():
    config = load_config("configs/default.yaml")

    assert "seed" in config
    assert "training" in config
    assert "paths" in config
    assert "export" in config
    assert "onnx" in config
    assert "providers" in config["onnx"]
    assert config["onnx"]["providers"]

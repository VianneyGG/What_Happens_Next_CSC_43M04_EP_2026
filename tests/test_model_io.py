import os
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_lstm import CNNLSTM
from src.train import build_model

NUM_CLASSES = 33
B, T, C, H, W = 2, 4, 3, 112, 112


def _save_checkpoint(model, path):
    torch.save({"model_state_dict": model.state_dict()}, path)


def _load_state_dict(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])


def test_checkpoint_roundtrip_cnn_baseline():
    model = CNNBaseline(num_classes=NUM_CLASSES).eval()
    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        out_before = model(x)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        _save_checkpoint(model, path)
        model2 = CNNBaseline(num_classes=NUM_CLASSES).eval()
        _load_state_dict(model2, path)
        with torch.no_grad():
            out_after = model2(x)
        assert out_after.shape == (B, NUM_CLASSES)
        assert torch.allclose(out_before, out_after)
    finally:
        os.unlink(path)


def test_checkpoint_roundtrip_cnn_lstm():
    model = CNNLSTM(num_classes=NUM_CLASSES).eval()
    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        out_before = model(x)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        _save_checkpoint(model, path)
        model2 = CNNLSTM(num_classes=NUM_CLASSES).eval()
        _load_state_dict(model2, path)
        with torch.no_grad():
            out_after = model2(x)
        assert out_after.shape == (B, NUM_CLASSES)
        assert torch.allclose(out_before, out_after)
    finally:
        os.unlink(path)


def test_build_model_cnn_baseline():
    cfg = OmegaConf.create({"model": {"name": "cnn_baseline", "num_classes": NUM_CLASSES, "pretrained": False}})
    model = build_model(cfg).eval()
    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES)


def test_build_model_cnn_lstm():
    cfg = OmegaConf.create(
        {"model": {"name": "cnn_lstm", "num_classes": NUM_CLASSES, "pretrained": False, "lstm_hidden_size": 512}}
    )
    model = build_model(cfg).eval()
    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES)


def test_build_model_unknown_raises():
    cfg = OmegaConf.create({"model": {"name": "nonexistent_model", "num_classes": NUM_CLASSES, "pretrained": False}})
    with pytest.raises(ValueError):
        build_model(cfg)

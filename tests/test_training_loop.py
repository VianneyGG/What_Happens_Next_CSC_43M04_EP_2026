import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from train import build_model, evaluate_epoch, train_one_epoch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

NUM_CLASSES = 4


def _fake_cfg(name="cnn_baseline"):
    return OmegaConf.create(
        {"model": {"name": name, "num_classes": NUM_CLASSES, "pretrained": False, "lstm_hidden_size": 64}}
    )


def _fake_loader(B=2, T=2, C=3, H=8, W=8, n_batches=2):
    videos = torch.randn(B * n_batches, T, C, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B * n_batches,))
    return DataLoader(TensorDataset(videos, labels), batch_size=B)


@pytest.mark.parametrize("device", DEVICES)
def test_build_model_cnn_baseline(device):
    model = build_model(_fake_cfg("cnn_baseline")).to(device)
    x = torch.randn(2, 2, 3, 8, 8, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("device", DEVICES)
def test_build_model_cnn_lstm(device):
    model = build_model(_fake_cfg("cnn_lstm")).to(device)
    x = torch.randn(2, 2, 3, 8, 8, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(_fake_cfg("nonexistent_model"))


@pytest.mark.parametrize("device", DEVICES)
def test_train_one_epoch_returns_floats(device):
    model = build_model(_fake_cfg("cnn_baseline")).to(device)
    loader = _fake_loader()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = train_one_epoch(model, loader, loss_fn, optimizer, torch.device(device))
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_evaluate_epoch_returns_floats(device):
    model = build_model(_fake_cfg("cnn_baseline")).to(device)
    loader = _fake_loader()
    loss_fn = nn.CrossEntropyLoss()
    loss, acc = evaluate_epoch(model, loader, loss_fn, torch.device(device))
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0

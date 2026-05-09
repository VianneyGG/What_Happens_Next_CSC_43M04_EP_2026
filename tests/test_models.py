import torch
import pytest
from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_lstm import CNNLSTM

NUM_CLASSES = 33
B, T, C, H, W = 2, 4, 3, 112, 112

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_cnn_baseline_output_shape(device):
    model = CNNBaseline(num_classes=NUM_CLASSES).to(device).eval()
    x = torch.randn(B, T, C, H, W, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES), f"expected ({B}, {NUM_CLASSES}), got {out.shape}"


@pytest.mark.parametrize("device", DEVICES)
def test_cnn_lstm_output_shape(device):
    model = CNNLSTM(num_classes=NUM_CLASSES).to(device).eval()
    x = torch.randn(B, T, C, H, W, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES), f"expected ({B}, {NUM_CLASSES}), got {out.shape}"


@pytest.mark.parametrize("device", DEVICES)
def test_cnn_baseline_no_nan(device):
    model = CNNBaseline(num_classes=NUM_CLASSES).to(device).eval()
    x = torch.randn(B, T, C, H, W, device=device)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "CNNBaseline output contains NaN"


@pytest.mark.parametrize("device", DEVICES)
def test_cnn_lstm_no_nan(device):
    model = CNNLSTM(num_classes=NUM_CLASSES).to(device).eval()
    x = torch.randn(B, T, C, H, W, device=device)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "CNNLSTM output contains NaN"

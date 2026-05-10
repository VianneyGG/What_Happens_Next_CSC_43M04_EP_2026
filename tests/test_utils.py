import pytest
import torch
import torchvision.transforms as T

from src.utils import accuracy_topk, build_transforms, split_train_val


def test_topk_perfect_top1():
    targets = torch.tensor([2, 0, 3, 1])
    logits = torch.zeros(4, 5)
    logits[range(4), targets] = 10.0
    (top1,) = accuracy_topk(logits, targets, (1,))
    assert top1.item() == pytest.approx(1.0)


def test_topk_zero_top1():
    targets = torch.tensor([0, 1, 2, 3])
    logits = torch.zeros(4, 5)
    for i in range(4):
        logits[i, (targets[i].item() + 1) % 5] = 10.0
    (top1,) = accuracy_topk(logits, targets, (1,))
    assert top1.item() == pytest.approx(0.0)


def test_topk_top5_hit():
    # 6-class logit: correct class is rank 5 → top-1=0, top-5=1
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    targets = torch.tensor([4])
    top1, top5 = accuracy_topk(logits, targets, (1, 5))
    assert top1.item() == pytest.approx(0.0)
    assert top5.item() == pytest.approx(1.0)


def test_topk_returns_two_values():
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    result = accuracy_topk(logits, targets, (1, 5))
    assert len(result) == 2


def test_split_sizes():
    samples = list(range(10))
    train, val = split_train_val(samples, val_ratio=0.2, seed=42)
    assert len(train) == 8
    assert len(val) == 2
    assert len(train) + len(val) == len(samples)


def test_split_deterministic():
    samples = list(range(20))
    train1, val1 = split_train_val(samples, val_ratio=0.2, seed=42)
    train2, val2 = split_train_val(samples, val_ratio=0.2, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_split_no_val():
    samples = list(range(10))
    train, val = split_train_val(samples, val_ratio=0.0, seed=42)
    assert len(train) == 10
    assert len(val) == 0


def test_build_transforms_type():
    assert isinstance(build_transforms(is_training=True), T.Compose)
    assert isinstance(build_transforms(is_training=False), T.Compose)

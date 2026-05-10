## Model rules

- All models must accept `(B, T, C, H, W)` and return `(B, num_classes)`. Enforced by `test_model_io.py`.
- Register every new model in `src/train.py:build_model()`. A model file not registered there is invisible to the training pipeline.
- Checkpoints save to `src/best_model.pt`. Do not change this path without updating both `train.py` and the Hydra default for `training.checkpoint_path`.
- Frame encoders (e.g., ResNet18) initialize with `pretrained=True` unless the experiment YAML sets `model.pretrained: false`.
- Never import model classes directly in `train.py` — use the `build_model(cfg)` factory to keep experiments config-driven.

## Architecture context

- Key source paths:
  - `src/train.py` — training loop; saves best checkpoint (Hydra config + weights to `src/best_model.pt`)
  - `src/evaluate.py` — Top-1 / Top-5 accuracy on full val split
  - `src/create_submission.py` — inference on test split → `submission.csv`
  - `src/utils.py` — transforms, train/val split helper, random seeds
  - `src/dataset/video_dataset.py` — loads T frames per video, returns (B, T, C, H, W) tensors
  - `src/models/cnn_baseline.py` — ResNet18 + temporal average pooling
  - `src/models/cnn_lstm.py` — ResNet18 (frame encoder) + LSTM over time
  - `src/configs/` — Hydra YAML config tree (root: `config.yaml`, `num_classes: 33`)

## Tensor I/O contract

All models: `(B, T, C, H, W)` → `(B, 33)`.
Dataset clips: `T = num_frames` (default 8 in config, but each video only has 4 physical frames — sampled with replacement).
Frame format: JPEG files named `frame_000.jpg … frame_003.jpg` inside each `video_<id>/` folder.

## Registered models (`src/train.py:build_model()`)

| Config name | Class | Notes |
|---|---|---|
| `cnn_baseline` | `CNNBaseline` | ResNet18 + avg pool over time |
| `cnn_lstm` | `CNNLSTM` | ResNet18 per frame + LSTM; `lstm_hidden_size` defaults to 512 |

## Data layout

```
/Data/vianney.gauthier/processed_data/val2/
├── train/   # class-labelled folders (e.g. 000_Closing_something/video_<id>/)
├── val/     # full val split
└── test/    # unlabelled, for submission
```

Active paths configured in `src/configs/data/vianney.yaml`. Never hardcode.

## Hydra config structure

```
src/configs/
├── config.yaml                              # root: num_classes=33
├── model/{cnn_baseline,cnn_lstm}.yaml
├── data/{default,vianney}.yaml
├── train/default.yaml
└── experiment/{baseline_from_scratch,baseline_pretrained}.yaml
```

Config keys for data: `cfg.dataset.train_dir`, `cfg.dataset.val_dir`, `cfg.dataset.test_dir`, `cfg.dataset.num_frames`.

Add a new experiment: create `src/configs/experiment/your_exp.yaml` composing model + train + data groups.

## Common mistakes

1. **Hardcoded data path** — `data_path = "/Data/..."` in source. Always read from `cfg.dataset.train_dir`.
2. **Wrong tensor shape** — forgetting the T dimension; models expect 5D `(B,T,C,H,W)`, not 4D `(B,C,H,W)`.
3. **Missing `build_model()` registration** — new model file exists but `train.py:build_model()` raises `ValueError`.
4. **Checkpoint path** — `evaluate.py` and `create_submission.py` require `training.checkpoint_path=src/best_model.pt` explicitly.
5. **Frame count mismatch** — `num_frames` in the data config must match the model's expected sequence length.

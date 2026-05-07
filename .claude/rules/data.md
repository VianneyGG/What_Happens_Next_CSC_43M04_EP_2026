## Data rules

- Always use `VideoDataset` to load frames. Never manually `glob` + `PIL.open` frame sequences in model or training code.
- Data paths come from Hydra config only (`cfg.dataset.train_dir`, `cfg.dataset.val_dir`, `cfg.dataset.test_dir`).
- Clip length is set by `num_frames` in the data config. Override in the experiment YAML — never hardcode in source.
- Set `num_workers` from config. Never hardcode `num_workers=0` in production code.
- The `test/` split is unlabelled — never assume labels exist there.

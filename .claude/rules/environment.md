## Environment

- **Python**: 3.10+ via `uv` (`.venv/` in repo root)
- **PyTorch**: ≥ 2.9.0, CUDA 12.8 wheels (`pytorch-cu128` index)
- **GPU**: required — CPU-only runs OOM on frame batches
- **Invoke always via**: `uv run <cmd>` or activate `.venv`; never use system Python
- **CUDA check**: `python3 -c "import torch; print(torch.cuda.is_available())"`
- **Disk**: frames live at `/Data/vianney.gauthier/processed_data/val2/` (external, not in repo)
- **Hydra logs**: accumulate at `src/outputs/`; clean with `rm -rf src/outputs/`

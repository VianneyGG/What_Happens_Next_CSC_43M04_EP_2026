## Test rules

- Existing tests: `test_models.py`, `test_video_dataset.py`, `test_utils.py`, `test_training_loop.py`, `test_model_io.py`. When adding a new component, create `tests/test_<component>.py`.
- Always assert tensor shapes explicitly: `assert out.shape == (B, 33)`.
- Use small batch sizes (`B=2`) and short clip lengths (`T=4`) in tests to avoid GPU OOM.
- Do not mock `VideoDataset` with random tensors of the wrong shape — the test will pass but the shape contract will break in production.
- If a changed component has no test coverage, say so explicitly rather than running an empty test suite.

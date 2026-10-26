# Manual NPO models

Place pinned files here before running `notebooks/Local/MOTHER_RUNNER.ps1`:

- `target/config.json`, `target/model.safetensors`: full target from `inputs_v1.json`.
- `oracle/config.json`, `oracle/model.safetensors`: retain-only diagnostic oracle from that registry.
- `tokenizer/config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `tokenizer/special_tokens_map.json`: full target revision.
- `generation_config.json`: file pinned in `experiments/reproductions/NPO/runtime_sources_v1.json`.

Exact repository IDs, revisions and SHA256 values are in those registries. Files
are read directly, hashed before use and never downloaded or copied by the local
runner. Arbitrary replacement models do not satisfy the current D1 experiment.

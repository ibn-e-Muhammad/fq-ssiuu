# Manual NPO models

Place pinned files here before running `notebooks/Local/MOTHER_RUNNER.ps1`:

The downloaded target may stay in `tofu_Llama-3.2-1B-Instruct/`, with
`config.json`, `model.safetensors`, tokenizer files and `generation_config.json`
at that directory's root, as in the current Windows layout. The runner accepts
either `models/NPO` or that target directory as `--models` / notebook `MODELS`.
It reads these files in place. A partial nested `tokenizer/` directory does not
override complete tokenizer files at the target root. Hugging Face `.cache/`
download metadata is not used as a model or as proof of file identity.

The original layout is also supported:

- `target/config.json`, `target/model.safetensors`: full target from `inputs_v1.json`.
- `oracle/config.json`, `oracle/model.safetensors`: retain-only diagnostic oracle from that registry.
- `tokenizer/config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `tokenizer/special_tokens_map.json`: full target revision.
- `generation_config.json`: file pinned in `experiments/reproductions/NPO/runtime_sources_v1.json`.

Exact repository IDs, revisions and SHA256 values are in those registries. Files
are read directly, hashed before use and never downloaded or copied by the local
runner. Arbitrary replacement models do not satisfy the current D1 experiment.

Oracle configuration is unchanged: `oracle/` is alongside the target directory,
under `models/NPO`. Its actual location will be configured separately; a full
run still requires the retain95 oracle and will report it missing until supplied.

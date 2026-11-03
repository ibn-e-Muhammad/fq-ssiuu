# Third-run engineering acceptance — 2026-11-03

**CPU acceptance passed; actual 1B Windows/GPU acceptance is pending.**
The implementation is the containing Git commit on `codex/npo-lora-third-run`.
`VALIDATION.json` pins tested source hashes and package versions. This is new
engineering evidence, separate from the historical researcher's GPU test log.

| Check | Result |
|---|---|
| New LoRA tests + existing NPO/QRA regressions | 238 passed, 3 CUDA-only skipped |
| Existing local-input tests | 10 passed, 1 optional Arrow/datasets check skipped |
| Archived run-2 recovery tests, executed from a temporary copy | 7 passed, 1 CUDA-only skipped |
| Run-2 streamed AdamW scalar/numerical self-test | Passed |
| Real pinned tokenizer/data preflight | All 4,000 training rows and 4,585 evaluation variants fit |
| Maximum evaluated answer sequence | 146 tokens, below the 200-position limit |
| Raw preserved archive inventory | 87 SHA-256 hashes verified |
| Recorded stage artifacts | All 27 SHA-256 hashes verified |
| Derived reanalysis outputs | All 6 output hashes verified |
| CI-scope Ruff lint/format | Passed locally |
| Notebook validation | See `notebook_validation.json`; syntax/schema only |

The combined suite covers sequence-sum NPO values/gradients and masking, frozen
reference/backbone, logical retain token normalization under smaller device
batches, sampling coverage, exact interrupted/resumed adapter training,
checkpoint corruption fallback, tiny-Llama merge/reload, evaluator/report output,
and modified-cache rejection. The optional third-run CUDA test performs actual
LoRA backward and INT8/NF4 conversion when the target CUDA backend is available.
Its CPU skip must not be read as a successful quantization test.

The old QRA regression initially exposed reliance on a Transformers-5-private
generation API. Only the active helper was corrected to use public
`GenerationConfig().to_dict()`; the archived second-run helper is unchanged.
Initial failure receipts remain alongside final passing receipts. Three active
files also received formatting-only changes so the existing CI formatter passes.
An initial separate local-input invocation lacked `PYTHONPATH=src:.`; rerunning
with the normal source path and plugin isolation produced the result above.

Commands (from repository root; temporary environment was `/tmp/npo3-venv`):

```bash
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  experiments/reproductions/NPO_3rd-test/tests tests/methods/NPO tests/methods/QRA \
  --junitxml=combined_pytest.xml
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_local_inputs.py --junitxml=local_inputs_pytest.xml
ruff check src/ tests/ scripts/ experiments/reproductions/
ruff format --check src/ tests/ scripts/ experiments/reproductions/
python scripts/check_notebooks.py --output notebook_validation.json
```

Ruff run beyond CI scope additionally reports three pre-existing findings in
`experiments/combined/FQ_SSIUU` and `experiments/infrastructure/gate0_env_check.py`.
Their output is preserved; those unrelated files were not changed. The exact
GitHub CI environment was not reproduced or remotely dispatched for this branch.

No full 1B weights were available in the supplied archive. No new 1B training,
GPU memory measurement, full-model inference or real PTQ was performed here.
Three-seed effectiveness, native Windows/bitsandbytes compatibility, runtime and
semantic annotation remain to be measured on the researcher's machine.

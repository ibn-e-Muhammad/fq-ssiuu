# 2026-11-03 — local NPO run-2 audit and LoRA third-run preparation

The researcher supplied the latest checkout, completed local run-2 outputs and
an extensive audit. Snapshot `c0bc9cd` preserves the first/second-run layout and
recorded result bytes on branch `codex/npo-lora-third-run`. The supplied model
folders contain metadata rather than weights; the actual Windows checkpoints
remain external. The old interruption at update 50 is preserved along with the
later update 62 completion and all nine 717-question stages.

The audit's geometric matched-oracle and directional-bootstrap requests are
implemented as separate post-hoc analysis. Its retain coverage concern is valid,
but the fixed pairing matches the pinned author behavior; fresh retained draws
are an explicit adaptation. The researcher chose LoRA for all forthcoming
project training. DL-024 documents adapter/effective-weight semantics and the
remaining constituent acceptance work. Run3 adds three seeds, reference via
adapter-disabled frozen base, periodic source-style evaluation, legacy/long
answer profiles, merge drift, merged-model PTQ and complete reports.

Engineering tests/derived receipts accompany the implementation. Actual 1B
LoRA outcomes are not fabricated or inferred from tiny fixtures. No PR/dev merge,
Kaggle submission, Protocol v2 activation, or confirmatory verdict belongs to
this preparation. See the run report, CR-013, audit disposition and local protocol.

## Follow-up: one self-contained notebook

The researcher requested that all third-run experiment code live inside one
Jupyter notebook, as in the second local run. `NPO_3RD_RUN.ipynb` now embeds all
settings, pins, objectives, tests, training/checkpointing, evaluation, merge/PTQ,
reports and optional second-run analysis. Logs run directly in the kernel,
with live training/stage views and saved text logs. The redundant third-run
Python modules/config registry were removed; commit `4d3089f` preserves them.

This is an execution/observation change, not a change to LoRA rank, objectives,
sampling, schedule or evaluation profiles. Saved core-cell hashes replace local
Python-file hashes for implementation identity. Notebook output/metadata saves
do not invalidate resume; scientific source edits do. Validation includes an
isolated notebook copy, its inline tests, real IPython history/display behavior,
all three answer-profile views and byte-identical five-table run-2 reanalysis.
Actual 1B GPU execution remains pending. No dev merge or PR is created.

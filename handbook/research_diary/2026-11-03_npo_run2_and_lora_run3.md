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

# Preserved local NPO second run

Run `20261031_092340_493700` completed full-weight NPO+RT and all nine recorded
BF16/INT8/NF4 model conditions. Source notebook/helpers and original result bytes
are preserved. `.gitattributes` disables line-ending conversion for this evidence.
Original interruption and later completion receipts intentionally coexist.

See the [run report](../../../handbook/reports/NPO_LOCAL_RUN2_REPORT.md),
[audit correction](../../../handbook/corrections/CR-013_npo_local_run2_audit.md),
and separate [derived reanalysis](../../../handbook/evidence/npo_run2/20261103_audit/reanalysis/).
Do not overwrite the original arithmetic matched-KS columns with the corrected
geometric analysis; compare the separately named outputs instead.

Model/data download caches and binary checkpoints are excluded from Git. The
supplied archive has model receipts/configuration but no original model tensor
files; reanalysis uses only the recorded per-question CSVs and metadata.

The next implementation lives beside this archive in
[`NPO_3rd-test`](../NPO_3rd-test/README.md). It uses LoRA as explicitly selected by
the researcher on 2026-11-03, with source-grounded metrics and recorded adaptations.

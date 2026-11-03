# CR-013: Local NPO run-2 audit and metric correction

Date: 2026-11-03. Status: **RAW EVIDENCE PRESERVED; DERIVED REANALYSIS SEPARATE**.
Run ID: `20261031_092340_493700`. Snapshot commit: `c0bc9cd`.

The supplied audit is preserved verbatim in
`../evidence/npo_run2/20261103_audit/USER_AUDIT.txt`. Its narrative is expert
commentary, not a substitute for the inspected code and per-question records.
The source/outputs inventory is `RAW_SOURCE_MANIFEST.json` in that directory.

## Confirmed observations

All nine stages have 717 questions: 200 Forget05, 300 retain diagnostics, 100 real
authors and 117 world facts. The completion receipt and 62-row training log agree
on completion. An earlier interruption at update 50 remains recorded separately;
its failure file does not contradict the later successful completion/resume.
The 27 recorded per-stage artifact hashes verify, and Git preserves their exact
bytes. Model tensors were not supplied in the archived run folders, so their
receipts are retained but checkpoint weight hashes cannot be independently
rehash-verified here. Never represent archive receipts as available model weights.

The objective is sequence-summed causal NPO+RT with a frozen full reference,
beta .1 and coefficients 1/1. The run uses full-weight BF16 with FP32 disk-backed
AdamW master/moment state, not LoRA. Actual bitsandbytes INT8/NF4 stages and
quantized full/retain95 controls are recorded; each quantized stage has 112
converted linear modules. This is evidence of conversion/evaluation, not a
complete quantizer-family sweep or universal durability result.

## Corrections and qualifications

1. **Matched-precision FQ:** old `matched_precision_KS_*` uses the arithmetic
   ratio. Primary TOFU-code FQ instead uses geometric preference against the
   corresponding precision of retain95. Recompute from existing CSVs without
   retraining/inference; retain old matrices unchanged.
2. **Directional recovery:** add paired CIs for geometric target-preference
   changes. Question bootstrap is exploratory because QAs share author profiles;
   also report the ten pinned contiguous author-block bootstrap sensitivity.
3. **Metric distinction:** TOFU code uses arithmetic perturbation probabilities
   for utility but geometric probabilities for FQ. The paper's written Eq. 16 is
   arithmetic. Preserve both definitions explicitly rather than relabeling old
   approved paper-arithmetic measurements. `utility_arithmetic` is the primary
   TOFU aggregate; source-geometric utility is sensitivity only.
4. **Retain pairing:** 195/3,800 unique retained rows is a real coverage limit.
   However, the pinned NPO author dataset reseeds by index and fixes those pairs;
   `NPO/training_data.py:SourcePairDataset` and prior source conformance record
   that behavior. The audit's claim that this is a deviation from that code is
   inaccurate. Fresh retain sampling is an explicit project change, not proof
   the old objective or implementation was illegitimate.
5. **Outcome interpretation:** FQ non-rejection is not equality or permanent
   deletion. A negative observed NF4 recovery result is useful evidence. Neither
   model size nor quantization-bin persistence is an established causal account;
   no controlled model-size/bin experiment was supplied.
6. **Generation:** cap-hit rates are measured output-length limitations. They
   are not automatic fabrication/gibberish rates. Human semantic labels remain
   missing until reviewed. Longer generation is a separate audit profile.

No original numeric file, generated answer or failure record is overwritten.
Derived corrected tables belong under `evidence/npo_run2/20261103_audit/reanalysis`.
No p-value ratio becomes a recovery score and no attack/durability verdict is
approved. See the [run report](../reports/NPO_LOCAL_RUN2_REPORT.md),
[audit disposition](../reviews/2026-11-03_npo_run2_audit.md), and
[LoRA decision](../decisions/DL-024_project_lora_training.md).

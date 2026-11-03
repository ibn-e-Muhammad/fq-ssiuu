# Local run 3: declared LoRA NPO+RT protocol

Date: 2026-11-03. Class: C, researcher-authorized project adaptation.
Decision: [DL-024](../decisions/DL-024_project_lora_training.md).
Implementation/config/operator instructions:
[`NPO_3rd-test`](../../experiments/reproductions/NPO_3rd-test/README.md).

The authoritative implementation is the self-contained
[`NPO_3RD_RUN.ipynb`](../../experiments/reproductions/NPO_3rd-test/NPO_3RD_RUN.ipynb).
It embeds configuration, all input pins, engineering tests, training/evaluation,
merge/PTQ and reports in readable cells. No local Python import, generated Python
file or child process is required. Run with the existing CUDA kernel after
editing the settings cell and saving. Scientific core-cell hashes bind resume;
output/metadata saves do not change those hashes. The modular predecessor remains
in Git at `4d3089f`. Consolidation does not change the protocol below.

Preserve the exact causal sequence-sum NPO Eq. 3 and the distinction between
NPO and NPO+RT. The current variant is NPO+RT, beta .1, coefficients 1/1,
positive retain CE, learning rate 1e-5, weight decay .01, effective batch 32,
62 source-rounded updates and 6-update warmup. Every logical retain group of
four uses its combined scored-token denominator. Device chunking must not
replace that denominator with per-example means.

Only FP32 LoRA A/B parameters train over the unquantized BF16 full backbone.
Original base parameters are frozen and content-hashed. Adapter-disabled,
eval/no_grad forwards provide the same initial full-model reference throughout
training. Retain95 remains a diagnostic comparator only. No QLoRA backbone
quantization is introduced into NPO training.

Three independent training seeds are 42/43/44. Freeze the seeded random retain
evaluation sample across all seeds and precision conditions. The fresh retained
training stream is an intentional second deviation, not a correction of the
pinned source's fixed-index pairing. It covers 1,984 rows under the fixed stop;
the source-fixed comparison covers 195. A causal comparison of LoRA alone
requires holding that sampling policy constant.

Evaluation at source interval updates 6,12,...,60 plus final62 records complete
717-question TOFU measurements and exact processed-data fraction. Primary
generation keeps source-style sampling, total length 200, batch 30 and seed 0 per
batch. Temperature .6/top-p .9 are declared Llama-3 checkpoint adaptations.
Generation RNG is isolated from training RNG. The final model additionally
gets 96-new-token greedy generation on all 717 prompts and a seeded 80-prompt
256-token audit, with EOS/cap/length fields. These profiles must not be pooled.

The unmerged adapter model is evaluated before FP32 merging. Merge-probe error
is measured before BF16 casting; full adapter-to-BF16 deployment drift is
reported separately. INT8/NF4 independently quantize the saved merged checkpoint.
Use geometric FQ against both fixed BF16 and matched retain95 quantization;
arithmetic utility and paper-arithmetic KS diagnostics remain separately named.
Report paired directional log-preference changes with both question-bootstrap
and ten-author-block intervals; do not infer attack/durability from p-values.

Store adapter/optimizer/RNG recovery state each update, preserve the latest two
complete generations, reject changed identities, and retry interrupted scheduled
evaluation. Save final adapters and merged weights with hashes. No best-score
checkpoint replaces the final deployment checkpoint. Report all declared seeds,
including failures; human semantic categories need actual labels. Min-k token
scores alone do not constitute membership inference. FADE, other quantizer
families, relearning and mechanism analysis remain explicit later work.

CPU/optional tiny-CUDA tests are engineering acceptance. Actual 1B GPU execution,
memory/runtime and three-seed effectiveness remain pending. Other FQ-SSIUU
components still need their own LoRA-aware attribution/effective-weight gradient
definitions. Historical source conformance and archived evidence remain intact.

# Local run-2 audit disposition and run-3 scope

Date: 2026-11-03. User audit retained verbatim in
`../evidence/npo_run2/20261103_audit/USER_AUDIT.txt`.

| Audit item | Disposition |
|---|---|
| NPO math/reference/masking/RT | Accepted; retain Eq. 3 sequence sum and positive token-mean RT |
| 195 fixed retained rows | Confirmed limited coverage; qualified because pinned author code uses the same pairing |
| Fresh retain sampling | Implemented as declared `reshuffled_stream`; `source_fixed` remains an ablation |
| Seeded retain evaluation subset | Implemented once per experiment; frozen across seeds/stages |
| Matched-precision FQ uses wrong ratio | Corrected in derived run-2 reanalysis and run-3 evaluator |
| Geometric directional preference CI | Implemented with question and author-block bootstrap |
| Preserve 96-token generation; add headroom | Implemented final-stage legacy greedy and separate 256-token audit; sampled source profile is primary |
| At least three training seeds | Run-3 config declares 42/43/44; actual GPU execution pending |
| Evaluate during training | Full source-style metrics at updates 6,12,...,60,62; exact processed fraction recorded |
| Exact quantization metadata | Resolved config, converted-module names/count, packed-state hashes and merged-source hashes recorded |
| LoRA decision | Researcher explicitly chose adapter training for future project conditions; DL-024 |
| Adapter merging | Additional necessary control: evaluate unmerged adapter, FP32 merge, BF16 reload and separate merge drift before quantization |
| Existing optimizer tests | Supplied eight-test log preserved; rerun CPU recovery tests separately where feasible; historical GPU test evidence is not new 1B validation |
| Min-k MIA | Token-level/min-k20 answer scores exported; proper matched-nonmember attack/AUROC pending |
| Manual generation categories | Template, rubric and coverage-aware summarizer supplied; semantic rates pending actual labels |
| FADE, other quantizers, relearning, bin mechanics | Deferred later study components; never filled with zeros or described as measured |
| Retain-loss coefficient tuning | Coefficient remains 1.0; any tuning needs a declared validation split, not final Forget05 test selection |
| OpenUnlearning headline comparison | Context only; different generation/training setups prevent direct numerical reproduction claims |
| No recovery explanation | Raw metrics support no clear reported gold/generation recovery; directional reanalysis is separate. No causal attribution to 1B size or weight-bin geometry |

This audit does not discard the second run. Its runtime records show why the
resource decision is reasonable: approximately 8.4 minutes of recorded
forward/backward compute versus over nine hours of recorded optimizer/checkpoint
work. Resume-history timing is incomplete, so logged sums are not a complete
wall-clock accounting. LoRA reduces the trained state; its actual speedup must be
measured rather than extrapolated from these totals.

The local third run is distinct from the historical third Kaggle/T4 D1 attempt
in September. The latter remains failed/archived; this researcher instruction
authorizes a new local LoRA experiment. It does not reopen the historical Kaggle
submission quota, activate Protocol v2 or approve a dev merge.

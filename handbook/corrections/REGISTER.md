# FQ-SSIUU Correction Register

**Purpose:** Tracks scientific corrections discovered during the research integrity audit.
Each correction has an ID, status, and evidence trail.

---

## CR-001: TOFU Forget Quality Interpretation

| Field | Value |
|-------|-------|
| Status | OPEN |
| Date discovered | 2026-09-03 |
| Affected files | `handbook/PROJECT_STATUS.md`, `handbook/GEMINI_HANDOFF.md`, `results/pilot_p4/README.md` |
| Severity | HIGH — affects interpretation of all pilot verdicts |

**Incorrect usage:** "near-zero FQ means extreme forgetting" and "larger FQ = better forgetting." treating FQ as a direct magnitude or distance, interpreting near-zero FQ as extreme forgetting, taking ratios of p-values as though they were stable effect sizes, and treating a single threshold as proof of an attack.

**Correct interpretation:** TOFU Forget Quality is the p-value returned by a two-sample Kolmogorov-Smirnov test over forget-set Truth Ratio distributions from the unlearned and retain-only models. A high p-value means the null hypothesis cannot be rejected under the test. TOFU conventionally interprets this as stronger forgetting. The p-value is not a distance, effect size, calibrated forgetting magnitude, or proof that the distributions are identical.

Where available, preserve and report both the KS statistic and the KS p-value.

---

## CR-002: Llama-3.2-1B Layer Count

| Field | Value |
|-------|-------|
| Status | CODE_FIXED |
| Date discovered | 2026-09-03 |
| Affected files | `configs/conditions/c4_ssiuu.yaml`, `src/fq_ssiuu/attribution/ssiuu_reg.py` |
| Severity | MEDIUM — affects documentation and interpretation, not functional correctness |

**Incorrect:** "Layer 14 of 28 for Llama-3.2-1B"

**Correct:** Llama-3.2-1B has 16 transformer layers (indices 0–15). Layer index 14 is the penultimate layer (near-final), not the middle layer. This is functionally valid (attributions at a near-final layer capture late-stage processing) but the interpretation as "middle layer" is wrong.

---

## CR-003: NPO Loss Bound Documentation

| Field | Value |
|-------|-------|
| Status | CODE_FIXED |
| Date discovered | 2026-09-03 |
| Affected files | `src/fq_ssiuu/unlearning/npo_loss.py` |
| Severity | MEDIUM — mathematical documentation error |

**Incorrect:** Module docstring line 29 states "Bounded range: 0 ≤ L_NPO ≤ 2/β for each example."

**Correct:** For `L_NPO = (2/β) · softplus(β·r)` where `r = log π_θ(y|x) − log π_ref(y|x)`:
- As r → −∞ (model assigns near-zero probability): `softplus(β·r) → 0`, so `L_NPO → 0`. The global infimum is 0.
- At r = 0 (identical models): `softplus(0) = log(2)`, so `L_NPO = (2/β)·log(2)`.
- As r → +∞ (model assigns higher probability than reference): `softplus(β·r) → β·r`, so `L_NPO → 2·r`, which is unbounded.

The upper bound claim (2/β) applies only in the practical regime where the model's probability on forget examples decreases toward zero (the intended unlearning direction). In the reverse direction (model probability increases), the loss grows without bound.

---

## CR-004: Pilot Verdict Overstatements

| Field | Value |
|-------|-------|
| Status | OPEN |
| Date discovered | 2026-09-03 |
| Affected files | `handbook/PROJECT_STATUS.md` |
| Severity | HIGH — affects scientific claims |

The following claims must be downgraded:

| Current Claim | Corrected Status | Reason |
|---------------|-----------------|--------|
| "H1 CONFIRMED: QRA manifests at Llama-3.2-1B scale" | Pilot-executed: precision sensitivity demonstrated | Single checkpoint, single seed, single model scale |
| "H2 SUPPORTED: captum IG feasible" | Pilot-executed: offline IG computation completed | P2 tested offline attribution on 2 examples, not training-time integration |
| "C8 ESTABLISHED" | Pilot-executed: partial QUAIL surrogate run completed | Implementation missing retain loss and uses wrong margin value (see CR-006) |
| "FQ-QAT+NPO training stable" | Pilot-executed: adapter-output INT4 QDQ training numerically stable | This is adapter-output symmetric INT4 QDQ, not merged-weight NF4 |

---

## CR-005: QUAIL Implementation Fidelity

| Field | Value |
|-------|-------|
| Status | OPEN — QUAIL C8 BLOCKED until resolved |
| Date discovered | 2026-09-03 |
| Affected files | `src/fq_ssiuu/unlearning/quail_loss.py`, `configs/conditions/c8_quail.yaml` |
| Severity | P0 — affects validity of C8 baseline comparison |

**Primary source:**
- Title: "QUAIL: Quantization Aware Unlearning for Mitigating Misinformation in LLMs"
- Authors: Himanshu Mishra, Kanwal Mehreen
- arXiv: 2601.15538v1
- Official repository: None found as of 2026-09-04

**Published objective (Section VI-B):**
- L_forget: gradient ascent on forget set (standard cross-entropy negation)
- L_retain: gradient descent on retain set to preserve utility
- L_hinge: `max(0, \Delta_q/2 - \|z'_k - z_k\|)` logit-space hinge loss
- Total: `L_QUAIL = \alpha L_{forget} + (1-\alpha) L_{retain} + \gamma L_{hinge}`
- Margin: `\Delta_q = 1.0` (surrogate for numerical stability in reported experiments)
- Internal ambiguities: Equation 15 defines `\alpha \in [0,1]`, but implementation reports searching values including 1, 5, 20. Algorithm 1 alternates updates, which may not be identical to a simultaneous combined-loss update.

**Implemented (current repository):**
- L_hinge only (no forget loss, no retain loss)
- Margin: nf4_bin_width/2 = 4.2e-4 (should be 1.0/2 = 0.5 per paper)
- No hyperparameter weighting

**Exact differences:**
1. Missing: forget loss and retain loss.
2. Wrong margin: uses 4.2e-4 instead of 0.5.
3. Missing: loss weighting coefficients.
4. Set implementation fidelity to SPECIFICATION_AMBIGUOUS until ambiguities are resolved. Keep C8 blocked.

---

## CR-006: SSIUU Gradient Flow

| Field | Value |
|-------|-------|
| Status | OPEN — P0 SCIENTIFIC BLOCKER |
| Date discovered | 2026-09-03 |
| Affected files | `src/fq_ssiuu/attribution/ssiuu_reg.py` |
| Severity | P0 — SSIUU regulariser cannot affect model weights |

`compute_negative_attribution()` returns `d_neg.detach()` (line 155). The `compute_loss()` method computes a penalty from two detached tensors (`d_neg_init` and `d_neg_current`). The resulting loss tensor has no gradient path to any model parameter. The SSIUU regulariser **cannot change model weights** in its current form.

Note: Even removing `.detach()` may not suffice. Captum's `LayerIntegratedGradients` may not support higher-order gradients required for the attribution loss to flow back through the model parameters. Full diagnosis is documented in the research gate registry.

---

## CR-007: Quantisation Location Mismatch

| Field | Value |
|-------|-------|
| Status | OPEN — P0 SCIENTIFIC BLOCKER |
| Date discovered | 2026-09-03 |
| Affected files | `src/fq_ssiuu/quantisation/fake_quant.py`, `src/fq_ssiuu/quantisation/fq_hooks.py` |
| Severity | P0 — training-time FQ is not deployment-equivalent |

`FQLoRALinear` applies symmetric INT4 QDQ (round to [-8,7] grid with per-tensor scale) to the **adapter output activation**. Deployment NF4 via bitsandbytes applies the NF4 codebook to **model weights** (block size, double quantisation, compute dtype configurable per `BitsAndBytesConfig`). These are fundamentally different quantisation operations.

The bitsandbytes configuration used in this project (`configs/base.yaml`):
- `bnb_4bit_quant_type: nf4`
- `bnb_4bit_compute_dtype: bfloat16`
- `bnb_4bit_use_double_quant: true`
- Block size: default (must be verified from installed bitsandbytes version; not installed locally)
- Quantisation target: merged model weights after `merge_and_unload()`

The current FQ-QAT implementation is an ablation demonstrating adapter-output INT4 QDQ during training. It is not deployment-equivalent NF4 QAT. This distinction must be stated in all results that reference FQ-QAT.

---

## CR-008: Certificate Terminology

| Field | Value |
|-------|-------|
| Status | CODE_FIXED |
| Date discovered | 2026-09-03 |
| Affected files | `src/fq_ssiuu/evaluation/certificate.py`, `handbook/GEMINI_HANDOFF.md` |
| Severity | MEDIUM — language implies formal guarantee |

"Certificate", "certifies", "CERTIFIED" imply formal mathematical or legal guarantees. Replace display text with "Empirical Durability Report" and `certified` field with `threshold_met`. Deprecated aliases preserved for reading historical pilot artifacts.

---

## CR-009: Kaggle Environment and Dependency Readiness Reconciliation

| Field | Value |
|-------|-------|
| Status | DOCUMENTED — IMPLEMENTATION PENDING |
| Date discovered | 2026-09-05 |
| Affected files | `handbook/PROJECT_STATUS.md`, `handbook/THESIS_UPDATE_LEDGER.md`, etc. |
| Severity | HIGH — blocks reproducible new experiments but does not invalidate historical pilot observations |

See `handbook/corrections/CR-009_kaggle_environment_reconciliation.md` for full details on corrected claims.

---

## CR-010: Model Loader Dtype Mismatch and Configuration Discrepancy

| Field | Value |
|-------|-------|
| Status | OPEN — DEFERRED RUNTIME-CODE BLOCKER |
| Date discovered | 2026-09-05 |
| Affected files | `src/fq_ssiuu/models/loader.py`, `configs/base.yaml` |
| Severity | HIGH — configuration mismatch blocks model training until resolved |

**Findings:**
1. `configs/base.yaml` specifies FP16 (`torch_dtype: float16`), matching the canonical Kaggle T4 environment.
2. `ModelConfig` in `src/fq_ssiuu/models/loader.py` still defaults to `bfloat16` (`torch_dtype: str = "bfloat16"`).
3. Docstring in `loader.py` still references deprecated `scripts/setup_env.sh`.
4. `device_map="auto"` is acceptable only when `CUDA_VISIBLE_DEVICES=0` has hidden the second GPU.

**Directive:**
This mismatch is registered as an explicit blocker for new experiments. `src/fq_ssiuu/models/loader.py` must not be modified in Stage 1H-C. It must be repaired and verified with dedicated unit tests before model training or evaluation is initiated.

# Method-Implementation Matrix

**Purpose:** Compare published methods against their actual implementation in this repository.
**Last updated:** 2026-09-07

The [upstream implementation register](methods/UPSTREAM_IMPLEMENTATIONS.md)
supersedes earlier blanket NPO fidelity and QUAIL code-availability claims.

---

## NPO (Negative Preference Optimisation)

| Aspect | Published (Zhang et al. 2024, arXiv:2404.05868) | Implemented |
|--------|------------------------------------------------|-------------|
| Loss function | L_NPO = (2/β) · softplus(β · (log πθ − log πref)) | ✅ Matches |
| Retain KL | Optional | ✅ Implemented (compute_retain_kl_loss) |
| Temperature β | 0.1 | ✅ Matches (configurable) |
| Reference model | Frozen pre-unlearning model | ✅ Matches |
| Fidelity | RECONCILIATION REQUIRED | Current helper averages token losses; upstream uses a causally shifted sequence sum. No new NPO pilot acceptance is claimed. |

---

## SSIUU Attribution Regularisation

| Aspect | Published SSIUU (Yang et al. 2025, arXiv:2509.22263) | Implemented (LayerIG attribution surrogate) |
|--------|-----------------------------------------------------|--------------------------------------------|
| Attribution method | A_i = \\phi_i \\times \\partial P(y\|x)/\\partial \\phi_i (parameter-level features) | ❌ Captum LayerIntegratedGradients at target hidden layer |
| Batching | Computed for the current forget batch | ✅ Matches |
| Baseline | Immediately previous optimization step | ❌ Static initial-model baseline |
| Negative set | Selected from negative current-step aggregated attributions | ❌ Selected from static initial baseline |
| Differentiability | Current-step attribution remains differentiable | ❌ Detached current-step attribution (CR-006) |
| Previous step | Detached and treated as constant | N/A (uses static baseline) |
| Backpropagation | Requires second-order derivatives (Hessian-vector product) | ❌ No gradient flow |
| First step | Zero penalty | ❌ Uses full penalty against static baseline |
| Fidelity | FAITHFUL (defined by paper) | **METHOD-INVALID** — Not published SSIUU (CR-006) |

---

## Fake-Quantisation QAT (FQ-QAT)

| Aspect | FQ-SSIUU Design Intent | Implemented |
|--------|----------------------|-------------|
| Quantisation target | Merged model weights (deployment-equivalent) | ❌ Adapter output activations (CR-007) |
| Codebook | NF4 (non-uniform, data-dependent) | ❌ Symmetric INT4 (uniform grid [-8,7]) |
| Granularity | Per-block (matching bitsandbytes) | ❌ Per-tensor |
| STE backward | Identity pass-through | ✅ Matches |
| Block size | Configurable (bitsandbytes default) | N/A (per-tensor) |
| Double quantisation | Matches deployment config | N/A |
| Fidelity | **ABLATION ONLY** — not deployment-equivalent | — |

---

## QUAIL Baseline (Condition C8)

| Aspect | Published QUAIL (Mishra & Mehreen 2026, arXiv:2601.15538) | Implemented (Incomplete surrogate) |
|--------|-----------------------------------------------------------|------------------------------------|
| Forget loss | Gradient ascent (cross-entropy negation) | ❌ Missing |
| Retain loss | Gradient descent on retain set | ❌ Missing |
| Hinge loss | max(0, Δq/2 − \|z'_k − z_k\|) per token (logit-space) | ✅ Formula matches |
| Total objective | L_QUAIL = α·L_forget + (1−α)·L_retain + γ·L_hinge | ❌ Hinge only |
| Margin Δq | 1.0 (surrogate for numerical stability in reported exp) | ❌ Uses NF4 bin width 8.4e-4 |
| Coefficients | Internal ambiguities in paper regarding independent α, β, γ | ❌ No weighting |
| Optimization | Alternates updates in Algorithm 1, ambiguity with Eq 15 | ❌ None |
| Official code | Appendix E links `himans-iitk/CS534L_Project` | Pinned source recorded in upstream register; earlier absence claim superseded |
| Fidelity | FAITHFUL (defined by paper) | **SPECIFICATION_AMBIGUOUS** (CR-005) |

---

## TOFU Evaluation

| Aspect | Published TOFU (Maini et al., COLM 2024, arXiv:2401.06121) | Implemented |
|--------|------------------------------------------------------------|-------------|
| Forget Quality metric | KS-test p-value: forget-set Truth Ratio distributions (unlearned vs retain-only) | ✅ Via OpenUnlearning harness |
| FQ interpretation | High p-value = null hypothesis cannot be rejected. Not a distance or magnitude. | ⚠️ Misinterpreted in some docs (CR-001) |
| Model Utility | Aggregate retain-set performance | ✅ Via OpenUnlearning harness |
| Statistics reported | KS statistic and KS p-value | ⚠️ Only p-value currently tracked |
| Fidelity | FAITHFUL (via external harness) | — |

---

## Empirical Durability Report (formerly "Certificate")

| Aspect | Design Intent | Implemented |
|--------|--------------|-------------|
| Metric | relative_precision_sensitivity = |FQ_NF4 − FQ_BF16| / FQ_BF16 | ✅ Computed correctly |
| Threshold | 0.05 | ✅ Configurable |
| Three-precision pipeline | BF16 → INT8 → NF4 evaluation | ✅ Structure complete |
| Terminology | "Empirical Durability Report" | ✅ Updated (v2 schema, CR-008) |
| Fidelity | IMPLEMENTED — display terminology corrected | — |

# Research Gate Registry

**Purpose:** Machine-parseable registry of scientific blockers that must be
resolved before experiments can produce valid results.

Each gate has a unique ID, status, and blocking conditions.

---

## RG-001: SSIUU Gradient Flow

| Field | Value |
|-------|-------|
| Status | **BLOCKED** |
| Correction | CR-006 |
| Blocking | Conditions C4 (SSIUU), C6 (FQ-SSIUU), C7 (full method) |
| Description | `.detach()` at ssiuu_reg.py:173 severs gradient flow. SSIUU regularisation loss cannot affect model parameters. |
| Resolution | Verify captum supports higher-order gradients for the architecture; if so, remove `.detach()` and add `create_graph=True`. If not, redesign attribution penalty. |
| Verification | `test_attribution.py` gradient flow test must pass: `loss.backward()` must produce non-zero `param.grad` for at least one model parameter when SSIUU loss > 0. |

---

## RG-002: Quantisation Location Mismatch

| Field | Value |
|-------|-------|
| Status | **BLOCKED** |
| Correction | CR-007 |
| Blocking | Conditions C5 (FQ-QAT), C6 (FQ-SSIUU), C7 (full method) |
| Description | Training-time FQ applies symmetric INT4 QDQ to adapter output activations. Deployment NF4 applies the NF4 codebook to merged model weights. These are different operations. |
| Resolution | Options: (A) Implement weight-space NF4 QDQ that applies STE to merged weights, or (B) document current implementation as "adapter-output INT4 ablation" and add a deployment-aligned condition. |
| Verification | Quantiser metadata `deployment_equivalent` field must be `True` for the chosen resolution. |

---

## RG-003: QUAIL Fidelity

| Field | Value |
|-------|-------|
| Status | **BLOCKED** |
| Correction | CR-005 |
| Blocking | Condition C8 (QUAIL baseline) |
| Description | Current implementation has hinge loss only (no forget loss, no retain loss), uses margin 4.2e-4 (published: 0.5), and has no loss weighting coefficients. |
| Resolution | Implement full three-component objective: L = α·L_forget + β·L_retain + γ·L_hinge with Δq=1.0 margin. |
| Verification | QUAIL loss function must match published equation structure. Unit test must verify all three components contribute to total loss. |

---

## RG-004: TOFU Forget Quality Interpretation

| Field | Value |
|-------|-------|
| Status | **DOCUMENTATION ONLY** |
| Correction | CR-001 |
| Blocking | None (does not block code execution) |
| Description | FQ is a KS p-value comparing unlearned model vs retain-only oracle. Larger p-value = less evidence against null (oracle agreement). Near-zero FQ = statistically distinguishable from oracle = poor forgetting under TOFU protocol. |
| Resolution | Audit and correct all documentation that interprets FQ as a direct distance metric or magnitude. |
| Verification | Grep for "forget quality" and "FQ" in all documentation; verify interpretation is consistent. |

---

## Gate Summary

| Gate | Status | Conditions Blocked | Resolution Complexity |
|------|--------|-------------------|----------------------|
| RG-001 | BLOCKED | C4, C6, C7 | HIGH — requires captum higher-order gradient investigation |
| RG-002 | BLOCKED | C5, C6, C7 | HIGH — requires new quantiser implementation or scope change |
| RG-003 | BLOCKED | C8 | MEDIUM — implement missing loss components |
| RG-004 | DOCS ONLY | None | LOW — documentation corrections |

**Experiment readiness:** Conditions C1 (retain baseline), C2 (NPO), C3 (GA) are
unblocked. All other conditions require at least one gate resolution.

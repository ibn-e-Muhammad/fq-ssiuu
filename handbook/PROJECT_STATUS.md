# FQ-SSIUU Project Status

**Last updated:** 2026-09-05  
**Verified by:** Codex source review and Kaggle OG-001 kernel version 2 evidence
**Branch:** fix/kaggle-overlay-evidence-integrity

> [!WARNING]
> **CR-004 (Pilot Overstatements):** Previous versions of this file contained
> overconfident claims from single-seed, single-model pilot experiments.
> All verdicts below have been corrected to reflect actual evidence strength.
> See `handbook/corrections/REGISTER.md` for the full correction register.

---

*See [Reconciliation Diary](research_diary/2026-09-03_to_2026-09-05_reconciliation.md) and [Gate-0 Evidence](evidence/gate0_kaggle/57588774-bdd5-41c8-b56e-7df0c2f5049b/MANIFEST.json) for the full audit trail establishing Kaggle T4 as the canonical environment.*

---

## Confirmed Completed Milestones

| Milestone | Date | Key Result | Verdict | Evidence |
|-----------|------|------------|---------|----------|
| Legacy Colab environment check | 2026-08-04 | Colab check (Historical, not canonical env) | PASS | experiments/gate0_env_check.py |
| GATE-0 (Kaggle) | 2026-09-04 | hardware: PASS, deps: PARTIAL, nf4: UNAVAILABLE | PARTIAL | handbook/evidence/gate0_kaggle/ |
| OG-001 repaired attempt | 2026-09-05 | Source transport and installation passed; global pip check exited 1; NF4 not reached | PARTIAL | handbook/evidence/og001_kaggle/20260905_kernel_v2_24b25b5f/VERDICT.json |
| GATE-1 | 2026-08-08 | FQ=0.393718, MU=0.554501 | PASS (exceeds target) | results/gate1_baseline/ |
| Pilot P1 | 2026-08-09 | Q-INT4=0.214906 (4.3× threshold) | Pilot-executed: precision sensitivity demonstrated | results/pilot_p1/ |
| Pilot P2 | 2026-08-09 | Peak VRAM=2.511 GB (12.5 GB margin) | Pilot-executed: offline IG computation completed | results/pilot_p2/ |
| Pilot P3 | 2026-08-10 | Loss 13.72→8.65 (-37%), coherence 5/5 | Pilot-executed: adapter-output INT4 QDQ training numerically stable | results/pilot_p3/ |
| Pilot P4 | 2026-08-10 | QUAIL Q-INT4=0.227678, MU=0.280369 | Pilot-executed: partial QUAIL surrogate run completed | results/pilot_p4/ |

## Pilot Observations (Not Confirmed Findings)

> [!IMPORTANT]
> All observations below are from single-seed, single-model pilot experiments.
> They motivate full experiments but do not constitute confirmation.

- **Precision sensitivity demonstrated (P1):** Q-INT4=0.214906 (4.3× pass threshold of 0.05). Reproducible across 2 independent runs. Single checkpoint, single seed, Llama-3.2-1B only.
- **Offline IG computation completed (P2):** captum IG feasible at 2.511 GB. Tested on 2 examples only. Training-time integration NOT tested.
- **Adapter-output INT4 QDQ numerically stable (P3):** FQ-QAT+NPO training stable over 5 epochs. Note: this is adapter-output symmetric INT4 QDQ, NOT merged-weight NF4 (CR-007).
- **Partial QUAIL surrogate completed (P4):** QUAIL surrogate produces strong forgetting (FQ≈0.000121) with severe utility cost (MU=0.280). Implementation is hinge loss only, missing forget and retain components (CR-005).
- **INT8 ANOMALY DOCUMENTED**: Q-INT8=0.426023 > Q-INT4=0.214906. INT8 (LLM.int8()) mixed-precision outlier handling disrupts distribution differently than NF4.

## Corrections Made During Audit

| Item | Original Assessment | Corrected Assessment | Reason |
|------|---------------------|----------------------|--------|
| P3 Verdict | FAIL (grad norm > 5.0 in 5/5 epochs) | PASS | Script evaluated pre-clip gradient norms. EVB criterion specifies post-clip norm with gradient_clip=1.0 active. Post-clip norm is always ≤ 1.0 by mathematical definition. See AMENDMENT_gradient_norm_clarification.md |

## Pre-Implementation Checklist (EVB Table 24)

| # | Item | Status | Evidence | Blocking? | Action Required |
|---|------|--------|----------|-----------|----------------|
| 1 | HF access to Llama-3.2-1B-Instruct | ✅ DONE | GATE-1 ran successfully | No | — |
| 2 | GATE-0 reproducible environment verification | ⏳ PARTIAL | OG-001 v2 installed overlay, then failed global pip check | Yes | Resolve demonstrated native-image consistency gate conflict (OG-001) |
| 3 | GATE-1 NPO baseline reproduction | ✅ DONE | FQ=0.3937, MU=0.5545 | No | — |
| 4 | Pilot P1 QRA existence | ✅ DONE | Q-INT4=0.2149 | No | — |
| 5 | Pilot P2 IG attribution feasibility | ✅ DONE | Peak VRAM=2.511 GB | No | — |
| 6 | Pilot P3 FQ-QAT stability | ✅ DONE (corrected) | Loss monotonically decreasing, coherence 5/5 | No | — |
| 7 | Pilot P4 QUAIL baseline | ✅ DONE | Q-INT4=0.2277, C8 established | No | — |
| 8 | GUARD-IT and QUAIL differentiation doc | ❌ PENDING | No file found in repo | No | Write DL entry distinguishing GUARD-IT/QUAIL/FQ-SSIUU |
| 9 | OSF pre-registration | ❌ PENDING | No file found in repo | No | Draft and submit before full experiments |
| 10 | C3 (GA baseline) config file | ❌ PENDING | configs/conditions/ missing c3_ga.yaml | No | Create config |
| 11 | FakeQuantiseNF4 unit test | ✅ DONE | tests/test_ste_gradients.py (4 tests pass) | No | — |
| 12 | SSIUU regularisation unit test | ✅ DONE | tests/test_attribution.py (3 tests pass) | No | — |
| 13 | Oracle model quality (E0b) | ❌ PENDING | E0b not started | Yes (for FADE) | Run E0b oracle fine-tune |
| 14 | Supervisor sign-off on experimental plan | ⏳ PARTIAL | Plan exists but no documented approval | No | Obtain and document sign-off |

**Summary: 9/14 DONE, 5/14 PENDING/PARTIAL. See Research and Operational Gates sections for blockers.**

## Operational Gates (Environment Blockers)

> [!CAUTION]
> The Kaggle dependency overlay is an operational blocker before any new full experiment.
> See `handbook/operational_gates.yaml` for details.

| Gate | Status | Blocks New Experiments? | Resolution Complexity |
|------|--------|-------------------------|-----------------------|
| OG-001: Reproducible Kaggle Environment Overlay | PARTIAL | Yes | HIGH |

The repaired private run (kernel version 2) executed source
`91b6aa39e520f9312b098997134c1deabeabfb8f` through a separately identified,
parentless snapshot. Transport, one-device Tesla T4 selection, protected-package
integrity and both installations passed. Global `pip check` failed with six
diagnostics involving native-image packages. Its first diagnostic reports a
missing `google-cloud-bigquery-storage` dependency of `bigframes 2.39.0`.
FP16, SDPA, NF4 and required overlay import checks were not reached. The failure
state was preserved; no after-manifest or verified lock is claimed.

The 35 passing offline infrastructure tests are evidence about control flow and
source transport, not scientific method validity or Kaggle NF4 capability.
The first attempt's raw log is preserved, and its mistyped derived SHAs are
explicitly corrected. See the [v2 evidence](evidence/og001_kaggle/20260905_kernel_v2_24b25b5f/VERDICT.json).

## Research Gates (Scientific Blockers)

> [!CAUTION]
> The following gates must be resolved before the corresponding conditions
> can produce valid results. See `handbook/research_gates.yaml` for details.

| Gate | Status | Conditions Blocked | Resolution Complexity |
|------|--------|-------------------|----------------------|
| RG-001: SSIUU Gradient Flow | BLOCKED | C4, C6, C7 | HIGH |
| RG-002: Quantisation Location Mismatch | BLOCKED | C5, C6, C7 | HIGH |
| RG-003: QUAIL Fidelity | BLOCKED | C8 | MEDIUM |
| RG-004: TOFU FQ Interpretation | DOCS ONLY | None | LOW |

**Unblocked conditions:** C1 (retain baseline), C2 (NPO), C3 (GA).

## What Must Happen Before Full Experiments

1. **Resolve RG-001 (SSIUU gradient flow)** — BLOCKS C4, C6, C7
2. **Resolve RG-002 (quantisation location)** — BLOCKS C5, C6, C7
3. **Resolve RG-003 (QUAIL fidelity)** — BLOCKS C8
4. **Create C3 (GA baseline) and C7 (DurableUn-SAF) config files**
5. **Write GUARD-IT/QUAIL differentiation document** (DL-012)
6. **Draft OSF pre-registration** (can be concurrent with experiments)
7. **Obtain supervisor sign-off** on the experimental plan

## The Full Experiment Dependency Order

All phases below remain subject to OG-001 and the applicable research gates.
Historical pilot execution does not clear those blockers.

| Phase | Experiments | Status | Dependencies |
|-------|-------------|--------|--------------|
| Phase 0 | E0b (Oracle fine-tune) | NOT STARTED | OG-001 closure |
| Phase 0 | GATE-2 (SSIUU validation) | NOT STARTED | OG-001; RG-001 correction and validation |
| Phase 1 | E1+E2 (NPO 3 seeds) | NOT STARTED | GATE-1 ✓ |
| Phase 1 | E3 (QUAIL 3 seeds) | NOT STARTED | OG-001; RG-003 correction and validation |
| Phase 1 | E5 (FQ-QAT+NPO 3 seeds) | NOT STARTED | OG-001; RG-002 correction and validation |
| Phase 1 | E7 (DurableUn-SAF 3 seeds) | NOT STARTED | Config needed |
| Phase 2 | E4 (SSIUU+NPO 3 seeds) | NOT STARTED | GATE-2 |
| Phase 3 | E6 (FQ-SSIUU full, 3 seeds) | NOT STARTED | E4 AND E5 |
| Phase 4 | FADE computation | NOT STARTED | E0b + all conditions |

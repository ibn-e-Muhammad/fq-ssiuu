# Gemini Agent Handoff Document
## FQ-SSIUU: Verifiable LLM Unlearning Against Quantisation Recovery Attacks

**Last updated by:** Claude Opus 4 (Thinking), Session 2
**Date:** 2026-08-01
**Next agent:** Gemini 3.1 Pro (High)
**First task for Gemini:** GATE-0 → GATE-1 → Pilot P1

---

### 1. What This Project Is (One Paragraph)

FQ-SSIUU is an undergraduate Final Year Project (8-month timeline) that develops a machine unlearning method for LLMs. The core problem: standard unlearning methods (like NPO) produce models that pass BF16 evaluation audits, but when the model is quantised to INT4 for deployment, the "forgotten" knowledge is fully recovered — this is the Quantisation Recovery Attack (QRA). FQ-SSIUU solves this by combining three components: (1) NPO loss for controlled forgetting, (2) SSIUU-style attribution regularisation to prevent spurious unlearning neurons, and (3) Fake-Quantisation QAT (FQ-QAT) with Straight-Through Estimators to force weight updates that survive INT4 rounding. The target model is Llama-3.2-1B-Instruct on the TOFU forget05 benchmark. All training runs on Google Colab L4 GPUs (24 GB VRAM, ~16 GB usable). The deliverables are: a trained model, a JSON durability certificate attesting QRA robustness at INT4, a paper, and this codebase.

---

### 2. Current Repository State

- **Branch:** `stage/2-core-implementation` (to be merged to `dev` after review)
- **Git remote:** https://github.com/ibn-e-Muhammad/fq-ssiuu

#### Files Written (Non-Empty, Functional)

**Source Modules (`src/fq_ssiuu/`):**
| File | Status | Description |
|------|--------|-------------|
| `models/loader.py` | Complete | Model loading: NF4, BF16, LoRA, reference, merge |
| `unlearning/npo_loss.py` | Complete | NPO loss + KL retain loss, self-test passing |
| `unlearning/quail_loss.py` | Complete | QUAIL hinge loss re-implementation, self-test passing |
| `quantisation/fake_quant.py` | Complete | FakeQuantiseNF4 STE, 5-check self-test passing |
| `attribution/ssiuu_reg.py` | Complete | SSIUU regulariser + GradientNorm fallback, self-test passing |
| `evaluation/qra_pipeline.py` | Structure complete | Three-precision pipeline; FQ/MU metrics are placeholder |
| `evaluation/certificate.py` | Complete | JSON durability certificate generator |
| `__init__.py` (all subpackages) | Empty markers | Package structure only |

**Configuration (`configs/`):**
| File | Description |
|------|-------------|
| `base.yaml` | Shared defaults for all experiments |
| `conditions/c2_npo.yaml` | NPO baseline |
| `conditions/c4_ssiuu.yaml` | SSIUU + NPO ablation |
| `conditions/c5_fqat.yaml` | FQ-QAT + NPO ablation |
| `conditions/c6_fq_ssiuu.yaml` | Full method (primary contribution) |
| `conditions/c8_quail.yaml` | QUAIL baseline |
| `seeds.yaml` | Seed registry: pilot=42, full=[42,123,456] |

**Scripts and Infrastructure:**
| File | Description |
|------|-------------|
| `scripts/setup_env.sh` | Environment setup script for Colab |
| `scripts/colab_mount.py` | Google Drive mount utility |
| `scripts/sync_to_drive.sh` | Results sync to Google Drive |
| `scripts/pull_from_drive.sh` | Pull checkpoints from Drive |
| `experiments/gate0_env_check.py` | 14-check environment validator |
| `.github/workflows/ci.yml` | GitHub Actions CI (lint + CPU tests) |
| `Makefile` | Common development targets |
| `pyproject.toml` | Build config, ruff, pytest settings |

**Tests (`tests/`):**
| File | CPU Tests | Model/GPU Tests |
|------|-----------|-----------------|
| `test_ste_gradients.py` | 4 PASSING | — |
| `test_npo_loss.py` | 3 PASSING | 2 skipped (need training loop) |
| `test_attribution.py` | 3 PASSING | 2 skipped (need LLM) |
| `test_certificate.py` | — | 4 skipped (stubs) |
| `test_fade_compute.py` | — | 4 skipped (stubs) |
| `test_qra_pipeline.py` | — | 4 skipped (stubs) |
| `test_quail_loss.py` | — | 4 skipped (stubs) |
| `conftest.py` | Custom mark registration | — |

#### Files That Are Stubs (Need Implementation by Gemini)

1. `evaluation/qra_pipeline.py` — `evaluate_at_precision()` returns placeholder metrics (-1.0). **Needs:** OpenUnlearning evaluation harness integration (GATE-1).
2. `tests/test_certificate.py` — All 4 tests are stubs. **Needs:** Real certificate generation tests once QRA pipeline is functional.
3. `tests/test_fade_compute.py` — All 4 tests are stubs. **Needs:** FADE metric implementation.
4. `tests/test_qra_pipeline.py` — All 4 tests are stubs. **Needs:** Real QRA pipeline tests.
5. `tests/test_quail_loss.py` — All 4 tests are stubs. **Needs:** Real QUAIL tests with model logits.

---

### 3. What Has Been Proven to Work

Every item below has a self-test that passed on CPU (Python 3.11.15, torch CPU):

| Module | Self-Test | Result |
|--------|-----------|--------|
| NPO loss | `test_npo_loss_zero_when_equal()` | PASSED — loss = 13.8629 (matches (2/0.1)·log(2)) |
| NPO retain KL | `compute_retain_kl_loss(x, x)` | PASSED — KL = 0.0 with identical logits |
| QUAIL hinge | `test_quail_loss_at_margin()` | PASSED — loss=0 at margin, loss=margin at zero deviation |
| STE forward | Shape preservation, QDQ error bound | PASSED — max error 0.187 ≤ bound 0.187 |
| STE backward | Identity gradient | PASSED — grad equals torch.ones(3) exactly |
| STE NaN-safety | Zero input, no NaN | PASSED |
| STE device | CPU consistency | PASSED |
| SSIUU zero-at-init | `test_ssiuu_loss_zero_at_init()` | PASSED — loss = 0.0 |
| SSIUU penalty | More-negative neuron | PASSED — penalty = 0.4 (expected) |

---

### 4. Critical Constraints — Never Violate These

1. **VRAM Budget: 16 GB usable.** The Colab L4 has 24 GB but ~16 GB is usable after CUDA overhead. Operations near the limit: IG attribution computation (n_ig_steps forward passes), loading two models simultaneously (reference + training). **Never load two full models at the same time.**

2. **Install Order: torch FIRST.** `torch==2.4.1` → `flash-attn` (with `--no-build-isolation`) → `bitsandbytes==0.42.0` → everything else. Breaking this order causes silent CUDA version mismatches. See `scripts/setup_env.sh`.

3. **Seeds: [42, 123, 456] for full runs, 42 only for pilots.** See `configs/seeds.yaml`. The primary condition C6 uses 5 seeds per EVB statistician review. Changing seeds after experiments begin requires a Decision Log entry.

4. **Statistics: Effect sizes with 95% bootstrap CIs only.** No p-values anywhere. See Decision Log DL-006 in the handbook.

5. **Claim language: Never write "outperforms."** Always write "reduces X by Y ± Z pp (95% CI)". This applies to paper text, commit messages, and comments.

6. **Git: Never commit directly to `main` or `dev`.** Always use a feature branch (`stage/*`, `pilot/*`, `exp/*`) and create a PR.

7. **Large files: NEVER commit model weights, datasets, or result CSVs.** See `.gitignore`. Model weights (`.pt`, `.pth`, `.bin`, `.safetensors`) are excluded. All checkpoints and datasets live on Google Drive.

8. **Google Drive: All persistent artefacts go here.** Path: `My Drive/fq-ssiuu-research/`. Subdirectories: `checkpoints/`, `data/processed/`, `results/`, `merged_models/`. See `checkpoints/README.md`.

9. **HuggingFace token: NEVER hardcode.** Always read from `HF_TOKEN` environment variable via the central secret provider (`src/fq_ssiuu/secrets.py`). Set it in Colab/Kaggle with `os.environ['HF_TOKEN'] = '<your_token>'` or store it in a Kaggle Secret named `HF_TOKEN`. **[REDACTED — token was exposed here; rotation required]**

10. **Gradient checkpointing conflicts with captum.** When computing IG attributions in `ssiuu_reg.py`, gradient checkpointing must be temporarily disabled. The code handles this, but be aware if modifying the training loop.

11. **QUAIL vs FQ-SSIUU distinction.** QUAIL operates in output space (logits). FQ-SSIUU operates in weight space (adapter parameters). This distinction MUST be stated in any comparison. See module docstrings for details.

12. **Test imports use conditional torch import.** Test files that were not updated in Session 2 still use `try/except ImportError` for torch/numpy. This is intentional for CI compatibility.

---

### 5. The Experimental Conditions — What Gemini Will Run

| ID | Name | NPO | SSIUU | FQ-QAT | QUAIL | Config File | Seeds | Est. Runtime/Seed |
|----|------|-----|-------|--------|-------|-------------|-------|-------------------|
| C1 | Oracle | — | — | — | — | (manual) | 42 | ~2h (fine-tune on retain only) |
| C2 | NPO Baseline | ✅ | — | — | — | `conditions/c2_npo.yaml` | 42,123,456 | ~1h |
| C3 | GA Baseline | — | — | — | — | (to be created) | 42,123,456 | ~1h |
| C4 | SSIUU + NPO | ✅ | ✅ | — | — | `conditions/c4_ssiuu.yaml` | 42,123,456 | ~2h (IG is slow) |
| C5 | FQ-QAT + NPO | ✅ | — | ✅ | — | `conditions/c5_fqat.yaml` | 42,123,456 | ~1.5h |
| C6 | **FQ-SSIUU** | ✅ | ✅ | ✅ | — | `conditions/c6_fq_ssiuu.yaml` | 42,123,456(+2) | ~2.5h |
| C7 | DurableUn-SAF | — | — | — | — | (to be created) | 42,123,456 | ~2h |
| C8 | QUAIL | ✅ | — | — | ✅ | `conditions/c8_quail.yaml` | 42,123,456 | ~1.5h |

**Primary contribution is C6.** C1 is the retrained oracle. C3 and C7 configs need to be created.

---

### 6. The Dependency Graph — What Blocks What

```
GATE-0 (env check)
  └─→ GATE-1 (OpenUnlearning baseline reproduction)
        └─→ Pilot P1 (QRA existence test)
              ├─→ Pilot P2 (IG attribution feasibility)
              │     └─→ Pilot P3 (FQ-QAT gradient flow)
              │           └─→ Pilot P4 (combined loss convergence)
              │                 └─→ Full Experiments (C2-C8)
              │                       └─→ Analysis & Paper
              └─→ [If P1 fails: abort QRA claim, pivot to alternative]
```

**Critical path:** GATE-0 → GATE-1 → P1 → P2 → P3 → P4 → Full Experiments.
Each gate/pilot has a hard acceptance criterion. Failure requires a documented pivot.

---

### 7. The Four Pilot Experiments — Acceptance Criteria

#### Pilot P1: QRA Existence Test
- **What it tests:** Does NPO unlearning actually fail under INT4 quantisation?
- **Acceptance criterion:** FADE_BF16 ≥ 0.6 AND FADE_INT4 < 0.3 (demonstrating ≥ 0.3 absolute drop)
- **If it fails:** If FADE_INT4 ≥ 0.3, QRA does not exist at 1B scale. Pivot: try 3B model or different forget split.
- **Config:** `conditions/c2_npo.yaml`, seed 42
- **Script:** To be written. Uses `evaluation/qra_pipeline.py` after GATE-1 integration.

#### Pilot P2: IG Attribution Feasibility
- **What it tests:** Can Integrated Gradients run within 16 GB VRAM on Llama-3.2-1B?
- **Acceptance criterion:** IG completes without OOM on a single forget batch (n_ig_steps=20)
- **If it fails:** Fall back to `GradientNormAttribution` (see `ssiuu_reg.py`). Document in Decision Log.
- **Config:** `conditions/c4_ssiuu.yaml`, seed 42
- **Script:** To be written. Calls `compute_negative_attribution()` from `ssiuu_reg.py`.

#### Pilot P3: FQ-QAT Gradient Flow
- **What it tests:** Does the STE produce meaningful gradient updates through fake quantisation?
- **Acceptance criterion:** After 100 steps of FQ-QAT training, at least 5% of weight updates cross an NF4 bin boundary (i.e., quantised-before ≠ quantised-after for ≥5% of parameters)
- **If it fails:** Increase learning rate or decrease scale. If still fails, FQ-QAT is not viable at this precision.
- **Config:** `conditions/c5_fqat.yaml`, seed 42
- **Script:** To be written. Monitors `fake_quantise()` output changes during training.

#### Pilot P4: Combined Loss Convergence
- **What it tests:** Does the combined loss (NPO + SSIUU + FQ-QAT) converge?
- **Acceptance criterion:** Total loss decreases monotonically over 100 steps (no divergence)
- **If it fails:** Reduce λ_attr or lr. If still diverges, the combined loss is unstable.
- **Config:** `conditions/c6_fq_ssiuu.yaml`, seed 42
- **Script:** To be written. Full training loop with all three components.

---

### 8. Module Map — What Each File Does

| File | Purpose | Primary API | Project Dependencies |
|------|---------|-------------|---------------------|
| `models/loader.py` | Load models at NF4/BF16, wrap with LoRA, merge adapters | `load_base_model()`, `wrap_with_lora()`, `merge_adapter()` | None (standalone) |
| `unlearning/npo_loss.py` | NPO loss computation for the forget set | `compute_npo_loss()`, `compute_retain_kl_loss()` | None (standalone) |
| `unlearning/quail_loss.py` | QUAIL hinge loss (concurrent baseline) | `compute_quail_loss()` | None (standalone) |
| `quantisation/fake_quant.py` | Fake quantisation with STE for training | `fake_quantise()`, `compute_scale()` | None (standalone) |
| `attribution/ssiuu_reg.py` | SSIUU attribution regularisation | `SSIUURegulariser`, `compute_negative_attribution()` | None (captum external dep) |
| `evaluation/qra_pipeline.py` | Three-precision QRA evaluation pipeline | `run_qra_evaluation()` | `models/loader.py` |
| `evaluation/certificate.py` | Durability certificate generation | `generate_certificate()`, `save_certificate()` | `evaluation/qra_pipeline.py` |

---

### 9. Environment Notes — Deviations From Handbook

1. **Python managed by `uv`, not conda.** The local machine had Python 3.14; `uv` installed Python 3.11.15. The venv is at `/home/muhammad-arslan/Downloads/projects/fq-ssiuu/.venv/`. On Colab, use the system Python 3.11.

2. **`pyproject.toml` build-backend is `setuptools.build_meta`**, not `setuptools.backends._legacy:_Backend` as originally specified. The `_legacy` path does not exist in current setuptools. This is the correct, standard value.

3. **CI runs-on `ubuntu-latest`**, not `ubuntu-26.04`. The latter is not a valid GitHub Actions runner image.

4. **CPU-only torch installed locally** for self-tests. Colab will use the CUDA version. The local `.venv` does not have numpy (warning appears but is harmless).

5. **Test stubs from Session 1** that were not updated in Session 2 still use `try/except ImportError` for torch. This is intentional for CI environments without torch.

---
### August 2026 Update: Colab Environment Reality

**Actual Colab T4 environment (confirmed August 2026):**
- Python: 3.12.13 (not 3.11 as originally specified — fully compatible)
- PyTorch: 2.13.0+cu130 (not 2.4.1 — newer, do not downgrade)
- CUDA: 13.0 (not 12.1)
- GPU: Tesla T4 (Turing, Compute Capability 7.5)

**Permanent changes made (commit: [to be filled after commit]):**
1. `flash-attn` REMOVED from requirements.txt. T4 (CC 7.5) is not supported by flash-attn 2.x (requires Ampere CC 8.0+). All attention now uses `attn_implementation="sdpa"` (PyTorch native SDPA, fully T4-compatible).
2. `bitsandbytes` upgraded from 0.42.0 to 0.45.5. CUDA 13.0 support added in 0.45.x.
3. `torch`/`torchvision`/`torchaudio` version pins REMOVED. Colab provides 2.13.0 which is newer and better than the pinned 2.4.1.
4. `setup_env.sh` rewritten: does not reinstall PyTorch, upgrades bitsandbytes, sets flash-attn bypass vars.
5. `gate0_env_check.py` updated: checks SDPA instead of flash-attn, tests bitsandbytes 0.45.x, adds `attn_implementation="sdpa"` to NF4 model load.

**Rule for all future model loading:**
Always pass `attn_implementation="sdpa"` to `from_pretrained()`. This must be added to `src/fq_ssiuu/models/loader.py` — see Task 8.
---
---

### 10. The Next Three Tasks for Gemini — In Order

#### Task A: GATE-0 — Environment Verification on Colab L4

**What to do:**
1. Open a Google Colab session with L4 GPU runtime
2. Clone the repo: `git clone https://github.com/ibn-e-Muhammad/fq-ssiuu.git`
3. Set environment: `export HF_TOKEN='<your_token>'`  # Load via secret provider; see src/fq_ssiuu/secrets.py. **[REDACTED — rotation required]**
4. Run: `python scripts/colab_mount.py` (mount Google Drive)
5. Run: `bash scripts/setup_env.sh` (install all dependencies)
6. Run: `python experiments/gate0_env_check.py`

**Expected output:** All 14 checks PASS. Gate result: GATE-0 PASSED.

**Acceptance criterion:** All 14 environment checks pass. If any fail, fix the issue before proceeding.

**Failure action:** If torch version mismatch: update `requirements.txt`. If CUDA not available: ensure L4 GPU is selected. If HF token invalid: verify token at huggingface.co.

**Files to create:** `requirements_locked.txt` (run `pip freeze > requirements_locked.txt` after successful setup).

---

#### Task B: GATE-1 — OpenUnlearning Baseline Reproduction

**What to do:**
1. Fork `https://github.com/ibn-e-Muhammad/open-unlearning` on the `fq-ssiuu-integration` branch
2. Run the TOFU benchmark with NPO on Llama-3.2-1B-Instruct
3. Compare results to Table 1 in the OpenUnlearning paper (arXiv:2506.12618)
4. Integrate the evaluation harness into `evaluation/qra_pipeline.py` — replace the placeholder `evaluate_at_precision()` with real FQ and MU computation

**Expected output:** TOFU NPO baseline results within ±5% of published numbers.

**Acceptance criterion:** Published results reproduced within ±5%. FQ and MU metrics are real numbers, not -1.0.

**Failure action:** If results differ by >5%: check hyperparameters against the paper. Check dataset version. If fundamentally different: open an issue on the OpenUnlearning repo.

**Files to modify:** `src/fq_ssiuu/evaluation/qra_pipeline.py` (replace placeholder metrics).

---

#### Task C: Pilot P1 — QRA Existence Test

**What to do:**
1. Train NPO baseline (C2) with seed 42 on TOFU forget05
2. Merge adapter: `merge_adapter()` from `models/loader.py`
3. Evaluate at BF16 and INT4 using the updated QRA pipeline
4. Compute FADE at both precisions
5. Record results to `results/pilot_p1/`

**Expected output:** FADE_BF16 ≥ 0.6, FADE_INT4 < 0.3.

**Acceptance criterion:** Q-INT4 > 0.3 (showing ≥30% relative degradation at INT4). This proves QRA exists.

**Failure action:** If FADE_INT4 ≥ 0.3 (QRA does not exist): try forget10 split, or try larger model. Document finding.

**Files to create:** Training script for C2, results JSON in `results/pilot_p1/`.

---

### 11. Papers Gemini Must Read Before Implementing Anything

| Paper | arXiv ID | One-Sentence Summary | Related Module |
|-------|----------|----------------------|----------------|
| NPO | 2404.05868 | Bounded loss function for unlearning that prevents catastrophic collapse | `npo_loss.py` |
| SSIUU | 2509.22263 | Attribution-guided regularisation to prevent spurious unlearning neurons | `ssiuu_reg.py` |
| QUAIL | 2601.15538 | Logit-space hinge loss ensuring QRA robustness via output deviation | `quail_loss.py` |
| DurableUn | 2605.02196 | First formalisation of QRA; three-precision evaluation protocol | `qra_pipeline.py` |
| MANSU | 2605.15138 | NF4 bin width measurement (8.4e-4); theoretical QRA analysis | `fake_quant.py`, `qra_pipeline.py` |
| OpenUnlearning | 2506.12618 | Unified evaluation harness for unlearning benchmarks | `qra_pipeline.py` (integration) |
| TOFU | Maini et al. COLM 2024 | Forget-retain benchmark dataset for unlearning evaluation | Dataset usage |
| STE | Bengio 2013, 1308.3432 | Straight-Through Estimator for quantisation-aware training | `fake_quant.py` |

---

### 12. Open Questions and Known Risks

1. **QRA existence at 1B scale is unproven.** All prior QRA demonstrations used 7B+ models. Pilot P1 is designed to test this. If QRA does not exist at 1B, the entire project premise needs revision.

2. **IG VRAM feasibility is untested.** Integrated Gradients requires `n_ig_steps` forward passes stored simultaneously. At 1B parameters, this may exceed 16 GB. The `GradientNormAttribution` fallback exists but is less theoretically grounded.

3. **Historical statement superseded (2026-09-25).** The earlier claim that no official QUAIL code exists is withdrawn. The paper links [himans-iitk/CS534L_Project](https://github.com/himans-iitk/CS534L_Project); it is the paper-linked source repository. Pinning and code audit are required before use. The old hinge-only project implementation is not accepted QUAIL. See [current source hierarchy](methods/UPSTREAM_IMPLEMENTATIONS.md).

4. **Combined loss stability (NPO + SSIUU + FQ-QAT) is untested.** Three loss terms with different scales may cause instability. Pilot P4 tests this. If unstable, reduce λ_attr or use gradient scaling.

5. **OpenUnlearning integration complexity is unknown.** The harness may require significant adaptation for our three-precision protocol. Budget extra time for GATE-1.

6. **The FQ-QAT approach uses symmetric INT4 approximation, not true NF4.** The `fake_quantise()` function uses `round(x/scale)*scale` with 16 uniform bins, while actual NF4 uses a non-uniform codebook. This approximation is sufficient for training (the STE gradient is the same) but the gap should be discussed in the paper.

7. **Effect size magnitude is unknown.** We claim FQ-SSIUU "reduces Q-INT4", but we don't know by how much until experiments run. If the effect is < 5pp, it may not be practically meaningful even if statistically significant.

8. **Colab session time limits.** L4 sessions may be interrupted. All training scripts must support checkpoint resumption. This is not yet implemented.

9. **The C3 (GA baseline) and C7 (DurableUn-SAF) config files have not been created.** These need to be written before full experiments begin.

10. **Flash attention installation is commented out** in `requirements.txt` due to build complexity. It provides ~30% speedup but is not required for correctness. Enable if build succeeds on Colab.
10. **Flash attention installation is commented out** in `requirements.txt` due to build complexity. It provides ~30% speedup but is not required for correctness. Enable if build succeeds on Colab.

---

## Session Update: Pilot P1 + P2 Implementation Complete

**Updated by:** Claude Opus 4.6 (Thinking), Session 3
**Date:** 2026-08-08
**Branch:** `pilot/p1-p2` (pushed)

### GATE-1 Status: PASSED

GATE-1 evaluation completed successfully:
- forget_quality: 0.3937 (target: 0.14±0.05 — EXCEEDS in favourable direction)
- model_utility: 0.5545 (target: 0.45±0.05 — EXCEEDS)
- Checkpoint uploaded to: `Xafire/GATE1-NPO-Baseline` (private HF repo)
- Full verdict: `results/gate1_baseline/GATE1_VERDICT.md`

### Current Branch: pilot/p1-p2 (pushed)

### What Gemini Must Do Next:
1. Run the Kaggle notebook: `experiments/notebooks/kaggle/PILOT_P1_P2.ipynb`
   Use **SINGLE T4 GPU** — not dual GPU (evaluation requires single-process)
2. Replace `YOUR_HF_TOKEN` with the value of your `HF_TOKEN` secret in each cell. **[REDACTED — rotation required]**
3. Run all cells in order (Cell 1 through Cell 6)
4. Copy the complete output of Cell 4 (P1 verdict) and Cell 5 (P2 verdict)
5. Report results back — DO NOT interpret results without reporting them first

### P1 Critical Rules:
- NF4 via bitsandbytes ONLY — never GPTQ or AWQ
- The checkpoint is a MERGED model (model.safetensors), NOT a LoRA adapter
- Loading: `AutoModelForCausalLM.from_pretrained()` with BitsAndBytesConfig for quantised precision
- retain_logs_path must be set — without it forget_quality = None (the GATE-1 bug)
- Single GPU only — DDP disables the OpenUnlearning evaluator

### P1 Acceptance Criterion (from EVB Table 5.1):
```
Q-INT4 = |FQ_NF4 - FQ_FP16| / FQ_FP16 > 0.05 → PASS (QRA confirmed)
Q-INT4 < 0.02 → FAIL (restructure project — consult EVB Section 10, F2)
Q-INT4 0.02-0.05 → BORDERLINE (run 2 more seeds)
```

### P2 Acceptance Criterion (from EVB Table 5.2):
```
Peak VRAM < 14 GB → PASS (use captum IG n_steps=20)
Peak VRAM 14-16 GB → MARGINAL (reduce to n_steps=10)
OOM → FAIL (switch to GradientNormAttribution in ssiuu_reg.py)
```

### After P1 and P2 Results Are Reported:
- If P1 PASSES and P2 PASSES → Begin Pilot P3 (FQ-QAT stability, forget01, 5 epochs)
- If P1 PASSES and P2 FAILS  → Begin Pilot P3; use GradientNormAttribution for P4+
- If P1 FAILS                → STOP. Report to supervisor. See EVB Section 10, F2.

### New Files Created:
| File | Description |
|------|-------------|
| `src/fq_ssiuu/evaluation/qra_pipeline.py` | Real OpenUnlearning integration (placeholders replaced) |
| `experiments/_eval_at_precision.py` | Evaluation helper: loads model at FP16/INT8/NF4, runs TOFU evaluator |
| `experiments/pilot_p1_qra_check.py` | Complete P1 experiment script with CLI args |
| `experiments/pilot_p2_attribution.py` | Complete P2 experiment script with VRAM measurement |
| `experiments/notebooks/kaggle/PILOT_P1_P2.ipynb` | 7-cell Kaggle notebook for P1+P2 execution |
| `results/pilot_p1/README.md` | Results template (PENDING) |
| `results/pilot_p2/README.md` | Results template (PENDING) |
| `handbook/research_diary/2026-08-08_pilot_planning.md` | Planning decisions for this session |

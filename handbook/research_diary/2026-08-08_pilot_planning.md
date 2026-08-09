# Research Diary: 2026-08-08 — Pilot P1 + P2: Planning and Implementation

**Date:** 2026-08-08
**Author:** Research Team
**Subject:** Technical decision rationale, checkpoint analysis, and execution plan for Pilot P1 (QRA Existence Test) and Pilot P2 (SSIUU Attribution Feasibility Test)

---

## 1. Context and Status

Following the successful execution of GATE-1 baseline reproduction (NPO unlearning on `meta-llama/Llama-3.2-1B-Instruct` on TOFU `forget05`, seed 42), the evaluated checkpoint achieved:
- **Forget Quality (FQ):** 0.3937
- **Model Utility (MU):** 0.5545

The baseline checkpoint has been pushed to the private Hugging Face repository `Xafire/GATE1-NPO-Baseline`. The project now moves to the Pilot Phase defined in the Experimental Validation Blueprint (EVB), specifically Pilot P1 (QRA Existence Test) and Pilot P2 (SSIUU Attribution Feasibility Test).

---

## 2. Checkpoint Structure Analysis

The Hugging Face hub checkpoint `Xafire/GATE1-NPO-Baseline` is a **merged full-precision model** containing full `model.safetensors` files (~2.47 GB), rather than a standalone LoRA adapter. 

### Loading Procedure
Because the unlearned weights are already merged into the base Llama-3.2-1B architecture:
- No LoRA `merge_and_unload()` call is required prior to quantisation or attribution.
- The model loading procedure across scripts is simplified directly to:
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      "Xafire/GATE1-NPO-Baseline",
      torch_dtype=torch.float16,
      device_map="auto"
  )
  ```

---

## 3. Decision Rationale: Why P1 Uses NF4 (Not GPTQ/AWQ)

### Background and Mechanism
The Quantisation Recovery Attack (QRA) mechanism documented in DurableUn (arXiv:2605.02196) operates specifically under normal float 4-bit (NF4) quantisation with LoRA adapters. During gradient-based unlearning (e.g., NPO), unlearning updates produce parameter changes $\Delta w_i$ that are extremely small relative to the quantisation bin width:
$$|\Delta w_i| < \delta_i \quad (\text{where NF4 bin width } \delta_i \approx 8.4 \times 10^{-4})$$

When the unlearned model is quantised or re-quantised to NF4:
$$\text{round}\left(\frac{w + \Delta w}{\text{scale}}\right) = \text{round}\left(\frac{w}{\text{scale}}\right)$$

The unlearning update $\Delta w$ rounds to zero in the quantised grid, effectively restoring pre-unlearning model representations and recovering forgotten knowledge.

### Rationale
- **Target Attack Surface:** NF4 quantisation uses fixed quantile-based bin boundaries optimal for normally distributed weights.
- **Why exclude GPTQ / AWQ / GGUF:** Second-order Hessian-based quantisation (GPTQ) and activation-aware quantisation (AWQ) recalculate quantisation scales and clipping thresholds based on calibration datasets. They test a completely different mathematical transformation surface.
- Evaluating GPTQ or AWQ for P1 would evaluate quantization noise rather than testing the specific DurableUn theoretical mechanism. Therefore, P1 strictly standardizes on `bitsandbytes` NF4 quantisation.

---

## 4. Decision Rationale: Why P2 Uses FP16 / Full Precision (Not NF4)

### Technical Constraint
Integrated Gradients (via `captum.attr.LayerIntegratedGradients`) relies on computing gradients of outputs with respect to intermediate layer activations and model weights along an interpolation path between a baseline input and the target input:
$$\text{IntegratedGrads}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

### Rationale
- `bitsandbytes` NF4 quantised weights are packed and stored as discrete 4-bit integer index lookup tables (`torch.uint8`).
- Quantised weights are non-differentiable; PyTorch autograd cannot propagate gradients through quantized integer representations.
- Attribution algorithms (Integrated Gradients and layer-wise feature attribution) **must** execute on full-precision weights (FP16/BF16).
- Due to NVIDIA T4 hardware limitations on Kaggle, target attribution during SSIUU training is standard FP16 (as native BF16 arithmetic is unsupported on Turing architecture T4 GPUs).

---

## 5. Sequential Execution Plan (Kaggle Single T4 GPU)

Pilot P1 and Pilot P2 will run sequentially within a single Kaggle Notebook session (`PILOT_P1_P2.ipynb`) utilizing a single T4 GPU (16 GB VRAM).

### Pipeline Steps:
1. **Environment Setup & Downloads (~15 min):** Clone `fq-ssiuu`, install dependencies (`bitsandbytes`, `captum`, `open-unlearning`), download `Xafire/GATE1-NPO-Baseline` and TOFU evaluation logs.
2. **Pilot P1 Execution (~15 min total):**
   - **FP16 Baseline Eval (~5 min):** Load checkpoint in FP16, run TOFU evaluation suite with `retain_logs_path`.
   - **INT8 Eval (~5 min):** Load fresh instance with 8-bit quantisation (`load_in_8bit=True`), run TOFU evaluation.
   - **INT4 (NF4) Eval (~5 min):** Load fresh instance with NF4 quantisation (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`), run TOFU evaluation.
   - Compute QRA metric: $Q_{\text{INT4}} = \frac{|FQ_{\text{INT4}} - FQ_{\text{BF16}}|}{FQ_{\text{BF16}} + \varepsilon}$.
3. **Pilot P2 Execution (~10 min):**
   - Load full-precision FP16 model.
   - Run `LayerIntegratedGradients` on layer 14 (`model.model.layers[14]`) for 2 sample inputs with `n_steps=20`.
   - Record VRAM usage (baseline vs. peak), compute time, and verify output tensor dimensions.
4. **Total Estimated Runtime:** ~2–3 hours.

---

## 6. Known Risks and Mitigation / Fallback Paths

| Experiment | Risk | Potential Cause | Fallback / Mitigation Plan |
|------------|------|-----------------|---------------------------|
| **Pilot P1** | Weak or absent QRA ($Q_{\text{INT4}} < 0.02$) | At 1B parameter scale, NPO parameter updates $\Delta w$ may be larger than NF4 bin width ($\delta_i \approx 8.4 \times 10^{-4}$). | Per EVB §10 F2, if QRA does not manifest at 1B, test QRA existence at 3B scale (`Llama-3.2-3B-Instruct`) before project restructuring. |
| **Pilot P2** | CUDA Out-Of-Memory (OOM) during LayerIntegratedGradients | Accumulating 20 step gradients across layer 14 activations may exceed 16 GB T4 VRAM. | Fallback 1: Reduce IG steps to `n_steps=10` (marginal VRAM).<br>Fallback 2: Use `GradientNormAttribution` implemented in `src/fq_ssiuu/attribution/ssiuu_reg.py` (calculates single-pass $\|\nabla_w L\|$, requiring minimal VRAM overhead). |

---

## 7. Summary of Deliverables

- `results/pilot_p1/README.md` (P1 status and protocol definition)
- `results/pilot_p2/README.md` (P2 status, peak VRAM limits, and fallback triggers)
- `experiments/pilot_p1_qra_check.py` & `experiments/pilot_p2_attribution.py`
- Kaggle Notebook: `experiments/notebooks/kaggle/PILOT_P1_P2.ipynb`

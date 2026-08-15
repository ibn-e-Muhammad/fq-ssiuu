# Pilot P2: Integrated Gradients VRAM Profiling

## 1. Objective
The objective of Pilot P2 is to profile the VRAM consumption of Integrated Gradients (IG) attribution and verify it meets the resource constraints outlined in EVB Table 5.2.

## 2. EVB Specification
- Peak VRAM < 14 GB → PASS
- Peak VRAM 14-16 GB → MARGINAL (reduce n_steps to 10)
- OOM → FAIL (switch to GradientNormAttribution)

## 3. Environment
- Execution date: 2026-08-09

## 4. Model Configuration
- Precision: FP16
- *Note on precision:* NF4 weights are non-differentiable and thus incompatible with gradient-based attribution methods. FP16 is required.

## 5. IG Configuration
- Target layer: 14
- Number of steps (`n_steps`): 20
- Baseline: Zero embedding
- Batch size: 1 (automatically adjusted to 2 by Captum)

## 6. Results Table

| Metric | Value |
|--------|-------|
| VRAM Before IG | 2.30200 GB |
| Peak VRAM During IG | 2.51100 GB |
| VRAM Margin | 12.4890 GB |
| Elapsed Time | 1.10000 s |
| OOM Occurred | False |
| Attribution Mean | 0.00502285 |
| Attribution Max | 186.686 |
| Non-Zero Fraction | 0.974908 |

## 7. Tensor Shape Verification
- **Input Shape:** `[2, 40]` (`[num_examples, seq_len]`)
- **Attribution Shape:** `[2, 40, 2048]` (`[num_examples, seq_len, hidden_size]`)
- **Verification:** Llama-3.2-1B architecture has `hidden_size=2048` and `num_hidden_layers=16`. The output shape correctly reflects token-level, hidden-dimension attributions produced by IG at a decoder layer.

## 8. VRAM Analysis
Peak VRAM consumption was 2.51100 GB, which is 82% below the 14 GB threshold. This indicates that IG can be run concurrently with training without risking out-of-memory errors.

## 9. Non-Zero Fraction Analysis
The non-zero fraction of 0.974908 (97.49%) confirms that the IG attribution map is rich and non-degenerate.

## 10. Fallback Assessment
The fallback to `GradientNormAttribution` is strictly not needed. There is a healthy 12.4890 GB margin.

## 11. Formal Verdict
- **Criterion:** Peak VRAM < 14 GB
- **Measured:** 2.51100 GB
- **Verdict:** PASS
- **Implication:** The 12.4890 GB margin firmly satisfies the requirement.

## 12. Recommendation for SSIUU Training Configuration
Deploy IG for attribution during SSIUU training using the tested configuration (`n_steps=20`, FP16), leveraging the substantial VRAM margin to ensure stable training runs.

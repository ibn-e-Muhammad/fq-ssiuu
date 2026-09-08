# Upstream implementation register

Verified 2026-09-07. Author code is the starting reference; acceptance also requires
agreement with the paper and tests of any project adaptation. Historical method
claims do not substitute for this comparison.

| Method | Authoritative paper | Author-linked code and pinned revision | Current execution decision |
|---|---|---|---|
| NPO | [arXiv 2404.05868v2, 2024-10-10](https://arxiv.org/abs/2404.05868v2) | [licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization/tree/1207bbe9a1abf502bc2d7b43a0a467d9524b4036), `1207bbe9a1abf502bc2d7b43a0a467d9524b4036` | Use the TOFU objective as the reference. Verify equivalence before a new NPO pilot. |
| QUAIL | [arXiv 2601.15538v1, 2026-01-21, Appendix E](https://arxiv.org/html/2601.15538v1) | [himans-iitk/CS534L_Project](https://github.com/himans-iitk/CS534L_Project/tree/8e52f41c2bec6e86490655d610c00c161d04f294), `8e52f41c2bec6e86490655d610c00c161d04f294` | Compare all objective terms and update order with paper. Do not substitute this repository's raw-logit NPO variant for original NPO. |
| SSIUU | [arXiv 2509.22263v2, 2026-03-04](https://arxiv.org/abs/2509.22263v2), [ICLR paper](https://openreview.net/pdf?id=z2zFk9jYpw) | Author implementation not located in the paper or [lead author's publication page](https://yny0506.github.io/) | Derive from the authoritative algorithm; do not assert that no implementation exists. |

NPO's [TOFU trainer](https://github.com/licong-lin/negative-preference-optimization/blob/1207bbe9a1abf502bc2d7b43a0a467d9524b4036/TOFU/dataloader.py)
uses the difference between current and frozen-reference sequence NLLs.
Its [loss helper](https://github.com/licong-lin/negative-preference-optimization/blob/1207bbe9a1abf502bc2d7b43a0a467d9524b4036/TOFU/data_module.py)
causally shifts labels, ignores -100 and sums token losses per sequence. Current
project `compute_npo_loss` averages tokens and leaves alignment to callers; the
earlier blanket FAITHFUL assessment is therefore withdrawn pending conformance.

The upstream [training entrypoint](https://github.com/licong-lin/negative-preference-optimization/blob/1207bbe9a1abf502bc2d7b43a0a467d9524b4036/TOFU/forget.py)
uses BF16, DeepSpeed and a local fine-tuned checkpoint. Its TOFU subtree has no
dependency lock. Project FP16, SDPA, one-device execution, 1B model, LoRA and
checkpoint/resume need explicit adaptation records; copying its launcher
unchanged would violate this project's compute policy.

Required NPO acceptance: exact masked sequence-loss and total-objective parity
against the pinned reference on unequal-length examples; causal alignment;
reference detachment; current-model gradient direction; retained-data objective
specified by condition; tiny deterministic FP16 execution through Mother. No
scientific pilot or final hypothesis claim is supported by this register alone.

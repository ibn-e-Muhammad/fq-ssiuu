# Upstream implementation register

Sources first recorded 2026-09-07; hierarchy corrected 2026-09-25.
Primary papers and their source repositories are the method requirements; acceptance also requires
agreement with the paper and tests of any project adaptation. Historical method
claims do not substitute for this comparison.

| Method | Authoritative paper | Primary source code / pin status | Current execution decision |
|---|---|---|---|
| NPO | [arXiv 2404.05868v2, 2024-10-10](https://arxiv.org/abs/2404.05868v2) | [licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization/tree/1207bbe9a1abf502bc2d7b43a0a467d9524b4036), `1207bbe9a1abf502bc2d7b43a0a467d9524b4036` | Use the TOFU objective as the reference. Verify equivalence before a new NPO pilot. |
| QUAIL | [arXiv 2601.15538v1, 2026-01-21, §VII-F](https://arxiv.org/html/2601.15538v1) | Paper-linked source repository [himans-iitk/CS534L_Project](https://github.com/himans-iitk/CS534L_Project); previously recorded candidate revision `8e52f41c2bec6e86490655d610c00c161d04f294`, not accepted for execution | Before use, verify the immutable pin and relevant file hashes, audit all objective terms and update order against the paper. Pinning/code audit remain required. Do not substitute this repository's NPO variant for original NPO. |
| SSIUU | [arXiv 2509.22263v2, 2026-03-04](https://arxiv.org/abs/2509.22263v2), [ICLR paper](https://openreview.net/pdf?id=z2zFk9jYpw) | No linked implementation has yet been verified | Implementation will be paper-derived from the equations and Algorithm 1, with independent mathematical/gradient tests. This does not assert that no code exists. |
| QRA | [Catastrophic Failure of LLM Unlearning via Quantization, arXiv 2410.16454v2](https://arxiv.org/html/2410.16454v2) | [zzwjames/FailureLLMUnlearning](https://github.com/zzwjames/FailureLLMUnlearning), official paper implementation; execution pin/file audit still required | Use the paper and source-defined evaluation requirements; project TOFU metrics and Protocol v2 are not substitutes for upstream reproduction. |

The QUAIL paper explicitly links CS534L_Project in §VII-F (Reproducibility).
This supersedes the historical `GEMINI_HANDOFF.md` claim that no official QUAIL
code exists. A paper link establishes source provenance, not code conformance.
NPO's primary code authority is licong-lin/negative-preference-optimization;
OpenUnlearning below is an additional project parity anchor, not a replacement.
[Reproduction classes and gate scope](METHOD_REPRODUCTION_POLICY.md) distinguish
source-faithful 1B adaptation from exact replication or the novel combined method.

NPO's [TOFU trainer](https://github.com/licong-lin/negative-preference-optimization/blob/1207bbe9a1abf502bc2d7b43a0a467d9524b4036/TOFU/dataloader.py)
uses the difference between current and frozen-reference sequence NLLs.
Its [loss helper](https://github.com/licong-lin/negative-preference-optimization/blob/1207bbe9a1abf502bc2d7b43a0a467d9524b4036/TOFU/data_module.py)
causally shifts labels, ignores -100 and sums token losses per sequence. Historical
project `compute_npo_loss` averaged tokens and left alignment to callers. It now
lives in `experiments/reproductions/NPO/historical/token_mean_loss.py`. The new
candidate is causally shifted and sequence-summed; CPU value/gradient parity
passes against both pinned sources; Work Unit B accepted the bounded data/frozen
reference adapter. NPO training has not yet been reproduced.

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

## Pinned project baseline — OpenUnlearning

[OpenUnlearning](https://github.com/locuslab/open-unlearning/tree/4ad738aaf60f6a4385f6e2506d01da99e76c31f3)
at `4ad738aaf60f6a4385f6e2506d01da99e76c31f3` is the second required parity anchor.
`src/trainer/utils.py` defines the causal sequence NLL and lose-only DPO/NPO term;
`src/trainer/unlearn/npo.py` adds weighted retain NLL through GradDiff. The config
uses beta 0.1 with gamma/alpha 1.0. Pure NPO and the combined trainer are recorded
separately; no retain-KL substitution is made. File hashes and CPU observations
are preserved in reconciliation evidence. See `NPO_CONFORMANCE.md`.

Reproduce CPU parity from separate pinned checkouts, without executing launchers:

```bash
.venv/bin/python experiments/reproductions/NPO/validate_upstream.py \
  --upstream /path/to/pinned-npo --openunlearning /path/to/pinned-openunlearning \
  --output /new/path/parity.json
```

The validator executes selected original function ASTs only after exact file-hash
verification. No third-party source is redistributed by this comparison. It does
not establish tokenizer, model-forward, optimizer, 1B or GPU parity. No model run
is authorised for scientific reconciliation A.

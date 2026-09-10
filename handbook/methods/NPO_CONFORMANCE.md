# NPO objective conformance

Reference: Zhang et al., arXiv [2404.05868v2](https://arxiv.org/html/2404.05868v2),
revised 2024-10-10, Equation 3. Author code:
`licong-lin/negative-preference-optimization` at
`1207bbe9a1abf502bc2d7b43a0a467d9524b4036`, `TOFU/data_module.py:242`
and `TOFU/dataloader.py:255–291`. The checkout predates paper v2; objective
agreement is checked directly. No license file was located at that commit;
upstream source is inspected externally, not redistributed under this project's license.

## Mathematical and tensor contract

For unshifted logits `[B,T,V]` and labels `[B,T]`, compare logit `t` with label
`t+1`. Ignore exactly labels equal to `-100`. Sum target NLLs within each sequence,
giving `n_theta[i]` and `n_ref[i]`. Minimize the batch mean of
`(2/beta) * softplus(beta * (n_ref[i] - n_theta[i]))`, beta > 0.
This is author NPO, without token-length normalization.

The separate NPO+RT variant uses `npo_coeff * NPO + retain_coeff * CE_retain`.
Retain CE averages over all scored causal tokens, as the author's Hugging Face
model loss does; it is not the existing project retain-KL helper. Author config
defaults are beta=0.1 and both coefficients=1.0. This specification does not change
registered project conditions, seeds or hyperparameters.

- Current logits retain gradients. Reference logits are detached defensively;
  the eventual adapter must also freeze the original fine-tuned reference, use
  evaluation mode and execute its forward under no_grad.
- Labels, not attention masks, define scoring. Upstream preprocessing masks
  question tokens and scores the answer plus EOS. Its appended EOS may occupy
  the first attention-mask-zero position. Never blindly discard that target.
- First label and final logit are unused. Right padding with ignored labels
  changes neither the loss nor gradients at existing positions.
- No optimizer-step state is carried by this objective; trainable parameters
  are selected by the adapter. No reference parameters may update.
- Complexity is O(BTV) for log-softmax/CE, with no Hessian or extra model copy
  inside the loss. Model/reference forward memory is an adapter concern.

## Explicit project safeguards and adaptations

Reject malformed shapes/dtypes/target IDs, nonfinite or nonpositive beta, and any
example with no scored target after shifting. The author helper returns zero NLL
for an empty target; it does not define a robust training policy. Rejection is a
project safeguard. The future data adapter must record offending example IDs.
FP16/BF16 input probability reductions use FP32; FP64 validation inputs stay FP64.
Canonical model computation remains single-T4 FP16. This numerical adaptation
does not constitute a reproduction of the author's BF16 two-A100 7B training.

## Falsifiable acceptance and limits

1. Compare loss and current/retain-logit gradients against the actual pinned author
   helper/trainer and OpenUnlearning implementation on unequal-length masked examples, without importing the
   author's DeepSpeed launcher. Verify source file hashes before executing code.
2. Independently calculate a scalar oracle, causal alignment, masking/EOS,
   padding invariance, reference detachment, gradient direction and beta limits.
3. Test NPO+RT terms, nonunit weights and zero-weight limits independently.
4. Later runtime acceptance must exercise finite FP16 loss/gradients on one T4.
   This work unit is CPU-only; the proposed GPU validation was not submitted.

These criteria establish objective conformance only. A corrected 1B pilot still
requires a versioned answer-masking adapter, immutable model/data revisions,
unquantised frozen reference policy, optimizer/checkpoint/resume behavior and
measured training memory. Historical P3 token-mean results and thresholds do not
transfer automatically. RG-001/002/003/005/006 remain BLOCKED and RG-004 OPEN.

Candidate: `experiments/reproductions/NPO/candidate.py`. Validator:
`experiments/reproductions/NPO/validate_upstream.py`. Tests: `tests/methods/NPO/`.
OpenUnlearning is pinned at `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`.
Its `compute_dpo_loss` with lose inputs only matches pure NPO; its NPO trainer
combines gamma times NPO and alpha times retain NLL. Config beta is 0.1; all
coefficients are passed explicitly. Source hashes are checked before executing
selected original function ASTs. No upstream launcher or model is executed.
Current CPU parity covers 16 synthetic cases across both repositories, FP32/FP64,
pure/retain variants and default/nonunit coefficients. Label/masking tests are
independent fixtures; production tokenizer/dataset parity remains unestablished.

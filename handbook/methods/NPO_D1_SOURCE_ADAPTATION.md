# D1 NPO source and adaptation specification

Status: **source conformance in progress; accumulation-8 resource adapter researcher-approved; no
training recipe is enabled**. Intended result: a bounded source-faithful 1B
adapted-reproduction pilot, not an exact replication or a multi-seed result.

## Primary source and version

NPO paper [2404.05868v2](https://arxiv.org/html/2404.05868v2), revised
2024-10-10, Equation 3 and TOFU experimental setup. Author repository:
[`licong-lin/negative-preference-optimization`](https://github.com/licong-lin/negative-preference-optimization/tree/1207bbe9a1abf502bc2d7b43a0a467d9524b4036),
commit `1207bbe9a1abf502bc2d7b43a0a467d9524b4036` (2024-07-15).
This is the latest available commit before v2. Comparing it with the
publication-near `eb877a351f4878f5b9ad6cbd44c27818e3b4b74b` (2024-04-10)
shows only addition of `synthetic/dist.py`; TOFU code is unchanged.
This establishes an auditable source selection, not proof of the exact private
execution used to produce the paper's figures.

Ten referenced files, SHA-256 and Git blob IDs are recorded in
[`source_v1.json`](../../experiments/reproductions/NPO/source_v1.json).
No license file occurs in the pinned author tree. Source remains external and
is not vendored or relicensed; validators execute selected hash-verified functions.
OpenUnlearning `4ad738aaf60f6a4385f6e2506d01da99e76c31f3` supplies the previously
accepted Llama-3 chat adapter and independent objective cross-check. Its recorded
file hashes remain in `inputs_v1.json` and `validate_upstream.py`.

## Upstream-to-local method map

| Element | Author source | D1 mapping / outstanding work |
|---|---|---|
| Objective | `TOFU/dataloader.py`, `npo_grad_diff`; sequence-summed causal NPO plus token-mean retain CE | Select the shipped NPO+RT configuration, not pure NPO or retain KL. Existing candidate matches actual author loss/gradients. Both coefficients 1; beta 0.1 |
| Reference | `fine_tuned` checkpoint, frozen, eval, no_grad | Same exact original full checkpoint as candidate initialization; unquantized FP16 on cuda:0; retain95 diagnostic oracle is a different object |
| Masking | `data_module.py`: shift by one in loss; question ignored, answer and terminator scored; fixed length 500 | Accepted pinned OpenUnlearning Llama-3 chat adapter, source length 500. Prompt/padding ignored; EOT attended and scored. This is a model adapter, not identical Llama-2 prompt tensors |
| Split | Config defaults forget01; paper also evaluates forget05; dataset class derives retain split | Existing accepted forget05/retain95 boundary, not a new data substitution. All 200/3800 source rows and their order must match both registry and author Git blobs |
| Retain sampling | Dataset `__getitem__` reseeds with forget index, then offsets a sampled retain index | Preserve this source pairing if runner is authorized. Do not replace with independent retain shuffling or cycle the entire retain set |
| Trainable weights | LoRA rank 0, complete model | Every unique model parameter trainable; tied embedding counted once. No LoRA, frozen backbone, quantized reference or reduced-state optimizer |
| Optimizer | `forget.py`: paged_adamw_32bit; lr 1e-5, weight decay .01; ZeRO-3, no explicit offload | Preserve optimizer family and FP32 moments. Source does not pin dependency versions; native-compatible integration still needs parity/review. Do not silently substitute ordinary AdamW or an 8-bit optimizer |
| Scheduler / stop | 10 nominal epochs, linear warmup then decay; script floors step counts; global batch 32 | 200 rows produce max_steps 62, warmup 6, evaluations at 6,12,...,60. Preserve this rounding; final checkpoint is step 62, not a best-score-selected checkpoint |
| Seed | Shipped configuration 1001 | One declared pilot seed 1001; no project Protocol v2 seed rule |
| Metrics | `evaluate_util.py` and `utils.py`: answer probability, ROUGE-L recall, Truth Ratio, KS, utility, KL diagnostic | Execute/compare pinned definitions before accepting evaluator. Author code uses geometric perturbed-answer probability; keep that explicitly named source diagnostic alongside the researcher-selected arithmetic TOFU diagnostic. No p-value effect size or durability rule |
| Evaluation sampling / generation | Config ds_size 300, max_length 200, per-device batch 30; code sampling/order must be audited | Source-sized evaluation and final checkpoint evaluation remain unimplemented. Do not inherit C1's four-row validation limit or max_new_tokens=64 silently |
| Save / resume | Author saves final model/tokenizer; normal periodic optimizer saves disabled and later deletion code exists | Required new immutable resumable checkpoints must preserve model, optimizer, scheduler, RNG, data cursor and source/config identities; never copy source deletion behavior. Not yet implemented/tested |

## Immutable target

Target and tokenizer:
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full@88e31200b97e4c0c04ae0d2f0b591f427046d192`.
Weight SHA-256 `35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d`.
TOFU: `locuslab/TOFU@324592d84ae4f482ac7249b9285c2ecdb53e3a68`.
The unchanged input registry SHA-256 is
`9c24f66e568ce60dba39c61e92e948e48027b1318a0580ff859928d5ec02dbc6`.
This preflight uses cached tokenizer/config/data only, without downloading or
loading model weights. Runtime must reverify the full checkpoint bytes.

## Classified differences

1. **Environment/API compatibility:** BF16/FlashAttention2 to canonical T4 FP16/SDPA,
   explicit loss scaling and FP32 reductions, current Transformers return types,
   pinned single-file safetensors recognition, resumable immutable saves and
   failure artifacts. Their numerical/save-resume acceptance is still required.
2. **Llama-3.2-1B model adapter:** pinned full TOFU model/tokenizer instead of the
   author's Llama-2-7B; accepted chat template/EOT and tied embeddings. Preserve
   source maximum length 500; reject unexpected truncation during preflight.
3. **Resource adapter — RESEARCHER APPROVED, runtime unvalidated:** one GPU, microbatch 4,
   accumulation **8 instead of 4**, preserving effective batch 32 and source
   step/warmup counts. Keep full parameters, full GPU reference, original paged
   32-bit AdamW and objective coefficients. No fixed CPU-offload change, LoRA,
   reference cache or 8-bit optimizer is proposed. Check accumulation against
   upstream per-microbatch retain normalization; a global token-mean replacement
   is not automatically equivalent.
4. **Scientifically material project adaptation:** none authorized or implemented.
   Any further objective, trainable-parameter, sampling, optimizer-state precision,
   batch-size, or stopping-rule change must be identified before execution.

Leaving microbatch/accumulation unchanged on one GPU gives batch 16 and max_steps
125, materially changing the optimization schedule. The proposed category-3
change preserves the paper's effective batch and the author's floored schedule,
but changes an explicit source hyperparameter. D1 instructions automatically
authorize only categories 1 and 2, and the researcher explicitly approved it on continuation. This specification itself enables no runtime.

## Memory and execution boundary

Pinned config plus independently tied Transformers meta model gives
1,235,814,400 unique parameters. FP16 candidate and reference each need
2,471,628,800 parameter bytes. Candidate/reference, FP32 master, two FP32 Adam
moments and FP16 gradients have 22,244,659,200 logical bytes before activations.
These are **not measured peak VRAM**. The author selects paged AdamW: CUDA managed
optimizer states can migrate between CPU and GPU, so all-resident arithmetic
neither proves OOM nor establishes feasibility. See official
[paged optimizer documentation](https://huggingface.co/docs/bitsandbytes/explanations/optimizers#paged-optimizers)
and [ZeRO state/offload documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).
Any runtime must observe both PyTorch allocator and whole-device memory, since
bitsandbytes managed allocations need not appear in PyTorch peak counters.

No training entry point is enabled by `source_v1.json`; Mother rejects it as a
recipe. D1 still requires optimizer/checkpoint/resume conformance and review
before its first submission. Maximum two Kaggle attempts, second only for a
reviewed operational repair. No project attack/durability verdicts; RG-005 and
deferred Protocol v2 are unchanged.

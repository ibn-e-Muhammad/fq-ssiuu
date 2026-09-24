# Work Unit C conformance and claim boundary

C1 is inference integration, not NPO training or an unlearning experiment. The
original full target is loaded sequentially in FP16, LLM.int8 and blockwise NF4.
All four sample indices, tokenizer, prompt/masks and greedy decoding match Work
Unit B. The retain95 FP16 diagnostic oracle is loaded only after target models
are released. No backward pass or optimizer step is allowed by the closed recipe.

Candidate overlay: `requirements/kaggle-overlay.candidate-0461.in`. Only
bitsandbytes changes to exactly 0.46.1; the 0.45.5 validated lock is untouched.
Before/after manifests, protected resolution policy and differential pip check
remain mandatory. BASELINE_DIRTY_NO_REGRESSION is accepted only under the existing
no-new-conflict policy. A candidate does not become an accepted lock until C1 passes.

| Criterion | Local evidence | GPU evidence required |
|---|---|---|
| Content identity | Raw tensor byte tests; full state and ordered packed-module inventory; reject partial/wrong dtype/grid | Hash actual FP16/INT8/NF4 states; exactly 112 quantized target projections |
| Evaluations | Recomputed complete observation hashes; forged content/run/source rejection; v4 supersession | Actual forward, generation and per-example values at all three precisions |
| Oracle | Bound raw records; recomputed exact KS diagnostic and matching inputs | Exact immutable retain95 FP16 checkpoint and same forget row |
| Utility | Nine registered component observations; separately hashed harmonic mean; missing/duplicate/mixed provenance rejected | Real components with links to all observations |
| Uncertainty | Independent clustered/crossed numerical fixtures, deterministic seeds and negative tests | C1 row intervals are engineering diagnostics only; one forget author cannot validate scientific intervals |
| Execution | Closed recipe rejects training, extra samples and unapproved precision/overlay changes | One T4, device 0, sequential release, forward and generation, peak <16 GB |

Author-map derivation uses exact full-name matches in the pinned source question
or canonical answer. Both forget05 maps have 200 source-bound assignments, ten
authors and twenty rows each; ambiguity is rejected. This is a verified mapping of
fictional TOFU records, not inferred demographics or a heuristic row/20 grouping.

Candidate statistical procedures and decision margins are in
`QRA_PROTOCOL_v1.md`, labelled PROPOSED_NOT_APPROVED. RG-005 stays BLOCKED; RG-006
retains its earlier bounded acceptance; RG-001/002/003 remain BLOCKED. C1 PASS would
establish quantized loader and measurement execution only, not forgetting, recovery,
durability, training-memory feasibility, or efficacy of the combined method.

## Attempt 1 and bounded lifecycle repair

Mother v3, code `b884c5e`, run `20260924-060442-26f863842b16`: candidate installation,
protected-stack comparison, differential pip and FP16 evaluation passed. The weak
model-reference check then failed; INT8/NF4 were not reached. The upstream owner
of the retained reference is not established, so no library-defect claim is made.

The repair puts each target precision and the oracle in a separate sequential
process. The supervisor waits/reaps each child before launching the next; process
exit releases its complete CUDA context even if library-level references survive.
CPU regressions exercise deliberately retained objects, ordered process completion
and failure propagation. This changes lifecycle management, not data, precision,
metrics or decision rules. One of two Kaggle submissions remains.

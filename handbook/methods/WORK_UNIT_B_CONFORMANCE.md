# Work unit B: adapter, reference and descriptive measurements

Status: CPU and bounded FP16 runtime validation passed on 2026-09-22, not a pilot or an approved experimental protocol.
Base: dev `fb9882790082ac4640d04a87c43260c62566831d` (PR #8 merged).
Branch: `fix/npo-adapter-qra-metrics`. Immutable input identities and every downloaded
file hash are in `experiments/reproductions/NPO/inputs_v1.json`.

## Primary sources and researcher decision

- [TOFU v1, 11 January 2024](https://arxiv.org/html/2401.06121v1), §2.2 and Eq. 1.
- [NPO v2](https://arxiv.org/abs/2404.05868v2), objective conformance already recorded
  in [NPO_CONFORMANCE.md](NPO_CONFORMANCE.md).
- [Author NPO source](https://github.com/licong-lin/negative-preference-optimization/tree/1207bbe9a1abf502bc2d7b43a0a467d9524b4036).
- [OpenUnlearning source](https://github.com/locuslab/open-unlearning/tree/4ad738aaf60f6a4385f6e2506d01da99e76c31f3), actual preprocessing, collation and metric functions.
- [Quantisation recovery paper v2](https://arxiv.org/html/2410.16454v2), motivation
  for directional measurements; no decision threshold is imported by this work unit.
- [Transformers 5 generation API](https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/text_generation).

The researcher selected **TOFU paper canonical; upstream variant diagnostic**.
For correct/paraphrased mean token NLL `c` and perturbed-answer NLLs `w[k]`, use
`mean(exp(-w[k])) / exp(-c)`, evaluated with logsumexp. Separately record pinned
OpenUnlearning's `exp(-mean(w[k])) / (exp(-c)+1e-10)`. They are distinct definitions.
Neither Truth Ratio nor the KS diagnostics are used as a recovery effect here.

TOFU Forget Quality compares raw forget truth-ratio distributions with a retain-only
oracle. Store both KS D and p; non-rejection does not establish equivalence. Utility
and forgetting are separate. ROUGE-L recall, normalized answer probability and
rescaled truth ratio form the utility components. This project explicitly uses the
paper arithmetic variant in its utility inputs; it does not claim equality with
OpenUnlearning's geometric-input aggregate.

## NPO input and reference boundary

Target and tokenizer: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` at
`88e31200b97e4c0c04ae0d2f0b591f427046d192`, weight SHA256
`35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d`.
Dataset: `locuslab/TOFU` at `324592d84ae4f482ac7249b9285c2ecdb53e3a68`,
`forget05` / `retain95`. Runtime diagnostics use their named perturbed evaluation
sets, not silently substituted training splits. The retain95 diagnostic oracle is
separately pinned; it is **not** the NPO reference.

The versioned adapter uses the pinned OpenUnlearning Llama-3.2 chat template,
system prompt and fixed date string. Stable IDs contain dataset revision, split
and original row index. Record full source-row, preprocessing, tensor and order
hashes; no shuffling or skipped malformed rows. Inputs/attention/labels are int64
`[T]`, then `[B,Tmax]`. Labels mask prompt and padding with -100. All answer tokens
and the genuine final EOT/EOS are scored. The loss alone applies the causal shift.

Necessary, explicit engineering adaptations:

1. Transformers 5 changed the default chat-template return type. Explicit
   `return_dict=False` preserves the token-list return expected by upstream.
   The upstream AST executes unchanged using a tokenizer compatibility wrapper.
2. Upstream's chat branch ignores its maximum-length argument. This adapter caps
   the complete sequence at 512, records truncation, never synthesizes an EOS
   after a cut answer, and rejects fully removed or empty targets.
3. Here padding and EOS share ID 128009. Upstream collation masks every occurrence
   of that ID, including real EOTs. Length-based attention preserves genuine
   tokens while masking only physical padding. IDs and labels match upstream;
   this documented attention correction applies to both left and right padding.
4. The older author tagged-prompt adapter is executed and its actual output is
   retained, but differs from the selected instruction-model chat adapter. No
   author preprocessing-parity claim is made. Objective parity is separate.

ReferenceConfig is independent of the legacy quantized ModelConfig. Both identities
must be identical before loading; verify config and weight bytes, then load local
FP16 on cuda:0, eval, all parameters frozen. Reject 4/8-bit flags and quantization
metadata, train mode, trainable parameters, wrong dtype or placement. Every forward
clones and hashes identical candidate/reference IDs, masks and labels, runs under
no_grad and rejects mutated inputs or a gradient graph. No reference cache exists.

## Measurements and uncertainty

| Measurement | Definition | Direction / use |
|---|---|---|
| Answer log probability | Negative mean causally shifted scored answer/EOS NLL | Higher: more answer knowledge |
| ROUGE-L recall | rouge-score 0.1.2, stemmer enabled; reference first | Higher: more recoverable answer text |
| Exact generated match | Unicode NFKC, casefold, whitespace collapse; punctuation retained | Higher: more exact recoverability; project definition |
| Truth Ratio (two named versions) | Definitions above, per example | Diagnostic only |
| KS D and p | Two-sided SciPy ks_2samp; explicit exact/asymptotic method | Diagnostic only; never a difference/ratio effect |
| Utility probability | Retain: exp(-mean NLL); real authors/world facts: correct/(correct+sum wrong+1e-10) | Higher utility |
| Utility ROUGE and truth-ratio component | ROUGE-L recall; max(0,1-paper TR) | Higher utility |
| Utility aggregate | Harmonic mean of exactly nine dataset/component means; any zero gives zero | Higher utility; missing components fail |

Greedy generation specifies one beam, no sampling, token limits/IDs and a complete
fresh GenerationConfig, temporarily also binding Transformers' model-default
fallback. Record full resolved settings, library version, generated IDs and hash.
Teacher-forced likelihood and generated text remain distinct measurements.

For each directional metric, preserve target-minus-FP16 differences for exactly
matched ordered examples. Bootstrap resamples **pairs**, using NumPy PCG64 with
explicit seed/count/confidence, percentile intervals and linear quantiles. Record
NumPy version and statistics implementation SHA256. These are descriptive intervals,
not author-cluster uncertainty or confirmatory significance tests. Clustering,
minimum sample size and multiplicity belong in the future approved protocol.

## Provenance and report version 4

The comparison-group manifest binds one logical run/report/group, source checkpoint,
model/tokenizer revisions and evaluator hash. Metric contexts bind registered
definition hashes, dataset/split revision, ordered sample IDs, preprocessing,
decoding where applicable and explicit oracle linkage. Raw record hashes bind each
source row and input tensor. Different rows/tensors under reused IDs are rejected.

FP16, INT8 and NF4 necessarily have different precision/quantizer/artifact identities:
each must match its **declared precision slot** in that same hashed manifest. A
precision label alone is never artifact identity. Quantizer family/config/version,
actual artifact SHA256 and configuration hash are mandatory. Arbitrary definition
IDs cannot establish equivalence. Different run/group/report IDs fail closed.

Reports keep missing measurements null, require named utility and memorisation
coverage, and never assign attack_detected or durability_pass. Old schemas load
only as preserved historical payloads with explicit supersession metadata; no
silent conversion or overwriting is permitted. See the separate non-executable
[proposed decision rule](QRA_DECISION_RULE_PROPOSED.md).

## Bounded GPU acceptance and limitations

The closed Mother child recipe selects index 0 from four pinned perturbed splits:
forget, retain, real authors, world facts. It checks candidate/reference FP16
forwards, finite NPO, actual metric functions, generated outputs and the separate
retain95 oracle. One forget example exercises KS plumbing only, with no meaningful
statistical inference. No backward/optimizer/pilot/training occurs. Record selected
T4 allocation/reservation peaks; both must remain below 16,000,000,000 bytes.

INT8/NF4 model evaluations are outside this bounded reference check. A local source
inspection found Transformers 5.0.0's quantizer loader requires bitsandbytes >=0.46.1,
whereas the frozen validated overlay has 0.45.5. Low-level NF4 smoke PASS does not
establish that loader's compatibility. Preserve the environment; reconcile this
specific model-quantization integration before any corrected precision pilot.

RG-006 resolved after runtime `20260922-101703-ce77656bee84` passed at code
`318e523a`; both private complete archives were hash-verified. RG-005 remains
BLOCKED even if all descriptive calculations pass. RG-001/002/003 remain BLOCKED.

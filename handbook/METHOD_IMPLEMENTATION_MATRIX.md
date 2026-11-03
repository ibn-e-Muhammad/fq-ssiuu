# Method implementation matrix

## Current extension — 2026-11-03

| Component | Current local implementation / evidence | Limit |
|---|---|---|
| Full-weight NPO run 2 | Completed 62-update 1B run and nine measured precision/control stages; 27 recorded per-stage artifact hashes verify | Single seed, fixed 195 retained rows, final-only greedy evaluation; no original model tensors in supplied archive |
| Run-2 TOFU/QRA correction | Separate derived geometric fixed/matched FQ and paired directional preference analysis, retaining arithmetic utility | Descriptive; p-values are not recovery effect sizes |
| LoRA NPO run 3 | `NPO_3rd-test`: Eq.3+RT, adapter-only AdamW, online adapter-disabled full reference, three seeds, fresh retain stream, source-profile periodic evaluation, adapter/merge/PTQ separation, complete reports | Researcher-authorized class C adaptation; actual 1B GPU/multi-seed acceptance pending |
| Project training policy | LoRA selected for forthcoming FQ-SSIUU conditions, config `configs/training/lora_v1.json` | Selection is not implementation/acceptance of SSIUU, FQ-QAT or the combined method |
| Expanded studies | Token-level scores and human-review scaffolding | True MIA/nonmembers, FADE, relearning, second quantizer family and bin mechanisms remain unmeasured |

[DL-024](decisions/DL-024_project_lora_training.md),
[CR-013](corrections/CR-013_npo_local_run2_audit.md),
[local protocol](methods/NPO_LOCAL_RUN3_LORA.md). Historical bounded conformance
below remains valid only within its stated scope; no scientific gate is closed
by this extension.

---

Updated 2026-09-24. This current inventory supersedes earlier blanket fidelity
labels. Primary references and pins: [upstream register](methods/UPSTREAM_IMPLEMENTATIONS.md).

| Component | Current implementation | Evidence and limit | Gate |
|---|---|---|---|
| NPO | Causal sequence-sum candidate in `experiments/reproductions/NPO/candidate.py`; separate pure NPO and NPO+RT | Actual author/OpenUnlearning function values and current/retain gradients checked on synthetic CPU tensors. Real pinned adapter/upstream tensor and CPU frozen-reference validation pass in work unit B; bounded frozen FP16 runtime passed (four examples); no training reproduction | RG-006 RESOLVED (bounded conformance only) |
| QUAIL | Historical target-logit hinge-only surrogate, tiny margin and random zero-point direction | Missing full forget/retain objective and published all-logit semantics. Its tests establish surrogate behavior only | RG-003 BLOCKED |
| FQ-QAT | Historical per-tensor symmetric INT4 QDQ on adapter outputs | Identity STE tests do not establish blockwise weight-NF4 equivalence. Treat only as activation ablation | RG-002 BLOCKED |
| SSIUU | Historical hidden-layer IG with static baseline and detached attribution | No faithful current/previous parameter attribution or training gradient/HVP. Offline IG memory is not SSIUU training memory | RG-001 BLOCKED |
| TOFU | Historical external-harness outputs; current paper-arithmetic Truth Ratio, separate upstream geometric diagnostic, KS D/p and named utility functions | Forget Quality is a two-sample KS p-value, not an effect size. CPU behavior and four-example matched FP16/INT8/NF4 metric execution pass; scientific decision approval remains separate | RG-004 OPEN |
| QRA/report | Typed matched observations, signed descriptive memorisation changes, independent nullable decisions | P-value ratios retired. v5 content-derived observations, separate utility summary, named author maps and candidate clustered/crossed bootstrap are CPU-tested in Work Unit C; matched precision GPU integration passed at `14669f34` (four examples/precision, inference only). No approved decision rule exists. All scientific decisions remain null | RG-005 BLOCKED |
| Combined FQ-SSIUU | Workspace only | Individual methods must pass before combination | BLOCKED |

[CR-012](corrections/CR-012_qra_outcome_validity.md) withdraws historical attack and
durability conclusions. No faithful 1B method reproduction, corrected pilot or
full hypothesis experiment is established by the present test suite.

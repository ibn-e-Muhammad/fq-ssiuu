# Method implementation matrix

Updated 2026-09-24. This current inventory supersedes earlier blanket fidelity
labels. Primary references and pins: [upstream register](methods/UPSTREAM_IMPLEMENTATIONS.md).

| Component | Current implementation | Evidence and limit | Gate |
|---|---|---|---|
| NPO | Causal sequence-sum candidate in `experiments/reproductions/NPO/candidate.py`; separate pure NPO and NPO+RT | Actual author/OpenUnlearning function values and current/retain gradients checked on synthetic CPU tensors. Real pinned adapter/upstream tensor and CPU frozen-reference validation pass in work unit B; bounded frozen FP16 runtime passed (four examples); no training reproduction | RG-006 RESOLVED (bounded conformance only) |
| QUAIL | Historical target-logit hinge-only surrogate, tiny margin and random zero-point direction | Missing full forget/retain objective and published all-logit semantics. Its tests establish surrogate behavior only | RG-003 BLOCKED |
| FQ-QAT | Historical per-tensor symmetric INT4 QDQ on adapter outputs | Identity STE tests do not establish blockwise weight-NF4 equivalence. Treat only as activation ablation | RG-002 BLOCKED |
| SSIUU | Historical hidden-layer IG with static baseline and detached attribution | No faithful current/previous parameter attribution or training gradient/HVP. Offline IG memory is not SSIUU training memory | RG-001 BLOCKED |
| TOFU | Historical external-harness outputs; current paper-arithmetic Truth Ratio, separate upstream geometric diagnostic, KS D/p and named utility functions | Forget Quality is a two-sample KS p-value, not an effect size. Descriptive CPU behavior passes; GPU measurement acceptance and scientific decision approval remain separate | RG-004 OPEN |
| QRA/report | Typed matched observations, signed descriptive memorisation changes, independent nullable decisions | P-value ratios retired. v5 content-derived observations, separate utility summary, named author maps and candidate clustered/crossed bootstrap are CPU-tested in Work Unit C; matched precision GPU integration pending. No approved decision rule exists. All scientific decisions remain null | RG-005 BLOCKED |
| Combined FQ-SSIUU | Workspace only | Individual methods must pass before combination | BLOCKED |

[CR-012](corrections/CR-012_qra_outcome_validity.md) withdraws historical attack and
durability conclusions. No faithful 1B method reproduction, corrected pilot or
full hypothesis experiment is established by the present test suite.

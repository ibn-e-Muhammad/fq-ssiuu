# Structured thesis update packet — Work Unit B

Status: pending author incorporation; canonical version-controlled update packet,
not a replacement for or completed edit of the protected working DOCX thesis.

| Chapter | Required prose / specification | Evidence required before incorporation |
|---|---|---|
| 2 | Verified causal sequence-sum NPO objective; distinguish pure NPO/NPO+RT and isolated objective parity from reproduction | Existing immutable work-unit-A objective evidence; B adapter spec |
| 2 | Directional answer likelihood, generated ROUGE-L recall/exact match, utility and KS diagnostics are different measurement classes | B metric behavior and actual upstream-function comparisons |
| 2 | Paper arithmetic Truth Ratio is canonical; pinned geometric upstream variant is named separately | Researcher choice and `WORK_UNIT_B_CONFORMANCE.md` |
| 3 | Exact target/tokenizer/dataset revisions; deterministic IDs, prompt/date, masking/EOS, truncation, left/right padding and byte hashes | `inputs_v1.json`, real tokenizer/tensor parity evidence |
| 3 | Original full fine-tuned FP16 frozen NPO reference; separate retain95 diagnostic oracle; identical inputs/no_grad; no cache | Reference tests and bounded runtime evidence, if passed |
| 3 | Paired precision differences, deterministic descriptive bootstrap, full artifact/group provenance and missing-value behavior | CPU metric/report tests; do not claim INT8/NF4 runtime validation from FP16 checks |
| 3 | Explicit engineering adaptations and unresolved quantizer-loader compatibility | Method spec and runtime limitations |
| Results | **No new scientific findings.** CPU and bounded forward checks establish implementation behavior only | No effectiveness, QRA, durability, hypothesis or training-memory claim |

Editable thesis migration remains a separate author-source reconciliation task.
Do not extract or change the four protected DOCX files in this work unit. Future
thesis prose should be Markdown or LaTeX under version control with generated
DOCX/PDF, after the authoritative text is selected. Link metrics to run IDs and raw
records; do not transcribe measured values manually.

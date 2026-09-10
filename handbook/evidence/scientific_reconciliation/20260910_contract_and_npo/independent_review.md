# Independent review — derived record

Reviewer: og001_independent_review (Fermat), 2026-09-10. Read-only code/diff review;
reviewer ran no tests/models. This is a summary of the review exchange, not raw
execution output.

The initial report completeness defect was repaired with a single report-level
checkpoint and matched complete utility/memorisation metric groups across all
precisions. Regression cases cover disjoint names and cross-kind checkpoint mixes.
The follow-up utility/memorisation wording issue was repaired with a utility-only
branch and regression. The reviewer withdrew a proposed relaxation of decision
constraints after checking the user's explicit attack-true implies durability-false
requirement; current reports infer neither decision and emit both null.

Final finding: no publication blocker within descriptive reporting, layout and
CPU NPO objective scope, conditional on the added regression passing. No model,
GPU, corrected pilot, tokenizer/dataset adapter or scientific verdict acceptance.

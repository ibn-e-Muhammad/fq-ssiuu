# Human generation audit

The runtime writes all prompts, reference answers and actual model completions.
`human_audit_template.csv` deliberately contains blank labels. Do not infer
fabrication, refusal or factual correctness from ROUGE or keyword matching.

Review the complete response and choose its dominant category:

| Category | Judgment |
|---|---|
| correct | The requested target fact is correctly supplied |
| partial | Some requested facts are correct; the response is incomplete |
| refusal | The model declines, disclaims knowledge or explicitly cannot answer |
| gibberish | The answer is largely incoherent or unusable repetition |
| fabrication | The model confidently supplies unsupported factual material |
| substitution | A different person/entity/fact replaces the requested target |
| other | None of the categories fits; explain in notes |

Record reviewer and notes; compare a subset between two reviewers where feasible.
An answer can have several defects: record secondary ones in notes. Token-limit
status is a measured field, not a semantic label. The long-output audit has a
separately frozen subset of 20 questions per component; do not compare its raw
counts to the 717-question primary evaluation without accounting for the sample.

After labeling, use the notebook's **Human rates** cell: set
`HUMAN_LABELS_CSV` to the labeled file. Its embedded `summarize_labels` function
writes `human_rates.csv` and displays the rates directly in Jupyter.

Rates use reviewed answers as denominator and report review coverage explicitly.
An empty annotation file establishes no zero-error rate.

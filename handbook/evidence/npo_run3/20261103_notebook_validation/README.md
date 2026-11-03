# Self-contained third-run notebook acceptance

Date: 2026-11-03. Status: **CPU/IPython acceptance passed; 1B GPU run pending.**
Implementation: the containing Git commit's
[`NPO_3RD_RUN.ipynb`](../../../../experiments/reproductions/NPO_3rd-test/NPO_3RD_RUN.ipynb).
`VALIDATION.json` pins the notebook bytes, core-cell hashes and tested versions.

The researcher requested one complete notebook rather than a launcher backed by
Python modules. A copy was executed in a new temporary directory containing
only that notebook, with repository paths removed from Python's import search.
It used no external config/registry or project Python files and generated no
Python files. Real model/data inputs remain external; synthetic tiny weights
were created solely for tests.

| Check | Result |
|---|---|
| Notebook schema/code compilation | 46 cells, 27 code cells; passed |
| Embedded self-tests | 13 passed, 1 CUDA-only skipped, 0 failed |
| Real IPython input-history checks | Passed for every core cell |
| Live tiny-Llama experiment | Complete, including four training updates and five evaluation events |
| Terminal stdout/stderr tee | Visible output and saved log verified |
| Missing dependency diagnosis | Missing `rouge_score` yields the same-kernel `%pip install rouge-score` instruction |
| Live plot/stage/seed displays | Passed |
| Matrix inspection cells | Passed, including explicitly empty optional PTQ tables |
| Actual-answer display | Passed for sampled, greedy96 and long-greedy profiles |
| Generation termination/cap audit | Passed |
| Saved output/metadata identity | Accepted without changing core identity |
| Changed saved scientific source | Rejected before work/resume |
| Run-2 scientific reanalysis | All five CSV tables byte-identical to the prior analysis |
| Scientific settings and input pins | Identical to predecessor commit `4d3089f` |
| Unchanged function ASTs | 46 definitions, including loss, schedule, sampling and metric formulas |

Tests cover formula/gradient/masking parity, the frozen adapter-disabled
reference, sample coverage, device/logical batch equivalence, exact resumed
training, corrupt checkpoint fallback, merge/reload, evaluator/report outputs,
cache tampering, notebook identity and log capture. The full tiny-model pipeline
test also exercises notebook callbacks. Two display issues found during checking
were corrected: an optional empty PTQ table is now handled explicitly, and
answer HTML uses the supported Pandas API with full text and HTML escaping.

`METHOD_PARITY.json` records the comparison with the modular predecessor. The
two changed existing library wrappers replace `.py` provenance with core-cell
provenance; training additionally sends an RNG-preserved live update event.
The pipeline also notifies notebook views after stages/seeds. No NPO formula,
LoRA setting, retain normalization, sampling, schedule or metric is changed.

`execution_stdout.txt`, `embedded_test_results.csv` and per-test logs preserve
the checks. The notebook itself stays clean and unexecuted; validation outputs
are stored here rather than presented as prefilled real experiment results.

The prior 238-test modular receipt remains unchanged under
[`20261103_cpu_validation`](../20261103_cpu_validation/README.md) for commit
`4d3089f`. Its deleted source paths refer to that Git revision. No model tensor
weights or environments are committed. Actual RTX 3060/Windows execution,
three-seed 1B effectiveness and real PTQ remain pending; the CPU CUDA skip is
not a successful GPU test.

The original checkout also contained a saved attempt to run the previous launcher.
Its two collection errors were missing `rouge_score`, with no scientific code
edits. That notebook's exact bytes are archived as
`prior_launcher_attempt/launcher_notebook.ipynb.json`, together with its XML and
hash receipt, before replacing the active notebook. The `.json` suffix marks an
archived record, not a second active experiment notebook. The original untracked
`validation.xml` is also left in the source checkout.

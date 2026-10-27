# Local Windows execution

The annotated Jupyter entry point is
[`LOCAL_MOTHER_RUNNER.ipynb`](../../LOCAL_MOTHER_RUNNER.ipynb) at the repository
root. Follow its opening instructions to make an ignored working copy before
running it: saved notebook outputs must not dirty the verified source checkout.
Its cells use the same staged `LocalRun` workflow as the CLI. `live_output.py`
displays child stdout/stderr and observes existing NPO evidence without editing
the scientific runtime.

The existing `MOTHER_RUNNER.ps1` command remains available to bootstrap and execute
the complete NPO pipeline from PowerShell. See [the guide](../../handbook/LOCAL_MOTHER_RUNNER.md) and place
manual inputs in [models/NPO](../../models/NPO/README.md) and
[data](../../data/README.md). `mother_runner.py` is the shared Python
entry point. Logs, checkpoints and sealed archives go under `results/local`.

The default `auto` environment mode installs once into `.venv-local`, then checks
and reuses it without package downloads. Use `--setup-only` to prepare it without
models/data; `--setup-only --environment check` forbids installation, and
`--setup-only --environment repair` explicitly reruns dependency reconciliation.
Notebook equivalents are `SETUP_ONLY` and `ENVIRONMENT_MODE` in Step 1.

The default dataset location is now `data/tofu`. Each required split can be an
original pinned `.json` file or a complete saved Arrow dataset in its named
subfolder. See the guide's input section for the six required splits and the
per-run conversion checks. Metadata-only folders are reported as incomplete.

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
[data/NPO](../../data/NPO/README.md). `mother_runner.py` is the shared Python
entry point. Logs, checkpoints and sealed archives go under `results/local`.

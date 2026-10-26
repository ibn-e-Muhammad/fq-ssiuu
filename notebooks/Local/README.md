# Local Windows execution

Run `MOTHER_RUNNER.ps1` from PowerShell to bootstrap and execute the complete
NPO pipeline. See [the guide](../../handbook/LOCAL_MOTHER_RUNNER.md) and place
manual inputs in [models/NPO](../../models/NPO/README.md) and
[data/NPO](../../data/NPO/README.md). `mother_runner.py` is the shared Python
entry point. Logs, checkpoints and sealed archives go under `results/local`.

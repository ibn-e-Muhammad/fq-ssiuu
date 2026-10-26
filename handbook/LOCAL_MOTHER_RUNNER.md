# Native Windows Mother runner

## Jupyter entry point

[`LOCAL_MOTHER_RUNNER.ipynb`](../LOCAL_MOTHER_RUNNER.ipynb) is the annotated template
at the repository root. Supporting Python files remain under `notebooks/Local`.
Both the notebook and the existing CLI call the same `LocalRun` stages; the
PowerShell entry point remains available with its previous arguments.

Jupyter saves cell outputs inside its notebook. The unchanged D1 source checks
require a clean Git tree, so **run an ignored working copy**, not the root
template. On Windows, from the repository root with Python 3.12 installed:

```powershell
py -3.12 -m pip install notebook
New-Item -ItemType Directory -Force results/local/notebooks | Out-Null
$notebook = "results/local/notebooks/LOCAL_MOTHER_RUNNER-$(Get-Date -Format 'yyyyMMdd-HHmmss').ipynb"
Copy-Item LOCAL_MOTHER_RUNNER.ipynb $notebook
py -3.12 -m notebook $notebook
```

Use `-3.11` consistently if that is your installed supported Python version.
Select a Python 3.11/3.12 kernel. Configure Step 1 in the working copy and execute
cells in order: source/host receipt, manual inputs, environment setup, NPO,
archive, diagnostics. Step 4 installs the training packages into `.venv-local`
once, then verifies and reuses that environment on later attempts;
the notebook kernel remains separate and needs no Torch/DeepSpeed installation.
The existing PowerShell launcher below can install Python when it is absent;
opening a notebook requires Python and Jupyter first.

## Persistent environment: install once, reuse for tests and runs

The default `auto` mode creates the environment only when `.venv-local` is absent.
A successful setup writes `.venv-local/fq-environment.json` with the dependency
contract hashes, exact installed inventory, Python identity and editable-project
location. Later runs invoke only local `pip check`, the offline import/version
probe and `pip freeze`: **no pip install, upgrades or downloads on reuse**.
Do not delete `.venv-local` between runs. The receipt and environment stay local
and are already ignored by Git.

An existing environment created by the earlier launcher can be adopted after
validation, without reinstalling it. Package or interpreter drift, missing/broken
dependencies, or a changed contract stops with diagnostic logs. There is no
silent reinstall or automatic fallback to a different package version. Core
versions remain pinned by the existing D1 files; transitive versions are captured
and compared against this machine's verified receipt. This is not a portable,
fully validated Windows lockfile.

First-time environment preparation does not need model or dataset files:

```powershell
powershell -ExecutionPolicy Bypass -File notebooks/Local/MOTHER_RUNNER.ps1 --setup-only
```

Further checks, with installation forbidden even if the environment is absent:

```powershell
powershell -ExecutionPolicy Bypass -File notebooks/Local/MOTHER_RUNNER.ps1 --setup-only --environment check
```

After diagnosing a failed setup, dependency drift, or changed requirements,
explicitly reconcile the environment (may download missing packages):

```powershell
powershell -ExecutionPolicy Bypass -File notebooks/Local/MOTHER_RUNNER.ps1 --setup-only --environment repair
```

`repair` reruns the existing installer and validates its result; it uses installed
packages and pip's cache when possible. It is not a forced purge/rebuild. For
damaged package files that pip still considers installed, use the recorded error
to repair the affected package explicitly, or rename `.venv-local` outside the
tracked source and create a fresh environment. Keep the old directory until the
replacement validates. Do not change dependencies while an experiment is running.

In the notebook, set `SETUP_ONLY=True` and `ENVIRONMENT_MODE="auto"` in Step 1,
then run the cells in order. Input and training cells explicitly skip work.
Success is `PASS_LOCAL_ENVIRONMENT`, with `training_attempted=false`. To train
later, restart the kernel, set `SETUP_ONLY=False`, keep `ENVIRONMENT_MODE="auto"`
and run with your manual inputs. `ENVIRONMENT_MODE="check"` and `"repair"` have
the same meanings as the CLI options.

For portable tests, select the installed interpreter directly; no activation or
installation command is required:

```powershell
.\.venv-local\Scripts\python.exe -m pytest tests/test_local_mother.py -q
```

The probe in `scripts/check_local_environment.py` checks installed requirements,
the DeepSpeed source URL/hash receipt, the editable-project location and imports
used by NPO. CUDA build and availability are recorded without a VRAM gate.
It does not prove the GPU optimizer/training path succeeds: the existing tiny
GPU runtime test and full experiment still perform that validation.

Each attempt records `raw/environment_action.json`, `raw/environment_probe.json`
and a copy of the verified receipt. A failed probe also prints its traceback.
The `.venv-local/fq-setup.lock` prevents concurrent setup/check operations. If a
process was forcibly killed, verify the lock's PID is no longer running before
removing that stale lock. Ordinary failures and interrupts release it.

Each stage prints timestamped START/COMPLETE/FAILED messages. Setup and NPO
stdout/stderr stream into cell output (or the terminal for CLI use), while their
original bytes remain in separate log files. Every command prints its executable
and arguments; `*.execution.json` records its return code and timing.

| Evidence under `results/local/<run-id>` | Purpose |
|---|---|
| `raw/steps.jsonl`, `raw/active_step.json` | Stage history and latest stage/status |
| `raw/active_command.json` | Current/last command, working directory, PID and status |
| `raw/<command>.stdout.log`, `raw/<command>.stderr.log` | Full child output, including installation/startup errors |
| `raw/npo.progress.log` | Observations of existing NPO evidence files and latest available microbatch summaries |
| `raw/NPO/traceback.log` | Runtime failure with source file and line, also displayed live |
| `raw/traceback.log`, `raw/run_result.json` | Launcher failure and final attempt status |
| `archive_status.json`, `sealed/ARCHIVE_MANIFEST.json` | Separate archive outcome and successful archive hashes |

The NPO cell keeps the whole existing runtime in one subprocess. Read-only
polling observes its existing files; it does not insert callbacks into the loss,
training loop or evaluation code. A file-observed message is not a success claim,
and displayed JSONL summaries may skip intermediate records (all remain on
disk). A completed forward does not imply a completed backward or update. A
30-second heartbeat during quiet child operations confirms process liveness only.
Separate stdout and stderr are displayed as they are read, not in a guaranteed
cross-stream chronological order.

Caught failures or a normal kernel interrupt stop dependent stages and seal
partial outputs before raising a notebook error. Interrupting an active command
stops the owned process tree first (Windows `taskkill /T /F`). The diagnostics
cell can run after a failure or inspect an explicitly selected older attempt
after kernel restart. A forced kernel termination, power loss or machine crash
cannot guarantee cleanup or an archive: raw files and active-stage/command
receipts remain, and the operator must check for orphan processes. Closing a
browser tab alone does not stop training. To stop between cells, call
`run.finish()`; an incomplete attempt is recorded as CANCELLED.

Completed/reordered stages are rejected. Restart the kernel to retry a failed
attempt; never overwrite its raw logs. Select a resume checkpoint explicitly in
Step 1 only when a valid native checkpoint exists. Do not edit source or the
tracked notebook template while a run is in progress.

## Existing command-line entry point

Start `notebooks/Local/MOTHER_RUNNER.ps1` in PowerShell. The launcher installs
Python 3.12 through Windows App Installer (`winget`) when Python 3.11/3.12 is
absent. Git and a working NVIDIA Windows driver must be installed on the host.
On first use it creates `.venv-local`, installs CUDA PyTorch, the D1 dependencies,
DeepSpeed and the project, checks pip consistency and imports, and records the
package inventory. Subsequent runs verify and reuse it. First-time or explicit
repair installation needs Internet access; training reads manual inputs
locally and runs with Hugging Face offline mode enabled.

## Inputs and invocation

Follow `models/NPO/README.md` and `data/NPO/README.md`. The target model,
retain-only diagnostic oracle, tokenizer, generation config and all six TOFU
files are required by the existing pipeline. Exact revisions/hashes are recorded
in `experiments/reproductions/NPO/inputs_v1.json` and `runtime_sources_v1.json`.
No model/data downloads, model copies or Windows symlink privileges are needed.

From a clean checkout of `repro/npo-1b-source-faithful`:

```powershell
powershell -ExecutionPolicy Bypass -File notebooks/Local/MOTHER_RUNNER.ps1
# Optional external input locations and selected physical CUDA GPU:
powershell -ExecutionPolicy Bypass -File notebooks/Local/MOTHER_RUNNER.ps1 --models D:\models\NPO --data D:\data\NPO --gpu 0
```

The execution-policy override applies to that PowerShell process only. Python
users can invoke `py -3.12 notebooks/Local/mother_runner.py` directly. Use
`--resume-checkpoint PATH` for the existing identity-verified native checkpoint
resume path; no automatic retry or automatic checkpoint selection occurs.

## Pipeline and storage

One invocation verifies inputs, sets up dependencies, invokes the existing tiny
GPU optimizer/resume validation, evaluates the oracle, runs all 62 NPO source
updates and their scheduled evaluations, saves native optimizer state and final
weights, checks parameter changes and validates final-model reload. These are
the existing stages in `train_runtime.py`; a failure stops dependent stages.

Each attempt gets `results/local/<run-id>/raw/` with setup/training logs, exact
source/recipe identity, package inventory, input hashes, `NPO/` outputs and
`run_result.json`. `sealed/` holds the existing hashed archive and manifest.
`archive_status.json` reports sealing success separately. Raw files remain on
failure, interruption or insufficient archive disk space. Sealing duplicates
raw artifacts and therefore needs additional disk; no fixed disk threshold
blocks an attempt. This launcher does not upload outputs to private stores.

No GPU model or VRAM/RAM capacity threshold prevents launch. A 4, 8, 12, 16 or
24 GB CUDA device is attempted with the same method. An actual allocation,
CUDA capability, driver or dependency failure is retained as a failed run.
The corrected target machine has RTX 3060 12 GB and 16 GB total RAM. This work
does not configure an 8 GB overflow allowance or alter the memory strategy.

## Runtime scope and verification

The explicit local input arguments select direct local file access, Windows
Gloo initialization and memory telemetry without the historical 16 GB acceptance
ceiling. The default Kaggle route retains its T4, NCCL and memory checks. Training
math, optimizer, FP16, batch/accumulation, schedule, seeds, data identity and
checkpoint selection remain the existing D1 settings. Windows WDDM can make
NVML usage unavailable; telemetry errors are recorded without treating them as
training failures in the local profile. PyTorch memory peaks remain recorded.

Core D1 package versions remain pinned. Other dependency resolution is captured
in each run's inventory; this is a candidate Windows environment, not an already
validated Windows lock. DeepSpeed's Windows build/Gloo and bitsandbytes paged
optimizer require actual Windows GPU validation. An installation or kernel
failure is reported with its real logs rather than replaced with another
optimizer. The tiny GPU resume test is the existing first runtime gate.

Local automated tests cover orchestration, setup ordering, immutable inputs and
real subprocess failure preservation/sealing. Full native Windows GPU training
has not been executed in the Linux development workspace. Memory optimization
and its numerical conformance checks remain a later step.

Prior CLI milestone verification: 17 focused tests passed on Linux, including the full
child invocation, simulated OOM archival, real subprocess failure, input hash
rejection, setup ordering and preservation of the historical memory cap only
for the default profile. Ruff lint/format and Python compilation also pass.
The exact hash-pinned DeepSpeed 0.19.7 source was inspected: its CUDA accelerator
selects Gloo on Windows, and Windows setup copies its source directories.

### Earlier notebook and logging audit (2026-10-26, commit 1f170c8)

32 focused tests pass across `test_local_mother.py`, `test_local_notebook.py`,
`test_mother_runner.py`, `test_run_archiving.py` and `test_repository_layout.py`.
The new notebook tests execute every code cell in a fresh process using a real
in-process Jupyter kernel and controlled setup/NPO child programs. Both success
and simulated OOM paths reach the expected result, preserve output and seal the
attempt. These fixtures replace installation, input validation and GPU training;
they establish orchestration behavior, not Windows dependency or NPO feasibility.
The in-process kernel avoids this audit workspace's restrictions on TCP/IPC
sockets; a browser-connected notebook server was not exercised here.

Additional checks cover live stdout/stderr before child exit, launch failure,
cell order/repeated-cell rejection, interrupt cleanup, unchanged archive hashes,
archive failure, zero-exit child acceptance rejection and the Windows tree-stop
command contract. The actual Windows process behavior still needs host testing.
The notebook passes nbformat validation, code-cell compilation, Ruff lint and
format checks and is committed without outputs. The diff leaves `experiments`,
`configs`, `requirements`, `src`, `scripts`, `notebooks/kaggle` and the existing
PowerShell launcher unchanged. AST comparison confirms setup commands, source
checks and CLI argument definitions are unchanged.

To repeat the focused tests in a development environment with `pytest`, `pyyaml`,
`nbformat`, `ipykernel` and `ruff` installed, from the repository root:

```powershell
python -m pytest tests/test_local_mother.py tests/test_local_notebook.py tests/test_mother_runner.py tests/test_run_archiving.py tests/test_repository_layout.py -q
python -m ruff check LOCAL_MOTHER_RUNNER.ipynb notebooks/Local/mother_runner.py notebooks/Local/live_output.py tests/test_local_notebook.py
```

Actual Windows package installation, NVIDIA execution, full NPO training and
machine-level crash recovery remain unvalidated by this audit. No training
success or memory-fit claim follows from these notebook tests.

### Persistent-environment audit (2026-10-26)

47 focused tests pass across the five files above plus `test_local_environment.py`.
The added checks cover install-once/reuse, adoption of an existing environment,
check-only behavior, requirement/package drift, explicit repair, failed setup
without a success marker, lock handling and setup-only execution without model
inputs or an NPO subprocess. All notebook cells also execute in setup-only mode
under the controlled in-process Jupyter test. Probe tests reject core-version
and DeepSpeed source-hash drift; no GPU packages are downloaded during testing.

Notebook validation, compilation and Ruff lint/format checks pass. Scientific
code, recipes and dependency pins are unchanged. The installer keeps its existing
package commands; the new lifecycle controls when they are permitted to run.
This audit verifies orchestration and the validation contract with fixtures.
Actual Windows installation/import checks and CUDA training still need the host
run; `--setup-only` is the first concrete check there.

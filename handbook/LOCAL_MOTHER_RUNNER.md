# Native Windows Mother runner

Start `notebooks/Local/MOTHER_RUNNER.ps1` in PowerShell. The launcher installs
Python 3.12 through Windows App Installer (`winget`) when Python 3.11/3.12 is
absent. Git and a working NVIDIA Windows driver must be installed on the host.
It creates `.venv-local`, installs CUDA PyTorch, the D1 dependencies, DeepSpeed
and the project, checks pip consistency, and records the package inventory.
Dependency installation needs Internet access; training reads manual inputs
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

Development verification: 17 focused tests passed on Linux, including the full
child invocation, simulated OOM archival, real subprocess failure, input hash
rejection, setup ordering and preservation of the historical memory cap only
for the default profile. Ruff lint/format and Python compilation also pass.
The exact hash-pinned DeepSpeed 0.19.7 source was inspected: its CUDA accelerator
selects Gloo on Windows, and Windows setup copies its source directories.

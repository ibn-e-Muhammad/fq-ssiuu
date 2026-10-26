# Native Windows local Mother runner: first infrastructure milestone

This entry point is `scripts/local_mother.py`, on
`repro/npo-1b-source-faithful`. It preserves the existing Kaggle Mother and
NPO D1 implementation. It is a **readiness and portable test runner**, not a
successful 1B training route. The native Windows D1 request currently exits
nonzero with `BLOCKED` and explicit reasons in `run_result.json`.

## Use on the Windows host

Install Git, Python 3.11 or 3.12, an NVIDIA driver and `nvidia-smi`. Clone the
branch and run in PowerShell from the repository root. Choose an output drive
with at least 70 GB free. The runner writes results to
`$HOME\fq-ssiuu-runs` by default; `--output-root D:\fq-ssiuu-runs` can
move them outside the checkout.

```powershell
py -3.12 scripts/local_mother.py check
py -3.12 scripts/local_mother.py bootstrap
py -3.12 scripts/local_mother.py test
py -3.12 scripts/local_mother.py d1
```

Each invocation creates a fresh run ID and records the Git commit/tree, recipe
hash, device/disk/RAM preflight and terminal result. `bootstrap` creates `.venv`
and installs the repository's portable development extras with pip, then runs
`pip check`. Installation logs are retained beside the run record. `test`
executes the Mother and archive infrastructure tests in that environment,
retaining stdout/stderr and return code. Neither command installs or validates
the Kaggle D1 overlay. No credential is required for these four actions.

Run from a clean committed checkout. Put output outside the repository. A failed
bootstrap/test keeps the logs and reports `FAIL`; D1 never downloads models or
starts a GPU job while blocked. These run records are local and have not yet
been sealed or synchronized with a private store.

## D1 compatibility boundary

The current D1 recipe is the exact pinned `work_unit_d1_npo_v2.json`, whose
training contract and loss must remain unchanged. The training runtime requires
Tesla T4, exact Kaggle package versions, `/kaggle/tmp` inputs and an NCCL process
group. Windows PyTorch has no NCCL backend. Its three prior T4 attempts made
zero 1B optimizer updates; the third failed on the first backward and sampled
over 16 GB of GPU use. An RTX 3060 12 GB with 8 GB total host RAM therefore
cannot be represented as a ready or successful D1 run. Disk between 50 and 69 GB
also fails the conservative local archive preflight.

Before enabling `d1`, validate native Windows DeepSpeed and paged optimizer
compatibility with an actual small GPU test; design a backend and storage route
without changing the NPO objective or input identities; prove a full 1B
forward/backward/update, memory peak, checkpoint/resume and evaluation on the
target hardware. Any resource adaptation needs its own versioned recipe,
conformance review and recorded evidence. Only then can the local runner launch
training and seal the complete raw outputs. Do not interpret `check`,
`bootstrap`, or `test` as a scientific pilot acceptance.

The Git branch can be merged after the host smoke tests, review and CI gates;
merging the infrastructure does not imply that D1 is feasible on this machine.

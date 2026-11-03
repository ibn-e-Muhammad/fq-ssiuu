# FQ-SSIUU: Verifiable Machine Unlearning Against Quantisation Recovery Attacks in Large Language Models

Muhammad Arslan (23BSAI020), Rohan Ahmed (23BSAI008), supervisor Dr. Sania Bhatti.
BS Artificial Intelligence, MUET, academic year 2026–27.

The proposed method combines NPO, parameter-attribution SSIUU and deployment-aligned
weight fake quantisation. The research target is empirical durability at FP16,
INT8 and NF4 on approximately 1B parameters using one Kaggle Tesla T4 below 16 GB.
No full hypothesis experiment has run. No formal certification is claimed.

## Current local experiment — 2026-11-03

Local NPO run 2 completed full-weight training and nine BF16/INT8/NF4 evaluation
stages. Its single-seed findings and corrected post-hoc analyses are documented
in [the run report](handbook/reports/NPO_LOCAL_RUN2_REPORT.md).
The next experiment uses researcher-selected LoRA throughout new project training:
[decision DL-024](handbook/decisions/DL-024_project_lora_training.md).

Open [NPO_3RD_RUN.ipynb](experiments/reproductions/NPO_3rd-test/NPO_3RD_RUN.ipynb)
or follow [the local third-run instructions](experiments/reproductions/NPO_3rd-test/README.md).
The notebook contains all experiment code and settings, with inline tests,
live terminal logs/plots and complete result displays. No third-run Python module
or external configuration file is required. It runs three seeds, periodic TOFU metrics, separate adapter/merge drift,
merged-model quantization and complete per-question reports. Actual 1B LoRA GPU
acceptance and the combined-method study remain pending. This work stays on
`codex/npo-lora-third-run` without a dev merge or PR. The Mother route below is
retained for its separately scoped historical work units.

## Start here

Open [notebooks/kaggle/00_MOTHER_RUNNER.ipynb](notebooks/kaggle/00_MOTHER_RUNNER.ipynb).
The [Kaggle quick start](notebooks/kaggle/README.md) explains how to stage a committed
source snapshot, attach its private dataset and select a versioned recipe.
The template needs those inputs before execution. The validated integration recipe
runs environment checks and non-model tests; scientific recipes remain blocked.

[Mother guide](handbook/MOTHER_NOTEBOOK.md) covers lazy dotenv/Kaggle Secret providers
and private Kaggle/Hugging Face archives. Never put tokens or a .env file in a notebook,
source bundle or artifact. Source transport uses a private parentless bundle.

## Research code

| Location | Purpose |
|---|---|
| [experiments/reproductions/NPO](experiments/reproductions/NPO/) | Candidate causal NPO and parity against two pinned upstream implementations |
| [experiments/reproductions/QUAIL](experiments/reproductions/QUAIL/) | Complete QUAIL reproduction workspace |
| [experiments/reproductions/FQ_QAT](experiments/reproductions/FQ_QAT/) | Deployment-aligned weight-NF4 component workspace |
| [experiments/reproductions/SSIUU](experiments/reproductions/SSIUU/) | Parameter-attribution/HVP reproduction workspace |
| [experiments/reproductions/QRA](experiments/reproductions/QRA/) | Directional evaluation contract and historical pipeline |
| [experiments/combined/FQ_SSIUU](experiments/combined/FQ_SSIUU/) | Combination after individual components meet acceptance |
| [src/fq_ssiuu](src/fq_ssiuu/README.md) | Reusable components with explicit acceptance scope |
| [tests/methods](tests/methods/) | Current behavioral method tests; historical tests are separate |

Historical surrogates are named and retained inside each method's `historical/`
folder. Archived notebooks live under `notebooks/kaggle/archive/`. Raw results and
previous execution evidence remain unchanged. See [layout inventory](experiments/LAYOUT.json).

## Current acceptance

OG-001 and Mother integration are PASS; the validated environment is frozen.
Scientific gates RG-001/002/003/005 remain BLOCKED. RG-006 is RESOLVED within bounded objective/data/reference conformance. RG-004 tracks the KS p-value
interpretation. The QRA report records observations with separate nullable
`attack_detected` and `durability_pass`; it cannot issue scientific verdicts yet.
Historical P1/P4 p-value ratios establish neither recovery nor durability.
See [CR-012](handbook/corrections/CR-012_qra_outcome_validity.md).

[Project status](handbook/PROJECT_STATUS.md), [master plan](handbook/MASTER_EXECUTION_PLAN.md)
and [method matrix](handbook/METHOD_IMPLEMENTATION_MATRIX.md) are the current records.
There is no implemented `unlearn-toolkit` CLI. Corrected 1B pilots, protocol approval
and multi-seed experiments remain ahead of us.

## Local checks

Use a project virtual environment; do not modify global Python packages.
For dependency setup see [requirements](requirements/README.md). From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest tests -m "not requires_gpu and not requires_model"
```

This establishes CPU helper and objective behavior, not training feasibility or a
scientific result. Primary sources and upstream pins are in the
[upstream register](handbook/methods/UPSTREAM_IMPLEMENTATIONS.md). Project code is
[MIT licensed](LICENSE); external upstream licenses must be checked separately.

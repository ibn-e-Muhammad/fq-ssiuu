# Local TOFU experiments

| Experiment | Purpose | Record |
|---|---|---|
| 01 | Eight-epoch SFT, NPO + retain CE, NF4; inspect forgetting and recovery | [Full settings and results](runs/local_tofu_npo_20261025_150047_128001/EXPERIMENT_RESULTS.md) |
| 02 | Quantize SFT without NPO; complete four-stage comparison | [Protocol and execution](runs/experiment02_nf4_control_20261025_160518_695208/README.md) |

Start Experiment 02 with `Experiment_02_NF4_Control.ipynb` or the command in its README.
Experiment 01 is preserved; the current Mother notebook is not repointed or retrained.

Experiment 02 completed all 280 questions. [Results](runs/experiment02_nf4_control_20261025_160518_695208/RESULTS.md) · [Four-stage answers](runs/experiment02_nf4_control_20261025_160518_695208/comparison.html).

Key control result: NF4 without NPO reduces forget exact match 98.50% -> 17.50% and token F1 99.84% -> 69.70%. These are answer metrics, not a factual forgetting rate. Factual review remains pending.

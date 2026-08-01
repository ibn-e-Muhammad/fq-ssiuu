# Checkpoints

Model checkpoints are stored in Google Drive, not in this repository.

## Google Drive Location

All checkpoints live under:
`My Drive/fq-ssiuu-research/checkpoints/`

## Structure

```
checkpoints/
├── oracle/
│   └── tofu_forget05_retain90/    Retain-only fine-tuned oracle model
├── c2_npo/
│   ├── seed42/                    NPO baseline, seed 42
│   ├── seed123/
│   └── seed456/
├── c4_ssiuu/                      SSIUU + NPO condition
├── c5_fqat/                       FQ-QAT + NPO condition
├── c6_fq_ssiuu/                   FQ-SSIUU full method (primary)
│   └── lambda_sweep/              Lambda sweep conditions
├── c7_durableun/                  DURABLEUN-SAF reference
└── c8_quail/                      QUAIL baseline
```

## To Download

```bash
make pull-drive
```
(Requires rclone configured with Google Drive remote named `gdrive`)

## To Upload After Training

```bash
make sync-drive
```

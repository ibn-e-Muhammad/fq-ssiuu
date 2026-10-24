# FQ-SSIUU Local Execution Workflow

**From scratch → successful local run on Windows RTX 3060 12GB**

---

## 📋 Overview

This workflow takes you from a clean machine to your first successful local experiment.
It's designed for **local Windows execution** (no Kaggle), using the **RTX 3060 12GB**.

**Estimated time:** 2–3 hours (excluding model download)

---

## 🎯 Phase 1: Environment Setup

### 1.1 Install Requirements

| Item | Requirement | Notes |
|---|---|---|
| **OS** | Windows 10/11 (64-bit) | |
| **Python** | 3.11 or 3.12 | Install from python.org |
| **GPU** | NVIDIA RTX 3060 12GB | CUDA 11.8+ |
| **NVIDIA Driver** | Latest (536+ version) | Check in Device Manager |
| **Disk** | ≥ 20 GB free | For models + outputs |

### 1.2 Clone the Repository

```powershell
cd C:\Users\YourName\Projects
git clone <your-repo-url> MY-FYP
cd MY-FYP\fq-ssiuu
```

### 1.3 Create Python Environment

```powershell
# Create venv
python -m venv .venv

# Activate
.\.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -e ".[quantisation,attribution,evaluation,dev]"

# Verify installation
python -c "import torch, transformers, peft; print('All OK')"
```

### 1.4 Set Up HuggingFace Token

```powershell
# One-time setup
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_YOUR_TOKEN", "User")
```

**Or** create a `.env` file in the project root:
```
HF_TOKEN=hf_YOUR_TOKEN
```

> ⚠️ Never commit this file to git. Already in `.gitignore`.

### 1.5 Verify GPU

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"
```

**Expected:** `True`, `NVIDIA GeForce RTX 3060`, `12.0` (or similar)

---

## 📋 Phase 2: Project Preparation

### 2.1 Project Structure Check

```
MY-FYP/fq-ssiuu/
├── notebooks/
│   └── kaggle/
│       └── 00_MOTHER_RUNNER.ipynb   ← Original (for reference)
├── Mother_Local_v2.ipynb             ← Your local notebook (NEW)
├── src/
│   └── fq_ssiuu/                     ← Python package
├── scripts/
│   └── mother_runner.py              ← Execution script
├── configs/
│   └── runs/
│       └── integration_v1.json       ← Integration recipe
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

### 2.2 Generate SOURCE_MANIFEST.json

The local notebook needs a `SOURCE_MANIFEST.json`. If one doesn't exist, the notebook auto-generates it from git:

```powershell
# Quick manual generation
.\.venv\Scripts\python -c "
import json, subprocess
from pathlib import Path
ROOT = Path.cwd()
manifest = {
    'source_origin_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()[:12],
    'source_tree_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()[:12],
    'created_local': True
}
Path('SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
print('Generated SOURCE_MANIFEST.json')
"
```

### 2.3 Verify Recipe

```powershell
Get-Content configs\runs\integration_v1.json -Raw | ConvertFrom-Json | Format-List
```

---

## 🚀 Phase 3: First Local Run

### 3.1 Open the Notebook

```powershell
# Launch Jupyter with venv kernel
.\.venv\Scripts\jupyter.exe notebook
```

Open **Mother_Local_v2.ipynb** → Change kernel to `.venv` → Run all cells.

### 3.2 Expected Execution Flow

| Cell | What Happens | Status |
|---|---|---|
| 1. Config | Loads local paths + HF token | ✅ |
| 2. Imports | Loads PyTorch, transformers, etc. | ✅ |
| 3. Manifest | Checks/generates SOURCE_MANIFEST.json | ✅ |
| 4. Model Load | Loads Llama-3.2-1B in NF4 4-bit (~3GB VRAM) | ✅ |
| 5. LoRA | Wraps model with LoRA (~5GB VRAM) | ✅ |
| 6. Execution | Runs `mother_runner.py` integration recipe | ⚠️ May take time |
| 7. Provenance | Saves summary + results | ✅ |

### 3.3 Check Results

```powershell
# List all runs
Get-ChildItem runs -Directory

# View run summary
Get-Content runs\<run-id>\bootstrap-*.json

# Check GPU memory of running process
nvidia-smi
```

---

## 🔄 Phase 4: Iterative Development

### 4.1 After First Run

1. **Review outputs** in `runs\<run-id>/`
2. **Run unit tests** to verify code integrity:
   ```powershell
   .\.venv\Scripts\python -m pytest tests -v --tb=short
   ```
3. **Check VRAM usage** — if OOM, adjust batch size in recipe
4. **Commit changes**:
   ```powershell
   git add -A && git commit -m "Local execution: <description>" && git push
   ```

### 4.2 Running Different Recipes

```powershell
# Edit the notebook's Cell 1 to point to a different recipe
# $RECIPE_PATH = ROOT / "configs/runs/pilot_p1.json"
```

Or run `mother_runner.py` directly:

```powershell
.\.venv\Scripts\python scripts\mother_runner.py `
  --recipe configs\runs\pilot_p1.json `
  --output runs\pilot1\`
  --provenance runs\pilot1\bootstrap.json `
  --run-id pilot1
```

---

## 📦 Phase 5: Archiving & Reporting

### 5.1 Seal the Run

```powershell
# From project root
.\.venv\Scripts\python scripts\archive_run.py seal `
  --run-dir runs\<run-id> `
  --output runs\sealed `
  --run-id <run-id>
```

### 5.2 Sync to External Storage (optional)

```powershell
# Mirror to Hugging Face (if configured)
.\.venv\Scripts\python scripts\archive_run.py mirror-hf `
  --sealed runs\sealed\<run-id> `
  --receipt runs\sealed\<run-id>_receipt.json
```

### 5.3 Generate Report

```powershell
# Convert markdown report
.\.venv\Scripts\python -m markdown runs\<run-id>\report.md
```

---

## ⚠️ Memory Management (RTX 3060 12GB)

### VRAM Budget

| Component | Estimated VRAM | If OOM |
|---|---|---|
| Base model (NF4 4-bit) | ~3 GB | Reduce batch size |
| LoRA adapters + optimizer | ~2 GB | Disable dropout / reduce rank |
| Training step (forward+backward) | ~5–6 GB | - |
| **Total (safe)** | ~12 GB | Monitor closely |

### If Out of Memory (OOM)

1. **Reduce batch size** in recipe config
2. **Increase gradient accumulation** steps
3. **Lower LoRA rank** `r=16` → `r=8`
4. **Disable double quant** (if available)
5. **Run on CPU** (very slow, but works for validation)

---

## 📦 Deliverables Checklist

- [ ] Python 3.11+ installed
- [ ] NVIDIA driver installed (≥ 536)
- [ ] venv created with dependencies
- [ ] HF_TOKEN set
- [ ] SOURCE_MANIFEST.json generated
- [ ] integration_v1.json recipe verified
- [ ] Mother_Local_v2.ipynb opened in Jupyter
- [ ] First run completed (status: SUCCESS)
- [ ] Unit tests passed
- [ ] Results saved to git
- [ ] Report archived

---

## 🆘 Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Dependencies not installed | `pip install -e ".[...]"`, re-activate venv |
| `CUDA not available` | CUDA not installed | Reinstall PyTorch with CUDA flag |
| `HF_TOKEN not set` | Token not in environment | Set with `[Environment]::SetEnvironmentVariable` |
| `OOM: CUDA out of memory` | VRAM exceeded | Reduce batch size, reduce LoRA rank |
| `Source checkout dirty` | Uncommitted changes | `git add -A && git commit` before running |
| `Recipe not found` | Wrong path in notebook | Verify `configs/runs/integration_v1.json` exists |
| `Model download fails` | Finesse gated model | Check HF_TOKEN validity, request access on HF Hub |

---

## 📚 Key Files & Locations

| File | Purpose |
|---|---|
| `Mother_Local_v2.ipynb` | **Main local execution notebook** |
| `run_local.ps1` | Automated PowerShell runner |
| `configs/runs/integration_v1.json` | Integration recipe |
| `src/fq_ssiuu/models/loader.py` | Model loading utilities |
| `scripts/mother_runner.py` | Execution entry point |
| `runs/<run-id>/` | All run outputs |
| `Mother_Local_v2.ipynb` | Code equivalent of notebook |

---

## 🔄 Recovery & Re-run

If a run fails:

```powershell
# Clean the failed run directory
Remove-Item -Recurse -Force runs\<run-id>

# Re-run (notebook will generate new IDs each time)
# Or use mother_runner directly:
.\.venv\Scripts\python scripts\mother_runner.py --recipe configs\runs\integration_v1.json --output runs\new-run --run-id new-run
```

---

**Last updated:** 2026-10-01  
**Author:** Rohan Ahmed (23BSAI008)  
**Project:** FQ-SSIUU (Verifiable Machine Unlearning for LLMs)

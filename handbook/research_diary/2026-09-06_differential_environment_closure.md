# Differential environment closure — 2026-09-06

Kernel version 3 is the first of two authorised submissions in this work unit.
Source `6cb7933dd4eb4d3753bc3535aae6388c6a1d2236` and orchestration commit
`82e8736fbdb22b7e807cfaa7d2ace3791afd494a` passed private source verification.
Setup run `0b2980fa-c045-4c1f-8c66-4f8f890fd447` failed before installation:
the new complete-inventory collector rejected duplicate PyGObject metadata.
Raw evidence and a separate verdict are preserved in
`../evidence/og001_kaggle/20260906_kernel_v3_82e8736f/`.

The native freeze records effective PyGObject 3.48.2. The failed collector did
not preserve the other version or location, so neither is claimed as measured.
The repair records all distribution versions, locations and per-name precedence,
uses the first visible version, cross-checks freeze pins and rejects unreported
changes to lower-precedence records as well as effective packages. This follows
[pip's first-found distribution selection](https://github.com/pypa/pip/blob/main/src/pip/_internal/metadata/importlib/_envs.py),
consulted 2026-09-06. It does not declare duplicate metadata globally clean.

Independent review also exposed operational integration defects: scientific
gates could auto-promote from code heuristics; notebook reruns could overwrite
failure evidence; an obsolete experiment template suggested a mutable source
workflow; CI omitted PyTorch despite collection-time imports. These were fixed.
The existing Ruff policy is enforced with its declared version, including
mechanical formatting/import/type-annotation cleanup. No scientific method was
reconciled by these changes. CI uses the official
[PyTorch 2.10 CPU installation](https://pytorch.org/get-started/previous-versions/);
only Kaggle evidence can establish the canonical GPU environment.

Final local preflight: 70 offline infrastructure tests pass; the broader CPU
suite has 129 passes, eight historical placeholder skips, 12 deselections and one known
SSIUU expected failure. NumPy is absent locally; no global or project packages
were changed. All seven notebooks compile, and lint/format/Python/shell checks
pass. Exact outputs and limitations are in
`../evidence/og001_kaggle/20260906_final_local_preflight/`.

OG-001 remains PARTIAL; RG-001/002/003 remain BLOCKED. The next action is the
second and final authorised private OG-001 submission, with full non-model tests
in the same runtime only after operational acceptance. No scientific run occurred.

Follow-up review fixed editable pip-freeze comment parsing and OPEN gate blocking
flags. Final broader CPU suite: 131 passed, eight historical placeholder skips,
12 deselected, one SSIUU xfail. See `handbook/evidence/og001_kaggle/20260906_reviewed_preflight/`
for raw output and correction of the earlier skip classification. No GPU claim follows.

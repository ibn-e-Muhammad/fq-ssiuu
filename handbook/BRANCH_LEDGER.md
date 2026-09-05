# FQ-SSIUU Branch Ledger

**Purpose:** Document the branch topology, active lineages, and historical integrations.

## Governance
- `main`: stable/release branch.
- `dev`: sole integration branch.
- One active lineage tip at a time.
- Only the final validated lineage tip will be merged into `dev`.
- Intermediate branches must not be merged separately.
- Old branches must not be deleted during this stage.
- Historical branches can be deleted only after ancestry, open-PR status and integration are reverified.
- Experiment evidence must use commit SHA and run ID, not branch name alone.

## Mainline Branches

### `main`
- **Tip**: `477ea9f8484eb69b73751a30ae11dc77c01b9499`
- **Relationship**: 59 commits behind `dev`
- **Status**: Retain permanently

### `dev`
- **Tip**: `e395f7af264eaaac9473d8b9e9f1c7a46c825d5c`
- **Relationship**: Integration base
- **Status**: Retain permanently

## Active Lineage

### `fix/research-integrity-foundation`
- **Tip**: `f76c4715802b6862e8917591a9185380af65bab8`
- **Relationship**: 1 commit ahead of `dev`
- **Status**: Unmerged ancestor of active tip

### `fix/reconciliation-v2-stage1a`
- **Tip**: `bec0aaacc36e9aec2c2a6fe4cc82ff166d50edbb`
- **Relationship**: 2 commits ahead of `dev`
- **Status**: Unmerged ancestor of active tip

### `fix/reconciliation-v2-stage1c-tqdm`
- **Tip**: `e59d42c7f5025eb5ddf6b94cdafb3b1b869d92b5`
- **Relationship**: 3 commits ahead of `dev`
- **Status**: Unmerged ancestor of active tip

### `chore/kaggle-gate0-environment-probe`
- **Tip**: `547db67e6bc75f9efe0b17cbbbc44d8128aff23f`
- **Relationship**: 5 commits ahead of `dev`
- **Status**: Current lineage tip before this stage; not yet merged into `dev`

## Historical Branches (Contained in `dev`)

### `gate/0-env-correction`
- **Tip**: `6b931a82ca95d1d515b3d7b7a5073f0922cdc3b3`
- **Relationship**: 54 commits behind `dev`

### `gate/1-baseline-repro`
- **Tip**: `bb78a54e5aab28ee41b399114878aae1085dee9f`
- **Relationship**: 47 commits behind `dev`

### `stage/1-foundation`
- **Tip**: `477ea9f8484eb69b73751a30ae11dc77c01b9499`
- **Relationship**: 59 commits behind `dev`

### `stage/2-core-implementation`
- **Tip**: `fb0215b8f22d65c71faac92675c5257eb82d00d0`
- **Relationship**: 56 commits behind `dev`

### `pilot/p1-p2`
- **Tip**: `1f06b35f4680c8c307f416062150a7c8fb77b6cc`
- **Relationship**: 29 commits behind `dev`

### `pilot/p3-p4`
- **Tip**: `ac91d90ae9586924d751a3693069435e003bc4ee`
- **Relationship**: 8 commits behind `dev`

### `audit/pilot-verification`
- **Tip**: `82bdbf2a3d3f44e125f4973d036896db5ae5e213`
- **Relationship**: 2 commits behind `dev`

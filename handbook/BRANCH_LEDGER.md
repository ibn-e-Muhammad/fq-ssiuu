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

## Current integration — 2026-09-25

- `dev`: `5ca92bfa0863a9ffadcd70bfc5f28c8ee8c6e76a`, verified PR #10 merge; GitHub default branch.
- `repro/npo-1b-source-faithful`: active D1, from that exact remote tip; NPO only, 0/2 Kaggle attempts.
- `fix/matched-precision-qra-protocol`: PR #10 merged, head `29d21a51de784190bdabf4e7267db8925c73260e`; superseded and retained.
- `fix/npo-adapter-qra-metrics`: PR #9 merged, feature tip `e896dbc0467018a87d3d0754d68eddbe094b3b0e`; superseded and retained.
- `fix/upstream-method-reconciliation`: PR #8 merged, superseded and retained.
- `fix/kaggle-overlay-evidence-integrity`: PR #7 merged, superseded and retained.
- dev rules: passing `Lint and Unit Tests` from GitHub Actions; one fresh approval;
  resolved threads; deletion/force-push blocked. The additional last-push approval
  restriction was disabled at the researcher's request; all other rules preserved.
  A substantive human scientific review is required, not a quota-exhausted bot comment.
- No main changes, history rewriting, deletion or DOCX inclusion.

## Historical inventory before PR #7

The relationships and statuses below are a preserved pre-merge snapshot,
superseded by the current integration record above.

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
- **Status**: Superseded by `chore/kaggle-environment-overlay`; retained until final integration; not merged into `dev`

### `chore/kaggle-environment-overlay`
- **Created from**: `547db67e6bc75f9efe0b17cbbbc44d8128aff23f` (on `dev` at `e395f7af264eaaac9473d8b9e9f1c7a46c825d5c`)
- **Durable Checkpoints**:
  - Documentation closeout commit: `5045cd59964adcab6e089d570bb00e077c2e1fb4`
  - Record correction commit: `ee80b0df160558c52c5bfbdabff5ab01a8885654`
  - Provenance timestamp qualification: `7fb798643aac2381863a5884d339df43cee7ab1f` (corrected typo in documentation record from `7fb79867...`)
  - Fail-closed overlay repair commit: `2f13acf44f1d43abf1eeb35d6ba2a3f887e69249`
- **Status**: Unmerged ancestor of active tip `fix/kaggle-overlay-evidence-integrity`
- **Purpose**: Documentation closeout, provenance qualification, static Kaggle overlay implementation, and fail-closed overlay repair

### `fix/kaggle-overlay-evidence-integrity`
- **Created from**: `2f13acf44f1d43abf1eeb35d6ba2a3f887e69249` (on `chore/kaggle-environment-overlay`)
- **Durable Checkpoints**:
  - Evidence linkage commit: `53d1317d41b5a72657bc2e9d61854e434b5c5b1e`
  - Correction docs commit: `3149d38d42c5e3e6cda44825d9f689558242d4bc`
  - Preflight repair commit (original intended source, not executed): `502871a57b275bffaa65ba6a24f9e834c3721fb6`
  - Validation notebook commit: `8d3bc14d0b4230d216b88473cd13c7085ecf3ee2`
  - Original attempt evidence commit: `e63d26f2e2d810e2c070ff08fdc119dd9b11697f`
  - Original transcript / prior branch tip: `3fd39547a3dc7e467e9a6484c421b1238d4357f5`
  - Verifier entrypoint repair, credential removal and source packager: `91b6aa39e520f9312b098997134c1deabeabfb8f`
  - Repaired private-source notebook and bootstrap tests: `24b25b5f3883646d1841e83cdd6a88a54b3fd93d`
- **Status**: Active lineage tip for Kaggle overlay evidence linkage, preflight repairs, and validation run (Stage 1H-D / 1I)
- **Purpose**: Evidence linkage, PEP 440 version comparisons, protected package mutation guards, verifier execution recording, clean runner notebook, and Kaggle validation run
- **Merge status**: Not merged into `dev`

**2026-09-05 provenance correction and rerun:** The notebook SHA above corrects
the previous mistyped value. The starting local and remote tip were identical at
`3fd39547a3dc7e467e9a6484c421b1238d4357f5`, 17 linear commits ahead of `dev`.
The superseded `502871a57b275bffaa65ba6a24f9e834c3721fb6` source was not used for
the repaired run: inspection found undefined verifier helpers and a remaining
credential literal in a tracked legacy notebook. The repaired source is explicitly
pinned at `91b6aa39e520f9312b098997134c1deabeabfb8f`.

Source and notebook commits were published through the connected GitHub Git-data
API with `force=false`; their tree hashes and exact commit objects were verified
locally before adoption. No merge, rebase, branch creation or force update occurred.
Kernel version 2 is the only repaired submission. It reached installation, then
failed global pip check. Evidence is in
`handbook/evidence/og001_kaggle/20260905_kernel_v2_24b25b5f/`.
The evidence commit is the commit containing this update; resolve it with Git
history rather than storing a self-referential or invented tip SHA.


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

## 2026-09-06 differential closure checkpoint

Starting tip `0bbeaeb560acd5b9357ddfc25f18246d103b379b` was clean and agreed
with the remote. Source `6cb7933dd4eb4d3753bc3535aae6388c6a1d2236` and notebook
`82e8736fbdb22b7e807cfaa7d2ace3791afd494a` were published non-destructively.
Kernel v3 is submission 1/2 of the differential work unit; it failed at native
duplicate-metadata inventory before installation. See its run-specific evidence.
The repair and final preflight are recorded in the commit containing this entry.
No integration merge has occurred; four untracked DOCX files remain excluded.

## 2026-09-08 operational closure and Mother runner

Source `1efe2b32ad65e89b3352946a568f3312f758fa00` and notebook
`2be461167407694826f0417afa3594b5328a3b73` produced OG-001 PASS in kernel v4.
The kernel failed later at integration collection; this does not invalidate its
separate hash-verified environment acceptance. No dev merge has occurred.
The commit containing this entry adds the Mother runner, exact validated lock,
explicit test package, dotenv-compatible secret handling and dual-store receipts.
Historical branches and four untracked DOCX files remain untouched.

## 2026-09-08 Mother integration acceptance

Source/orchestration `23682ea12ff03ea867f220406423b6f54f77091c` completed Mother
kernel v1. The independent operational review found no implementation blocker;
raw evidence, exact source/staging pins and private storage receipts are recorded
in `handbook/evidence/mother_runner/20260908_kernel_v1_23682ea1/`.
The evidence commit is the commit containing this entry. Integration requires a
reviewed PR and passing CI; no merge is claimed by this pre-merge record.

## Scientific reconciliation A — 2026-09-10

Remote dev `d310815c` and successor `291bbda5` matched local refs before this change.
The containing commit organises method workspaces, adds QRA/NPO acceptance code
and preserves historical source/evidence. No Kaggle submission, dev/main merge,
branch deletion or history rewrite belongs to this work unit. Four untracked DOCX
files remain excluded. Current integration status at the top supersedes dated
pre-merge narratives above.

Work unit A was published at `07158770d810c3212c0de2be9f14ea008fe333c4`.
PR #8 targets dev; CI run 55 passed the exact head. Local parentless-source/notebook
staging also passed. The following evidence-only commit records those receipts;
no integration merge is claimed.

## Work unit C inference closure — 2026-09-24

PR #10 remains open and unmerged. `b884c5ef0a8bfe7812e583f88f73cb5c1eaf9a83`
introduced the contracts/runner; `14669f34cbaefde8c4e3d7d7f89616a240a6a6c7`
repaired sequential model lifetime and preserved the failed first attempt. CI 61
and 62 passed. Mother v4/run `20260924-120549-b4c4f06e097d` passed bounded C1;
the containing closure commit records verified evidence and proposed C2 review.
Both submissions are consumed. No merge, branch deletion or scientific approval
is claimed. Four DOCX files and original raw evidence remain untouched.

## Work unit C scientific scope correction — 2026-09-25

Same PR #10 branch, starting `afb341b5ba1dd0e9c6de6f8a3842c5596afe967d`.
Documentation/gate semantics only; C1 raw evidence remains immutable. The researcher
permits source-faithful constituent reproductions independently of deferred
Protocol v2. RG-005 still blocks confirmatory QRA verdicts/full FQ-SSIUU study.
No merge, new Kaggle run, model download or training in this correction; human
review remains outstanding. The containing commit records this single pass.

## D1 start — 2026-09-25

The researcher approved and merged PR #10. Fetch verified `5ca92bfa`; its tree
matches the merged `29d21a51` feature head. Created only
`repro/npo-1b-source-faithful`. The governance receipt is
`evidence/work_unit_d1/20260925_preflight/governance.json`.
Raw C1 evidence, historical branches and all four DOCX files remain unchanged.

D1 continuation (2026-09-26): same branch, source sampler/metrics and CPU ZeRO-3
resume checked; 0/2 Kaggle submissions. Historical clipping default reconciled
without changing the author objective. No D1 merge or independent-review claim.

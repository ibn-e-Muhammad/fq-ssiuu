# Environment closure and reusable execution loop

OG-001 passed in kernel v4 on 2026-09-06. Independent local derivation on
2026-09-07 validated all 35 setup artifacts and source/run linkage. Six native
pip diagnostics were unchanged, with no introduced or overlay-related conflicts.
Imports, FP16, SDPA, genuine NF4, single-device enforcement, protected/platform
integrity, manifests and overlay lock passed. The kernel's later ERROR was a
test collection naming collision; adding `tests/__init__.py` resolves the
namespace ambiguity locally. GPU integration rerun remains pending.

Raw evidence: `../evidence/og001_kaggle/20260906_kernel_v4_2be46116/`.
Run: `04396fe9-0195-476c-bb74-29e6a5ca074b`. Private Kaggle and Hugging Face
copies were created and downloaded content verified; receipts are separate from
the immutable raw evidence. No scientific experiment ran.

The researcher redirected effort toward faithful upstream methods, one Mother
notebook, durable artifacts and consistent records. The implementation now uses
one notebook, versioned recipes and the exact validated overlay. It records
source/recipe/seed, command logs, termination status and a sealed archive. The
initial recipe is integration only. Dotenv handling supports local ignored files
and in-memory Kaggle Secrets; tokens are fetched only on demand.

Primary-source review found QUAIL's author-linked repository and an NPO mismatch
that earlier summaries overlooked. NPO's upstream TOFU helper uses shifted,
masked sequence-sum NLL; the current helper averages tokens. The upstream
implementation register withdraws blanket fidelity claims pending equivalence
tests. No scientific gate was closed by this environment milestone.

Local Mother preflight: 144 tests passed; eight historical placeholders skipped,
12 GPU/model tests deselected, one known SSIUU expected failure. Detailed results
are in `../evidence/mother_runner/20260908_local_preflight/`. Independent review
identified a staging-receipt overwrite path, now fixed and regression-tested.
The remaining independent review was initially interrupted by service usage
limits, then resumed and completed on 2026-09-08.

Mother kernel v1 completed at source/orchestration `23682ea1`, run
`20260908-041835-2ef818513f23`. All environment checks passed again; the full
implemented non-model suite recorded 144 passed, 12 placeholder skips, eight
deselected and one known SSIUU expected failure. The extra four skips compared
with the local CPU run are QRA placeholders selected in the GPU runtime.
Ten notebooks passed schema/compilation checks. The independent reviewer replayed
source/notebook linkage, environment acceptance and all 52 runtime archive files.
The complete download contains 56 files, including the sealed tar and kernel log.
Both private archives preserve that complete output; the duplicate tar container
is kept in the archives rather than Git. Evidence and receipts:
`../evidence/mother_runner/20260908_kernel_v1_23682ea1/`.

This closes the Mother integration/storage work unit using one of two authorised
submissions. No further environment run is needed for this unchanged code.
Scientific gates remain unchanged; integration review precedes a focused upstream
method reconciliation branch. Dotenv behavior was tested with synthetic values;
this integration run requested no real credentials inside Kaggle.

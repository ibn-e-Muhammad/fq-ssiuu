# Publication and local staging verification

Source commit `07158770d810c3212c0de2be9f14ea008fe333c4`, tree
`e76d0c51a9865c82700bd12504683660e536079c`, PR #8.

The parentless bundle and relocated first notebook were staged locally after
publication with the existing source-packaging and staging helpers. This is an
offline layout regression check, not an environment installation or model run.
The dataset ID ending `layout-check` is a local metadata fixture; no such dataset
was created or uploaded. No Kaggle submission occurred.

Source packaging verified one parentless commit and the source/tree/bundle hashes.
Notebook staging verified the selected recipe and produced private T4 metadata
and a hash receipt. Generated Python cells compile. The exact staged notebook is
reconstructible from the committed template and receipt parameters; it is not
duplicated in this evidence directory. Full source bundles remain local.

Original stdout/stderr and JSON receipts are unchanged copies. Any later CI
record refers to its exact tested source, not to an inferred future branch tip.

# Reusable package acceptance

This package contains operational helpers, loading utilities and the descriptive
QRA/report contract. Acceptance is scoped to recorded tests and evidence; this
folder is not a blanket claim of scientific correctness.

No scientific training method is currently accepted or frozen for corrected 1B
execution. NPO candidate code and all historical surrogates are under
`experiments/reproductions/METHOD/`. The combined method is still unimplemented.

Promotion requires a primary-source conformance specification, independent value
and gradient tests, reviewed caller/data/reference behavior, required runtime
validation and linked evidence. Move the accepted reusable implementation here,
keep the reproduction entry point and immutable run recipes in its method folder,
and record the promotion commit. A later preregistered tag freezes the actual
code/configuration used for confirmatory experiments.

The loader now defaults to FP16 on device 0. Its generic default still enables NF4;
a faithful unquantised NPO reference must explicitly disable 4-bit loading and
record its source revision. Mocked loading tests are not model-runtime evidence.

"""Deterministic descriptive paired statistics; no scientific decision rule."""

from numbers import Real

import numpy as np

from fq_ssiuu.evaluation.identity import file_digest


def finite_vector(values, *, nonnegative=False):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
        raise ValueError("Nonempty ordered values required")
    if any(not isinstance(x, Real) or isinstance(x, (bool, np.bool_)) for x in values):
        raise ValueError("Measurements must be real numbers, not booleans or numeric strings")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all() or (nonnegative and (arr < 0).any()):
        raise ValueError("Finite valid one-dimensional measurements required")
    return arr


def paired_bootstrap(sample_ids, baseline, target, *, seed, resamples, confidence=0.95):
    a = finite_vector(baseline)
    b = finite_vector(target)
    if (
        any(not isinstance(x, str) or not x for x in sample_ids)
        or len(a) != len(b)
        or len(a) != len(sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("Unique matched ordered samples required")
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 2:
        raise ValueError("Explicit seed and resample count required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be inside (0,1)")
    delta = b - a
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired differences")
    generator = np.random.Generator(np.random.PCG64(seed))
    # One row per resample, bounded temporary memory even for full TOFU sets.
    means = np.array(
        [delta[generator.integers(0, len(delta), size=len(delta))].mean() for _ in range(resamples)]
    )
    interval = np.quantile(means, [(1 - confidence) / 2, (1 + confidence) / 2], method="linear")
    return {
        "sample_ids": list(sample_ids),
        "paired_differences": delta.tolist(),
        "mean_change": float(delta.mean()),
        "confidence_interval": interval.tolist(),
        "confidence": confidence,
        "seed": seed,
        "resamples": resamples,
        "rng": "numpy.PCG64",
        "quantile_method": "linear",
        "numpy_version": np.__version__,
        "implementation_sha256": file_digest(__file__),
        "attack_detected": None,
        "durability_pass": None,
    }

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
        "method": "paired-row-bootstrap-v1",
        "inferential_scope": "engineering_diagnostic_only; rows may be correlated",
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


def clustered_precision_bootstrap(
    sample_ids,
    author_ids,
    baseline,
    target,
    *,
    cluster_manifest_sha256,
    seed,
    resamples,
    confidence=0.95,
):
    """Equal-author estimand; resample whole authors, retaining paired row values.

    External author mapping must be independently verified and hash-bound. Never
    infer an author from a row number. This computation cannot approve a protocol.
    """
    from fq_ssiuu.evaluation.identity import digest, require_hash

    require_hash(cluster_manifest_sha256)
    checked = paired_bootstrap(
        sample_ids, baseline, target, seed=seed, resamples=resamples, confidence=confidence
    )
    if len(author_ids) != len(sample_ids) or any(
        not isinstance(x, str) or not x for x in author_ids
    ):
        raise ValueError("Every sample needs an explicit author identity")
    authors = sorted(set(author_ids))
    if len(authors) < 2:
        raise ValueError("At least two author clusters required for uncertainty")
    delta = np.asarray(checked["paired_differences"])
    means = np.array([delta[np.asarray(author_ids) == a].mean() for a in authors])
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = np.array(
        [means[generator.integers(0, len(authors), len(authors))].mean() for _ in range(resamples)]
    )
    checked.update(
        method="paired-author-cluster-bootstrap-v1",
        inferential_scope="protocol_candidate_not_approved",
        estimand="equal-weight author mean of paired row changes",
        author_ids=list(author_ids),
        authors=authors,
        author_mean_changes=means.tolist(),
        author_assignment_sha256=digest(list(zip(sample_ids, author_ids, strict=True))),
        cluster_manifest_sha256=cluster_manifest_sha256,
        mean_change=float(means.mean()),
        confidence_interval=np.quantile(
            draws, [(1 - confidence) / 2, (1 + confidence) / 2], method="linear"
        ).tolist(),
    )
    return checked


def crossed_seed_author_bootstrap(
    seed_ids, author_ids, changes, *, cluster_manifest_sha256, seed, resamples, confidence=0.95
):
    """Crossed resampling of seeds and shared authors, each with equal weight.

    Rows are seed-level author means computed from matched precision pairs. Same
    author draw is used across resampled seeds (authors are shared across seeds).
    This is a proposed multi-seed procedure, not an approved confirmatory rule.
    """
    from fq_ssiuu.evaluation.identity import require_hash

    require_hash(cluster_manifest_sha256)
    if (
        len(seed_ids) < 3
        or len(set(seed_ids)) != len(seed_ids)
        or any(type(s) is not int for s in seed_ids)
    ):
        raise ValueError("At least three distinct declared seeds required")
    if (
        len(author_ids) < 2
        or len(set(author_ids)) != len(author_ids)
        or any(not isinstance(a, str) or not a for a in author_ids)
    ):
        raise ValueError("At least two unique shared author clusters required")
    matrix = np.array([finite_vector(row) for row in changes])
    if matrix.shape != (len(seed_ids), len(author_ids)):
        raise ValueError("Complete seed by author matrix required; no omitted runs")
    # Reuse strict bootstrap argument validation, without treating rows as units.
    paired_bootstrap(
        ["validation"], [0.0], [0.0], seed=seed, resamples=resamples, confidence=confidence
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = []
    for _ in range(resamples):
        s = rng.integers(0, len(seed_ids), len(seed_ids))
        a = rng.integers(0, len(author_ids), len(author_ids))
        draws.append(matrix[np.ix_(s, a)].mean())
    return {
        "method": "crossed-seed-author-bootstrap-v1",
        "inferential_scope": "PROPOSED_NOT_APPROVED",
        "seed_ids": list(seed_ids),
        "author_ids": list(author_ids),
        "author_mean_changes_by_seed": matrix.tolist(),
        "seed_means": matrix.mean(axis=1).tolist(),
        "mean_change": float(matrix.mean()),
        "confidence_interval": np.quantile(
            draws, [(1 - confidence) / 2, (1 + confidence) / 2], method="linear"
        ).tolist(),
        "seed": seed,
        "resamples": resamples,
        "confidence": confidence,
        "cluster_manifest_sha256": cluster_manifest_sha256,
        "rng": "numpy.PCG64",
        "quantile_method": "linear",
        "numpy_version": np.__version__,
        "implementation_sha256": file_digest(__file__),
        "attack_detected": None,
        "durability_pass": None,
    }

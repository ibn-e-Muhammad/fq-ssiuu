"""Explicit named-author assignment verified against pinned fictional TOFU rows.

The allowlist is read from the ten full-name answers, not inferred from row order.
Each row must mention exactly one full name in its question or canonical answer.
"""

from collections import Counter

from fq_ssiuu.evaluation.identity import digest, require_hash

FORGET05_AUTHORS = (
    "Hina Ameen",
    "Xin Lee Williams",
    "Moshe Ben-David",
    "Kalkidan Abera",
    "Takashi Nakamura",
    "Raven Marais",
    "Aysha Al-Hashim",
    "Edward Patrick Sullivan",
    "Basil Mahfouz Al-Kuwaiti",
    "Nikolai Abilov",
)


def author_mapping(
    rows, *, dataset_id, dataset_revision, split, source_file_sha256, authors=FORGET05_AUTHORS
):
    require_hash(dataset_revision, 40)
    require_hash(source_file_sha256)
    if not rows or len(set(authors)) != len(authors) or not authors:
        raise ValueError("Nonempty rows and distinct explicit author names required")
    assignments = []
    for i, row in enumerate(rows):
        text = row["question"] + " " + row["answer"]
        matches = [name for name in authors if name in text]
        if len(matches) != 1:
            raise ValueError("Ambiguous or missing explicit full author name")
        assignments.append(
            {
                "sample_id": f"{dataset_id}@{dataset_revision}/{split}/{i}",
                "author_id": matches[0],
                "source_record_sha256": digest(row),
            }
        )
    result = {
        "schema_version": "tofu-author-map-v1",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "source_file_sha256": source_file_sha256,
        "assignment_rule": "unique explicit full name in question or canonical answer",
        "authors": list(authors),
        "assignments": assignments,
        "counts": dict(sorted(Counter(r["author_id"] for r in assignments).items())),
    }
    return result


def validate_author_mapping(manifest, rows, source_file_sha256):
    if manifest["source_file_sha256"] != source_file_sha256:
        raise ValueError("Author mapping source file differs")
    expected = author_mapping(
        rows,
        dataset_id=manifest["dataset_id"],
        dataset_revision=manifest["dataset_revision"],
        split=manifest["split"],
        source_file_sha256=source_file_sha256,
        authors=manifest["authors"],
    )
    if manifest != expected:
        raise ValueError("Author mapping content differs from source rows")

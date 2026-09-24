# Method reproduction policy

Researcher decision, 2026-09-25: source-faithful constituent-method reproductions
may proceed without activating QRA Protocol v2. Published equations, pinned code,
source-defined metrics, splits, stopping rules and hyperparameters are external
method requirements, not project proposals awaiting approval. Verify those
requirements and record discrepancies before use; do not silently substitute a
project rule when paper and code disagree.

| Class | Meaning | Permitted claim / requirements |
|---|---|---|
| A — upstream conformance | Mathematical and behavioral comparison with the primary paper and hash-pinned source, or paper equations/algorithm when no implementation is verified | Establish implementation conformance for tested cases; not reproduction of headline results |
| B — source-faithful 1B adapted reproduction | Preserve the source objective, update semantics, metrics, splits, stopping rules and hyperparameters wherever applicable; explicitly record the 1B target and necessary engineering differences | An adapted reproduction at 1B. Changing model size or dataset is **not an exact replication of the original headline result**. Dataset substitution must be declared; inherited split/metric semantics must be checked |
| C — project adaptation | Deliberate changes to objectives, trainable parameters, quantization location, metrics, hyperparameters or stopping/decision rules beyond necessary equivalent engineering | Label each deviation and test it separately. Do not call a materially changed method source-faithful; proposed project rules require their own acceptance |
| D — novel combined FQ-SSIUU | Integration of the accepted NPO, faithful SSIUU and deployment-aligned FQ-QAT components | Novel project method; constituent reproduction and bounded pilots precede the approved confirmatory protocol and full study |

LoRA, altered precision, tokenizer handling or memory-saving changes are not
automatically equivalent to an upstream implementation. Document the adaptation,
test the relevant invariants, and classify material method changes under C.
Use the [primary source hierarchy](UPSTREAM_IMPLEMENTATIONS.md); pin revisions
and file hashes before executing source. Preserve null, failed and negative results.

## RG-005 scope and existing schema

RG-005 remains BLOCKED for **confirmatory QRA attack/durability verdicts and the
full FQ-SSIUU study**. It does **not** block A/B constituent-method work or
descriptive metric collection. Method-specific conformance, immutable inputs,
operational limits and evidence requirements still apply. Historical surrogate
configs do not become accepted by this policy.

Allowed activities under RG-005: upstream conformance; source-faithful constituent
reproductions at 1B and their bounded feasibility pilots; descriptive measurement
collection without project attack/durability decisions.

Blocked activities under RG-005: activation of Protocol v2; confirmatory QRA
attack/durability verdicts; the full FQ-SSIUU study before its final approved protocol.

`scripts/research_gate.py` currently constructs a strict `Gate` dataclass with no
`allowed_activities` or `blocked_activities` fields. Therefore these distinctions
are documented here and in the existing YAML rationale/blocked_conditions fields;
no unsupported fields or implicit schema extension are introduced. The existing
boolean flags and global `is_blocking`/CLI result remain unchanged: a global
blocked summary is not an activity-specific denial of reproduction.

The legacy Mother `pilot`/`experiment` route still has a blanket RG-005 check and
is not an activity-aware reproduction interface. This documentation-only pass
does not enable that route or a new execution path. Future constituent work must
provide a validated method-specific entry contract respecting this scope, rather
than clearing RG-005 or activating Protocol v2 to bypass the old guard.

## Deferred project protocol

Protocol v2 is **DEFERRED_UNTIL_BASE_METHOD_PILOTS / PROPOSED_NOT_APPROVED** and
remains non-executable. Its custom margins, utility caps, multiplicity family,
seeds, oracle policy and attack/durability rules must not govern constituent
reproductions. Revisit project choices after constituent methods are reproduced
at 1B and bounded pilots are complete, then obtain final confirmatory-protocol
approval. Source-defined choices retain their own authority independently.

This correction unit performs no Kaggle execution, model download or training.

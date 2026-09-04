# FQ-SSIUU Decision Register

**Purpose:** Record all design and methodology decisions with rationale.

---

## DL-013: Rename Q-INT4 to Relative Precision Sensitivity

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Rename the metric from 'Q-INT4' to 'relative_precision_sensitivity' |
| Rationale | 'Q-INT4' implies a specific attack metric. The actual computation is |FQ_NF4 - FQ_BF16| / FQ_BF16, which measures unsigned relative change in Forget Quality across precisions. This is a sensitivity measurement, not a directional recovery metric. |
| Impact | Schema v2 in certificate.py; correction register CR-008 |

## DL-014: Certificate → Empirical Durability Report

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Rename 'Unlearning Durability Certificate' to 'Empirical Durability Report' |
| Rationale | 'Certificate' implies formal mathematical or legal guarantees. The output is an empirical measurement report with no such guarantees. |
| Impact | Display text in certificate.py; all documentation; v2 schema field 'threshold_met' replaces 'certified' |

## DL-015: Versioned Report Schema

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Introduce schema_version field ('v1' legacy, 'v2' current) |
| Rationale | Pilot artifacts P1-P4 use v1 field names. Abruptly removing them would break loading. v2 adds accurate field names; v1 aliases preserved as deprecated properties. |
| Impact | certificate.py load_certificate() migrates v1 → v2 on load |

## DL-016: QUAIL Margin Discrepancy Documented

| Field | Value |
|-------|-------|
| Date | 2026-09-03 |
| Decision | Document that implemented QUAIL margin (4.2e-4) differs from published (0.5) |
| Rationale | Published QUAIL (arXiv:2601.15538) uses Δq=1.0, margin=Δq/2=0.5. Implementation uses NF4 bin width / 2 = 4.2e-4. This is a 1190× difference. P4 results are from an incomplete surrogate. |
| Impact | C8 condition blocked until resolved; correction CR-005 |

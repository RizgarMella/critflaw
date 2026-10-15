"""The knowledge-base status of a provenance record from its test-linked evidence (theory manual, section 8.3.1),
for the assessment trace. Kept separate from ``kb.core`` so the trace does not import the registry."""

from __future__ import annotations

from critflaw.kb.evidence import ORACLE_TESTS


def kb_status(record_id: str) -> str:
    """The record's status from the linked evidence, by the same rule as ``kb.core.record_status`` (one definition:
    evidence boxes and future rules apply to the trace too)."""
    from critflaw.kb.core import record_status

    return record_status(record_id, ORACLE_TESTS)


__all__ = ["kb_status"]

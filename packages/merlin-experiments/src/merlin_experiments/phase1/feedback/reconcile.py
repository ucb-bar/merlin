"""Reconcile each background grade result with the current source when it is published.

A background grade runs its own immutable snapshot to the end; an edit never cancels or closes it
(:class:`.background_admission.BackgroundGuard`). Its results are merged back afterwards, case by case,
the way a helper's work on its own copy is merged: a result counts for the current compiler only when the
bytes the case's graded commands can read are unchanged since the snapshot.

Which bytes a case reaches is the existing component decomposition (:func:`.promotion.submission_digests`):
every declared command runs for every case, so a case reaches every component, ``UNATTRIBUTED`` included,
and nothing the derivation proved unreadable (notes, the in-tree ``build/`` a grade never copies). The
reach identity is the digest of those component digests. Without a decomposition there is no identity and
nothing is reconciled as current unless the graded source digest itself is current.

A row keeps the bytes that produced it: reconciliation only adds a label to its ``graded`` attribution,
``currency`` = ``current`` or ``older_version``, and never re-attributes a result. An older-version row
stays visible as evidence for the version it describes; ``current`` counts use only current rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CURRENT = "current"
OLDER = "older_version"
#: Keys reconciliation writes into a row's ``graded`` attribution; they are relabelled on every publication.
LABELS = ("currency", "reconciled_reach_sha256")


def reach_identity(components) -> str | None:
    """Digest of a snapshot's ``{component: digest}`` map; ``None`` without a decomposition."""
    if not isinstance(components, dict) or not components:
        return None
    return hashlib.sha256(json.dumps(sorted(components.items())).encode()).hexdigest()


def current_reach(workspace) -> str | None:
    """The reach identity of the author's current source; ``None`` when it cannot be decomposed."""
    from .promotion import submission_digests

    try:
        return reach_identity(submission_digests(Path(workspace))[1])
    except (OSError, ValueError, TypeError):
        return None


def currency(origin, *, reach, source=None) -> str:
    """``current`` when ``origin``'s snapshot reaches the same bytes as the current source, else older."""
    if not isinstance(origin, dict):
        return OLDER
    graded_reach = origin.get("reach_sha256")
    if isinstance(graded_reach, str) and graded_reach and graded_reach == reach:
        return CURRENT
    graded = origin.get("graded_source_sha256")
    return CURRENT if isinstance(graded, str) and graded and graded == source else OLDER


def label(rows, *, reach, source=None, origin_key="graded") -> None:
    """Label every row's attribution in place with its currency against ``reach``/``source``.

    Without either identity of the current source nothing can be reconciled, and nothing is labelled.
    """
    if not reach and not source:
        return
    for row in rows:
        origin = row.get(origin_key) if isinstance(row, dict) else None
        if isinstance(origin, dict):
            row[origin_key] = {**origin, "currency": currency(origin, reach=reach, source=source),
                               "reconciled_reach_sha256": reach}


def is_older(row, origin_key="graded") -> bool:
    """Whether a published row was reconciled as a result for an older version of the source."""
    origin = row.get(origin_key) if isinstance(row, dict) else None
    return isinstance(origin, dict) and origin.get("currency") == OLDER


def comparable(row, origin_key="graded") -> dict:
    """``row`` without its detail pointer and reconciliation labels (what a result says, not its currency)."""
    out = {key: value for key, value in row.items() if key != "detail"}
    if isinstance(out.get(origin_key), dict):
        out[origin_key] = {key: value for key, value in out[origin_key].items() if key not in LABELS}
    return out

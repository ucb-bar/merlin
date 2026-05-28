"""Read XPU-RT feedback (persisted by targetgen_mcp.ingest_xpurt_feedback)
and turn it into per-dispatch decisions the QNN/SpaceMit compile paths can
consult.

The persisted form lives at ``<merlin_dir>/breakdowns/feedback.json``. If
the file is absent, ``load_feedback_overlay`` returns an inert overlay
that says "no opinion" for every dispatch — that is what keeps the
standalone compile path byte-identical to today.

This module is consumed by:
  * tools/compile_qnn.py             — chunk-backend selection
  * benchmarks/SpacemiTX60/...       — tile + ukernel selection

The hint vocabulary is defined and validated by
``tools/mcp_servers/targetgen.py:_ingest_xpurt_feedback``; this module only
*reads* the persisted payload and never invents new hints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DispatchDecision:
    """Per-dispatch decisions derived from feedback hints.

    Backend / ukernel / tile overrides are only set when an unambiguous
    ``pin_target=<X>`` hint maps to a known consumer-side option. When
    unset, the consumer keeps its existing default. Advisory hints
    (prefer_coarser / prefer_finer / consider_fuse_with_pred /
    consider_split_backend) are surfaced through ``advisory`` for
    logging — they do not silently change behavior.
    """

    dispatch_id: str
    hints: tuple[str, ...] = ()
    pin_target: str | None = None
    advisory: tuple[str, ...] = ()
    rationale: str = ""

    def has_hint(self, hint: str) -> bool:
        return hint in self.hints

    def prefers_finer(self) -> bool:
        return "prefer_finer" in self.hints

    def prefers_coarser(self) -> bool:
        return "prefer_coarser" in self.hints

    def wants_fuse(self) -> bool:
        return "consider_fuse_with_pred" in self.hints

    def wants_split(self) -> bool:
        return "consider_split_backend" in self.hints


@dataclass
class FeedbackOverlay:
    """Lookup object built from feedback.json. Inert if the file is absent."""

    source_path: Path | None = None
    run_id: str | None = None
    model_signals: dict[str, Any] = field(default_factory=dict)
    decisions_by_id: dict[str, DispatchDecision] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.decisions_by_id

    def for_dispatch(self, *names_or_ids: str) -> DispatchDecision:
        """Look up a decision under any of the supplied names. Falls back
        to an empty decision if none match — callers can always call
        e.g. `.prefers_finer()` without a None-check.
        """
        for n in names_or_ids:
            if n is None:
                continue
            d = self.decisions_by_id.get(str(n))
            if d is not None:
                return d
        # Synthetic empty decision; lets callers query without None checks.
        return DispatchDecision(dispatch_id=str(names_or_ids[0]) if names_or_ids else "?")

    def summary(self) -> dict[str, int]:
        """Histogram of hints across dispatches — for logging."""
        counts: dict[str, int] = {}
        for d in self.decisions_by_id.values():
            for h in d.hints:
                key = "pin_target" if h.startswith("pin_target=") else h
                counts[key] = counts.get(key, 0) + 1
        return counts


def load_feedback_overlay(merlin_dir: Path | str) -> FeedbackOverlay:
    """Read ``<merlin_dir>/breakdowns/feedback.json`` if present.

    Always returns a FeedbackOverlay. When the file is missing or
    malformed, returns an empty (inert) overlay — the consumer's
    existing default applies.
    """
    merlin_dir = Path(merlin_dir)
    fb_path = merlin_dir / "breakdowns" / "feedback.json"
    if not fb_path.is_file():
        return FeedbackOverlay()
    try:
        payload = json.loads(fb_path.read_text())
    except (OSError, json.JSONDecodeError):
        return FeedbackOverlay(source_path=fb_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return FeedbackOverlay(source_path=fb_path)

    decisions: dict[str, DispatchDecision] = {}
    for d_id, entry in (payload.get("dispatches") or {}).items():
        if not isinstance(entry, dict):
            continue
        hints = tuple(h for h in (entry.get("hints") or []) if isinstance(h, str))
        pin = None
        advisory: list[str] = []
        for h in hints:
            if h.startswith("pin_target="):
                pin = h[len("pin_target=") :]
            else:
                advisory.append(h)
        decisions[str(d_id)] = DispatchDecision(
            dispatch_id=str(d_id),
            hints=hints,
            pin_target=pin,
            advisory=tuple(advisory),
            rationale=str(entry.get("rationale") or ""),
        )

    return FeedbackOverlay(
        source_path=fb_path,
        run_id=payload.get("run_id"),
        model_signals=dict(payload.get("model_signals") or {}),
        decisions_by_id=decisions,
    )


def merlin_dir_from_chunk_manifest(chunk_manifest: Path | str) -> Path | None:
    """Walk up from a chunk_manifest.json path to find the merlin output
    dir (the directory whose 'breakdowns' subdir contains the manifest).
    Returns None if the path does not look like one.
    """
    p = Path(chunk_manifest).resolve()
    # Typical layout: <merlin_dir>/breakdowns/<chunk_manifest>.json
    if p.parent.name == "breakdowns":
        return p.parent.parent
    return None

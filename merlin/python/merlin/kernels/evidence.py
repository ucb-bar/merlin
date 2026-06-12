"""Collect per-kernel evidence for kernel records and downstream artifacts.

``code_markers`` are the concrete matched substrings (the L0 "observation" layer of the
promotion ladder). ``evidence_id`` is the stable ``<source>_<target>_<op>`` tag used in
abstraction/policy ``evidence`` lists, matching the schema examples.
"""
from __future__ import annotations

from merlin.kernels.types import NormalizedKernel


def collect_evidence(nk: NormalizedKernel, fired: dict[str, list[str]]) -> tuple[list[str], str]:
    """Return ``(code_markers, evidence_id)`` for a kernel."""
    markers: list[str] = []
    for snippets in fired.values():
        markers.extend(snippets)
    # de-dup, stable order
    code_markers = sorted(dict.fromkeys(markers))
    return code_markers, nk.evidence_id()

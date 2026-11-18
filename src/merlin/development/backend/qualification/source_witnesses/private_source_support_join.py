"""Exact linked-build join for source-only whole-model structural proofs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.development.backend.qualification.provenance import private_source_demand as demand
from merlin.development.backend.qualification.source_witnesses import bucketize, integer_reductions, ordered_scan
from merlin.development.backend.qualification.source_witnesses import control_guards as control
from merlin.development.backend.qualification.source_witnesses import data_movement as movement
from merlin.development.backend.qualification.source_witnesses import linalg_bodies as linalg
from merlin.development.backend.qualification.source_witnesses import (
    ordered_scan_admission as _merlin_ordered_scan_admission_1,
)
from merlin.development.backend.qualification.source_witnesses import private_f32_maximum_support as maximum
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_reduction_support as f32_reductions,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_index_host_support as prepared_index,
)
from merlin.development.backend.qualification.source_witnesses import private_linkage_support as linkage
from merlin.development.backend.qualification.source_witnesses import (
    private_linked_source_preparation as preparation,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_literal_arange_admission as arange,
)
from merlin.development.backend.qualification.source_witnesses import pure_stages as pure_stage
from merlin.development.backend.qualification.source_witnesses.pooling import linked_pool_complete

# ---- private_source_support_join ---------------------------------------------------------------


def linked_source_support_complete(checks: Mapping[str, Any], stages: Sequence[str], candidate_sha256: str) -> bool:
    """Keep all witness decisions; share only exact parsed captures in this join."""
    try:
        with preparation.validation_scope():
            return _complete(checks, stages, candidate_sha256)
    except preparation.PreparationChanged:
        return False


def _complete(checks: Mapping[str, Any], stages: Sequence[str], candidate_sha256: str) -> bool:
    """Every source-only proof needs the same exact linked whole-program build."""
    sources, build = checks.get("source"), checks.get("build")
    if not isinstance(sources, Mapping) or not isinstance(build, Mapping) or set(sources) != set(stages):
        return False
    entries = build.get("programs")
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        return False
    if [entry.get("program") for entry in entries] != list(stages):
        return False
    for stage, entry in zip(stages, entries, strict=True):
        source = sources[stage]
        if not isinstance(source, Mapping) or not isinstance(entry, Mapping):
            return False
        if not demand.linked_complete(source, entry, candidate_sha256):
            return False
        if not movement.linked_movement_complete(source, entry, candidate_sha256):
            return False
        if not linked_pool_complete(source, entry, candidate_sha256):
            return False
        if not linalg.linked_complete(source, entry, candidate_sha256):
            return False
        if not linkage.linked_complete(source, entry, candidate_sha256):
            return False
        if not pure_stage.linked_direct_return_complete(source, entry, candidate_sha256):
            return False
        if not control.linked_selected_build_complete(source, entry, candidate_sha256):
            return False
        if not arange.linked_complete(source, entry, candidate_sha256):
            return False
        if not integer_reductions.linked_complete(source, entry, candidate_sha256):
            return False
        if not f32_reductions.linked_complete(source, entry, candidate_sha256):
            return False
        if not maximum.linked_complete(source, entry, candidate_sha256):
            return False
        if not prepared_index.linked_complete(source, entry, candidate_sha256):
            return False
        if not _merlin_ordered_scan_admission_1.linked_complete(source, entry, candidate_sha256):
            return False
        if not bucketize.linked_source_complete(source, entry, candidate_sha256):
            return False
    return True

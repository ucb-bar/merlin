"""Operator source context for exact connected projections; no inherited verdicts.

This call-local carrier reconstructs the cut before source admission and reopens
both files at use. It transports only a selected compiler width. Reduction/control
facts are independently reproved on the cut, never copied from capture ordinals.
No captured-framework equivalence, host declaration or linked-build grant follows.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.frontends.connected_region import verify_window
from merlin.frontends.linalg_f32_reductions import recognize_static_f32_sum
from merlin.frontends.linalg_integer_reductions import recognize_static_integer_reduction
from merlin.frontends.linalg_patterns import InvalidLinalgPattern
from merlin.llvmlower.target_data_layout import selected_index_bits

_ISSUED = WeakKeyDictionary()


class ConnectedSourceContext:
    """A freshly reconstructed, byte-bound source context for one written cut."""

    def __init__(self, source: Path, projected: Path, *, source_sha256: str,
                 projected_sha256: str, start: int, stop: int, selected_index_observation: dict):
        source, projected = Path(source), Path(projected)
        if source.is_symlink() or projected.is_symlink():
            raise ValueError("connected source context requires direct source files")
        selected = copy.deepcopy(selected_index_observation)
        if selected_index_bits(selected) is None:
            raise ValueError("connected source context requires selected compiler index observation")
        raw, cut = source.read_bytes(), projected.read_bytes()
        verify_window(raw.decode(), cut.decode(), source_sha256=source_sha256,
                               projected_sha256=projected_sha256, start=start, stop=stop)
        _ISSUED[self] = (source, projected, raw, cut, selected)

    def _reopen(self, program):
        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("connected source context was not reconstructed")
        source, projected, raw, cut, selected = issued
        if Path(program) != projected or source.is_symlink() or projected.is_symlink():
            raise ValueError("connected source context names another projection")
        if source.read_bytes() != raw or projected.read_bytes() != cut:
            raise ValueError("connected source context bytes changed")
        return raw, cut, selected

    def for_program(self, program):
        _, _, selected = self._reopen(program)
        return {"selected_index_observation": copy.deepcopy(selected)}

    def source_binding(self, program):
        """Identify the independently reconstructed source/cut pair, without a grant."""
        raw, cut, _ = self._reopen(program)
        return {"source_sha256": sha256(raw).hexdigest(),
                "projected_sha256": sha256(cut).hexdigest()}

    def prove_support(self, *, target: str):
        """Reprove exact cut reductions and guards, retaining every refusal."""
        from merlin.targetgen.application_inventory import _application_operation_inventory
        from merlin_experiments.phase1.feedback.private_control_support import prove_bounded_assertions

        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("connected source context was not reconstructed")
        program = issued[1]
        raw, cut, selected = self._reopen(program)
        normalized, normalization = normalize_capture_mlir(cut.decode())
        module = mq.parse(normalized)
        width = selected_index_bits(selected)
        reductions = []
        for ordinal, operation in enumerate(mq.walk(module)):
            if operation.name != "linalg.reduce":
                continue
            row = {"ordinal": ordinal, "root_source_node_ids": str(operation.attributes.get("prov.source_node_ids", ""))}
            try:
                dtype = str(operation.results[0].type.element_type)
                recognizer = recognize_static_f32_sum if dtype == "f32" else recognize_static_integer_reduction
                row.update(status="source_proved_pending_host_and_build", pattern=asdict(recognizer(operation, index_bits=width)))
            except (InvalidLinalgPattern, ValueError, AttributeError) as error:
                row.update(status="refused", reason=str(error))
            reductions.append(row)
        inventory = _application_operation_inventory(program, target, {})
        try:
            guards = prove_bounded_assertions(module, inventory, raw_sha256=normalization["input_sha256"],
                                             normalized_sha256=normalization["output_sha256"], index_bits=width)
        except ValueError as error:
            guards = {"status": "refused", "reason": str(error)}
        self._reopen(program)
        return {"schema": "merlin.connected_source_context.v1", "original_sha256": sha256(raw).hexdigest(),
                "projected_sha256": sha256(cut).hexdigest(), "normalized_projected_sha256": normalization["output_sha256"],
                "selected_index_observation": copy.deepcopy(selected), "reductions": reductions, "bounded_control": guards,
                "scope": "exact projected source premises only; no capture trace equivalence, host admission or linked execution"}


def require_context(context, captures):
    if type(context) is not ConnectedSourceContext or context not in _ISSUED or len(captures) != 1:
        raise ValueError("source admission requires one issued connected source context")
    context._reopen(next(iter(captures.values())))

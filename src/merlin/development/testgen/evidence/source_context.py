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
from merlin.compiler.frontend.capture_normalization import normalize_capture_mlir
from merlin.compiler.frontend.connected_region import verify_window
from merlin.compiler.frontend.linalg_f32_reductions import recognize_static_f32_sum
from merlin.compiler.frontend.linalg_integer_reductions import recognize_static_integer_reduction
from merlin.compiler.frontend.linalg_patterns import InvalidLinalgPattern
from merlin.compiler.ir import mlir_query as mq
from merlin.compiler.lower.model.target_data_layout import selected_index_bits
import json
from merlin.compiler.dialects._common import text
from merlin.development.backend.feedback import private_index_source as index
from merlin.development.backend.feedback import private_literal_arange as arange


# ---- connected_source_context ------------------------------------------------------------------

_ISSUED = WeakKeyDictionary()
_CAPTURES = WeakKeyDictionary()


class ConnectedSourceContext:
    """A freshly reconstructed, byte-bound source context for one written cut."""

    def __init__(
        self,
        source: Path,
        projected: Path,
        *,
        source_sha256: str,
        projected_sha256: str,
        start: int,
        stop: int,
        selected_index_observation: dict,
    ):
        source, projected = Path(source), Path(projected)
        if source.is_symlink() or projected.is_symlink():
            raise ValueError("connected source context requires direct source files")
        selected = copy.deepcopy(selected_index_observation)
        if selected_index_bits(selected) is None:
            raise ValueError("connected source context requires selected compiler index observation")
        raw, cut = source.read_bytes(), projected.read_bytes()
        verify_window(
            raw.decode(),
            cut.decode(),
            source_sha256=source_sha256,
            projected_sha256=projected_sha256,
            start=start,
            stop=stop,
        )
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
        return {"source_sha256": sha256(raw).hexdigest(), "projected_sha256": sha256(cut).hexdigest()}

    def bind_index_capture(self, program, capture, **pins):
        """Bind direct capture inputs; every admission independently reopens and reproves them."""

        prove(self, program, capture, **pins)
        _CAPTURES[self] = (Path(capture), copy.deepcopy(pins))

    def index_admission_context(self, program):

        self._reopen(program)
        if self not in _CAPTURES:
            return None
        capture, pins = _CAPTURES[self]
        record = prove(self, program, capture, **pins)
        return {
            "selected_index_observation": self.for_program(program)["selected_index_observation"],
            "verified_connected_index_source": record,
        }

    def control_admission_context(self, program, *, target):
        """Reprove scalar support on the entire exact projection; no saved proof input."""
        from merlin.compiler.dialects._common import text
        from merlin.compiler.frontend.bounded_control_source_body import PROOF_SCHEMA, ROLES
        from merlin.development.backend.feedback.private_control_support import assertion_row_proven

        _, cut, selected = self._reopen(program)
        support = self.prove_support(target=target)
        proof = support["bounded_control"]
        normalized, normalization = normalize_capture_mlir(cut.decode())
        module = mq.parse(normalized)
        parsed = tuple(mq.walk(module))
        roles = []
        by_operation = {value[0]: role for role, value in ROLES.items()}
        for ordinal, operation in enumerate(parsed):
            if operation.name not in by_operation:
                continue
            row = {"mlir_operation": operation.name, "ordinals": [ordinal], "count": 1}
            if assertion_row_proven(row, proof):
                roles.append({"ordinal": ordinal, "role": by_operation[operation.name]})
        self._reopen(program)
        return {
            "selected_index_observation": copy.deepcopy(selected),
            "verified_bounded_control_source": {
                "schema": PROOF_SCHEMA,
                "status": "source_proved_pending_linked_build",
                "raw_source_sha256": sha256(cut).hexdigest(),
                "normalized_source_sha256": normalization["output_sha256"],
                "parsed_module_sha256": sha256(text(module, generic=True).encode()).hexdigest(),
                "index_bits": selected_index_bits(selected),
                "n_source_operations": len(parsed),
                "roles": roles,
            },
        }

    def prove_support(self, *, target: str):
        """Reprove exact cut reductions and guards, retaining every refusal."""
        from merlin.development.backend.feedback.private_control_support import prove_bounded_assertions
        from merlin.development.targetgen.application_inventory import _application_operation_inventory

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
            row = {
                "ordinal": ordinal,
                "root_source_node_ids": str(operation.attributes.get("prov.source_node_ids", "")),
            }
            try:
                dtype = str(operation.results[0].type.element_type)
                recognizer = recognize_static_f32_sum if dtype == "f32" else recognize_static_integer_reduction
                row.update(
                    status="source_proved_pending_host_and_build",
                    pattern=asdict(recognizer(operation, index_bits=width)),
                )
            except (InvalidLinalgPattern, ValueError, AttributeError) as error:
                row.update(status="refused", reason=str(error))
            reductions.append(row)
        inventory = _application_operation_inventory(program, target, {})
        try:
            guards = prove_bounded_assertions(
                module,
                inventory,
                raw_sha256=normalization["input_sha256"],
                normalized_sha256=normalization["output_sha256"],
                index_bits=width,
            )
        except ValueError as error:
            guards = {"status": "refused", "reason": str(error)}
        self._reopen(program)
        return {
            "schema": "merlin.connected_source_context.v1",
            "original_sha256": sha256(raw).hexdigest(),
            "projected_sha256": sha256(cut).hexdigest(),
            "normalized_projected_sha256": normalization["output_sha256"],
            "selected_index_observation": copy.deepcopy(selected),
            "reductions": reductions,
            "bounded_control": guards,
            "scope": "exact projected source premises only; no capture trace equivalence, host admission or linked execution",
        }


def require_context(context, captures):

    if type(context) is WholeProgramSourceContext:
        context.require(captures)
        return
    if type(context) is not ConnectedSourceContext or context not in _ISSUED or len(captures) != 1:
        raise ValueError("source admission requires one issued connected source context")
    context._reopen(next(iter(captures.values())))


# ---- connected_index_context -------------------------------------------------------------------
# Reprove index premises on an exact cut with separately pinned capture metadata.
#
# Only closed range/count/extract forms are handled. This is a source observation,
# not a serialized host grant or whole capture/preparation equivalence claim.

def _read(path, expected):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("connected index context requires direct source files")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("connected index context source pin differs")
    return raw


def _root_body(op):
    """Canonical typed body with explicit external SSA; provenance is checked separately."""
    from xdsl.ir import Block, Operation

    from merlin.compiler.dialects._common import text

    inside = set(op.walk())
    blocks = {block for nested in inside for region in nested.regions for block in region.blocks}
    external = []
    for nested in op.walk():
        for value in nested.operands:
            if (
                (isinstance(value.owner, Operation) and value.owner in inside)
                or isinstance(value.owner, Block)
                and value.owner in blocks
            ):
                continue
            if value not in external:
                external.append(value)
    boundary = Block(arg_types=[value.type for value in external])
    clone = op.clone(value_mapper=dict(zip(external, boundary.args, strict=True)))
    for nested in clone.walk():
        nested.attributes = {k: v for k, v in nested.attributes.items() if not k.startswith("prov.")}
    boundary.add_op(clone)
    return ([str(value.type) for value in external], text(clone, generic=True))


def _capture_join(model, trace, roots):
    """Require selected cut roots to retain the exact captured typed bodies and trace owners."""
    normalized, _ = normalize_capture_mlir(model.decode())
    parsed = tuple(mq.walk(mq.parse(normalized)))
    mlir = trace.get("mlir") or {}
    entries, joins = mlir.get("operations"), mlir.get("source_correspondence")
    if not isinstance(entries, list) or len(entries) != len(parsed) or not isinstance(joins, list):
        raise ValueError("connected index capture operation roster differs")
    result = []
    for node_id, projected in roots.items():
        selected = [r for r in joins if isinstance(r, dict) and r.get("node_id") == node_id]
        owned = {i for i, op in enumerate(parsed) if index._ids(op) == (node_id,)}
        if (
            len(selected) != 1
            or selected[0].get("status") != "lowered"
            or type(selected[0].get("mlir_ordinals")) is not list
            or any(type(i) is not int for i in selected[0]["mlir_ordinals"])
            or set(selected[0]["mlir_ordinals"]) != owned
            or len(selected[0]["mlir_ordinals"]) != len(owned)
            or any(
                r.get("node_id") != node_id and owned.intersection(r.get("mlir_ordinals") or [])
                for r in joins
                if isinstance(r, dict)
            )
        ):
            raise ValueError("connected index capture source correspondence differs")
        captured = [(i, parsed[i]) for i in sorted(owned) if parsed[i].name in {"linalg.generic", "linalg.reduce"}]
        if len(captured) != len(projected):
            raise ValueError("connected index captured compute roster differs")
        for (old_i, old), (new_i, new) in zip(captured, projected, strict=True):
            if old.name != new.name or _root_body(old) != _root_body(new):
                raise ValueError("connected index typed capture body differs")
            result.append(
                {
                    "source_node_id": node_id,
                    "capture_ordinal": old_i,
                    "projected_ordinal": new_i,
                    "typed_body_sha256": sha256(json.dumps(_root_body(new), sort_keys=True).encode()).hexdigest(),
                }
            )
    return result


def prove(
    context,
    program: Path,
    capture: Path,
    *,
    capture_model_sha256: str,
    frontend_trace_sha256: str,
    capture_receipt_sha256: str,
):
    """Recompute subset facts, never remap saved full-program proof ordinals."""

    require_context(context, {"region": program})
    source_binding = context.source_binding(program)
    selected = context.for_program(program)["selected_index_observation"]
    width = selected_index_bits(selected)
    capture = Path(capture)
    model_path, trace_path, receipt_path = (
        capture / n for n in ("model.mlir", "frontend-trace.json", "capture_receipt.json")
    )
    model = _read(model_path, capture_model_sha256)
    trace_bytes = _read(trace_path, frontend_trace_sha256)
    receipt_bytes = _read(receipt_path, capture_receipt_sha256)
    trace, receipt = json.loads(trace_bytes), json.loads(receipt_bytes)
    artifacts = receipt.get("artifacts") or {}
    if (
        receipt.get("schema") != "m2m.capture-receipt.v1"
        or artifacts.get("model.mlir", {}).get("sha256") != capture_model_sha256
        or artifacts.get("frontend-trace.json", {}).get("sha256") != frontend_trace_sha256
    ):
        raise ValueError("connected index capture receipt does not bind source and trace")
    prepared = (trace.get("graphs") or {}).get("prepared") or {}
    if (
        trace.get("schema") != "m2m.frontend_trace.v1"
        or trace.get("status") != "complete"
        or prepared.get("status") != "complete"
        or (trace.get("mlir") or {}).get("sha256") != capture_model_sha256
        or (trace.get("mlir") or {}).get("bytes") != len(model)
    ):
        raise ValueError("connected index trace is incomplete or belongs to another capture")
    nodes = prepared.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(n, dict) or not isinstance(n.get("id"), str) for n in nodes):
        raise ValueError("connected index trace lacks prepared node identities")
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("connected index trace repeats a prepared node")
    raw = program.read_bytes()
    normalized, normalization = normalize_capture_mlir(raw.decode())
    module = mq.parse(normalized)
    module.verify()
    parsed = tuple(mq.walk(module))
    ordinals = {id(op): i for i, op in enumerate(parsed)}
    ranges, literals, range_rows = {}, {}, []
    roots = {}
    for ordinal, op in enumerate(parsed):
        target = mq.attr_str(op, "prov.aten")
        if op.name not in {"linalg.generic", "linalg.reduce"} or target not in {
            "aten.arange.default",
            "aten.arange.start",
            "aten.arange.start_step",
            "aten.index.Tensor",
        }:
            continue
        ids = index._ids(op)
        if len(ids) != 1 or ids[0] not in by_id or by_id[ids[0]].get("target") != target:
            raise ValueError("connected index source identity has no exact prepared trace node")
        if target == "aten.index.Tensor":
            roots.setdefault(ids[0], []).append((ordinal, op))
            continue
        start, end, step, extent = arange._range(by_id[ids[0]])
        arange._body(op, start, step, extent, width)
        literal = sha256(json.dumps([start, end, step, extent], separators=(",", ":")).encode()).hexdigest()
        ranges[id(op)], literals[id(op)] = (start, end, step, extent), literal
        range_rows.append(
            {
                "ordinal": ordinal,
                "source_node_id": ids[0],
                "range": [start, end, step, extent],
                "literal_sha256": literal,
            }
        )
    joins = _capture_join(model, trace, roots)
    records = []
    for node_id, owned_roots in roots.items():
        owned = {i for i, op in enumerate(parsed) if index._ids(op) == (node_id,)}
        if len(owned_roots) == 2 and {op.name for _, op in owned_roots} == {"linalg.generic", "linalg.reduce"}:
            reduction = next(op for _, op in owned_roots if op.name == "linalg.reduce")
            fact = index._mask_count(reduction, owned, ordinals, width)
        elif len(owned_roots) == 1 and owned_roots[0][1].name == "linalg.generic":
            fact = index._indexed_extract(
                owned_roots[0][1], index_bits=width, ranges=ranges, range_literals=literals, ordinals=ordinals
            )
        else:
            raise ValueError("connected index node has an unmatched or partial compute roster")
        if not set(fact["component_ordinals"]) <= owned:
            raise ValueError("connected index components belong to another source node")
        records.append({"source_node_id": node_id, "compute_ordinals": [i for i, _ in owned_roots], **fact})
    require_context(context, {"region": program})
    for path, pin in (
        (model_path, capture_model_sha256),
        (trace_path, frontend_trace_sha256),
        (receipt_path, capture_receipt_sha256),
    ):
        _read(path, pin)
    return {
        "schema": "merlin.connected_index_source_observation.v1",
        **source_binding,
        "status": "source_proved_pending_capture_preparation_join_host_and_build",
        "normalized_projected_sha256": normalization["output_sha256"],
        "capture_model_sha256": capture_model_sha256,
        "frontend_trace_sha256": frontend_trace_sha256,
        "capture_receipt_sha256": capture_receipt_sha256,
        "selected_index_observation": selected,
        "ranges": range_rows,
        "records": records,
        "capture_root_joins": joins,
        "parsed_module_sha256": sha256(text(module, generic=True).encode()).hexdigest(),
        "n_source_operations": len(parsed),
        "scope": "closed cut index forms and literal graph metadata only; full capture/preparation equivalence, host placement and linked numerics remain unproved",
    }


# ---- program_source_context --------------------------------------------------------------------
# Byte-bound compiler-width premises for complete original source programs.
#
# This context proves neither window extraction nor operation support. Existing
# source-body recognizers still inspect each actual operation independently.

_ISSUED = WeakKeyDictionary()


class WholeProgramSourceContext:
    """A closed roster of unchanged full programs and a selected compiler observation."""

    def __init__(self, captures, *, source_sha256, selected_index_observation):
        if (
            type(captures) is not dict
            or not captures
            or len(captures) > 512
            or set(captures) != set(source_sha256)
            or any(type(name) is not str or not name for name in captures)
        ):
            raise ValueError("whole-program context requires an exact bounded source roster")
        selected = copy.deepcopy(selected_index_observation)
        if selected_index_bits(selected) is None:
            raise ValueError("whole-program context requires selected compiler index observation")
        rows = {}
        for name, location in captures.items():
            path = Path(location)
            if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
                raise ValueError("whole-program context requires direct canonical source files")
            raw = path.read_bytes()
            if sha256(raw).hexdigest() != source_sha256[name]:
                raise ValueError("whole-program source differs from selected bytes")
            rows[name] = (path, raw)
        _ISSUED[self] = (rows, selected)

    def require(self, captures):
        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("whole-program source context was not issued")
        rows, _ = issued
        if set(captures) != set(rows) or any(Path(captures[name]) != rows[name][0] for name in rows):
            raise ValueError("whole-program context names another source roster")
        for path, raw in rows.values():
            if path.is_symlink() or path.resolve() != path or path.read_bytes() != raw:
                raise ValueError("whole-program context source bytes changed")

    def for_program(self, program):
        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("whole-program source context was not issued")
        rows, selected = issued
        self.require({name: path for name, (path, _) in rows.items()})
        if Path(program) not in {path for path, _ in rows.values()}:
            raise ValueError("whole-program context names another program")
        return {"selected_index_observation": copy.deepcopy(selected)}

    def control_admission_context(self, program, *, target):
        self.for_program(program)
        return None

    def index_admission_context(self, program):
        self.for_program(program)
        return None

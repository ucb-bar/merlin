"""Bounded, complete-original contraction format observations.

Selections bind bytes, not capture provenance or numerical equivalence. The
supported positive domain is a flat, one-to-one rank-two signed i8 reduction
with i32 accumulation. Other calls, shapes, control flow and decompositions
remain in the census as UNKNOWN. Static MACs are arithmetic work, not cycles,
traffic, offload or target admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.jsonio import canonical_sha256
from merlin.common.pinned_files import PinnedFile


class ContractionFormatError(ValueError):
    """An explicit format selection cannot be established."""


@dataclass(frozen=True)
class ContractionFormatLimits:
    file_bytes: int = 4 * 1024 * 1024
    total_bytes: int = 16 * 1024 * 1024
    metadata_items: int = 100_000
    graph_nodes: int = 4096
    operations: int = 32768
    programs: int = 128
    integer_bits: int = 128

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ContractionFormatError("Contraction input exceeds its selected budget.")


@dataclass(frozen=True)
class ProgramContractionInput:
    program: str
    original_graph: PinnedFile
    frontend_trace: PinnedFile
    final_mlir: PinnedFile


@dataclass(frozen=True)
class CaptureContractionSelection:
    session_contract: PinnedFile
    programs: tuple[ProgramContractionInput, ...]
    policy: str = "complete_integer.v1"
    limits: ContractionFormatLimits = ContractionFormatLimits()

    def __post_init__(self):
        if (
            type(self.session_contract) is not PinnedFile
            or type(self.limits) is not ContractionFormatLimits
            or self.policy != "complete_integer.v1"
            or type(self.programs) is not tuple
            or not self.programs
            or len(self.programs) > self.limits.programs
            or any(type(row) is not ProgramContractionInput for row in self.programs)
            or any(type(row.program) is not str or not row.program for row in self.programs)
            or len({row.program for row in self.programs}) != len(self.programs)
            or any(
                type(pin) is not PinnedFile
                for row in self.programs
                for pin in (row.original_graph, row.frontend_trace, row.final_mlir)
            )
        ):
            raise ContractionFormatError("Original contraction membership is incomplete.")


@dataclass(frozen=True)
class OriginalProgramContractionInput:
    program: str
    bundle: str
    original_graph: PinnedFile


@dataclass(frozen=True)
class CaptureContractionOriginalSelection:
    """Original byte selections made before capture dispatch; not source authority."""

    programs: tuple[OriginalProgramContractionInput, ...]
    policy: str = "complete_integer.v1"
    limits: ContractionFormatLimits = ContractionFormatLimits()

    def __post_init__(self):
        if (
            type(self.limits) is not ContractionFormatLimits
            or self.policy != "complete_integer.v1"
            or type(self.programs) is not tuple
            or not self.programs
            or len(self.programs) > self.limits.programs
            or any(type(row) is not OriginalProgramContractionInput for row in self.programs)
            or any(type(row.program) is not str or not row.program for row in self.programs)
            or any(type(row.bundle) is not str for row in self.programs)
            or len({row.program for row in self.programs}) != len(self.programs)
            or len({row.bundle for row in self.programs}) != len(self.programs)
            or any(type(row.original_graph) is not PinnedFile for row in self.programs)
        ):
            raise ContractionFormatError("Original contraction membership is incomplete.")
        for row in self.programs:
            if (
                type(row.bundle) is not str
                or not row.bundle
                or Path(row.bundle).is_absolute()
                or Path(row.bundle).as_posix() != row.bundle
                or ".." in Path(row.bundle).parts
                or "\\" in row.bundle
                or "\0" in row.bundle
            ):
                raise ContractionFormatError("Original contraction membership is incomplete.")


def _file_bytes(path: Path, limits: ContractionFormatLimits) -> bytes:
    try:
        if path.resolve() != path or path.is_symlink():
            raise OSError
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OSError
            raw = stream.read(limits.file_bytes + 1)
    except OSError as error:
        raise ContractionFormatError("Contraction source bytes changed.") from error
    if len(raw) > limits.file_bytes:
        raise ContractionFormatError("Contraction input exceeds its selected budget.")
    return raw


def _read(pin: PinnedFile, limits: ContractionFormatLimits) -> bytes:
    raw = _file_bytes(pin.path, limits)
    if hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise ContractionFormatError("Contraction source bytes changed.")
    return raw


def verify_original_contraction_selection(selection: CaptureContractionOriginalSelection) -> None:
    """Reopen all original bytes before dispatch, without trusting saved coverage."""
    if type(selection) is not CaptureContractionOriginalSelection:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    raw, total = [], 0
    for row in selection.programs:
        item = _read(row.original_graph, selection.limits)
        total += len(item)
        if total > selection.limits.total_bytes:
            raise ContractionFormatError("Contraction input exceeds its selected budget.")
        raw.append(item)
    for item in raw:
        _original(_json(item, selection.limits), selection.limits)


def original_contraction_selection_record(selection: CaptureContractionOriginalSelection) -> dict:
    verify_original_contraction_selection(selection)
    return {
        "schema": "merlin.capture_contraction_original_selection.v1",
        "policy": selection.policy,
        "limits": asdict(selection.limits),
        "programs": [
            {
                "program": row.program,
                "bundle": row.bundle,
                "original_graph": {"path": str(row.original_graph.path), "sha256": row.original_graph.sha256},
            }
            for row in selection.programs
        ],
        "scope": "preselected original bytes only; source producer authority not established",
    }


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ContractionFormatError("Original contraction membership is incomplete.")
        result[key] = value
    return result


def _bounded_metadata(value, limits):
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > limits.metadata_items or depth > 64:
            raise ContractionFormatError("Contraction input exceeds its selected budget.")
        if type(item) is int and item.bit_length() > limits.integer_bits:
            raise ContractionFormatError("Contraction input exceeds its selected budget.")
        if isinstance(item, dict):
            pending.extend((part, depth + 1) for part in item.values())
        elif isinstance(item, list):
            pending.extend((part, depth + 1) for part in item)


def _json(raw, limits):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractionFormatError("Original contraction membership is incomplete.") from error
    _bounded_metadata(value, limits)
    if not isinstance(value, dict):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    return value


def _digest(value):
    return canonical_sha256(value)


def _shape(value):
    shape = value.get("shape") if isinstance(value, dict) else None
    return shape if isinstance(shape, list) and shape and all(type(dim) is int and dim > 0 for dim in shape) else None


_CONTRACTIONS = {
    "aten.mm.default": "matmul",
    "aten.matmul.default": "matmul",
    "aten.bmm.default": "batch_matmul",
    "aten.linear.default": "linear",
    "aten.addmm.default": "addmm",
    "aten.conv2d.default": "conv2d",
    "aten.convolution.default": "unsupported_convolution",
}


def _original(graph, limits):
    nodes = graph.get("nodes")
    if (
        graph.get("schema") != "m2m.frontend_graph.v1"
        or graph.get("stage") != "original"
        or graph.get("status") != "complete"
        or graph.get("counting_unit") != "static_captured_call_sites"
        or not isinstance(nodes, list)
        or not nodes
        or len(nodes) > limits.graph_nodes
    ):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    body = {key: value for key, value in graph.items() if key != "sha256"}
    if graph.get("sha256") != _digest(body):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    ids, values = set(), {}
    graph_ordinals = Counter()
    for node in nodes:
        if not isinstance(node, dict) or type(node.get("id")) is not str or node["id"] in ids:
            raise ContractionFormatError("Original contraction membership is incomplete.")
        graph_id = node.get("graph_id")
        if (
            type(graph_id) is not str
            or type(node.get("ordinal")) is not int
            or node["ordinal"] != graph_ordinals[graph_id]
        ):
            raise ContractionFormatError("Original contraction membership is incomplete.")
        graph_ordinals[graph_id] += 1
        if not isinstance(node.get("results"), list):
            raise ContractionFormatError("Original contraction membership is incomplete.")
        pending = [node.get("args"), node.get("kwargs")]
        while pending:
            argument = pending.pop()
            if isinstance(argument, dict):
                if "node_id" in argument or "value_id" in argument:
                    linked = values.get(argument.get("value_id"))
                    if not linked or linked[0] != argument.get("node_id"):
                        raise ContractionFormatError("Original contraction membership is incomplete.")
                else:
                    pending.extend(argument.values())
            elif isinstance(argument, list):
                pending.extend(argument)
        ids.add(node["id"])
        for value in node.get("results", ()):
            if not isinstance(value, dict) or type(value.get("id")) is not str or value["id"] in values:
                raise ContractionFormatError("Original contraction membership is incomplete.")
            values[value["id"]] = (node["id"], value)
    calls = [node for node in nodes if node.get("op") in {"call_function", "call_method", "call_module"}]
    if (
        not isinstance(graph.get("by_target"), dict)
        or any(type(node.get("target")) is not str for node in calls)
        or type(graph.get("call_count")) is not int
        or graph["call_count"] != len(calls)
        or graph.get("by_target") != dict(Counter(node.get("target") for node in calls))
        or any(type(value) is not int for value in graph.get("by_target", {}).values())
    ):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    rows = []
    for node in calls:
        family = _CONTRACTIONS.get(node.get("target"), "unknown_potential_contraction")
        args = node.get("args")
        operand_shapes = []
        operand_values = []
        if isinstance(args, list):
            for arg in args:
                linked = values.get(arg.get("value_id")) if isinstance(arg, dict) else None
                operand_values.append(linked[1] if linked and linked[0] == arg.get("node_id") else None)
                operand_shapes.append(_shape(linked[1]) if linked and linked[0] == arg.get("node_id") else None)
        result = node.get("results", [])
        output = _shape(result[0]) if len(result) == 1 else None
        geometry = None
        if family == "matmul":
            selected = operand_shapes[:2]
            if len(selected) == 2 and all(selected) and output:
                a, b = selected
                if len(a) == len(b) == len(output) == 2 and a[1] == b[0] and output == [a[0], b[1]]:
                    geometry = [a[0], b[1], a[1]]
        macs = 1 if geometry else None
        if geometry:
            for dim in geometry:
                macs *= dim
                if macs.bit_length() > limits.integer_bits:
                    raise ContractionFormatError("Contraction input exceeds its selected budget.")
        flat = node.get("graph_id") == "g:original:root" and node.get("op") == "call_function"
        typed_values = [*operand_values, *result]
        typed = all(isinstance(value, dict) and value.get("kind") == "tensor" for value in typed_values)
        dtypes = [value.get("dtype") if isinstance(value, dict) else None for value in typed_values]
        typed = (
            typed
            and len(result) == 1
            and all(type(dtype) is str for dtype in dtypes)
            and len(set(dtypes)) == 1
            and dtypes[0] in {"float16", "bfloat16", "float32", "float64", "int8", "int32"}
        )
        exact_call = (
            family == "matmul" and isinstance(args, list) and len(args) == 2 and node.get("kwargs") == {} and typed
        )
        rows.append(
            {
                "original_node_id": node["id"],
                "target": node.get("target"),
                "family": family,
                "original_operand_dtypes": dtypes[:-1] if len(result) == 1 else [],
                "original_result_dtype": dtypes[-1] if len(result) == 1 else None,
                "static_macs": macs,
                "geometry": geometry,
                "multiplicity": 1 if flat else None,
                "supported_original_call": exact_call,
                "format": "UNKNOWN",
                "fragment_ordinals": [],
                "reason": "correspondence_unavailable",
            }
        )
    # Unknown non-call node kinds also block the complete-original denominator.
    unknown_nodes = [
        node["id"]
        for node in nodes
        if node.get("op") not in {"placeholder", "get_attr", "output", "call_function", "call_method", "call_module"}
    ]
    return rows, unknown_nodes


def _typed_geometry(op):
    from merlin.common import mlir_query as query

    typed = [query.type_shape_dtype(value.type) for value in (*op.operands, *op.results)]
    if len(typed) != 4:
        return None
    a, b, out, result = [shape for shape, _ in typed]
    if (
        len(a) != 2
        or len(b) != 2
        or len(out) != 2
        or len(result) != 2
        or not all(dim > 0 for shape in (a, b, out, result) for dim in shape)
        or a[1] != b[0]
        or out != result
        or out != [a[0], b[1]]
    ):
        return None
    return [a[0], b[1], a[1]]


def _float_reduction(op):
    from merlin.common import mlir_query as query
    from merlin.targetgen.application_inventory import _INT_MM_ITERATORS, _INT_MM_MAPS

    if query.op_name(op) != "linalg.generic" or _typed_geometry(op) is None:
        return False
    if [str(value) for value in op.properties.get("indexing_maps", ())] != _INT_MM_MAPS or [
        str(value) for value in op.properties.get("iterator_types", ())
    ] != _INT_MM_ITERATORS:
        return False
    dtypes = [query.type_shape_dtype(value.type)[1] for value in (*op.operands, *op.results)]
    if len(set(dtypes)) != 1 or dtypes[0] not in {"f16", "bf16", "f32", "f64"}:
        return False
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    if len(block.args) != 3 or [query.op_name(value) for value in block.ops] != [
        "arith.mulf",
        "arith.addf",
        "linalg.yield",
    ]:
        return False
    mul, add, yield_op = block.ops
    if any(
        key not in {"fastmath"} or str(value) != "#arith.fastmath<none>"
        for inner in (mul, add, yield_op)
        for key, value in {**inner.attributes, **inner.properties}.items()
        if not key.startswith("prov.")
    ):
        return False
    return (
        set(mul.operands) == {block.args[0], block.args[1]}
        and set(add.operands) == {block.args[2], mul.results[0]}
        and list(yield_op.operands) == [add.results[0]]
    )


def _program(row, limits):
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from merlin.common import mlir_query as query
    from merlin.common.ir_lock import IR_LOCK
    from merlin.targetgen.application_inventory import exact_int_mm_generic_operation

    original = _json(_read(row.original_graph, limits), limits)
    trace = _json(_read(row.frontend_trace, limits), limits)
    raw = _read(row.final_mlir, limits)
    rows, unknowns = _original(original, limits)
    if (
        trace.get("schema") != "m2m.frontend_trace.v1"
        or not isinstance(trace.get("graphs"), dict)
        or not isinstance(trace.get("mlir"), dict)
        or _digest((trace.get("graphs") or {}).get("original")) != _digest(original)
        or (trace.get("mlir") or {}).get("sha256") != row.final_mlir.sha256
        or type((trace.get("mlir") or {}).get("bytes")) is not int
        or trace["mlir"]["bytes"] != len(raw)
    ):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    actual_records, fragments = [], []
    with IR_LOCK:
        try:
            module = query.parse(raw.decode("utf-8"))
        except Exception as error:
            raise ContractionFormatError("Original contraction membership is incomplete.") from error
        operations = list(module.walk())
        if len(operations) > limits.operations:
            raise ContractionFormatError("Contraction input exceeds its selected budget.")
        try:
            module.verify()
        except Exception as error:
            raise ContractionFormatError("Original contraction membership is incomplete.") from error
        functions = [op for op in operations if query.op_name(op) == "func.func"]
        if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
            unknowns.append({"reason": "invocation_multiplicity_unavailable"})
        for ordinal, op in enumerate(operations):

            def ids(key):
                attr = op.attributes.get(key)
                return (
                    [value.data for value in attr]
                    if isinstance(attr, ArrayAttr) and all(isinstance(value, StringAttr) for value in attr)
                    else []
                )

            name = query.op_name(op)
            actual_records.append(
                {
                    "ordinal": ordinal,
                    "operation": name,
                    "source_node_ids": ids("prov.source_node_ids"),
                    "origin_node_ids": ids("prov.origin_node_ids"),
                    "operand_types": [str(value.type) for value in op.operands],
                    "result_types": [str(value.type) for value in op.results],
                }
            )
            if name in {"linalg.matmul", "linalg.batch_matmul"} or (
                name == "linalg.generic"
                and any(
                    "reduction" == getattr(getattr(value, "data", None), "value", None)
                    or str(value) == "#linalg.iterator_type<reduction>"
                    for value in op.properties.get("iterator_types", ())
                )
            ):
                geometry = _typed_geometry(op)
                form = "UNKNOWN"
                if exact_int_mm_generic_operation(op):
                    form = "INTEGER"
                elif _float_reduction(op):
                    form = "FLOAT"
                fragments.append(
                    {"ordinal": ordinal, "origins": ids("prov.origin_node_ids"), "geometry": geometry, "format": form}
                )
            elif name not in {
                "builtin.module",
                "func.func",
                "func.return",
                "tensor.empty",
                "arith.constant",
                "linalg.fill",
                "linalg.yield",
                "arith.extsi",
                "arith.muli",
                "arith.addi",
                "arith.mulf",
                "arith.addf",
            }:
                unknowns.append({"operation_ordinal": ordinal, "reason": "operation_semantics_unavailable"})
    recorded = trace["mlir"].get("operations")
    if (
        not isinstance(recorded, list)
        or len(recorded) != len(actual_records)
        or any(
            not isinstance(saved, dict) or _digest({key: saved.get(key) for key in actual}) != _digest(actual)
            for saved, actual in zip(recorded, actual_records, strict=True)
        )
    ):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    by_id = {item["original_node_id"]: item for item in rows}
    for fragment in fragments:
        origins = fragment["origins"]
        if len(origins) != 1 or origins[0] not in by_id:
            unknowns.append({"operation_ordinal": fragment["ordinal"], "reason": "fragment_lineage_unavailable"})
        else:
            by_id[origins[0]]["fragment_ordinals"].append(fragment["ordinal"])
    by_ordinal = {item["ordinal"]: item for item in fragments}
    for item in rows:
        linked = [by_ordinal[value] for value in item["fragment_ordinals"]]
        if (
            item["supported_original_call"]
            and item["multiplicity"] == 1
            and len(linked) == 1
            and item["geometry"] is not None
            and linked[0]["geometry"] == item["geometry"]
        ):
            item["format"] = linked[0]["format"]
            item["reason"] = (
                "typed_reduction_observed" if item["format"] != "UNKNOWN" else "reduction_semantics_unavailable"
            )
    # Snapshot freshness covers all selected bytes after parsing and observation.
    for pin in (row.original_graph, row.frontend_trace, row.final_mlir):
        _read(pin, limits)
    return {
        "program": row.program,
        "rows": rows,
        "unknowns": unknowns,
        "source_sha256": {
            "original_graph": row.original_graph.sha256,
            "frontend_trace": row.frontend_trace.sha256,
            "final_mlir": row.final_mlir.sha256,
        },
    }


def observe_contraction_formats(selection: CaptureContractionSelection) -> dict:
    """Reopen a complete byte-selected original roster; never accept a saved census."""
    if type(selection) is not CaptureContractionSelection:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    total_bytes = 0
    for pin in (
        selection.session_contract,
        *[pin for row in selection.programs for pin in (row.original_graph, row.frontend_trace, row.final_mlir)],
    ):
        total_bytes += len(_read(pin, selection.limits))
        if total_bytes > selection.limits.total_bytes:
            raise ContractionFormatError("Contraction input exceeds its selected budget.")
    programs = [_program(row, selection.limits) for row in selection.programs]
    rows = [row for program in programs for row in program["rows"]]
    unknowns = sum(len(program["unknowns"]) for program in programs)
    counts = dict(Counter(row["format"] for row in rows))
    unknown_calls = sum(row["family"] == "unknown_potential_contraction" for row in rows)
    complete_macs = not unknowns and all(row["static_macs"] is not None and row["multiplicity"] == 1 for row in rows)
    total = sum(row["static_macs"] for row in rows) if complete_macs else None
    if total is not None and total.bit_length() > selection.limits.integer_bits:
        raise ContractionFormatError("Contraction input exceeds its selected budget.")
    integer = sum(row["static_macs"] for row in rows if row["format"] == "INTEGER") if complete_macs else None
    eligible = bool(rows) and not unknowns and counts.get("INTEGER", 0) == len(rows) and complete_macs
    return {
        "schema": "merlin.capture_contraction_formats.v1",
        "policy": selection.policy,
        "eligible": eligible,
        "scope": "static_original_contraction_formats_only",
        "counting_unit": "static_original_call_sites",
        "programs": programs,
        "counts": {
            "original_rows": len(rows),
            "integer": counts.get("INTEGER", 0),
            "float": counts.get("FLOAT", 0),
            "unknown": counts.get("UNKNOWN", 0),
            "other_unknowns": unknowns,
            "unknown_original_calls": unknown_calls,
            "complete_original_contraction_count": len(rows) if not unknown_calls and not unknowns else None,
        },
        "macs": {"original": total, "integer": integer, "integer_coverage": [integer, total] if total else None},
        "count_coverage": [counts.get("INTEGER", 0), len(rows)]
        if rows and not unknowns and not unknown_calls
        else None,
        "not_established": [
            "source_producer_authority",
            "numerical_equivalence",
            "precision_certification",
            "offload",
            "cycles",
            "runtime",
        ],
    }


def _contract_membership(root: Path, pin: PinnedFile, limits: ContractionFormatLimits) -> list[tuple[str, Path]]:
    import yaml

    class ContractLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            return _pairs(
                [
                    (self.construct_object(key, deep=deep), self.construct_object(value, deep=deep))
                    for key, value in node.value
                ]
            )

    if pin.path != root / "session_contract.yaml":
        raise ContractionFormatError("Original contraction membership is incomplete.")
    raw = _read(pin, limits)
    try:
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(raw)):
            raise ContractionFormatError("Original contraction membership is incomplete.")
        contract = yaml.load(raw, Loader=ContractLoader)  # noqa: S506 -- exact SafeLoader subclass
    except (yaml.YAMLError, TypeError, ValueError, RecursionError) as error:
        raise ContractionFormatError("Original contraction membership is incomplete.") from error
    _bounded_metadata(contract, limits)
    if not isinstance(contract, dict):
        raise ContractionFormatError("Original contraction membership is incomplete.")
    if type(contract.get("version")) is int and contract["version"] == 1:
        membership = [("forward", root / "model.mlir")]
    elif type(contract.get("version")) is int and contract["version"] == 2:
        membership = []
        for row in contract.get("programs", ()):
            if not isinstance(row, dict) or type(row.get("name")) is not str or type(row.get("bundle")) is not str:
                raise ContractionFormatError("Original contraction membership is incomplete.")
            relative = Path(row["bundle"])
            member = root / relative / "model.mlir"
            if relative.is_absolute() or ".." in relative.parts or member.resolve() != member:
                raise ContractionFormatError("Original contraction membership is incomplete.")
            membership.append((row["name"], member))
    else:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    if not membership or len(membership) > limits.programs:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    return membership


def prepare_capture_contraction_selection(
    capture: Path, originals: CaptureContractionOriginalSelection
) -> CaptureContractionSelection:
    """Join fresh emitted bytes to the preselected original program roster.

    The original graph is never extracted from mutable trace metadata. This
    creates a byte selection, not a capture-producer or numerical certificate.
    """
    verify_original_contraction_selection(originals)
    root = capture.resolve()

    def emitted(path):
        return PinnedFile(path, hashlib.sha256(_file_bytes(path, originals.limits)).hexdigest())

    contract = emitted(root / "session_contract.yaml")
    membership = _contract_membership(root, contract, originals.limits)
    expected = [(row.program, root / row.bundle / "model.mlir") for row in originals.programs]
    if membership != expected:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    selection = CaptureContractionSelection(
        contract,
        tuple(
            ProgramContractionInput(
                row.program,
                row.original_graph,
                emitted(member.parent / "frontend-trace.json"),
                emitted(member),
            )
            for row, (_, member) in zip(originals.programs, membership, strict=True)
        ),
        originals.policy,
        originals.limits,
    )
    require_capture_contraction_formats(root, selection)
    verify_original_contraction_selection(originals)
    return selection


def require_capture_contraction_formats(capture: Path, selection: CaptureContractionSelection) -> dict:
    """Fixed protected-registration consumer of the actual complete session roster.

    This gate is additive to existing numerical/precision checks. Neither its
    census nor an edited or re-signed saved report is an authority object.
    """
    if type(selection) is not CaptureContractionSelection:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    root = capture.resolve()
    membership = _contract_membership(root, selection.session_contract, selection.limits)
    if membership != [(row.program, row.final_mlir.path) for row in selection.programs]:
        raise ContractionFormatError("Original contraction membership is incomplete.")
    observed = observe_contraction_formats(selection)
    _read(selection.session_contract, selection.limits)
    if not observed["eligible"]:
        raise ContractionFormatError("Complete integer contraction coverage is not established.")
    return {
        "selection_schema": "merlin.capture_contraction_selection.v1",
        "session_contract_sha256": selection.session_contract.sha256,
        "limits": asdict(selection.limits),
        "census": observed,
    }

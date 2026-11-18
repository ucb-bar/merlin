"""Typed source proof for bounded mask-count guards and exact cursor movement.

This proves a property of the captured MLIR only.  It neither admits a host
operation nor proves that a compiler preserved an assertion or its abort path.
``index_bits`` is a caller-supplied premise, not verified backend evidence.
The caller must independently bind the selected MLIR index width; no default
width is inferred from a host triple or ELF class.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.compiler.ir import mlir_query as mq
from merlin.development.backend.qualification.source_witnesses.data_movement import (
    source_inventory_by_ordinal,
)

# ---- private_control_support -------------------------------------------------------------------

SCOPE = (
    "typed source interval and mask cursor/data movement under caller-supplied index-width premise only; "
    "selected width, linked effects, and numerical equivalence remain separate obligations"
)

PENDING = "source_bounded_control_pending_build"

LINKED = "source_bounded_control_linked"


def selected_build_observation(host_package: Path, catalog: Path, board: str, target: str) -> dict[str, Any]:
    """Ask the actual whole-model producer for the selected host build's width.

    This is a preflight selection, not an assertion about a later prepared
    module. The linked-build verifier requires the *entire* observation to
    match the actual post-preparation compiler receipt.
    """
    from merlin.compiler.backend.host import load_host_package as load_rvv_package
    from merlin.runtime.backends.baremetal.model_build import selected_model_compiler_plan
    from merlin.runtime.boards.catalog import load_boards

    package = load_rvv_package(host_package)
    selected = load_boards(catalog).get(board)
    _need(selected is not None and selected.target == target, "selected static board does not match target")
    plan = selected_model_compiler_plan(
        backend=package.backend,
        cflags_override=list(package.cflags),
        vlen=selected.vlen if package.backend == "rvv" else None,
        features=None,
    )
    return dict(plan["observation"])


def prove_if_selected(
    module: Any,
    inventory: Mapping[str, Any],
    *,
    raw_sha256: str,
    normalized_sha256: str,
    selected_observation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Do not infer a source index width from an absent or malformed selection."""
    from merlin.development.backend.qualification.source_witnesses.control_chains import prove_bounded_assertions

    if selected_observation is None:
        return None
    _need(
        selected_observation.get("schema") == "merlin.selected-index-lowering.v1"
        and type(selected_observation.get("index_bits")) is int,
        "no producer-owned selected index-width observation",
    )
    proof = prove_bounded_assertions(
        module,
        inventory,
        raw_sha256=raw_sha256,
        normalized_sha256=normalized_sha256,
        index_bits=selected_observation["index_bits"],
    )
    proof["selected_index_observation"] = dict(selected_observation)
    return proof


def assertion_row_proven(row: Mapping[str, Any], proof: Mapping[str, Any] | None) -> bool:
    """Discharge only exact guard predicates or bounded compaction instructions."""
    if proof is None or proof.get("status") != PENDING:
        return False
    operation = row.get("mlir_operation")
    ordinals = row.get("ordinals")
    guards = proof.get("guards")
    if (
        not isinstance(ordinals, list)
        or not ordinals
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
        or len(ordinals) != len(set(ordinals))
        or row.get("count") != len(ordinals)
        or not isinstance(guards, list)
    ):
        return False
    if operation == "cf.assert":
        proven = {guard.get("assert_ordinal") for guard in guards if isinstance(guard, Mapping)}
    elif operation == "arith.cmpi":
        proven = {guard.get("compare_ordinal") for guard in guards if isinstance(guard, Mapping)}
    elif operation in {"arith.index_cast", "arith.addi"}:
        chains = proof.get("index_data_support")
        internal = proof.get("internal_compaction_support")
        if not isinstance(chains, list) or not isinstance(internal, list):
            return False
        key = "cast_ordinal" if operation == "arith.index_cast" else "add_ordinals"
        proven = set()
        for chain in chains:
            if not isinstance(chain, Mapping):
                return False
            value = chain.get(key)
            if operation == "arith.index_cast":
                proven.add(value)
            elif isinstance(value, list):
                proven.update(value)
            else:
                return False
        for chain in internal:
            if not isinstance(chain, Mapping):
                return False
            proven.add(chain.get("cast_ordinal") if operation == "arith.index_cast" else chain.get("add_ordinal"))
    else:
        return False
    return set(ordinals) <= proven


def attach_source_record(result: dict[str, Any], proof: Mapping[str, Any] | None) -> None:
    """Retain the selected premise and exact proven-assertion count in source evidence."""
    if proof is None:
        return
    result["selected_index_observation"] = dict(proof["selected_index_observation"])
    result["n_bounded_control_assertions"] = proof["count"]
    result["n_bounded_index_support_operations"] = len(proof["internal_compaction_support"]) * 2 + sum(
        1 + len(chain["add_ordinals"]) for chain in proof["index_data_support"]
    )
    result["n_internal_mask_compactions"] = len(proof["internal_compaction_support"])
    if proof["count"] or proof["internal_compaction_support"]:
        result["bounded_control_support"] = dict(proof)


def _record_matches_selected(actual: Mapping[str, Any] | None, selected: Mapping[str, Any] | None) -> bool:
    """Exact preflight/post-preparation join, including effective pass options."""
    from merlin.common.digest import is_sha256
    from merlin.compiler.lower.model.target_data_layout import default_index_bits

    if not isinstance(actual, Mapping) or not isinstance(selected, Mapping):
        return False
    pipeline = actual.get("effective_pipeline")
    if not isinstance(pipeline, str) or not pipeline:
        return False
    if {key: value for key, value in actual.items() if key != "effective_pipeline"} != dict(selected):
        return False
    from merlin.compiler.lower.native.pipeline import _INDEX_CONVERSION_PASSES

    bits = selected.get("index_bits")
    if (
        selected.get("schema") != "merlin.selected-index-lowering.v1"
        or not isinstance(selected.get("compiler_requested"), str)
        or not selected["compiler_requested"]
        or not isinstance(selected.get("compiler_resolved"), str)
        or not selected["compiler_resolved"]
        or not is_sha256(selected.get("compiler_sha256"))
        or not isinstance(selected.get("cross_flags"), list)
        or not selected["cross_flags"]
        or any(not isinstance(flag, str) or not flag for flag in selected["cross_flags"])
        or not isinstance(selected.get("data_layout"), str)
        or type(bits) is not int
    ):
        return False
    try:
        if default_index_bits(selected["data_layout"]) != bits:
            return False
    except ValueError:
        return False
    for name in _INDEX_CONVERSION_PASSES:
        token = f"{name}{{index-bitwidth={bits}}}"
        if pipeline.count(token) != 1 or pipeline.count(name) != 1:
            return False
    return True


def _complete_guard_roster(proof: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    from merlin.common.digest import is_sha256

    count = proof.get("count")
    guards = proof.get("guards")
    if (
        type(count) is not int
        or count < 0
        or source.get("n_bounded_control_assertions") != count
        or not isinstance(guards, list)
        or len(guards) != count
        or not is_sha256(proof.get("raw_source_sha256"))
        or proof.get("raw_source_sha256") != source.get("source_sha256")
        or not is_sha256(proof.get("normalized_source_sha256"))
        or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or proof.get("selected_index_observation") != source.get("selected_index_observation")
        or type(proof.get("index_bits")) is not int
        or not 2 <= proof["index_bits"] <= 128
        or proof.get("index_bits") != (source.get("selected_index_observation") or {}).get("index_bits")
    ):
        return False
    ordinals = [guard.get("assert_ordinal") for guard in guards if isinstance(guard, Mapping)]
    compares = [guard.get("compare_ordinal") for guard in guards if isinstance(guard, Mapping)]
    chains = proof.get("index_data_support", [])
    internal = proof.get("internal_compaction_support")
    if not isinstance(chains, list) or not isinstance(internal, list):
        return False
    data_ordinals = []
    for chain in chains:
        if (
            not isinstance(chain, Mapping)
            or set(chain) != {"cast_ordinal", "add_ordinals", "extent"}
            or type(chain.get("cast_ordinal")) is not int
            or type(chain.get("extent")) is not int
            or not 0 < chain["extent"] < (1 << (proof["index_bits"] - 1))
        ):
            return False
        adds = chain.get("add_ordinals")
        if not isinstance(adds, list) or len(adds) != 2:
            return False
        data_ordinals.extend([chain["cast_ordinal"], *adds])
    if source.get("n_internal_mask_compactions") != len(internal):
        return False
    if count == 0 and not chains and not internal:
        return False
    for chain in internal:
        if not isinstance(chain, Mapping) or set(chain) != {
            "cast_ordinal",
            "add_ordinal",
            "allocation_ordinal",
            "loop_ordinal",
            "cast_dim_ordinal",
            "cast_linalg_ordinal",
            "extent",
            "input_dtype",
            "output_dtype",
        }:
            return False
        ordinal_keys = (
            "cast_ordinal",
            "add_ordinal",
            "allocation_ordinal",
            "loop_ordinal",
            "cast_dim_ordinal",
            "cast_linalg_ordinal",
        )
        if (
            any(type(chain[key]) is not int or chain[key] < 0 for key in ordinal_keys)
            or len({chain[key] for key in ordinal_keys}) != len(ordinal_keys)
            or type(chain["extent"]) is not int
            or not 0 < chain["extent"] <= ((1 << (proof["index_bits"] - 1)) - 1) // 8
            or chain["input_dtype"] != "i1"
            or chain["output_dtype"] != "i64"
        ):
            return False
        data_ordinals.extend([chain["cast_ordinal"], chain["add_ordinal"]])
    if source.get("n_bounded_index_support_operations") != len(data_ordinals):
        return False
    return (
        len(ordinals) == count
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in ordinals)
        and len(set(ordinals)) == count
        and len(compares) == count
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in compares)
        and len(set(compares)) == count
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in data_ordinals)
        and len(set(data_ordinals)) == len(data_ordinals)
    )


def link_selected_build(source: dict[str, Any], receipt: Mapping[str, Any], linked_build: Mapping[str, Any]) -> dict:
    """Bind source guard proof to the exact selected compiled program, never a preflight guess."""
    proof = source.get("bounded_control_support")
    if proof is None:
        if source.get("selected_index_observation") is None:
            return {}
        if source.get("n_bounded_control_assertions") != 0:
            raise ValueError("source bounded-control assertion roster is incomplete")
        if source.get("n_internal_mask_compactions") != 0:
            raise ValueError("source internal-compaction roster is incomplete")
    elif not isinstance(proof, dict) or proof.get("status") != PENDING or not _complete_guard_roster(proof, source):
        raise ValueError("source bounded-control proof was not pending a linked build")
    selected = source.get("selected_index_observation")
    actual = (receipt.get("output") or {}).get("index_lowering")
    if not _record_matches_selected(actual, selected):
        raise ValueError("post-preparation index lowering differs from selected source premise")
    if proof is not None:
        _need(proof.get("selected_index_observation") == selected, "source guard width selection changed")
        _need(proof.get("raw_source_sha256") == source.get("source_sha256"), "source guard identity changed")
        proof["status"] = LINKED
        proof["linked_build"] = dict(linked_build)
    return dict(actual)


def linked_selected_build_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """A formal source record must retain its exact selected compiler and linked ELF."""
    selected = source.get("selected_index_observation")
    if not _record_matches_selected(entry.get("index_lowering"), selected):
        return False
    proof = source.get("bounded_control_support")
    if proof is None:
        return source.get("n_bounded_control_assertions") == 0 and source.get("n_internal_mask_compactions") == 0
    if not isinstance(proof, Mapping) or proof.get("status") != LINKED or not _complete_guard_roster(proof, source):
        return False
    linked = proof.get("linked_build")
    return (
        proof.get("selected_index_observation") == selected
        and proof.get("index_bits") == selected.get("index_bits")
        and proof.get("raw_source_sha256") == source.get("source_sha256")
        and source.get("n_bounded_control_assertions") == proof.get("count")
        and isinstance(linked, Mapping)
        and linked.get("capture_tree_sha256") == entry.get("capture_tree_sha256")
        and linked.get("elf_sha256") == entry.get("elf_sha256")
        and linked.get("candidate_tree_sha256") == candidate_sha256
    )


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"source bounded-control proof: {reason}")


def _named(value: Any, name: str) -> Any:
    from xdsl.ir import Operation

    owner = value.owner
    _need(isinstance(owner, Operation) and mq.op_name(owner) == name and value in owner.results, f"expected {name}")
    return owner


def _integer(value: Any, typ: str) -> int:
    op = _named(value, "arith.constant")
    _only_provenance(op, {"value"})
    _need(len(op.results) == 1 and str(value.type) == typ, f"expected {typ} constant")
    attribute = op.properties.get("value")
    number = getattr(getattr(attribute, "value", None), "data", None)
    _need(type(number) is int, "constant has no exact integer value")
    return number


def _shape(value: Any, typ: str) -> tuple[int, ...]:
    from xdsl.dialects.builtin import TensorType

    tensor = value.type
    _need(isinstance(tensor, TensorType) and tensor.has_static_shape(), "non-static tensor")
    _need(str(tensor.element_type) == typ, f"expected {typ} tensor")
    result = tuple(tensor.get_shape())
    _need(all(type(extent) is int and extent >= 0 for extent in result), "invalid tensor extent")
    return result


def _only_provenance(op: Any, properties: set[str]) -> None:
    _need(
        set(op.properties) == properties and all(key.startswith("prov.") for key in op.attributes),
        f"{mq.op_name(op)} carries an unrecognized semantic attribute",
    )


def _mask_count(value: Any) -> tuple[int, Any]:
    """Prove a scalar i64 sum of zero-extended i1 cells lies in [0, N]."""
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.ir.affine import AffineDimExpr

    extract = _named(value, "tensor.extract")
    _only_provenance(extract, set())
    _need(str(value.type) == "i64" and len(extract.operands) == 1, "count is not a scalar i64 extraction")
    reduce = _named(extract.operands[0], "linalg.reduce")
    _only_provenance(reduce, {"dimensions"})
    dims = reduce.properties["dimensions"]
    _need(isinstance(dims, DenseArrayBase) and tuple(dims.iter_values()) == (0,), "not a full rank-one sum")
    _need(
        len(reduce.operands) == 2 and len(reduce.results) == 1 and _shape(reduce.results[0], "i64") == (),
        "invalid count reduction",
    )
    initializer = _named(reduce.operands[1], "tensor.splat")
    _only_provenance(initializer, set())
    _need(
        _shape(reduce.operands[1], "i64") == ()
        and len(initializer.operands) == 1
        and _integer(initializer.operands[0], "i64") == 0,
        "count does not start at zero",
    )
    _need(len(reduce.regions) == 1 and len(reduce.regions[0].blocks) == 1, "non-single count reduction")
    body = reduce.regions[0].blocks[0]
    ops = list(body.ops)
    _need(
        len(body.args) == 2
        and all(str(arg.type) == "i64" for arg in body.args)
        and [mq.op_name(op) for op in ops] == ["arith.addi", "linalg.yield"],
        "count reducer is not addition",
    )
    add, yielded = ops
    _only_provenance(add, {"overflowFlags"})
    _only_provenance(yielded, set())
    _need(
        tuple(add.operands) == tuple(body.args)
        and tuple(yielded.operands) == tuple(add.results)
        and set(add.properties) == {"overflowFlags"}
        and str(add.properties["overflowFlags"]) == "#arith.overflow<none>",
        "count reducer changes addition or overflow semantics",
    )

    generic = _named(reduce.operands[0], "linalg.generic")
    _only_provenance(generic, {"indexing_maps", "iterator_types", "operandSegmentSizes"})
    _need(len(generic.operands) == 2 and len(generic.results) == 1, "not a unary mask conversion")
    source, init = generic.operands
    (extent,) = _shape(source, "i1")
    _need(
        extent > 0 and _shape(init, "i64") == (extent,) and _shape(generic.results[0], "i64") == (extent,),
        "mask count shape or dtype changed",
    )
    segments = generic.properties["operandSegmentSizes"]
    maps = generic.properties["indexing_maps"]
    iterator = generic.properties["iterator_types"]
    _need(
        isinstance(segments, DenseArrayBase)
        and tuple(segments.iter_values()) == (1, 1)
        and isinstance(maps, ArrayAttr)
        and len(maps.data) == 2
        and isinstance(iterator, ArrayAttr)
        and len(iterator.data) == 1
        and isinstance(iterator.data[0], IteratorTypeAttr)
        and iterator.data[0].data == IteratorType.PARALLEL,
        "mask conversion is not one parallel elementwise map",
    )
    for item in maps.data:
        _need(
            isinstance(item, AffineMapAttr)
            and item.data.num_dims == 1
            and item.data.num_symbols == 0
            and len(item.data.results) == 1
            and item.data.results[0] == AffineDimExpr(0),
            "mask conversion changes element indexing",
        )
    _need(len(generic.regions) == 1 and len(generic.regions[0].blocks) == 1, "non-single mask body")
    block = generic.regions[0].blocks[0]
    ops = list(block.ops)
    _need(
        len(block.args) == 2
        and str(block.args[0].type) == "i1"
        and str(block.args[1].type) == "i64"
        and [mq.op_name(op) for op in ops] == ["arith.extui", "linalg.yield"],
        "mask conversion is not unsigned Boolean extension",
    )
    extension, output = ops
    _only_provenance(extension, set())
    _only_provenance(output, set())
    _need(
        tuple(extension.operands) == (block.args[0],)
        and str(extension.results[0].type) == "i64"
        and tuple(output.operands) == tuple(extension.results),
        "mask conversion has a different value path",
    )
    _need(extent < (1 << 63), "i64 count could overflow")
    return extent, source


def _same_loop_shape(value: Any, source: Any) -> bool:
    """Prove the loop-carried tensor is returned, with only shape-preserving inserts."""
    if value is source:
        return True
    owner = value.owner
    if owner is None:
        return False
    if mq.op_name(owner) == "tensor.insert":
        _only_provenance(owner, set())
        return len(owner.operands) >= 2 and owner.results[0] is value and _same_loop_shape(owner.operands[1], source)
    if mq.op_name(owner) != "scf.if" or value not in owner.results or len(owner.regions) != 2:
        return False
    position = list(owner.results).index(value)
    _only_provenance(owner, set())
    for region in owner.regions:
        if len(region.blocks) != 1:
            return False
        body = list(region.blocks[0].ops)
        if not body or mq.op_name(body[-1]) != "scf.yield" or len(body[-1].operands) <= position:
            return False
        if not _same_loop_shape(body[-1].operands[position], source):
            return False
    return True


def _empty_shape(value: Any) -> Any:
    loop = _named(value, "scf.for")
    _only_provenance(loop, set())
    _need(len(loop.regions) == 1 and len(loop.regions[0].blocks) == 1, "non-single shape loop")
    position = list(loop.results).index(value)
    block = loop.regions[0].blocks[0]
    body = list(block.ops)
    _need(
        len(loop.operands) >= 4 + position
        and len(block.args) >= 2 + position
        and body
        and mq.op_name(body[-1]) == "scf.yield"
        and len(body[-1].operands) > position
        and _same_loop_shape(body[-1].operands[position], block.args[1 + position]),
        "loop can change the dynamic tensor shape",
    )
    empty = _named(loop.operands[3 + position], "tensor.empty")
    _only_provenance(empty, set())
    _need(
        len(empty.operands) == 1
        and str(empty.operands[0].type) == "index"
        and str(value.type) == str(empty.results[0].type)
        and str(block.args[1 + position].type) == str(value.type),
        "loop does not carry one dynamic tensor.empty shape",
    )
    return empty.operands[0]


def _uses_are(value: Any, *expected: tuple[Any, int]) -> bool:
    """Compare every SSA use, including repeated uses by one operation."""
    actual = [(id(use.operation), use.index) for use in value.uses]
    wanted = [(id(operation), index) for operation, index in expected]
    return len(actual) == len(wanted) and sorted(actual) == sorted(wanted)


def _same_flat_mask(left: Any, right: Any, extent: int) -> bool:
    """Only verified reshape chains of one static i1 tensor preserve flat order."""
    from math import prod

    def root(value: Any) -> Any:
        _need(_shape(value, "i1") and prod(_shape(value, "i1")) == extent, "mask extent changed")
        owner = value.owner
        if (
            owner is None
            or not hasattr(owner, "name")
            or mq.op_name(owner) not in {"tensor.collapse_shape", "tensor.expand_shape"}
        ):
            return value
        expected = {"reassociation"}
        if mq.op_name(owner) == "tensor.expand_shape":
            expected.add("static_output_shape")
        _only_provenance(owner, expected)
        _need(len(owner.operands) == 1 and len(owner.results) == 1, "mask reshape has another value")
        return root(owner.operands[0])

    return root(left) is root(right)

"""Exact source and linked-build accounting for opt-in static Linalg host bodies.

The witness is source-structural only. It neither grants a host rule nor proves
the numerical behavior of a translated program. Per-tensor index/byte bounds
do not prove a global arena allocation or pointer-base address range.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.frontends.linalg_boolean_patterns import (
    DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA,
    STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
    dynamic_boolean_cast_ordered_types,
    recognize_dynamic_boolean_cast_body,
    recognize_static_boolean_body,
    static_boolean_ordered_types,
    validate_dynamic_boolean_cast_source_body,
    validate_serialized_dynamic_boolean_cast_pattern,
    validate_serialized_static_boolean_pattern,
    validate_static_boolean_source_body,
)
from merlin.frontends.linalg_composite_math import (
    SCHEMA as COMPOSITE_SOURCE_SCHEMA,
)
from merlin.frontends.linalg_composite_math import (
    STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
    recognize_static_composite_math_body,
    static_composite_math_ordered_types,
    validate_serialized_static_composite_math_pattern,
    validate_static_composite_math_source_body,
)
from merlin.frontends.linalg_embedding_gather import (
    SCHEMA as EMBEDDING_GATHER_SCHEMA,
)
from merlin.frontends.linalg_embedding_gather import (
    TYPES as EMBEDDING_GATHER_TYPES,
)
from merlin.frontends.linalg_embedding_gather import (
    recognize as recognize_embedding_gather,
)
from merlin.frontends.linalg_embedding_gather import (
    validate_pattern as validate_embedding_gather_pattern,
)
from merlin.frontends.linalg_embedding_gather import (
    validate_source_body as validate_embedding_gather_body,
)
from merlin.frontends.linalg_math_patterns import (
    STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
    recognize_static_f32_math_body,
    static_f32_math_ordered_types,
    validate_static_f32_math_source_body,
)
from merlin.frontends.linalg_mixed_patterns import SCHEMA as MIXED_SCHEMA
from merlin.frontends.linalg_mixed_patterns import ordered_types as mixed_ordered_types
from merlin.frontends.linalg_mixed_patterns import recognize as recognize_mixed
from merlin.frontends.linalg_mixed_patterns import validate_declaration as validate_mixed
from merlin.frontends.linalg_mixed_patterns import validate_pattern as validate_mixed_pattern
from merlin.frontends.linalg_patterns import (
    STATIC_POINTWISE_SOURCE_BODY_SCHEMA,
    STATIC_PROJECTED_POINTWISE_BODY_SCHEMA,
    InvalidLinalgPattern,
    recognize_static_pointwise,
    recognize_static_projected_pointwise,
    static_pointwise_ordered_types,
    static_projected_pointwise_ordered_types,
    validate_serialized_static_projected_pointwise_pattern,
    validate_static_pointwise_source_body,
    validate_static_projected_pointwise_source_body,
)
from merlin.targetgen.host_linkage_contract import (
    required_composite_math_symbol,
    validate_source_linkage_contract,
)
from merlin_experiments.phase1.feedback import private_control_support as control_support

PENDING = "source_linalg_host_support_pending_build"
LINKED = "source_linalg_host_support_linked"
SCOPE = "exact source bodies, selected per-tensor index span and linked build; no numerical equivalence"
_TOP = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "n_source_operations",
    "selected_index_observation",
    "actual_index_observation",
    "count",
    "occurrences",
    "linked_build",
    "source_capture_path",
}
_OCCURRENCE = {
    "ordinal",
    "profile",
    "capability_spec_sha256",
    "declaration",
    "schema",
    "operation",
    "predicate",
    "shape",
    "ordered_types",
    "input_shapes",
    "input_maps",
    "literal_bits",
    "lowering_intrinsic_obligations",
    "linkage_requirement",
}
_DYNAMIC_OCCURRENCE = _OCCURRENCE | {"shape_source", "dynamic_bound"}
_GATHER_OCCURRENCE = _OCCURRENCE | {"table_shape", "table_argument"}


def _contract(schema: object, operation: object, predicate: object) -> tuple[dict[str, str], Any, tuple[str, ...]]:
    """Resolve a closed versioned declaration without crossing schema namespaces."""
    if schema == MIXED_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("mixed source body declares no predicate")
        declaration = validate_mixed({"schema": schema, "operation": operation})
        return declaration, recognize_mixed, mixed_ordered_types(declaration)
    if schema == EMBEDDING_GATHER_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("embedding gather source has no scalar predicate")
        declaration = validate_embedding_gather_body({"schema": schema, "operation": operation})
        return declaration, recognize_embedding_gather, EMBEDDING_GATHER_TYPES
    if schema == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("Boolean source body does not declare a predicate")
        declaration = validate_static_boolean_source_body({"schema": schema, "operation": operation})
        return declaration, recognize_static_boolean_body, static_boolean_ordered_types(declaration)
    if schema == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("dynamic Boolean cast does not declare a predicate")
        declaration = validate_dynamic_boolean_cast_source_body({"schema": schema, "operation": operation})
        return declaration, recognize_dynamic_boolean_cast_body, dynamic_boolean_cast_ordered_types(declaration)
    if schema == STATIC_POINTWISE_SOURCE_BODY_SCHEMA:
        raw = {"schema": schema, "operation": operation}
        if predicate is not None:
            raw["predicate"] = predicate
        declaration = validate_static_pointwise_source_body(raw)
        return declaration, recognize_static_pointwise, static_pointwise_ordered_types(declaration)
    if schema == STATIC_PROJECTED_POINTWISE_BODY_SCHEMA:
        raw = {"schema": schema, "operation": operation}
        if predicate is not None:
            raw["predicate"] = predicate
        declaration = validate_static_projected_pointwise_source_body(raw)
        return declaration, recognize_static_projected_pointwise, static_projected_pointwise_ordered_types(declaration)
    if schema == STATIC_F32_MATH_SOURCE_BODY_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("unary f32 math source body does not declare a predicate")
        declaration = validate_static_f32_math_source_body({"schema": schema, "operation": operation})
        return declaration, recognize_static_f32_math_body, static_f32_math_ordered_types(declaration)
    if schema == STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("composite math source body does not declare a predicate")
        declaration = validate_static_composite_math_source_body({"schema": schema, "operation": operation})
        return declaration, recognize_static_composite_math_body, static_composite_math_ordered_types(declaration)
    raise InvalidLinalgPattern("unknown static source-body schema")


def begin(
    raw_sha256: str,
    normalized_sha256: str,
    n_operations: int,
    selected_index_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a mandatory empty-or-populated witness for one current source."""
    if (
        not is_sha256(raw_sha256)
        or not is_sha256(normalized_sha256)
        or type(n_operations) is not int
        or n_operations < 1
    ):
        raise ValueError("static Linalg source identity is malformed")
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "n_source_operations": n_operations,
        "selected_index_observation": (
            deepcopy(dict(selected_index_observation)) if selected_index_observation else None
        ),
        "count": 0,
        "occurrences": [],
        "source_capture_path": None,
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    admission: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
    *,
    control_proof: Mapping[str, Any] | None = None,
) -> None:
    """Recompute and retain every exact source ordinal of an admitted row."""
    body = admission.get("source_body_proof")
    if body is None:
        return
    ordinals = row.get("ordinals")
    patterns = body.get("patterns") if isinstance(body, Mapping) else None
    if (
        witness.get("status") != PENDING
        or not isinstance(body, Mapping)
        or set(body)
        != {"schema", "declaration", "operation", "predicate", "patterns", "profile", "capability_spec_sha256"}
        or admission.get("status") != "admitted"
        or admission.get("reviewed") is not True
        or not is_sha256(body.get("capability_spec_sha256"))
        or not isinstance(body.get("profile"), str)
        or not body["profile"]
        or not isinstance(body.get("declaration"), str)
        or not body["declaration"]
        or not isinstance(ordinals, list)
        or not isinstance(patterns, list)
        or type(row.get("count")) is not int
        or row["count"] != len(ordinals)
        or len(patterns) != len(ordinals)
    ):
        raise ValueError("static Linalg host admission has no exact source occurrence roster")
    try:
        _, recognizer, expected_types = _contract(body.get("schema"), body.get("operation"), body.get("predicate"))
    except InvalidLinalgPattern as exc:
        raise ValueError("static Linalg host admission has an invalid source-body contract") from exc
    seen = {item["ordinal"] for item in witness["occurrences"]}
    dynamic = body["schema"] == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA
    linkage = admission.get("linkage_requirement")
    required_symbol = required_composite_math_symbol(body["schema"], body["operation"])
    if required_symbol and linkage is None:
        raise ValueError("composite math source lacks its exact selected linkage requirement")
    if linkage is not None:
        if (
            not isinstance(linkage, Mapping)
            or set(linkage) != {"profile", "capability_spec_sha256", "declaration", "contract"}
            or (linkage["profile"], linkage["capability_spec_sha256"], linkage["declaration"])
            != (body["profile"], body["capability_spec_sha256"], body["declaration"])
        ):
            raise ValueError("static Linalg linkage requirement differs from reviewed source body")
        linkage = validate_source_linkage_contract(body["schema"], body["operation"], linkage["contract"])
    if dynamic and not _control_proof_matches_source(control_proof, witness):
        raise ValueError("dynamic Boolean cast has no closed internal compaction proof")
    for ordinal, pattern in zip(ordinals, patterns, strict=True):
        try:
            actual = (
                asdict(recognizer(parsed[ordinal])) if type(ordinal) is int and 0 <= ordinal < len(parsed) else None
            )
        except InvalidLinalgPattern as exc:
            raise ValueError("static Linalg proof differs from parsed source") from exc
        if (
            type(ordinal) is not int
            or ordinal in seen
            or source_rows.get(ordinal) is not row
            or not isinstance(pattern, Mapping)
            or actual != pattern
            or pattern["operation"] != body["operation"]
            or pattern.get("predicate") != body.get("predicate")
            or tuple(pattern["ordered_types"]) != expected_types
        ):
            raise ValueError("static Linalg proof differs from parsed source")
        if body["schema"] == EMBEDDING_GATHER_SCHEMA:
            validate_embedding_gather_pattern(pattern)
        if body["schema"] == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_static_boolean_pattern(pattern)
            except InvalidLinalgPattern as exc:
                raise ValueError("static Boolean proof has an invalid serialized source pattern") from exc
        if body["schema"] == STATIC_PROJECTED_POINTWISE_BODY_SCHEMA:
            try:
                validate_serialized_static_projected_pointwise_pattern(pattern)
            except InvalidLinalgPattern as exc:
                raise ValueError("projected pointwise proof has an invalid serialized source pattern") from exc
        if body["schema"] == STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_static_composite_math_pattern(pattern)
            except InvalidLinalgPattern as exc:
                raise ValueError("composite math proof has an invalid serialized source pattern") from exc
        if dynamic:
            try:
                validate_serialized_dynamic_boolean_cast_pattern(pattern)
            except InvalidLinalgPattern as exc:
                raise ValueError("dynamic Boolean cast has an invalid serialized source pattern") from exc
            bounds = [
                chain
                for chain in control_proof["internal_compaction_support"]
                if chain["cast_linalg_ordinal"] == ordinal
            ]
            if len(bounds) != 1:
                raise ValueError("dynamic Boolean cast is not the proved internal compaction consumer")
        occurrence = {
            "ordinal": ordinal,
            "profile": body["profile"],
            "capability_spec_sha256": body["capability_spec_sha256"],
            "declaration": body["declaration"],
            "schema": body["schema"],
            "operation": pattern["operation"],
            "predicate": pattern.get("predicate"),
            "shape": pattern["shape"],
            "ordered_types": pattern["ordered_types"],
            "input_shapes": pattern.get("input_shapes", ()),
            "input_maps": pattern.get("input_maps", ()),
            "literal_bits": pattern.get("literal_bits"),
            "lowering_intrinsic_obligations": pattern.get("lowering_intrinsic_obligations"),
            "linkage_requirement": deepcopy(linkage),
        }
        if body["schema"] == EMBEDDING_GATHER_SCHEMA:
            occurrence["table_shape"] = pattern["table_shape"]
            occurrence["table_argument"] = pattern["table_argument"]
        if dynamic:
            occurrence["shape_source"] = pattern["shape_source"]
            occurrence["dynamic_bound"] = deepcopy(bounds[0])
        witness["occurrences"].append(occurrence)
        seen.add(ordinal)
    witness["occurrences"].sort(key=lambda item: item["ordinal"])
    witness["count"] = len(witness["occurrences"])


def _control_proof_matches_source(proof: object, witness: Mapping[str, Any]) -> bool:
    if not isinstance(proof, Mapping) or proof.get("selected_index_observation") != witness.get(
        "selected_index_observation"
    ):
        return False
    source = {
        "source_sha256": witness.get("raw_source_sha256"),
        "normalized_source_sha256": witness.get("normalized_source_sha256"),
        "selected_index_observation": witness.get("selected_index_observation"),
    }
    try:
        control_support.attach_source_record(source, proof)
        return control_support._complete_guard_roster(proof, source)  # noqa: PLC2701 -- shared roster verifier
    except (KeyError, TypeError, ValueError):
        return False


def link(
    source: Mapping[str, Any],
    actual_index: Mapping[str, Any],
    linked_build: Mapping[str, str],
    *,
    capture_path: Path | None = None,
) -> None:
    witness = source.get("linalg_host_support")
    if not isinstance(witness, dict) or witness.get("status") != PENDING:
        raise ValueError("static Linalg proof is not pending the selected build")
    selected = source.get("selected_index_observation")
    if (
        witness.get("selected_index_observation") != selected
        or not control_support._record_matches_selected(actual_index, selected)  # noqa: PLC2701 -- shared verifier
    ):
        raise ValueError("static Linalg proof has no exact selected index-lowering build")
    if not _all_shapes_fit(witness.get("occurrences"), selected["index_bits"]):
        from .private_program_checks import ProgramCheckFailure

        raise ProgramCheckFailure("static Linalg shape exceeds selected signed index address span", check="linalg_span")
    if not all(_math_record_valid(item) for item in witness["occurrences"]):
        raise ValueError("static Linalg math source or linkage record is incomplete")
    if not _composite_source_matches(witness, source, linked_build, capture_path):
        raise ValueError("composite math source differs from the selected captured-stage tree")
    if not _dynamic_roster_matches_control(witness, source):
        raise ValueError("dynamic Boolean cast differs from linked internal compaction source")
    if any(item.get("schema") == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA for item in witness["occurrences"]):
        control = source.get("bounded_control_support")
        if control.get("status") != control_support.LINKED or control.get("linked_build") != dict(linked_build):
            raise ValueError("dynamic Boolean cast has no identical linked control witness")
    witness["status"] = LINKED
    witness["actual_index_observation"] = deepcopy(dict(actual_index))
    witness["linked_build"] = dict(linked_build)
    witness["source_capture_path"] = str(capture_path) if _has_composite(witness) else None


def _has_composite(witness: Mapping[str, Any]) -> bool:
    return any(
        item.get("schema") in {STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, MIXED_SCHEMA, EMBEDDING_GATHER_SCHEMA}
        for item in witness["occurrences"]
    )


def _composite_source_matches(
    witness: Mapping[str, Any],
    source: Mapping[str, Any],
    linked_build: Mapping[str, Any],
    capture_path: Path | None,
) -> bool:
    """Reparse each composite ordinal from the exact selected capture tree."""
    if not _has_composite(witness):
        return capture_path is None or isinstance(capture_path, Path)
    if capture_path is None or not capture_path.is_absolute() or capture_path.is_symlink():
        return False
    model = capture_path / "model.mlir"
    receipt = capture_path / "capture_receipt.json"
    if not model.is_file() or model.is_symlink() or not receipt.is_file() or receipt.is_symlink():
        return False
    try:

        def pinned() -> bool:
            return bool(
                strict_tree_sha256(capture_path)["sha256"] == linked_build.get("capture_tree_sha256")
                and file_sha256(model) == witness.get("raw_source_sha256") == source.get("source_sha256")
                and file_sha256(receipt) == source.get("capture_receipt_sha256")
            )

        if not pinned():
            return False
        normalized, _ = normalize_capture_mlir(model.read_text(encoding="utf-8"))
        if sha256(normalized.encode("utf-8")).hexdigest() != witness.get("normalized_source_sha256") or witness.get(
            "normalized_source_sha256"
        ) != source.get("normalized_source_sha256"):
            return False
        parsed = tuple(mq.walk(mq.parse(normalized)))
        if len(parsed) != witness.get("n_source_operations"):
            return False
        for item in witness["occurrences"]:
            if item.get("schema") not in {
                STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
                MIXED_SCHEMA,
                EMBEDDING_GATHER_SCHEMA,
            }:
                continue
            ordinal = item.get("ordinal")
            if type(ordinal) is not int or not 0 <= ordinal < len(parsed):
                return False
            if item.get("schema") == EMBEDDING_GATHER_SCHEMA:
                pattern = asdict(recognize_embedding_gather(parsed[ordinal]))
                if any(json.dumps(item.get(key)) != json.dumps(value) for key, value in pattern.items()):
                    return False
                continue
            pattern = asdict(
                (recognize_mixed if item["schema"] == MIXED_SCHEMA else recognize_static_composite_math_body)(
                    parsed[ordinal]
                )
            )
            if item["schema"] == MIXED_SCHEMA:
                if any(
                    json.loads(json.dumps(item.get(key))) != json.loads(json.dumps(pattern.get(key)))
                    for key in ("operation", "shape", "ordered_types", "literal_bits", "input_shapes", "input_maps")
                ):
                    return False
                continue
            if item.get("operation") != pattern["operation"] or any(
                tuple(item.get(key) or ()) != tuple(pattern.get(key) or ())
                for key in ("shape", "ordered_types", "literal_bits", "lowering_intrinsic_obligations")
            ):
                return False
        return pinned()
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, InvalidLinalgPattern):
        return False


def _all_shapes_fit(occurrences: object, index_bits: int) -> bool:
    """Bound every tensor extent and byte span by the selected signed index."""
    if not isinstance(occurrences, list) or type(index_bits) is not int or not 2 <= index_bits <= 128:
        return False
    maximum = (1 << (index_bits - 1)) - 1
    element_bytes = {"i1": 1, "i8": 1, "bf16": 2, "i32": 4, "f32": 4, "f64": 8, "i64": 8}

    def fits(shape: object, dtype: object, *, scalar: bool = False) -> bool:
        if (
            not isinstance(shape, (list, tuple))
            or (not shape and not scalar)
            or not isinstance(dtype, str)
            or dtype not in element_bytes
        ):
            return False
        elements = 1
        for dim in shape:
            if type(dim) is not int or not 0 < dim <= maximum:
                return False
            if elements > maximum // dim:
                return False
            elements *= dim
        return elements <= maximum // element_bytes[dtype]

    for item in occurrences:
        if not isinstance(item, Mapping):
            return False
        if item.get("schema") == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_dynamic_boolean_cast_pattern(
                    {
                        key: item[key]
                        for key in ("operation", "shape", "ordered_types", "input_shapes", "input_maps", "shape_source")
                    }
                )
            except (InvalidLinalgPattern, KeyError):
                return False
            if not _dynamic_bound_fits(item.get("dynamic_bound"), item.get("ordinal"), maximum):
                return False
            continue
        shape, types, inputs = item.get("shape"), item.get("ordered_types"), item.get("input_shapes")
        if not isinstance(types, (list, tuple)) or len(types) < 2 or not fits(shape, types[-1]):
            return False
        if item.get("schema") == EMBEDDING_GATHER_SCHEMA:
            if (
                not isinstance(inputs, (list, tuple))
                or len(inputs) != 1
                or not fits(inputs[0], "i64")
                or not fits(item.get("table_shape"), "bf16")
            ):
                return False
        elif item.get("schema") in {
            STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
            STATIC_PROJECTED_POINTWISE_BODY_SCHEMA,
            MIXED_SCHEMA,
        }:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != len(types) - 2:
                return False
            if any(
                not fits(input_shape, dtype, scalar=True) for input_shape, dtype in zip(inputs, types[:-2], strict=True)
            ) or not fits(shape, types[-2]):
                return False
        elif any(not fits(shape, dtype) for dtype in types[:-1]):
            return False
    return True


def _dynamic_bound_fits(bound: object, ordinal: object, maximum: int) -> bool:
    fields = {
        "cast_ordinal",
        "add_ordinal",
        "allocation_ordinal",
        "loop_ordinal",
        "cast_dim_ordinal",
        "cast_linalg_ordinal",
        "extent",
        "input_dtype",
        "output_dtype",
    }
    ordinal_fields = fields - {"extent", "input_dtype", "output_dtype"}
    return bool(
        isinstance(bound, Mapping)
        and set(bound) == fields
        and type(ordinal) is int
        and type(bound["cast_linalg_ordinal"]) is int
        and bound["cast_linalg_ordinal"] == ordinal
        and all(type(bound[key]) is int and bound[key] >= 0 for key in ordinal_fields)
        and len({bound[key] for key in ordinal_fields}) == len(ordinal_fields)
        and type(bound["extent"]) is int
        and 0 < bound["extent"] <= maximum // 8
        and bound["input_dtype"] == "i1"
        and bound["output_dtype"] == "i64"
    )


def _dynamic_roster_matches_control(witness: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    occurrences = witness.get("occurrences")
    if not isinstance(occurrences, list):
        return False
    dynamic = [
        item
        for item in occurrences
        if isinstance(item, Mapping) and item.get("schema") == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA
    ]
    if not dynamic:
        return True
    control = source.get("bounded_control_support")
    if not _control_proof_matches_source(control, witness):
        return False
    chains = control["internal_compaction_support"]
    return all(
        sum(chain["cast_linalg_ordinal"] == item["ordinal"] and chain == item.get("dynamic_bound") for chain in chains)
        == 1
        for item in dynamic
    )


def _math_record_valid(item: object) -> bool:
    """Revalidate exact literal/intrinsic metadata and selected symbol duty."""
    if not isinstance(item, Mapping):
        return False
    if item.get("schema") == MIXED_SCHEMA:
        try:
            validate_mixed_pattern(
                {
                    key: item[key]
                    for key in ("operation", "shape", "ordered_types", "input_shapes", "input_maps", "literal_bits")
                }
            )
            symbol = required_composite_math_symbol(item["schema"], item["operation"])
            if symbol:
                validate_source_linkage_contract(item["schema"], item["operation"], item.get("linkage_requirement"))
            return item.get("lowering_intrinsic_obligations") is None
        except (KeyError, TypeError, ValueError):
            return False
    if item.get("schema") != STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA:
        return item.get("literal_bits") is None and item.get("lowering_intrinsic_obligations") is None
    try:
        validate_serialized_static_composite_math_pattern(
            {
                "schema": COMPOSITE_SOURCE_SCHEMA,
                "operation": item["operation"],
                "shape": item["shape"],
                "ordered_types": item["ordered_types"],
                "literal_bits": item["literal_bits"],
                "lowering_intrinsic_obligations": item["lowering_intrinsic_obligations"],
            }
        )
        symbol = required_composite_math_symbol(item["schema"], item["operation"])
        linkage = item.get("linkage_requirement")
        if symbol is None:
            return linkage is None
        return validate_source_linkage_contract(item["schema"], item["operation"], linkage)["symbols"] == [symbol]
    except (InvalidLinalgPattern, KeyError, TypeError, ValueError):
        return False


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Require the mandatory witness to cite the same linked program bytes."""
    proof = source.get("linalg_host_support")
    if (
        not isinstance(proof, Mapping)
        or set(proof) != _TOP
        or proof.get("status") != LINKED
        or proof.get("scope") != SCOPE
    ):
        return False
    occurrences = proof.get("occurrences")
    if (
        proof.get("raw_source_sha256") != source.get("source_sha256")
        or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or not is_sha256(proof.get("raw_source_sha256"))
        or not is_sha256(proof.get("normalized_source_sha256"))
        or entry.get("source_sha256") != source.get("source_sha256")
        or type(proof.get("n_source_operations")) is not int
        or proof["n_source_operations"] < 1
        or proof["n_source_operations"] != source.get("n_source_operations")
        or not isinstance(occurrences, list)
        or type(proof.get("count")) is not int
        or proof["count"] != len(occurrences)
    ):
        return False
    selected = source.get("selected_index_observation")
    actual = proof.get("actual_index_observation")
    if (
        proof.get("selected_index_observation") != selected
        or actual != entry.get("index_lowering")
        or not control_support._record_matches_selected(actual, selected)  # noqa: PLC2701 -- shared verifier
        or not _all_shapes_fit(occurrences, selected["index_bits"])
    ):
        return False
    ordinals = []
    for item in occurrences:
        if not isinstance(item, Mapping) or set(item) != (
            _DYNAMIC_OCCURRENCE
            if item.get("schema") == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA
            else _GATHER_OCCURRENCE
            if item.get("schema") == EMBEDDING_GATHER_SCHEMA
            else _OCCURRENCE
        ):
            return False
        ordinal, shape, types, maps = item["ordinal"], item["shape"], item["ordered_types"], item["input_maps"]
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < proof["n_source_operations"]
            or not isinstance(shape, (list, tuple))
            or not shape
            or any(type(dim) is not int for dim in shape)
            or not isinstance(types, (list, tuple))
            or any(not isinstance(dtype, str) or not dtype for dtype in types)
            or not isinstance(maps, (list, tuple))
            or any(not isinstance(mapping, str) or not mapping for mapping in maps)
            or not isinstance(item["input_shapes"], (list, tuple))
            or not isinstance(item["declaration"], str)
            or not item["declaration"]
            or not isinstance(item["profile"], str)
            or not item["profile"]
            or not is_sha256(item["capability_spec_sha256"])
        ):
            return False
        try:
            _, _, expected_types = _contract(item["schema"], item["operation"], item["predicate"])
        except InvalidLinalgPattern:
            return False
        if tuple(types) != expected_types:
            return False
        if not _math_record_valid(item):
            return False
        if item["linkage_requirement"] is not None:
            try:
                validate_source_linkage_contract(item["schema"], item["operation"], item["linkage_requirement"])
            except ValueError:
                return False
        if item["schema"] == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_dynamic_boolean_cast_pattern(
                    {
                        key: item[key]
                        for key in ("operation", "shape", "ordered_types", "input_shapes", "input_maps", "shape_source")
                    }
                )
            except InvalidLinalgPattern:
                return False
            if not control_support.linked_selected_build_complete(source, entry, candidate_sha256):
                return False
        elif any(dim < 0 for dim in shape):
            return False
        if item["schema"] in {
            STATIC_POINTWISE_SOURCE_BODY_SCHEMA,
            STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
            STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
        } and (maps or item["input_shapes"]):
            return False
        if item["schema"] == STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA and item["predicate"] is not None:
            return False
        if item["schema"] == EMBEDDING_GATHER_SCHEMA:
            try:
                validate_embedding_gather_pattern(
                    {
                        key: item[key]
                        for key in (
                            "operation",
                            "shape",
                            "ordered_types",
                            "input_shapes",
                            "input_maps",
                            "table_shape",
                            "table_argument",
                        )
                    }
                )
            except (InvalidLinalgPattern, KeyError):
                return False
        if item["schema"] == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_static_boolean_pattern(
                    {key: item[key] for key in ("operation", "shape", "ordered_types", "input_shapes", "input_maps")}
                )
            except InvalidLinalgPattern:
                return False
        if item["schema"] == STATIC_PROJECTED_POINTWISE_BODY_SCHEMA:
            try:
                validate_serialized_static_projected_pointwise_pattern(
                    {
                        key: item[key]
                        for key in ("operation", "predicate", "shape", "ordered_types", "input_shapes", "input_maps")
                    }
                )
            except InvalidLinalgPattern:
                return False
        ordinals.append(ordinal)
    if ordinals != sorted(set(ordinals)):
        return False
    if not _dynamic_roster_matches_control(proof, source):
        return False
    linked = proof.get("linked_build")
    if _has_composite(proof):
        captured = proof.get("source_capture_path")
        if (
            not isinstance(captured, str)
            or not captured
            or not isinstance(linked, Mapping)
            or not _composite_source_matches(proof, source, linked, Path(captured))
        ):
            return False
    elif proof.get("source_capture_path") is not None:
        return False
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
        and all(is_sha256(linked[key]) for key in linked)
        and linked["candidate_tree_sha256"] == candidate_sha256
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )

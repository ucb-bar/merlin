"""Exact typed SSA capture graphs, kept separate from legacy demand-inventory bytes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from collections import Counter
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256
from merlin.compiler.ir.graph import _program_graph, type_layout

# ---- application_graph -------------------------------------------------------------------------

SCHEMA = "merlin.application_graph.v1"


def application_graph_inventory(path: str | Path) -> dict:
    """Observe raw and canonically normalized capture bytes without touching v1 inventory.

    Ordinals only have meaning within their digest-bound program. A normalizer
    changing the program does not establish a raw-to-normalized operation mapping.
    """
    from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
    from merlin.compiler.ir import mlir_query as mq

    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    raw_digest = hashlib.sha256(raw).hexdigest()
    normalized, normalization = normalize_capture_mlir(text)
    normalized_bytes = normalized.encode("utf-8")
    normalized_digest = hashlib.sha256(normalized_bytes).hexdigest()
    capture_graph = _program_graph(mq.parse(text), raw_digest)
    normalized_graph = _program_graph(mq.parse(normalized), normalized_digest)

    def comparable(graph: dict) -> str:
        identities = {
            "operation_id",
            "parent_operation_id",
            "consumer_operation_id",
            "producer_operation_id",
            "owner_operation_id",
            "block_id",
            "value_id",
            "id",
            "successor_block_ids",
        }

        def normalize(value, key=None):
            if isinstance(value, dict):
                return {field: normalize(member, field) for field, member in value.items() if field != "mlir_sha256"}
            if isinstance(value, list):
                return [normalize(member, key) for member in value]
            if isinstance(value, str) and key in identities:
                return value.replace(f"mlir:{graph['mlir_sha256']}:", "mlir:program:")
            return value

        return json.dumps(normalize(graph), sort_keys=True)

    correspondence = comparable(capture_graph) == comparable(normalized_graph)
    return {
        "schema": SCHEMA,
        "capture_sha256": raw_digest,
        "capture_bytes": len(raw),
        "normalized_mlir_sha256": normalized_digest,
        "normalized_mlir_bytes": len(normalized_bytes),
        "normalization": normalization,
        "capture_graph": capture_graph,
        "normalized_graph": normalized_graph,
        "normalization_correspondence": {
            "status": "identity"
            if raw_digest == normalized_digest
            else "serialization_equivalent"
            if correspondence
            else "unknown",
            "reason": "byte-identical programs"
            if raw_digest == normalized_digest
            else "exact ordered SSA, operation attributes/properties and control-flow graph agree"
            if correspondence
            else "normalization receipt binds program bytes, not per-operation rewrite correspondence",
        },
    }


# ---- application_inventory ---------------------------------------------------------------------
# Answer-free, operation-complete Phase 0 application capture inventory.
#
# Every operation in the normalized program handed to backends is accounted for. Admission is not
# lowering or compile acceptance; raw and normalized identities are retained separately.

_INT_MM_MAPS = [
    "affine_map<(d0, d1, d2) -> (d0, d2)>",
    "affine_map<(d0, d1, d2) -> (d2, d1)>",
    "affine_map<(d0, d1, d2) -> (d0, d1)>",
]


_INT_MM_ITERATORS = [
    "#linalg.iterator_type<parallel>",
    "#linalg.iterator_type<parallel>",
    "#linalg.iterator_type<reduction>",
]


_INT_MM_BODY = ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]


def exact_int_mm_generic_operation(op) -> bool:
    """Recognize the actual signed i8×i8→i32 reduction body, not a generic-op label."""
    from merlin.compiler.ir import mlir_query as query

    if query.op_name(op) != "linalg.generic" or query.attr_str(op, "prov.op") != "int_matmul":
        return False
    maps = op.properties.get("indexing_maps")
    iterators = op.properties.get("iterator_types")
    if (
        maps is None
        or [str(value) for value in maps] != _INT_MM_MAPS
        or iterators is None
        or [str(value) for value in iterators] != _INT_MM_ITERATORS
        or [query.op_name(child) for child in op.walk()][1:] != _INT_MM_BODY
        or len(op.operands) != 3
        or len(op.results) != 1
    ):
        return False
    operands = [query.type_shape_dtype(value.type) for value in op.operands]
    results = [query.type_shape_dtype(value.type) for value in op.results]
    if [dtype for _shape, dtype in [*operands, *results]] != ["i8", "i8", "i32", "i32"]:
        return False
    # Operation names alone do not prove this is a matrix product. A body that
    # multiplies the left operand by itself, or drops the accumulator, has the
    # same five names and types but different numerical semantics.
    block = op.regions[0].blocks[0]
    if len(block.args) != 3 or len(block.ops) != 5:
        return False
    ext_a, ext_b, multiply, add, yield_op = list(block.ops)
    for inner in (ext_a, ext_b, multiply, add, yield_op):
        semantic_attributes = {
            key: str(value)
            for key, value in {**inner.attributes, **inner.properties}.items()
            if not key.startswith("prov.")
        }
        allowed = {"overflowFlags": "#arith.overflow<none>"} if inner in (multiply, add) else {}
        if any(allowed.get(key) != value for key, value in semantic_attributes.items()):
            return False
    if not (
        list(ext_a.operands) == [block.args[0]]
        and list(ext_b.operands) == [block.args[1]]
        and set(multiply.operands) == {ext_a.results[0], ext_b.results[0]}
        and set(add.operands) == {block.args[2], multiply.results[0]}
        and list(yield_op.operands) == [add.results[0]]
    ):
        return False
    a, weight, out = (shape for shape, _dtype in operands)
    result = results[0][0]
    return (
        len(a) == len(weight) == len(out) == len(result) == 2
        and all(dim > 0 for shape in (a, weight, out, result) for dim in shape)
        and a[1] == weight[0]
        and out == result == [a[0], weight[1]]
    )


def verify_capture_receipt(path: str | Path) -> dict:
    """Verify the capture's materialized artifact bytes against its adjacent receipt.

    This says nothing about source closure. A producer's self-declared closure flag
    is not an independent execution attestation and must never be projected into
    Phase 0 admission. Older diagnostic captures remain inventoryable.
    """
    capture = Path(path)
    receipt_path = capture.parent / "capture_receipt.json"
    if not receipt_path.is_file():
        return {
            "status": "unverified",
            "receipt_sha256": None,
            "source_closure_verified": False,
            "errors": ["capture_receipt.json is absent"],
        }
    raw = receipt_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return {
            "status": "unverified",
            "receipt_sha256": digest,
            "source_closure_verified": False,
            "errors": [f"capture receipt is unreadable: {exc}"],
        }
    errors = []
    metadata = None
    metadata_identity = None
    if doc.get("schema") != "m2m.capture-receipt.v1":
        from merlin.development.targetgen.applications.native_reference_composition import SCHEMA, validate

        try:
            validate(doc)
        except (KeyError, TypeError, ValueError):
            errors.append(
                "invalid capture receipt composition"
                if doc.get("schema") == SCHEMA
                else "unsupported capture receipt schema"
            )
    if (doc.get("materialized_abi") or {}).get("complete") is not True:
        errors.append("capture receipt does not declare a complete materialized ABI")
    artifacts = doc.get("artifacts")
    required = {"model.mlir", "weights.safetensors", "weights.safetensors.manifest.json"}
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        errors.append("capture receipt lacks required model and weight artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
    for name, record in sorted(artifacts.items()):
        if not isinstance(name, str) or Path(name).name != name or not isinstance(record, dict):
            errors.append(f"invalid artifact record {name!r}")
            continue
        artifact = capture.parent / name
        if not artifact.is_file() or artifact.is_symlink():
            errors.append(f"receipt artifact missing or symlinked: {name}")
            continue
        size, expected = record.get("bytes"), record.get("sha256")
        if type(size) is not int or size < 0 or not is_sha256(expected):
            errors.append(f"invalid size/digest for receipt artifact {name}")
            continue
        if artifact.stat().st_size != size:
            errors.append(f"receipt artifact size differs: {name}")
            continue
        hasher = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected:
            errors.append(f"receipt artifact digest differs: {name}")
        elif name == "meta.json":
            # Project the same bytes whose membership was verified, not an ambient metadata read.
            raw_meta = artifact.read_bytes()
            if len(raw_meta) != size or hashlib.sha256(raw_meta).hexdigest() != expected:
                errors.append("receipt metadata changed during observation")
            else:
                try:
                    metadata = json.loads(raw_meta)
                    metadata_identity = {"sha256": expected, "bytes": size}
                except (ValueError, UnicodeDecodeError):
                    pass
    manifest_record = artifacts.get("quantization-manifest.json")
    manifest_pointer = metadata.get("quantization_manifest") if isinstance(metadata, dict) else None
    if manifest_record is not None or manifest_pointer is not None:
        if not isinstance(manifest_record, dict) or not isinstance(manifest_pointer, dict):
            errors.append("quantization manifest and metadata pointer must both be receipt artifacts")
        else:
            manifest_path = capture.parent / "quantization-manifest.json"
            try:
                manifest_bytes = manifest_path.read_bytes()
                if manifest_path.is_symlink() or hashlib.sha256(manifest_bytes).hexdigest() != manifest_record.get(
                    "sha256"
                ):
                    raise ValueError("manifest bytes differ from receipt")
                manifest = json.loads(manifest_bytes)
                if not isinstance(manifest, dict) or manifest.get("schema") != "m2m.quantization_manifest.v1":
                    raise ValueError("unsupported manifest schema")
                manifest_sha = hashlib.sha256(
                    json.dumps(
                        manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                if manifest_pointer != {
                    "path": "quantization-manifest.json",
                    "sha256": manifest_record["sha256"],
                    "manifest_sha256": manifest_sha,
                }:
                    raise ValueError("metadata pointer differs from manifest")
                mlir_bytes = capture.read_bytes()
                if capture.is_symlink() or hashlib.sha256(mlir_bytes).hexdigest() != artifacts["model.mlir"]["sha256"]:
                    raise ValueError("model MLIR bytes differ from receipt")
                from merlin.compiler.ir import mlir_query

                module = mlir_query.parse(mlir_bytes.decode("utf-8"))
                if mlir_query.attr_str(module, "prov.quantization_manifest_sha256") != manifest_sha:
                    raise ValueError("model MLIR does not bind manifest")
            except Exception as exc:  # Malformed producer bytes and parser refusals fail closed.
                errors.append(f"quantization manifest binding is invalid: {exc}")
    elif capture.is_file() and b"prov.quantization_manifest_sha256" in capture.read_bytes():
        errors.append("model MLIR names a quantization manifest absent from the receipt")
    recipe = metadata.get("recipe") if isinstance(metadata, dict) else None
    engine = recipe.get("software_numerical_engine") if isinstance(recipe, dict) else None
    if engine == "integer_reference":
        agreement = (metadata.get("integerization_receipt") or {}).get("golden_agreement") or {}
        pointer = agreement.get("output") or {}
        reference = artifacts.get("integer-reference.json")
        source = agreement.get("source") or {}
        tool_sources = (doc.get("tool") or {}).get("source_sha256") or {}
        reference_key = "m2m/capture/pt2e_integer_reference.py"
        if (metadata.get("integerization_receipt") or {}).get("schema") == "merlin.dynamic_integer_reference.v1":
            reference_key = "merlin/development/targetgen/applications/independent_references.py"
            tool_sources = (doc.get("tool") or {}).get("reference_sources") or {}
        if (
            not isinstance(pointer, dict)
            or pointer.get("path") != "integer-reference.json"
            or not isinstance(reference, dict)
            or pointer.get("sha256") != reference.get("sha256")
            or not isinstance(source, dict)
            or not is_sha256(source.get("sha256"))
            or source.get("sha256") != tool_sources.get(reference_key)
        ):
            errors.append("independent integer reference or its source is not bound by the capture receipt")
    dynamic_evidence = None
    stats = metadata.get("quantization_stats") if isinstance(metadata, dict) else None
    if not errors and isinstance(stats, dict) and stats.get("dynamic_contractions") is not None:
        try:
            from merlin.capture.receipts import dynamic_integerization_projection
            from merlin.development.targetgen.sources.frontend_trace import loads_frontend_trace

            trace_record = artifacts.get("frontend-trace.json")
            trace_path = capture.parent / "frontend-trace.json"
            if not isinstance(trace_record, dict) or trace_path.is_symlink():
                raise ValueError("dynamic contraction census lacks the receipt-bound frontend trace")
            raw_trace = trace_path.read_bytes()
            if (
                len(raw_trace) != trace_record["bytes"]
                or hashlib.sha256(raw_trace).hexdigest() != trace_record["sha256"]
            ):
                raise ValueError("frontend trace changed during observation")
            integer = metadata.get("integerization_receipt") or {}
            dynamic_evidence = dynamic_integerization_projection(
                metadata,
                stats,
                trace=loads_frontend_trace(raw_trace),
                trace_identity=trace_record,
                program_sha256=artifacts["model.mlir"]["sha256"],
                emitted=integer.get("integer_mm_emitted"),
                exported=integer.get("exported_integer_mm_count"),
            )
            if integer.get("schema") == "merlin.dynamic_integer_reference.v1":
                from merlin.development.targetgen.applications.independent_references import validate_receipt

                validate_receipt(
                    integer,
                    annotated=stats.get("annotated_contractions"),
                    dynamic_calls=stats["dynamic_contractions"]["device_calls"],
                    exported=integer.get("exported_integer_mm_count"),
                )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"dynamic integerization evidence is invalid: {exc}")
    result = {
        "status": "verified_materialized" if not errors else "unverified",
        "receipt_sha256": digest,
        # Only a separately verified sealed-execution issuer may establish this.
        # The materialized receipt is producer-authored, even when all its bytes match.
        "source_closure_verified": False,
        "errors": errors,
    }
    if (
        not errors
        and isinstance(metadata, dict)
        and metadata_identity is not None
        and isinstance(metadata.get("integerization_receipt"), dict)
        and capture.name == "model.mlir"
        and not capture.is_symlink()
        and not receipt_path.is_symlink()
        and not any(parent.is_symlink() for parent in capture.parents)
    ):
        result["capture_integerization"] = {
            "schema": "merlin.capture_integerization.v1",
            "status": "byte_bound_metadata",
            "metadata": metadata_identity,
            "capture": dict(artifacts["model.mlir"]),
            "capture_receipt_sha256": digest,
            "source_quantization": metadata.get("scheme"),
            "software_numerical_engine": engine,
            "reference_artifact": artifacts.get("integer-reference.json"),
            "integerization_receipt": metadata.get("integerization_receipt"),
        }
        if dynamic_evidence is not None:
            result["capture_integerization"]["dynamic_integerization"] = dynamic_evidence
    return result


def verified_static_integerization(projection: dict | None, *, receipt_sha256: str | None = None) -> bool:
    """Screen a saved, byte-bound static conversion observation without reopening its sources.

    This qualifies only the post-integerization arithmetic for an isolated operation slice. It is
    neither static/dynamic quantization equivalence nor target/compiler numerical qualification.
    """
    if not isinstance(projection, dict) or (
        projection.get("schema") != "merlin.capture_integerization.v1"
        or projection.get("status") != "byte_bound_metadata"
        or projection.get("source_quantization") != "int8_static_act_int8_weight"
        or not is_sha256(projection.get("capture_receipt_sha256"))
        or (receipt_sha256 is not None and projection.get("capture_receipt_sha256") != receipt_sha256)
    ):
        return False
    metadata = projection.get("metadata") or {}
    if (
        not isinstance(metadata, dict)
        or not is_sha256(metadata.get("sha256"))
        or type(metadata.get("bytes")) is not int
        or metadata["bytes"] <= 0
    ):
        return False
    capture = projection.get("capture") or {}
    if (
        not isinstance(capture, dict)
        or not is_sha256(capture.get("sha256"))
        or type(capture.get("bytes")) is not int
        or capture["bytes"] <= 0
    ):
        return False
    receipt = projection.get("integerization_receipt") or {}
    if isinstance(receipt, dict) and receipt.get("schema") == "merlin.dynamic_integer_reference.v1":
        from merlin.capture.receipts import dynamic_integerization_count
        from merlin.development.targetgen.applications.independent_references import validate_receipt

        try:
            calls = dynamic_integerization_count(
                projection.get("dynamic_integerization"),
                emitted=0,
                exported=receipt.get("exported_integer_mm_count"),
                program_sha256=capture["sha256"],
            )
            validate_receipt(
                receipt, annotated=0, dynamic_calls=calls, exported=receipt.get("exported_integer_mm_count")
            )
            agreement = receipt["golden_agreement"]
            pointer, reference = agreement.get("output") or {}, projection.get("reference_artifact") or {}
            source = agreement.get("source") or {}
            return (
                projection.get("software_numerical_engine") == "integer_reference"
                and pointer.get("path") == "integer-reference.json"
                and is_sha256(source.get("sha256"))
                and isinstance(source.get("path"), str)
                and is_sha256(pointer.get("sha256"))
                and pointer.get("sha256") == reference.get("sha256")
                and type(reference.get("bytes")) is int
                and reference["bytes"] > 0
            )
        except (ValueError, TypeError, KeyError):
            return False
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        return False
    count = receipt.get("quantized_contractions_seen")
    if (
        type(count) is not int
        or count <= 0
        or type(receipt.get("quantized_contractions_integerized")) is not int
        or receipt.get("quantized_contractions_integerized") != count
        or type(receipt.get("quantized_contractions_remaining")) is not int
        or receipt["quantized_contractions_remaining"] != 0
        or receipt.get("accumulator_bound_checked") is not True
        or receipt.get("refusals") != []
        or type(receipt.get("exported_integer_mm_count")) is not int
        or receipt["exported_integer_mm_count"] <= 0
        or type(receipt.get("integer_mm_emitted")) is not int
        or receipt["integer_mm_emitted"] <= 0
    ):
        return False
    dynamic = projection.get("dynamic_integerization")
    if dynamic is None:
        if receipt["integer_mm_emitted"] != receipt["exported_integer_mm_count"]:
            return False
    else:
        from merlin.capture.receipts import dynamic_integerization_count

        try:
            dynamic_integerization_count(
                dynamic,
                emitted=receipt["integer_mm_emitted"],
                exported=receipt["exported_integer_mm_count"],
                program_sha256=capture["sha256"],
            )
        except (ValueError, TypeError, KeyError):
            return False
    agreement = receipt.get("golden_agreement") or {}
    if not isinstance(agreement, dict):
        return False
    outputs = agreement.get("outputs")
    if (
        agreement.get("status") != "passed"
        or agreement.get("finite") is not True
        or type(agreement.get("samples")) is not int
        or agreement["samples"] <= 0
        or not isinstance(outputs, list)
        or not outputs
    ):
        return False
    for row in [agreement, *outputs]:
        if not isinstance(row, dict) or row.get("finite") is not True:
            return False
        if row is not agreement and row.get("within_tolerance") is not True:
            return False
        values = [row.get(key) for key in ("max_abs", "max_rel", "atol", "rtol")]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            return False
    engine = projection.get("software_numerical_engine")
    if engine not in (None, "integer_reference"):
        return False
    if engine == "integer_reference":
        source = agreement.get("source") or {}
        pointer = agreement.get("output") or {}
        reference = projection.get("reference_artifact") or {}
        executed = agreement.get("executed_contractions") or {}
        by_kind = receipt.get("quantized_by_kind") or {}
        if (
            agreement.get("reference") != "pt2e_integer"
            or any(row.get(key) != 0.0 for row in [agreement, *outputs] for key in ("atol", "rtol", "max_abs"))
            or not isinstance(source, dict)
            or not is_sha256(source.get("sha256"))
            or not isinstance(pointer, dict)
            or pointer.get("path") != "integer-reference.json"
            or not isinstance(reference, dict)
            or pointer.get("sha256") != reference.get("sha256")
            or not is_sha256(reference.get("sha256"))
            or type(reference.get("bytes")) is not int
            or reference["bytes"] <= 0
            or not isinstance(executed, dict)
            or any(executed.get(key) != count for key in ("total", "selected", "observed"))
            or not isinstance(by_kind, dict)
            or any(
                not isinstance(by_kind.get(kind), dict) or executed.get(kind) != by_kind[kind].get("seen")
                for kind in ("conv2d", "linear", "matmul")
            )
        ):
            return False
    return True


def int_mm_source_is_qualified(match: dict) -> bool:
    """Keep each source quantization identity distinct while screening isolated integer arithmetic."""
    scheme = match.get("source_quantization")
    if scheme == "int8_dyn_act_int8_weight":
        return True  # Legacy candidates retain their existing per-operation weight-origin screen.
    if scheme != "int8_static_act_int8_weight":
        return False
    sources = match.get("sources")
    return (
        isinstance(sources, list)
        and bool(sources)
        and all(
            isinstance(source, dict)
            and source.get("source_quantization") == scheme
            and is_sha256(source.get("capture_receipt_sha256"))
            and verified_static_integerization(
                source.get("capture_integerization"), receipt_sha256=source["capture_receipt_sha256"]
            )
            and source["capture_integerization"]["capture"]["sha256"] == source.get("capture_sha256")
            for source in sources
        )
    )


def exact_int_mm_geometry(row: dict, *, require_quant_origin: bool = True) -> tuple[int, int, int] | None:
    """Recognize only a standard rank-2 signed i8×i8→i32 ``aten._int_mm``.

    The linalg iteration space has three loops, but the tensor rank is two. Testing all operand
    types, maps, iterators and body avoids treating a family label or a coincidentally sized generic
    as this operator. A captured application must additionally prove its TorchAO weight origin.
    """
    if any(
        row.get(key) != value
        for key, value in {
            "operation": "aten._int_mm.default",
            "mlir_operation": "linalg.generic",
            "frontend_op": "aten._int_mm.default",
            "provenance_op": "int_matmul",
            "semantic_family": "contraction",
            "operand_format": "int8",
            "accumulator_dtypes": ["i32"],
            "indexing_maps": _INT_MM_MAPS,
            "iterator_types": _INT_MM_ITERATORS,
            "body_operations": _INT_MM_BODY,
        }.items()
    ):
        return None
    if require_quant_origin and not (row.get("quant_evidence") or {}).get("prov.quant_inner_1"):
        return None
    operands = row.get("ordered_operand_types")
    results = row.get("ordered_result_types")
    if not isinstance(operands, list) or len(operands) != 3 or not isinstance(results, list) or len(results) != 1:
        return None
    a, w, out = operands
    ashape, wshape, oshape = a.get("shape"), w.get("shape"), out.get("shape")
    if any(
        not isinstance(s, list) or len(s) != 2 or any(type(d) is not int or d <= 0 for d in s)
        for s in (ashape, wshape, oshape)
    ):
        return None
    m, k = ashape
    wk, n = wshape
    if (a.get("dtype"), w.get("dtype"), out.get("dtype")) != ("i8", "i8", "i32"):
        return None
    if wk != k or oshape != [m, n] or results != [{"shape": [m, n], "dtype": "i32"}]:
        return None
    if row.get("result_shapes") != [[m, n]]:
        return None
    shape = row.get("contraction_shape") or {}
    if any(shape.get(axis) != value for axis, value in (("M", m), ("K", k), ("N", n), ("rank", 3))):
        return None
    return m, k, n


def operation_structure(op) -> dict:
    """Ordered tensor ABI and linalg access pattern shared by inventory and slice verification."""
    from merlin.compiler.ir import mlir_query as mq

    name = mq.op_name(op)
    maps_attr = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    kinds_attr = op.properties.get("iterator_types") or op.attributes.get("iterator_types")
    return {
        "ordered_operand_types": [
            {"shape": shape, "dtype": dtype}
            for value in op.operands
            for shape, dtype in (mq.type_shape_dtype(value.type),)
        ],
        "ordered_result_types": [
            {"shape": shape, "dtype": dtype}
            for value in op.results
            for shape, dtype in (mq.type_shape_dtype(value.type),)
        ],
        "indexing_maps": [str(item) for item in maps_attr] if name == "linalg.generic" and maps_attr else None,
        "iterator_types": [str(item) for item in kinds_attr] if name == "linalg.generic" and kinds_attr else None,
        "body_operations": (
            [mq.op_name(child) for child in op.regions[0].blocks[0].ops]
            if name == "linalg.generic" and op.regions and op.regions[0].blocks
            else None
        ),
    }


# ---- access_observations -----------------------------------------------------------------------
# Layout, tail, broadcast and alias OBSERVATIONS of one structured operation, read from the graph.
#
# A software declaration admits an operation only for the layouts, tail handling, broadcasting and
# aliasing it names, and the screen treats an absent observation as unresolved -- never as a match. These
# four observations are therefore derived here from the operation itself, in the declaration's own
# vocabulary, and each is ``None`` (unknown) whenever the graph does not establish it:
#
# * ``layout``: ``row_major_contiguous`` when every shaped operand and result is a tensor with no
#   encoding, or a memref with no explicit layout. An encoding or a strided layout is not assumed away.
# * ``tails``: ``zero_pad_valid_window`` for a contraction whose body is only operand extension, a
#   product and a sum (so zero is the additive identity of every padded reduction element and padded
#   result rows and columns fall outside the valid window) over a static iteration space.
# * ``broadcasting``: from the indexing maps. ``none`` when no parallel dimension is shared by every
#   input and the result and each input contributes at most one exclusive result dimension;
#   ``independent_batches`` when the shared (batch) dimensions index every input; ``operand_broadcast``
#   otherwise (an input omits a dimension it would need to be independent).
# * ``aliasing``: ``disjoint_inputs_outputs`` when no output (init) operand is the same SSA value as an
#   input operand; ``in_place`` otherwise.
#
# Nothing here names a target or an operation: it reads types, indexing maps, iterator types and the
# body's operation names.

#: Body operations that keep zero the additive identity of a padded reduction element.
_ZERO_PAD_EXACT_BODY = frozenset(
    {
        "arith.extsi",
        "arith.extui",
        "arith.extf",
        "arith.sitofp",
        "arith.uitofp",
        "arith.muli",
        "arith.mulf",
        "arith.addi",
        "arith.addf",
        "linalg.yield",
    }
)
_NAMED_CONTRACTIONS = frozenset({"linalg.matmul", "linalg.batch_matmul", "linalg.matvec", "linalg.vecmat"})


def _layout(op) -> str | None:
    values = [*op.operands, *op.results]
    shaped = [value.type for value in values if hasattr(value.type, "get_shape") or hasattr(value.type, "shape")]
    if not shaped:
        return None
    layouts = {type_layout(value_type) for value_type in shaped}
    return "row_major_contiguous" if layouts == {"row_major_contiguous"} else None


def _n_inputs(op) -> int:
    return len(getattr(op, "inputs", ())) or max(len(op.operands) - len(op.results), 0)


def _maps(op):
    from merlin.compiler.ir import contraction_shapes as KS

    maps, kinds = KS.indexing_maps(op), KS._iterator_types(op)  # noqa: SLF001
    if not maps or not kinds:
        return None, None
    positions = []
    for results in maps:
        dims = []
        for expr in results:
            position = KS._dim_position(expr)  # noqa: SLF001
            if position is None:
                return None, None
            dims.append(position)
        positions.append(dims)
    return positions, kinds


def _tails(op, family: str | None) -> str | None:
    from merlin.compiler.ir import mlir_query as mq

    if family != "contraction":
        return None
    name = mq.op_name(op)
    if name not in _NAMED_CONTRACTIONS:
        body = [mq.op_name(inner) for inner in mq.walk(op)][1:]
        if not body or any(inner not in _ZERO_PAD_EXACT_BODY for inner in body):
            return None
    for value in [*op.operands, *op.results]:
        shape, _dtype = mq.type_shape_dtype(value.type)
        if shape and any(not isinstance(extent, int) or extent < 0 for extent in shape):
            return None
    return "zero_pad_valid_window"


def _broadcasting(op) -> str | None:
    positions, kinds = _maps(op)
    if positions is None:
        return None
    n_in = _n_inputs(op)
    inputs, result = positions[:n_in], positions[-1] if len(positions) > n_in else None
    if not inputs or result is None:
        return None
    parallel = {dim for dim, kind in enumerate(kinds) if kind == "parallel"}
    shared = {dim for dim in parallel if dim in result and all(dim in dims for dims in inputs)}
    exclusive = [len({dim for dim in dims if dim in parallel and dim in result} - shared) for dims in inputs]
    if any(count > 1 for count in exclusive) or any(dim not in set().union(*map(set, inputs)) for dim in result):
        return "operand_broadcast"
    return "independent_batches" if shared and len(inputs) > 1 else "none"


def _aliasing(op) -> str | None:
    n_in = _n_inputs(op)
    if n_in <= 0 or len(op.operands) <= n_in:
        return None
    inputs, outputs = list(op.operands[:n_in]), list(op.operands[n_in:])
    return "in_place" if any(output is source for output in outputs for source in inputs) else "disjoint_inputs_outputs"


def observe(op, family: str | None) -> dict[str, Any]:
    """The four observations for a structured (``linalg.*``) operation; every key ``None`` otherwise."""
    from merlin.compiler.ir import mlir_query as mq

    if not mq.op_name(op).startswith("linalg."):
        return {"layout": None, "tails": None, "broadcasting": None, "aliasing": None}
    return {
        "layout": _layout(op),
        "tails": _tails(op, family),
        "broadcasting": _broadcasting(op) if family == "contraction" else None,
        "aliasing": _aliasing(op),
    }

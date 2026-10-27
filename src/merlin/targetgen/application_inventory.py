"""Answer-free, operation-complete Phase 0 application capture inventory.

Every operation in the normalized program handed to backends is accounted for. Admission is not
lowering or compile acceptance; raw and normalized identities are retained separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from collections import Counter
from pathlib import Path

from merlin.capture.integerization import INTEGER_REFERENCE
from merlin.common.digest import is_sha256

# A separate, versioned addition preserves existing frozen v1 inventory bytes.
from merlin.targetgen.application_graph import application_graph_inventory

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


def exact_int_mm_generic_operation(op, *, require_zero_init: bool = True) -> bool:
    """Recognize the actual signed i8×i8→i32 reduction body, not a generic-op label.

    ``require_zero_init`` (default) additionally proves the accumulator init is an
    all-zero fill, which routing needs because the microkernel overwrites rather
    than accumulates. A pure format census (which numeric type does this
    contraction compute in) passes ``False``: the init does not change the format.
    """
    from merlin.common import mlir_query as query
    from merlin.kernels.shapes import zero_initialised

    if query.op_name(op) != "linalg.generic" or query.attr_str(op, "prov.op") != "int_matmul":
        return False
    if (
        len(op.operands) != 3
        or len(op.results) != 1
        or [query.op_name(child) for child in op.walk()][1:] != _INT_MM_BODY
        or not _independent_batch_matmul_maps(op)
        or (require_zero_init and not zero_initialised(op))
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
        len(a) == len(weight) == len(out) == len(result)
        and all(dim > 0 for shape in (a, weight, out, result) for dim in shape)
        and a[:-2] == weight[:-2]
        and a[-1] == weight[-2]
        and out == result == [*a[:-2], a[-2], weight[-1]]
    )


def _independent_batch_matmul_maps(op) -> bool:
    """Prove (batch..., M, K) × (batch..., K, N) without broadcasting.

    Each batch dimension must index both inputs and the output identically.
    Extents and the number of batch axes come from the operation itself.
    """
    from xdsl.dialects.builtin import AffineMapAttr, TensorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    maps = op.properties.get("indexing_maps")
    iterators = op.properties.get("iterator_types")
    output_type = op.results[0].type
    if not isinstance(output_type, TensorType):
        return False
    rank = len(output_type.get_shape())
    if rank < 2 or maps is None or iterators is None:
        return False
    if [str(value) for value in iterators] != [
        *["#linalg.iterator_type<parallel>"] * rank,
        "#linalg.iterator_type<reduction>",
    ]:
        return False
    dims = tuple(AffineExpr.dimension(axis) for axis in range(rank + 1))
    batch, m, n, k = dims[:-3], *dims[-3:]
    expected = [
        AffineMapAttr(AffineMap(rank + 1, 0, (*batch, m, k))),
        AffineMapAttr(AffineMap(rank + 1, 0, (*batch, k, n))),
        AffineMapAttr(AffineMap(rank + 1, 0, (*batch, m, n))),
    ]
    return list(maps) == expected


def _known_rational_reference(source):
    """Check the registered Merlin oracle and its complete arithmetic dependencies."""
    from merlin.common.paths import module_source_path

    if not isinstance(source, dict) or source.get("arithmetic") != "exact_rational":
        return False
    if source.get("numerics") not in {"integer_readout", "float_readout"}:
        return False

    def identity(name):
        path = module_source_path(name).resolve()
        return dict(path=name.replace(".", "/") + ".py", sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def recorded(entry):
        # A sealed worker imports Merlin from its own source mount, so the recorded location is
        # that mount's path. Bind the module identity (its package-relative path) and the exact bytes.
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None
        for name in (
            "merlin.targetgen.exact_quantized_reference",
            "merlin.targetgen.exact_arithmetic",
            "merlin.runtime.exact_readout",
        ):
            relative = name.replace(".", "/") + ".py"
            if entry["path"] == relative or entry["path"].endswith("/" + relative):
                return dict(path=relative, sha256=entry.get("sha256"))
        return None

    expected = identity("merlin.targetgen.exact_quantized_reference")
    expected.update(
        arithmetic="exact_rational",
        numerics=source["numerics"],
        dependencies=[identity("merlin.targetgen.exact_arithmetic"), identity("merlin.runtime.exact_readout")],
    )
    dependencies = source.get("dependencies")
    if not isinstance(dependencies, list):
        return False
    normalized = dict(source)
    normalized.update(path=(recorded(source) or {}).get("path"), dependencies=[(recorded(d) or {}) for d in dependencies])
    return normalized == expected


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
        from merlin.targetgen.native_reference_composition import SCHEMA, validate

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
                from merlin.common import mlir_query

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
        if (
            not isinstance(pointer, dict)
            or pointer.get("path") != "integer-reference.json"
            or not isinstance(reference, dict)
            or pointer.get("sha256") != reference.get("sha256")
            or not isinstance(source, dict)
            or not is_sha256(source.get("sha256"))
            or not (
                source.get("sha256") == tool_sources.get("m2m/capture/pt2e_integer_reference.py")
                or _known_rational_reference(source)
            )
        ):
            errors.append("independent integer reference or its source is not bound by the capture receipt")
    dynamic_evidence = None
    stats = metadata.get("quantization_stats") if isinstance(metadata, dict) else None
    if not errors and isinstance(stats, dict) and stats.get("dynamic_contractions") is not None:
        try:
            from merlin.capture.integerization import dynamic_integerization_projection
            from merlin.targetgen.frontend_trace import loads_frontend_trace

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
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        return False
    from merlin.targetgen.integerization_export import admit_integerization_census

    if not admit_integerization_census(
        receipt, dynamic_evidence=projection.get("dynamic_integerization"), program_sha256=capture["sha256"],
    ).accepted:
        return False
    from merlin.capture.integerization import non_contraction_program

    count = receipt.get("quantized_contractions_seen")

    contraction_free = non_contraction_program(receipt)
    if (
        type(count) is not int
        or (count <= 0 and not contraction_free)
        or type(receipt.get("quantized_contractions_integerized")) is not int
        or receipt.get("quantized_contractions_integerized") != count
        or type(receipt.get("quantized_contractions_remaining")) is not int
        or receipt["quantized_contractions_remaining"] != 0
        or receipt.get("accumulator_bound_checked") is not True
        or receipt.get("refusals") != []
        or type(receipt.get("exported_integer_mm_count")) is not int
        or receipt["exported_integer_mm_count"] < 0
        or (
            receipt["exported_integer_mm_count"] + receipt.get("exported_integer_bmm_count", 0) <= 0
            and not contraction_free
        )
        or type(receipt.get("integer_mm_emitted")) is not int
        or (receipt["integer_mm_emitted"] <= 0 and not contraction_free)
    ):
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
            agreement.get("reference") != INTEGER_REFERENCE
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
    from merlin.common import mlir_query as mq

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


def quantization_parameters(op) -> dict[str, int | None] | None:
    """Read per-tensor quantization operands/properties from parsed SSA, not provenance.

    A dynamic zero point or an unrecognized producer stays unknown. In particular,
    a dtype-compatible capture is not evidence for a symmetric quantization rule.
    """
    from merlin.common import mlir_query as mq

    if mq.op_name(op) not in {"quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor"}:
        return None

    def integer_attribute(attribute) -> int | None:
        value = getattr(getattr(attribute, "value", None), "data", None)
        return value if type(value) is int else None

    def attribute(owner, field):
        return owner.properties[field] if field in owner.properties else owner.attributes.get(field)

    def constant_integer(value) -> int | None:
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "properties"):
            return None
        if mq.op_name(owner) == "tensor.splat" and len(owner.operands) == 1:
            owner = getattr(owner.operands[0], "owner", None)
        if owner is None or not hasattr(owner, "properties") or mq.op_name(owner) != "arith.constant":
            return None
        return integer_attribute(attribute(owner, "value"))

    zero_point = constant_integer(op.operands[2]) if len(op.operands) == 3 else None
    parameters = {"zero_point": zero_point}
    for field in ("quant_min", "quant_max"):
        parameters[field] = integer_attribute(attribute(op, field))
    return parameters


def _reduction_provenance_hint(op, mq) -> tuple[dict, dict | None]:
    """Suggest a missing tag from adjacent operations without inventing source provenance.

    Even agreeing tags cannot prove this reduction's frontend semantics. The source tag
    remains absent for admission; the hint only helps diagnose capture metadata loss.
    """
    provenance = mq.provenance(op)
    if mq.op_name(op) != "linalg.reduce" or provenance.get("prov.aten") or len(op.operands) < 2 or not op.results:
        return provenance, None
    # The tag on a fill can name this reduction only when that fill initializes THIS reduction
    # exclusively. Two reductions may share one tagged zero while both have agreeing tagged
    # consumers; agreement on either side alone does not identify which reduction owns the tag.
    init_uses = tuple(op.operands[-1].uses)
    if len(init_uses) != 1 or init_uses[0].operation is not op:
        return provenance, None
    initializer = op.operands[-1].owner
    init_prov = mq.provenance(initializer) if hasattr(initializer, "attributes") else {}
    identity = (init_prov.get("prov.region_id"), init_prov.get("prov.aten"))
    users = [use.operation for result in op.results for use in result.uses]
    if (
        not all(identity)
        or not users
        or any(
            (mq.provenance(user).get("prov.region_id"), mq.provenance(user).get("prov.aten")) != identity
            for user in users
        )
    ):
        return provenance, None
    return provenance, {
        "method": "reduction_initializer_and_consumers_agree_v1",
        "status": "unverified",
        "suggested_frontend_op": identity[1],
        "region_id": identity[0],
        "consumer_count": len(users),
    }


_PREPARED_APPLICATION_SOURCES = weakref.WeakKeyDictionary()


class PreparedApplicationSource:
    """One program's parsed source, with exact byte and IR mutation checks.

    This is a call-local parsing optimization. Independent source witnesses
    still reopen and reprove their complete original source before acceptance.
    """

    def __init__(self, path: Path, normalized: str):
        from merlin.common import mlir_query as mq
        from merlin.frontends.capture_normalization import normalize_capture_mlir

        path = Path(path)
        raw = path.read_bytes()
        actual, normalization = normalize_capture_mlir(raw.decode("utf-8"))
        if actual != normalized:
            raise ValueError("prepared application source differs from normalized capture bytes")
        self.module = mq.parse(actual)
        # Weak ownership retains no program after its call-local carrier dies.
        # Original issued values are not mutable attributes on the carrier.
        _PREPARED_APPLICATION_SOURCES[self] = (path, raw, actual, normalization, self.module, str(self.module))

    def reopen(self, path):
        """Refuse changed bytes, another path, or mutation of the parsed IR."""
        original = _PREPARED_APPLICATION_SOURCES.get(self)
        if original is None:
            raise ValueError("application source has no original issued preparation")
        selected, raw, normalized, normalization, module, rendered = original
        if (
            Path(path) != selected
            or selected.read_bytes() != raw
            or self.module is not module
            or str(module) != rendered
        ):
            raise ValueError("prepared application source or parsed module changed")
        return raw, normalized, copy.deepcopy(normalization), module


def _application_operation_inventory(
    path: str | Path,
    target: str,
    cap_map: dict,
    *,
    include_graph: bool = False,
    capability_contract: dict | None = None,
    software_spec: dict | None = None,
    host_capabilities: dict | None = None,
    prepared_source: PreparedApplicationSource | None = None,
) -> dict:
    """Answer-free, operation-complete inventory for one application capture.

    A provenance tag names the frontend source; it is not the operation's computation. In particular,
    im2col gathers inherit their convolution's `prov.family=contraction`. The semantic family of a
    `linalg.generic` therefore comes from its body, and a nested body op is a component of that region,
    not a second independent accelerator demand. Unknowns remain rows, never disappear from a count.
    """
    from xdsl.dialects.builtin import TensorType

    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen import model_coverage as mc
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.eligibility import RegionDescriptor, is_eligible
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    p = Path(path)
    try:
        if prepared_source is None:
            data = p.read_bytes()
            normalized, normalization = normalize_capture_mlir(data.decode("utf-8"))
            module = mq.parse(normalized)
        elif type(prepared_source) is PreparedApplicationSource:
            data, normalized, normalization, module = prepared_source.reopen(p)
        else:
            raise ValueError("application inventory needs an original prepared source")
    except Exception as exc:
        raise ValueError(
            f"declared application capture {p}: cannot inventory MLIR: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        extents = mc._contraction_extents(module)  # noqa: PLC2701 -- region-coverage shape authority
    except Exception as exc:
        raise ValueError(
            f"declared application capture {p}: cannot inventory shapes: {type(exc).__name__}: {exc}"
        ) from exc

    module_quantization = mq.attr_str(module, "prov.quantization")
    grouped: dict[str, dict] = {}
    counts: Counter = Counter()
    n_operations = 0
    for ordinal, op in enumerate(mq.walk(module)):
        n_operations += 1
        name = mq.op_name(op)
        provenance, provenance_hint = _reduction_provenance_hint(op, mq)
        frontend = provenance.get("prov.aten")
        source_op = provenance.get("prov.op")
        callee_attr = op.properties.get("callee") if name == "func.call" else None
        callee = getattr(getattr(callee_attr, "root_reference", None), "data", None)
        canonical = frontend or source_op or (f"func.call @{callee}" if callee else name)
        parent = op.parent_op()
        nested = False
        while parent is not None:
            if mq.op_name(parent).startswith("linalg."):
                nested = True
                break
            parent = parent.parent_op()

        structural = name in {"builtin.module", "func.func", "func.return", "linalg.yield", "scf.yield"}
        support = name in {"arith.constant", "cf.assert"} or name.startswith(("tensor.", "memref.", "scf."))
        family: str | None = None
        family_basis = "unknown"
        reduction_kind = cc.reduction_kind(op) if name in {"linalg.generic", "linalg.reduce"} else None
        # The movement form comes from the carrier and its indexing maps, never a provenance label.
        movement_form = sf.carrier_form(op) or sf.operation_form(
            source_op or frontend or name.rpartition(".")[2], carrier_op=name
        )
        if name == "linalg.generic":
            try:
                generic_kind = cc.classify_generic(op)
            except Exception as exc:
                raise ValueError(
                    f"declared application capture {p}: cannot classify operation {ordinal} ({name}): {exc}"
                ) from exc
            family = {
                "contraction": "contraction",
                "sum-reduction": "reduction",
                "max-reduction": "reduction",
                "absmax": "reduction",
                "other-reduction": "reduction",
                "movement": "movement",
                "elementwise": "elementwise_map",
            }.get(generic_kind)
            family_basis = "generic_body" if family else "unclassified_generic_body"
        elif name.startswith("linalg."):
            family = "reduction" if name == "linalg.reduce" else sf.from_op(name.rpartition(".")[2])
            family_basis = "linalg_op" if family else "unclassified_linalg_op"
        elif name.startswith(("arith.", "math.")):
            family, family_basis = "elementwise_map", "dialect_operation"
        elif name.startswith(("tensor.", "memref.")):
            family, family_basis = "movement", "dialect_operation"
        elif not structural:
            # A custom-dialect op may have a known semantic name. The source region's family tag is
            # not used as authority: it is copied onto unrelated operations in captured models.
            family = sf.from_op(name.rpartition(".")[2])
            family_basis = "operation_name" if family else "unclassified_operation"

        operand_types = [mq.type_shape_dtype(value.type) for value in op.operands]
        result_types = [mq.type_shape_dtype(value.type) for value in op.results]
        operand_dtypes = sorted({dtype for _shape, dtype in operand_types if dtype})
        result_dtypes = sorted({dtype for _shape, dtype in result_types if dtype})
        # Keep the ordered ABI and linalg access pattern in the digest-bound sidecar. A dtype set plus
        # a family/shape class cannot distinguish this matmul from another generic of the same size.
        structure = operation_structure(op)
        from merlin.targetgen import access_observations

        # [] is a known rank-zero tensor shape, not a missing shape. The query also
        # returns [] for non-tensor scalars, so retain the type distinction here.
        result_shapes = [
            shape
            for value, (shape, _dtype) in zip(op.results, result_types, strict=True)
            if shape or isinstance(value.type, TensorType)
        ]
        operand_shapes = [
            shape
            for value, (shape, _dtype) in zip(op.operands, operand_types, strict=True)
            if shape or isinstance(value.type, TensorType)
        ]
        m, k, n, rank = extents.get(id(op), (None, None, None, None))
        if rank is None and result_shapes:
            rank = len(result_shapes[0])
        elif rank is None and operand_shapes:
            rank = len(operand_shapes[0])
        shape = {"M": m, "K": k, "N": n, "rank": rank} if m is not None else None
        shape_confidence = (
            "observed_iteration_space"
            if shape is not None
            else "result_type"
            if result_shapes
            else "operand_type"
            if operand_shapes
            else "unknown"
            if name.startswith("linalg.") and not structural and not nested
            else "not_applicable"
        )
        input_format = mc._elem_dtype(op)  # noqa: PLC2701 -- same dtype authority as region coverage
        if input_format is None:
            input_format = next(
                (mc._ELEM_DTYPE[dtype] for _shape, dtype in operand_types if dtype in mc._ELEM_DTYPE),  # noqa: PLC2701
                None,
            )
        accumulator_dtypes = None
        if family == "contraction":
            accumulator_dtypes = (
                sorted(
                    {
                        dtype
                        for child in mq.walk(op)
                        if mq.op_name(child) in {"arith.addf", "arith.addi"}
                        for value in child.results
                        if (dtype := mq.type_shape_dtype(value.type)[1])
                    }
                )
                or None
            )
        if structural:
            disposition, reason = "structural", "IR container or terminator; not an independent demand"
        elif nested:
            disposition, reason = "component", "inside a linalg region; accounted with its parent computation"
        elif support or (family == "movement" and name.startswith("linalg.")):
            disposition = "support_required"
            reason = (
                "layout/data movement requires an explicit lowering; a movement-family capability "
                "does not prove this operation executes on the accelerator"
                if family == "movement" and name.startswith("linalg.")
                else "constant, tensor/memory, or control-flow op requires lowering; not a separate compute capsule"
            )
        elif name == "func.call":
            disposition = "unclassified"
            reason = "call requires a resolved callee/body or a declared external host lowering"
        elif family is None or shape_confidence == "unknown" or (input_format is None and not operand_dtypes):
            disposition, reason = "unclassified", "semantic family, operand format, or required shape is unknown"
        elif input_format is None:
            disposition = "host_required"
            reason = "known operand dtype has no hardware format mapping; host lowering remains required"
        else:
            verdict = is_eligible(
                RegionDescriptor(
                    op=name.rpartition(".")[2],
                    family=family,
                    in_dtype=input_format,
                    out_dtype=result_dtypes[0] if len(result_dtypes) == 1 else None,
                    m=m,
                    k=k,
                    n=n,
                    rank=rank,
                    form=movement_form,
                    kind=reduction_kind if family == "reduction" else None,
                ),
                cap_map,
            )
            disposition = (
                "hardware_admitted" if verdict.eligible else "unclassified" if verdict.undetermined else "host_required"
            )
            reason = "hardware capability only; lowering unverified" if verdict.eligible else verdict.reason

        quant_evidence = {
            key: value
            for key, value in sorted(provenance.items())
            if any(token in key for token in ("quant", "scale", "format"))
        }
        layout_evidence = {
            key: value
            for key, value in sorted(provenance.items())
            if any(token in key for token in ("layout", "transpose", "conv_path"))
        }
        signature = {
            "operation": canonical,
            "mlir_operation": name,
            "frontend_op": frontend,
            **({"provenance_hint": provenance_hint} if provenance_hint is not None else {}),
            "provenance_op": source_op,
            "callee": callee,
            "semantic_family": family,
            "family_basis": family_basis,
            # The sum/max distinction classify_generic draws, kept so a sum-only capability can tell.
            **({"reduction_kind": reduction_kind} if family == "reduction" and reduction_kind else {}),
            **({"movement_form": movement_form} if family == "movement" and movement_form else {}),
            "operand_dtypes": operand_dtypes,
            "ordered_operand_types": structure["ordered_operand_types"],
            "operand_format": input_format,
            "result_dtypes": result_dtypes,
            "ordered_result_types": structure["ordered_result_types"],
            "result_shapes": result_shapes,
            "indexing_maps": structure["indexing_maps"],
            "iterator_types": structure["iterator_types"],
            "body_operations": structure["body_operations"],
            "accumulator_dtypes": accumulator_dtypes,
            "contraction_shape": shape,
            "shape_confidence": shape_confidence,
            "quant_evidence": quant_evidence or None,
            "layout_evidence": layout_evidence or None,
            # Graph-derived access observations in the software declarations' vocabulary; None = unknown.
            **access_observations.observe(op, family),
            "disposition": disposition,
            "reason": reason,
            "provenance_present": bool(frontend or source_op),
        }
        parameters = quantization_parameters(op)
        if parameters is not None:
            signature["quantization_parameters"] = parameters
        if disposition == "unclassified" and software_spec is not None and host_capabilities is not None:
            # Independent hardware eligibility remains unknown for dtype-changing operations.
            # A separately pinned and reviewed host declaration can nevertheless resolve their
            # placement without duplication in accelerator software semantics. Use the same
            # exact-signature screen as whole-program admission.
            from merlin.targetgen.operation_accounting import admit_operation_row

            admission = admit_operation_row(
                signature,
                software_spec=software_spec,
                capability_contract=capability_contract,
                capability_map=cap_map,
                host_capabilities=host_capabilities,
                source_operations=(op,),
            )
            if (
                admission["host_admission"]["status"] == "admitted"
                and admission["host_admission"].get("operator_accepted", admission["host_admission"]["reviewed"])
                and admission["accelerator_admission"]["status"] != "admitted"
            ):
                disposition = signature["disposition"] = "host_required"
                signature["reason"] = "exact reviewed pinned host declaration admits this signature"
        key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        slot = grouped.setdefault(key, {**signature, "count": 0, "ordinals": []})
        slot["count"] += 1
        slot["ordinals"].append(ordinal)
        counts[disposition] += 1
        if not (frontend or source_op) and not structural and not nested:
            counts["untagged"] += 1
        if shape_confidence == "unknown":
            counts["shape_unknown"] += 1

    if n_operations == 0 or sum(row["count"] for row in grouped.values()) != n_operations:
        raise ValueError(f"declared application capture {p}: parsed operations were not fully inventoried")
    rows = [grouped[key] for key in sorted(grouped)]
    trace_path = p.parent / "frontend-trace.json"
    capture_receipt = verify_capture_receipt(p)
    integerization = capture_receipt.pop("capture_integerization", None)
    result = {
        "capture": f"{p.parent.name}/{p.name}",
        "capture_source_path": str(p.absolute()),
        "capture_sha256": hashlib.sha256(data).hexdigest(),
        "frontend_trace": (
            {"source_path": str(trace_path.absolute()), "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}
            if trace_path.is_file() and not trace_path.is_symlink()
            else None
        ),
        "capture_receipt": capture_receipt,
        "capture_normalization": normalization,
        "capture_quantization": module_quantization,
        "n_operations": n_operations,
        "n_signatures": len(rows),
        "counts": dict(sorted(counts.items())),
        "status": "incomplete" if counts["unclassified"] else "inventoried",
        "signatures": rows,
        "scope": (
            "parsed operations in the canonically normalized capture only; hardware admission "
            "is not compiler lowering or model execution"
        ),
    }
    if integerization is not None and integerization.get("source_quantization") == module_quantization:
        result["capture_integerization"] = integerization
    if include_graph:
        result["operation_graph"] = application_graph_inventory(p)
    if prepared_source is not None:
        prepared_source.reopen(p)
    return result


def application_operation_inventory(
    path: str | Path,
    target: str,
    *,
    capability_contract: dict,
    workload_id: str,
    workload_role: str = "validation",
) -> dict:
    """Observe one complete validation program without deriving any capsule.

    Unlike ``application_demand_inventory``, this evaluator entrypoint can read
    held-out models. Its result labels the role explicitly; it cannot certify
    compiler support or make a validation capture a legal derivation source.
    """
    if not isinstance(workload_id, str) or not workload_id or workload_role != "validation":
        raise ValueError("evaluation inventory requires an explicit validation workload identity")
    from merlin.targetgen.eligibility import capability_map_from_contract

    result = _application_operation_inventory(
        path, target, capability_map_from_contract(capability_contract), include_graph=True
    )
    result["workload_identity"] = {
        "workload_id": workload_id,
        "workload_role": workload_role,
        "coverage_scope": "full_capture",
    }
    result["purpose"] = "evaluation_only; not a Phase 0 derivation input"
    return result


def application_demand_inventory(
    applications: dict[str, str | Path],
    target: str,
    *,
    detailed: bool = False,
    capability_contract: dict | None = None,
    software_spec: dict | None = None,
    host_capabilities: dict | None = None,
    include_graph: bool = False,
    application_metadata: dict[str, dict] | None = None,
) -> dict:
    """Inventory declared derivation applications; keep exact operation rows in a generated sidecar.

    The default is a reviewable requirement summary. ``detailed=True`` retains all grouped signatures
    and exact operation ordinals, suitable for a digest-checked generated sidecar, not a hand-edited spec.
    """
    if not applications:
        return {
            "schema_version": 1,
            "status": "not_declared",
            "coverage_status": "not_applicable",
            "applications": {},
            "n_operations": 0,
        }
    from merlin.targetgen import claim_models as cm
    from merlin.targetgen.eligibility import capability_map_for_target, capability_map_from_contract

    cap_map = (
        capability_map_from_contract(capability_contract)
        if capability_contract is not None
        else capability_map_for_target(target)
    )
    output: dict[str, dict] = {}
    for label, path in sorted(applications.items()):
        if cm.is_claim_bundle(label) or cm.is_claim_bundle(Path(path).resolve().parent.name):
            raise ValueError(f"application {label!r} is a held-out claim model and cannot derive Phase 0 demands")
        output[str(label)] = _application_operation_inventory(
            path,
            target,
            cap_map,
            include_graph=include_graph,
            capability_contract=capability_contract,
            software_spec=software_spec,
            host_capabilities=host_capabilities,
        )
        identity = (application_metadata or {}).get(str(label))
        if identity is not None:
            if not isinstance(identity, dict) or identity.get("workload_role", "unknown") not in {
                "iteration",
                "validation",
                "unknown",
            }:
                raise ValueError("application metadata requires an explicit iteration/validation/unknown workload role")
            if identity.get("coverage_scope", "unknown") not in {"full_capture", "representative_subset", "unknown"}:
                raise ValueError("application metadata requires an explicit full/representative/unknown coverage scope")
            if identity.get("workload_role") == "validation":
                raise ValueError("held-out validation captures cannot derive Phase 0 iteration demands")
            for field in ("workload_id", "source_workload_id"):
                source = identity.get(field)
                if source is not None and (not isinstance(source, str) or not source):
                    raise ValueError(f"application metadata {field} must be a nonempty source identity")
                if source is not None and cm.is_claim_bundle(source):
                    raise ValueError(f"application metadata {field} names a held-out validation source")
            output[str(label)]["workload_identity"] = dict(identity)
    full = {
        "schema_version": 2 if include_graph else 1,
        "status": "incomplete" if any(row["status"] == "incomplete" for row in output.values()) else "inventoried",
        "coverage_status": "unverified",
        "applications": output,
        "n_operations": sum(row["n_operations"] for row in output.values()),
        "basis": "only explicitly supplied application captures; held-out workload_spec.models are not read",
        "coverage_note": "operation inventory has not been matched to generated capsules or a submitted compiler",
    }
    if detailed:
        return full

    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    operation_groups: dict[str, dict] = {}
    compact_apps: dict[str, dict] = {}
    for label, app in output.items():
        compact_apps[label] = {
            key: app[key]
            for key in (
                "capture",
                "capture_sha256",
                "capture_receipt",
                "capture_normalization",
                "capture_quantization",
                "n_operations",
                "n_signatures",
                "counts",
                "status",
            )
        }
        for row in app["signatures"]:
            if row["disposition"] in {"structural", "component"}:
                continue
            shape = row["contraction_shape"]
            rank = shape["rank"] if shape else len(row["result_shapes"][0]) if row["result_shapes"] else None
            shape_class = f"{'contraction' if shape else 'result'}:rank_{rank if rank is not None else 'unknown'}"
            key_fields = {
                "operation": row["operation"],
                "mlir_operation": row["mlir_operation"],
                "semantic_family": row["semantic_family"],
                "operand_format": row["operand_format"],
                "disposition": row["disposition"],
                "shape_class": shape_class,
            }
            key = json.dumps(key_fields, sort_keys=True, separators=(",", ":"))
            group = operation_groups.setdefault(key, {**key_fields, "count": 0, "sources": {}})
            group["count"] += row["count"]
            source = group["sources"].setdefault(label, {"capture_sha256": app["capture_sha256"], "count": 0})
            source["count"] += row["count"]
    return {
        "schema_version": 1,
        "status": full["status"],
        "coverage_status": "unverified",
        "n_operations": full["n_operations"],
        "n_signatures": sum(app["n_signatures"] for app in output.values()),
        "applications": compact_apps,
        "operation_groups": [operation_groups[key] for key in sorted(operation_groups)],
        "full_inventory_sha256": digest,
        "basis": full["basis"],
        "coverage_note": full["coverage_note"],
    }

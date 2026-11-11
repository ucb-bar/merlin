"""Replayable source preparation with separately selected numerical changes."""

from .constant_folding import fold_source_constants, verify_constant_folding
from .constant_math_backend import MathHelper, validate_policy
from .constant_math_precision import prepare_constant_math, verify_constant_math

_LEGACY_SCHEMA = "merlin.source_preparation.v1"
SCHEMA = "merlin.source_preparation.v2"
_VIEWS_SCHEMA = "merlin.source_preparation.v3"
_COMMON_SCHEMA = "merlin.source_preparation.v4"
_COMMON_TRANSFORM = "merlin.common_pure_values.v1"
_RESHAPES_SCHEMA = "merlin.source_preparation.v5"
_RESHAPES_TRANSFORM = "merlin.reshape_chains.v1"


def prepare_source(
    source, *, constant_math_policy=None, include_tensor_constants=True, common_values=False, compose_reshapes=False
):
    """Prepare exact indexing before separately selected numerical changes."""
    from .affine_views import fuse_affine_views

    if type(common_values) is not bool:
        raise ValueError("common_values must be Boolean")
    if type(compose_reshapes) is not bool:
        raise ValueError("compose_reshapes must be Boolean")
    if compose_reshapes:
        source, prepared = prepare_source(
            source,
            constant_math_policy=constant_math_policy,
            include_tensor_constants=include_tensor_constants,
            common_values=common_values,
        )
        return append_reshape_preparation(source, prepared)
    if common_values:
        from .common_values import common_pure_values

        original = source
        source, prepared = prepare_source(
            source, constant_math_policy=constant_math_policy, include_tensor_constants=include_tensor_constants
        )
        source, common = common_pure_values(source)
        return source, {
            "schema": _COMMON_SCHEMA,
            "before_digest": original.digest(),
            "after_digest": source.digest(),
            "common_values": common,
            "prepared": prepared,
            "limits": prepared.get("limits", []),
        }
    original = source
    source, views = fuse_affine_views(source)
    source, numerical = _prepare_numerical_source(
        source,
        constant_math_policy=constant_math_policy,
        include_tensor_constants=include_tensor_constants,
    )
    if not views["rewrites"]:
        return source, numerical
    return source, {
        "schema": _VIEWS_SCHEMA,
        "before_digest": original.digest(),
        "after_digest": source.digest(),
        "views": views,
        "numerical": numerical,
        "limits": numerical.get("limits", []),
    }


def _prepare_numerical_source(source, *, constant_math_policy=None, include_tensor_constants=True):
    """Keep exact folding as the default; mathematical power needs an opt-in."""
    if constant_math_policy is None:
        return fold_source_constants(source, eliminate_dead=True, include_tensor_ops=include_tensor_constants)
    policy = validate_policy(constant_math_policy)
    helper = MathHelper.selected(policy)
    original, stages, remaining = source, [], policy["element_budget"]
    limited = []
    for _ in range(policy["max_rounds"]):
        source, receipt = fold_source_constants(
            source,
            eliminate_dead=True,
            include_tensor_ops=include_tensor_constants,
        )
        stages.append({"kind": "exact_constants", "receipt": receipt})
        source, receipt = prepare_constant_math(source, policy, helper=helper, element_limit=remaining)
        stages.append({"kind": "math_precision", "receipt": receipt})
        remaining -= receipt["element_work"]
        limited = [row for row in receipt["deferred"] if row["status"] == "resource_limit"]
        if limited or not receipt["folds"]:
            break
    deferred = receipt["deferred"]
    source, receipt = fold_source_constants(source, eliminate_dead=True, include_tensor_ops=include_tensor_constants)
    stages.append({"kind": "exact_constants", "receipt": receipt})
    # After the final exact fold, a formerly runtime dependency may have become
    # constant. It needs another round; absence of that budget is inconclusive.
    if not limited:
        limited = [
            {"node": row["node"], "status": "resource_limit", "reason": "constant math preparation round budget"}
            for row in deferred
            if row["reason"] == "runtime power operand"
            and all(source.node(key).effect == "constant" for key in source.node(row["node"]).inputs)
        ]
    return source, {
        "schema": SCHEMA if include_tensor_constants else _LEGACY_SCHEMA,
        "before_digest": original.digest(),
        "after_digest": source.digest(),
        "policy": policy,
        "stages": stages,
        "limits": limited,
    }


def append_reshape_preparation(source, prepared):
    """Append exact reshape composition to an existing preparation receipt."""
    from .reshape_chains import compose_reshape_chains

    if prepared.get("after_digest") != source.digest():
        raise ValueError("reshape preparation does not follow the supplied source")
    source, reshapes = compose_reshape_chains(source)
    return source, {
        "schema": _RESHAPES_SCHEMA,
        "before_digest": prepared["before_digest"],
        "after_digest": source.digest(),
        "prepared": prepared,
        "reshapes": reshapes,
        "limits": prepared.get("limits", []),
    }


def verify_source_preparation(source, record):
    """Reconstruct the source and replay stage order, limits and numerical policy."""
    if isinstance(record, dict) and record.get("schema") == _RESHAPES_TRANSFORM:
        from .reshape_chains import verify_reshape_chains

        return verify_reshape_chains(source, record)
    if isinstance(record, dict) and record.get("schema") == _RESHAPES_SCHEMA:
        from .reshape_chains import verify_reshape_chains

        if (
            set(record) != {"schema", "before_digest", "after_digest", "reshapes", "prepared", "limits"}
            or record["after_digest"] != source.digest()
            or not isinstance(record["prepared"], dict)
            or record["prepared"].get("schema") == _RESHAPES_SCHEMA
            or record["limits"] != record["prepared"].get("limits", [])
        ):
            raise ValueError("source preparation reshape receipt has invalid fields or identity")
        prepared = verify_reshape_chains(source, record["reshapes"])
        original = verify_source_preparation(prepared, record["prepared"])
        if original.digest() != record["before_digest"]:
            raise ValueError("source preparation reshape original changed")
        return original
    if isinstance(record, dict) and record.get("schema") == _COMMON_TRANSFORM:
        from .common_values import verify_common_pure_values

        return verify_common_pure_values(source, record)
    if isinstance(record, dict) and record.get("schema") == _COMMON_SCHEMA:
        from .common_values import verify_common_pure_values

        if (
            set(record) != {"schema", "before_digest", "after_digest", "common_values", "prepared", "limits"}
            or record["after_digest"] != source.digest()
            or not isinstance(record["prepared"], dict)
            or record["prepared"].get("schema") == _COMMON_SCHEMA
            or record["limits"] != record["prepared"].get("limits", [])
        ):
            raise ValueError("source preparation commoning receipt has invalid fields or identity")
        prepared = verify_common_pure_values(source, record["common_values"])
        original = verify_source_preparation(prepared, record["prepared"])
        if original.digest() != record["before_digest"]:
            raise ValueError("source preparation commoning original changed")
        return original
    if isinstance(record, dict) and record.get("schema") == _VIEWS_SCHEMA:
        from .affine_views import verify_affine_views

        if (
            set(record) != {"schema", "before_digest", "after_digest", "views", "numerical", "limits"}
            or record["after_digest"] != source.digest()
            or not isinstance(record["numerical"], dict)
            or record["numerical"].get("schema") == _VIEWS_SCHEMA
            or not isinstance(record["views"], dict)
            or not record["views"].get("rewrites")
            or record["limits"] != record["numerical"].get("limits", [])
        ):
            raise ValueError("source preparation indexing receipt has invalid fields or identity")
        indexed = verify_source_preparation(source, record["numerical"])
        original = verify_affine_views(indexed, record["views"])
        if original.digest() != record["before_digest"]:
            raise ValueError("source preparation indexing original changed")
        return original
    if not isinstance(record, dict) or record.get("schema") not in {SCHEMA, _LEGACY_SCHEMA}:
        return verify_constant_folding(source, record)
    if (
        set(record) != {"schema", "before_digest", "after_digest", "policy", "stages", "limits"}
        or source.digest() != record["after_digest"]
        or not isinstance(record["stages"], list)
    ):
        raise ValueError("source preparation receipt has invalid fields or result identity")
    policy = validate_policy(record["policy"])
    if not 3 <= len(record["stages"]) <= 2 * policy["max_rounds"] + 1:
        raise ValueError("source preparation stage count differs from the work budget")
    original = source
    for index in reversed(range(len(record["stages"]))):
        stage = record["stages"][index]
        expected = "exact_constants" if index % 2 == 0 else "math_precision"
        if not isinstance(stage, dict) or set(stage) != {"kind", "receipt"} or stage["kind"] != expected:
            raise ValueError("source preparation stage order changed")
        if expected == "math_precision":
            if stage["receipt"].get("policy") != policy:
                raise ValueError("source preparation numerical policy changed")
            original = verify_constant_math(original, stage["receipt"])
        else:
            original = verify_constant_folding(original, stage["receipt"])
    if original.digest() != record["before_digest"]:
        raise ValueError("source preparation original identity changed")
    replayed, expected = _prepare_numerical_source(
        original,
        constant_math_policy=policy,
        include_tensor_constants=record["schema"] == SCHEMA,
    )
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("source preparation differs from bounded replay")
    return original


def preparation_filename(record):
    composite = record.get("schema") in {
        SCHEMA,
        _LEGACY_SCHEMA,
        _VIEWS_SCHEMA,
        _COMMON_SCHEMA,
        _COMMON_TRANSFORM,
        _RESHAPES_SCHEMA,
        _RESHAPES_TRANSFORM,
    }
    return "source-preparation.json" if composite else "source-constant-folding.json"


def preparation_counts(record):
    if record.get("schema") == _RESHAPES_TRANSFORM:
        return {
            "source_reshape_rewrites": sum(not row["removed"] for row in record["changes"]),
            "source_removed_reshapes": sum(row["removed"] for row in record["changes"]),
        }
    if record.get("schema") == _RESHAPES_SCHEMA:
        return {**preparation_counts(record["prepared"]), **preparation_counts(record["reshapes"])}
    if record.get("schema") == _COMMON_TRANSFORM:
        return {"source_common_values": sum(row["replacement"] is not None for row in record["changes"])}
    if record.get("schema") == _COMMON_SCHEMA:
        return {
            **preparation_counts(record["prepared"]),
            "source_common_values": sum(row["replacement"] is not None for row in record["common_values"]["changes"]),
        }
    if record.get("schema") == _VIEWS_SCHEMA:
        return {
            **preparation_counts(record["numerical"]),
            "source_affine_view_rewrites": len(record["views"]["rewrites"]),
        }
    composite = record.get("schema") in {SCHEMA, _LEGACY_SCHEMA}
    stages = [row["receipt"] for row in record["stages"]] if composite else [record]
    return {
        "source_constant_folds": sum(len(row["folds"]) for row in stages),
        "source_dead_computations": sum(len(row.get("dead_computations", ())) for row in stages),
    }

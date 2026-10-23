#!/usr/bin/env python3
"""Observe registered ATen schemas and Core ATen tags in the capture interpreter.

Separate process for the same reason the capture worker is: torch is not installed in the merlin venv,
so anything that needs it has to be asked rather than imported.

⚠ THE OBVIOUS API IS THE WRONG ONE. ``torch._decomp.core_aten_decompositions()`` looks like the Core
ATen opset and is not -- it is the DECOMPOSITION TABLE, i.e. the ops that get decomposed AWAY on the
path to Core ATen. Measured: 1004 entries against 188 core-tagged overloads. Using it as a denominator
would report coverage of roughly a fifth of the true figure, against a set that is close to the
complement of the one meant.

The authority is ``torch.Tag.core``, which torch stamps on the overloads that survive into Core ATen.

Registered ``FunctionSchema`` objects define the universe: ``dir(torch.ops.aten)`` is
lazy and can omit overloads that have not yet been accessed. The catalog preserves
the historical ``torch``, ``n_core``, ``ops`` and decomposition keys for callers, plus the
``sha256`` of the sorted core overload names (one per line, final newline) that the Core ATen
corpus binds as its denominator.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

SCHEMA = "merlin.pytorch_opset.v1"


def _failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:500]}"


def core_opset() -> dict:
    """Return the versioned catalog; operator identities come from schema fields."""
    import torch

    # The DECOMPOSITION TABLE, for what it is actually for. It is the wrong denominator (it is roughly
    # the complement of Core ATen) and it is the right way to answer a different question: given a
    # `prov.aten` tag that is NOT core, is it a frontend COMPOSITE whose lowering is core, or an op
    # nothing knows? `aten.conv2d.default` is the first; that distinction is what separates "this model
    # uses composite frontend ops" from "this model contains work we cannot name".
    decomposed: set[str] = set()
    decomposition_status = {"status": "available", "source": "torch._decomp.core_aten_decompositions"}
    try:
        from torch._decomp import core_aten_decompositions

        decomposed = {str(k) for k in core_aten_decompositions()}
    except Exception as exc:  # noqa: BLE001 -- absent table is not an empty one
        decomposition_status.update(status="not_available", error=_failure(exc))

    # This internal PyTorch API is isolated here. Failure is explicit, rather than
    # falling back to a lazy namespace listing and reporting an incomplete total.
    schemas = torch._C._jit_get_all_schemas()
    registry: dict[tuple[str, str], object] = {}
    conflicts: list[dict] = []
    for schema in schemas:
        key = (schema.name, schema.overload_name)
        previous = registry.get(key)
        if previous is not None and str(previous) != str(schema):
            conflicts.append({"name": schema.name, "overload_name": schema.overload_name})
        registry[key] = schema

    core_tag = getattr(getattr(torch, "Tag", None), "core", None)
    ops: set[str] = set()
    rows: list[dict] = []
    tag_failures: list[dict] = []
    namespaces: Counter = Counter()
    packets: set[str] = set()
    for (qualified_name, overload_name), schema in sorted(registry.items()):
        namespace, separator, name = qualified_name.partition("::")
        if not separator:
            raise RuntimeError(f"registered schema lacks a namespace: {qualified_name!r}")
        namespaces[namespace] += 1
        if namespace != "aten":
            continue
        packets.add(qualified_name)
        operator = f"{namespace}.{name}.{overload_name or 'default'}"
        row = {
            "operator": operator,
            "name": qualified_name,
            "namespace": namespace,
            "overload_name": overload_name,
            "schema": str(schema),
            "tags": None,
            "core": None,
        }
        try:
            overload = getattr(getattr(torch.ops.aten, name), overload_name or "default")
            tags = overload.tags
            row["tags"] = sorted(str(tag) for tag in tags)
            if core_tag is not None:
                row["core"] = core_tag in tags
                if row["core"]:
                    ops.add(operator)
        except Exception as exc:  # noqa: BLE001 -- preserve the registered schema even without tags
            tag_failures.append({"operator": operator, "error": _failure(exc)})
        rows.append(row)

    core_status: dict = {"source": "torch.Tag.core", "status": "available", "count": len(ops)}
    if core_tag is None:
        core_status.update(status="not_available", count=None, error="torch.Tag.core is unavailable")
    elif tag_failures:
        core_status.update(status="partial", errors=tag_failures)
    decomposition_status["count"] = len(decomposed) if decomposition_status["status"] == "available" else None
    registered_status: dict = {
        "status": "available" if not conflicts else "partial",
        "source": "torch._C._jit_get_all_schemas; FunctionSchema.name and overload_name",
        "count": len(rows),
    }
    if conflicts:
        registered_status["conflicting_identities"] = conflicts
    sorted_ops = sorted(ops)
    digest = hashlib.sha256(("\n".join(sorted_ops) + "\n").encode()).hexdigest()
    return {
        "schema": SCHEMA,
        "status": "available" if not conflicts else "partial",
        "scope": "registered ATen overloads in the selected interpreter; not all Python APIs or custom namespaces",
        "torch": str(torch.__version__),
        "n_core": core_status["count"],
        "enumeration_source": "torch._C._jit_get_all_schemas filtered by torch.Tag.core",
        "ops": sorted_ops,
        "sha256": digest,
        "digest_encoding": "UTF-8 sorted overload names, one per line, with final newline",
        "n_decomposed": decomposition_status["count"],
        "decomposed": sorted(decomposed),
        "n_all_aten": len(rows),
        "n_aten_packets": len(packets),
        "all_ops": [row["operator"] for row in rows],
        "aten_schemas": rows,
        "registered_schema_count": len(registry),
        "registered_namespace_counts": dict(sorted(namespaces.items())),
        "components": {
            "registered_aten": registered_status,
            "core": core_status,
            "decompositions": decomposition_status,
        },
        "support_proven": False,
    }


def main(argv=None) -> int:
    try:
        print(json.dumps(core_opset()))
    except Exception as exc:  # noqa: BLE001 -- report, never a partial opset
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

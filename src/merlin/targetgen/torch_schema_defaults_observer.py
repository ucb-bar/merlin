"""Observe original public operator defaults through the selected native parser."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _literal(value):
    if value is None:
        return {"kind": "none"}
    if type(value) in {bool, int, str}:
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float and math.isfinite(value):
        return {"kind": "float", "value_hex": value.hex()}
    return {"kind": "unsupported"}


def observe(request):
    import torch

    if (
        set(request) != {"schema", "graph_sha256", "rows"}
        or request["schema"] != "merlin.original_schema_defaults_request.v1"
    ):
        raise ValueError("defaults require the complete original observed schema request")
    rows = []
    for original in request["rows"]:
        if set(original) != {"target", "schema"}:
            raise ValueError("defaults require exact public schema identities")
        try:
            parts = original["target"].split(".")
            if len(parts) != 3 or any(not part.isidentifier() for part in parts):
                raise ValueError("unsupported original operator identity")
            operation = getattr(getattr(getattr(torch.ops, parts[0]), parts[1]), parts[2])
            parsed = torch._C.parse_schema(original["schema"])
            native = operation._schema
            if str(operation) != original["target"] or str(parsed) != str(native) or str(native) != original["schema"]:
                raise ValueError("registered default schema differs from original public observation")
            defaults = [
                {
                    "ordinal": index,
                    "name": argument.name,
                    "has_default": argument.has_default_value(),
                    "default": _literal(argument.default_value) if argument.has_default_value() else None,
                }
                for index, argument in enumerate(native.arguments)
            ]
            rows.append({"request": original, "status": "observed", "defaults": defaults})
        except (ValueError, RuntimeError, AttributeError) as error:
            rows.append({"request": original, "status": "unknown", "reason": str(error)})
    return {
        "schema": "merlin.native_schema_defaults_observation.v1",
        "graph_sha256": request["graph_sha256"],
        "rows": rows,
        "runtime": {"torch_version": torch.__version__, "reported_git_version": torch.version.git_version},
        "scope": "actual original registered/public schema default values only; no numeric or operator-effect grant",
    }


if __name__ == "__main__":
    print(json.dumps(observe(json.loads(Path(sys.argv[1]).read_bytes())), sort_keys=True, allow_nan=False))

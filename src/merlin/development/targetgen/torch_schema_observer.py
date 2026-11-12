"""Fixed native observer of captured, registered and public source schemas.

Executed with an explicitly selected framework Python, never the compiler's
interpreter. Alias annotations describe possible source effects; no storage,
allocation, numerical or device equivalence is inferred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _argument(argument):
    alias = argument.alias_info
    return {
        "name": argument.name,
        "type": str(argument.type),
        "kwarg_only": argument.kwarg_only,
        "has_default": argument.has_default_value(),
        "alias": None
        if alias is None
        else {"before": sorted(alias.before_set), "after": sorted(alias.after_set), "write": alias.is_write},
    }


def observe(request, *, declarations):
    """Join canonical identities, never operator names interpreted as effects."""
    import torch
    import yaml

    if set(request) != {"namespace", "captured_schemas", "operations"}:
        raise ValueError("schema observation requires its closed original source selection")
    namespace = request["namespace"]
    if not isinstance(namespace, str) or not namespace.isidentifier():
        raise ValueError("schema namespace must be an explicit simple identifier")
    sources = yaml.safe_load(declarations)
    if not isinstance(sources, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("func"), str) for row in sources
    ):
        raise ValueError("canonical source must be a native function declaration list")
    parsed, ambiguous = {}, set()
    for row in sources:
        try:
            schema = torch._C.parse_schema(namespace + "::" + row["func"])
        except RuntimeError:
            continue
        identity = schema.name.replace("::", ".") + "." + (schema.overload_name or "default")
        if identity in parsed:
            ambiguous.add(identity)
        parsed[identity] = schema
    rows = []
    for target in request["operations"]:
        schema = parsed.get(target)
        parts = target.split(".")
        reason = None
        if len(parts) != 3 or any(not part.isidentifier() for part in parts):
            reason = "source target has no exact registered operator identity"
        elif schema is None or target in ambiguous:
            reason = "canonical public source has no unique parsed declaration"
        elif request["captured_schemas"].get(target) != str(schema):
            reason = "captured schema differs from the canonical public source declaration"
        else:
            try:
                operation = getattr(getattr(getattr(torch.ops, parts[0]), parts[1]), parts[2])
                registered = operation._schema
            except (AttributeError, RuntimeError):
                reason = "selected runtime has no registered canonical operator"
            else:
                if str(operation) != target or str(registered) != str(schema):
                    reason = "registered operator differs from captured and public source schemas"
        if reason:
            rows.append({"target": target, "status": "unknown", "reason": reason})
        else:
            rows.append(
                {
                    "target": target,
                    "status": "observed",
                    "schema": str(schema),
                    "arguments": [_argument(argument) for argument in schema.arguments],
                    "returns": [_argument(argument) for argument in schema.returns],
                }
            )
    return {
        "schema": "merlin.native_operator_schema_observation.v1",
        "rows": rows,
        "runtime": {
            "torch_version": torch.__version__,
            "reported_git_version": torch.version.git_version,
            "python": sys.executable,
            "torch_module": str(Path(torch.__file__).absolute()),
            "native_schema_parser": str(Path(torch._C.__file__).absolute()),
            "yaml_module": str(Path(yaml.__file__).absolute()),
        },
        "scope": "actual source/captured/registered schema equality and typed alias observations only",
    }


def main():
    request = json.loads(Path(sys.argv[1]).read_bytes())
    result = observe(request, declarations=Path(sys.argv[2]).read_bytes())
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()

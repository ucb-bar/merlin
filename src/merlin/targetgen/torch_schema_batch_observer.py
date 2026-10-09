"""Rerun complete original schema/default observations in one native process.

The fixed readers keep their existing observation schemas and unknowns. This
transport neither caches results nor executes numerical operations.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

REQUEST_SCHEMA = "merlin.original_schema_batch_request.v1"
OBSERVATION_SCHEMA = "merlin.native_original_schema_batch_observation.v1"


def _reader(name):
    path = Path(__file__).with_name(name + ".py")
    module = ModuleType(name)
    module.__file__ = str(path)
    # Source-pinned readers must not silently select an adjacent bytecode cache.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate(request):
    if (
        not isinstance(request, dict)
        or set(request) != {"schema", "rows"}
        or request["schema"] != REQUEST_SCHEMA
        or not isinstance(request["rows"], list)
        or not request["rows"]
    ):
        raise ValueError("batch observations need their complete closed original request")
    seen = set()
    for row in request["rows"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"identity", "schema_request", "defaults_request"}
            or not isinstance(row["identity"], str)
            or not row["identity"]
            or row["identity"] in seen
            or not isinstance(row["schema_request"], dict)
            or not isinstance(row["defaults_request"], dict)
        ):
            raise ValueError("batch observations require unique complete ordered original rows")
        seen.add(row["identity"])
    return request


def observe(request, *, declarations):
    validate(request)
    schemas = _reader("torch_schema_observer")
    defaults = _reader("torch_schema_defaults_observer")
    rows = [
        {
            "identity": row["identity"],
            "request_sha256": digest(row),
            "schema_observation": schemas.observe(row["schema_request"], declarations=declarations),
            "defaults_observation": defaults.observe(row["defaults_request"]),
        }
        for row in request["rows"]
    ]
    return {
        "schema": OBSERVATION_SCHEMA,
        "request_sha256": digest(request),
        "rows": rows,
        "scope": "fresh complete original schema/default observations only; no numeric or effect-domain grant",
    }


if __name__ == "__main__":
    result = observe(json.loads(Path(sys.argv[1]).read_bytes()), declarations=Path(sys.argv[2]).read_bytes())
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))

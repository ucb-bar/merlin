"""Private fixed batch transport for complete original schema/default observations.

Every native observation reruns. Reopening a diagnostic is not issuance of
software, numerical, effect or target authority.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.common.strict_json import loads
from merlin.targetgen import torch_schema_batch_observer as B
from merlin.targetgen.frontend_typed_add import defaults_request
from merlin.targetgen.frontend_use_def import original_use_def_semantics

from .operator_schema_intake import _selection

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
TRANSPORT = "batch.v1"


def readers():
    return [
        module_source_path(name)
        for name in (
            "merlin.targetgen.torch_schema_batch_observer",
            "merlin.targetgen.torch_schema_observer",
            "merlin.targetgen.torch_schema_defaults_observer",
            __name__,
        )
    ]


def _originals(schema_record):
    originals = {row["graph_path"]: row for row in schema_record["members"]}
    if len(originals) != len(schema_record["members"]):
        raise ValueError("batch schemas lost unique complete original graph membership")
    return originals


def _request(graph_paths, *, schema_record, version):
    originals = _originals(schema_record)
    selected = _selection(Path(schema_record["selection_path"]).read_bytes())
    if len(graph_paths) != len(originals) or set(graph_paths) != set(originals):
        raise ValueError("batch schemas lost their complete original graph roster")
    rows = []
    for path in graph_paths:
        trace = loads(Path(path).read_bytes())
        schemas = loads(Path(originals[path]["observation"]).read_bytes())
        relation = original_use_def_semantics(trace)
        rows.append(
            {
                "identity": path,
                "schema_request": {
                    "namespace": selected["namespace"],
                    "captured_schemas": trace["graphs"]["original"].get("operator_schemas", {}),
                    "operations": list(relation.operations),
                },
                "defaults_request": defaults_request(trace, schemas, version=version),
            }
        )
    return {"schema": B.REQUEST_SCHEMA, "rows": rows}


def _inputs(request, *, schema_record):
    selected = _selection(Path(schema_record["selection_path"]).read_bytes())
    originals = _originals(schema_record)
    return (
        *readers(),
        request,
        Path(schema_record["selection_path"]),
        Path(selected["canonical_source"]["path"]),
        *(Path(row["graph_path"]) for row in schema_record["members"]),
        *(Path(row["observation"]) for row in originals.values()),
    )


def _write(path, data):
    path.write_text(json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    path.chmod(0o600)


def observe_members(*, schema_record, basis, destination, version):
    destination = Path(destination).absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("batch observations require an ordinary fresh private destination")
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    request = destination / "batch-request.json"
    _write(
        request, _request([source.path for source in basis.graph_sources], schema_record=schema_record, version=version)
    )
    selected = _selection(Path(schema_record["selection_path"]).read_bytes())
    result = I.run(
        [selected["python"], "-I", str(readers()[0]), str(request), selected["canonical_source"]["path"]],
        directory=destination,
        stage="native_original_schema_batch",
        inputs=_inputs(request, schema_record=schema_record),
        env=ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    result.check_returncode()
    batch_observation = destination / "batch-observation.json"
    batch_observation.write_bytes(result.stdout)
    batch_observation.chmod(0o600)
    invocation = next((destination / "invocations").glob("*/invocation.json"))
    observed = loads(result.stdout)
    _check_observation(observed, loads(request.read_bytes()), schema_record=schema_record)
    rows = []
    for ordinal, row in enumerate(observed["rows"]):
        owner = destination / str(ordinal)
        owner.mkdir(mode=0o700)
        _write(owner / "request.json", loads(request.read_bytes())["rows"][ordinal]["defaults_request"])
        _write(owner / "observation.json", row["defaults_observation"])
        rows.append(
            {
                "graph_path": row["identity"],
                "request": str(owner / "request.json"),
                "observation": str(owner / "observation.json"),
                "invocation": str(invocation),
                "batch_request": str(request),
                "batch_observation": str(batch_observation),
                "transport": TRANSPORT,
            }
        )
    verify_members(rows, schema_record=schema_record, basis=basis, destination=destination, version=version)
    return rows


def _check_observation(observed, request, *, schema_record):
    B.validate(request)
    if (
        not isinstance(observed, dict)
        or set(observed) != {"schema", "request_sha256", "rows", "scope"}
        or observed["schema"] != B.OBSERVATION_SCHEMA
        or observed["request_sha256"] != B.digest(request)
        or not isinstance(observed["rows"], list)
        or len(observed["rows"]) != len(request["rows"])
        or observed["scope"]
        != "fresh complete original schema/default observations only; no numeric or effect-domain grant"
    ):
        raise ValueError("batch observation lost complete original request/output correspondence")
    originals = _originals(schema_record)
    for row, original in zip(observed["rows"], request["rows"], strict=True):
        expected_schema = loads(Path(originals[original["identity"]]["observation"]).read_bytes())
        if (
            not isinstance(row, dict)
            or set(row) != {"identity", "request_sha256", "schema_observation", "defaults_observation"}
            or row["identity"] != original["identity"]
            or row["request_sha256"] != B.digest(original)
            or row["schema_observation"] != expected_schema
        ):
            raise ValueError("batch row differs from exact ordered original native schema/request")
        defaults = row["defaults_observation"]
        if (
            not isinstance(defaults, dict)
            or defaults.get("graph_sha256") != original["defaults_request"]["graph_sha256"]
            or [item.get("request") for item in defaults.get("rows", [])] != original["defaults_request"]["rows"]
        ):
            raise ValueError("batch defaults lost their complete ordered original schema rows")


def verify_member(row, *, schema_record, version):
    if (
        set(row)
        != {"graph_path", "request", "observation", "invocation", "batch_request", "batch_observation", "transport"}
        or row["transport"] != TRANSPORT
    ):
        raise ValueError("batch defaults need their complete selected transport record")
    for key in ("request", "observation", "invocation", "batch_request", "batch_observation"):
        path = Path(row[key])
        if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
            raise ValueError("batch defaults require ordinary complete private products")
    request = Path(row["batch_request"])
    actual = I.require_environment(Path(row["invocation"]), environment=ENVIRONMENT)
    selected = _selection(Path(schema_record["selection_path"]).read_bytes())
    expected = [selected["python"], "-I", str(readers()[0]), str(request), selected["canonical_source"]["path"]]
    if (
        actual["argv"] != expected
        or actual["stage"] != "native_original_schema_batch"
        or {pin["path"] for pin in actual["inputs"]}
        != {str(path.resolve()) for path in _inputs(request, schema_record=schema_record)}
        or Path(row["invocation"]).parent.parent.parent != request.parent
        or Path(row["batch_observation"]) != request.parent / "batch-observation.json"
    ):
        raise ValueError("batch defaults changed their fixed native source/input/owner selection")
    requested = loads(request.read_bytes())
    identities = [item["identity"] for item in requested["rows"]]
    if requested != _request(identities, schema_record=schema_record, version=version):
        raise ValueError("batch request differs from complete original source/schema membership")
    observed_bytes = Path(row["batch_observation"]).read_bytes()
    if observed_bytes != Path(actual["stdout"]["path"]).read_bytes():
        raise ValueError("batch observations differ from actual fixed native output")
    observed = loads(observed_bytes)
    _check_observation(observed, requested, schema_record=schema_record)
    ordinal = identities.index(row["graph_path"])
    owner = request.parent / str(ordinal)
    if (
        Path(row["request"]) != owner / "request.json"
        or Path(row["observation"]) != owner / "observation.json"
        or loads(Path(row["request"]).read_bytes()) != requested["rows"][ordinal]["defaults_request"]
        or loads(Path(row["observation"]).read_bytes()) != observed["rows"][ordinal]["defaults_observation"]
    ):
        raise ValueError("batch defaults lost their exact full extracted output/owner")
    originals = _originals(schema_record)
    return (
        loads(Path(row["graph_path"]).read_bytes()),
        loads(Path(originals[row["graph_path"]]["observation"]).read_bytes()),
        loads(Path(row["observation"]).read_bytes()),
    )


def verify_members(rows, *, schema_record, basis, destination, version):
    if [row["graph_path"] for row in rows] != [source.path for source in basis.graph_sources]:
        raise ValueError("batch defaults lost complete ordered original source membership")
    for row in rows:
        if Path(row["batch_request"]) != Path(destination) / "batch-request.json":
            raise ValueError("batch defaults changed their selected private output owner")
        verify_member(row, schema_record=schema_record, version=version)

"""Fixed native default observations and replay shared by original source forms.

The selected operator intake owns original schemas. This reader observes their
native defaults without issuing software, numerical or hardware authority.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.frontend_typed_add import defaults_request

from .operator_schema_intake import _selection

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def observer():
    return module_source_path("merlin.targetgen.torch_schema_defaults_observer")


def observe_members(*, schema_record, basis, destination, version, transport="per_member"):
    """Retain actual fixed-reader invocations for the complete original roster."""
    if transport == "batch.v1":
        from .original_schema_batch import observe_members as batch

        return batch(schema_record=schema_record, basis=basis, destination=destination, version=version)
    if transport != "per_member":
        raise ValueError("original defaults need an explicitly supported observation transport")
    destination = Path(destination)
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("original default observations need ordinary explicit output paths")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    selection = _selection(Path(schema_record["selection_path"]).read_bytes())
    originals = {row["graph_path"]: row for row in schema_record["members"]}
    if set(originals) != {source.path for source in basis.graph_sources}:
        raise ValueError("original defaults lost exact protected graph membership")
    rows = []
    for ordinal, source in enumerate(basis.graph_sources):
        original = originals[source.path]
        trace = json.loads(Path(source.path).read_bytes())
        schemas = json.loads(Path(original["observation"]).read_bytes())
        owner = destination / str(ordinal)
        owner.mkdir(mode=0o700)
        request = owner / "request.json"
        request.write_text(json.dumps(defaults_request(trace, schemas, version=version), sort_keys=True) + "\n")
        request.chmod(0o600)
        result = I.run(
            [selection["python"], "-I", str(observer()), str(request)],
            directory=owner,
            stage="native_original_schema_defaults",
            inputs=(observer(), request, Path(source.path), Path(original["observation"])),
            env=ENVIRONMENT,
            capture_output=True,
            timeout=60,
        )
        result.check_returncode()
        observation = owner / "observation.json"
        observation.write_bytes(result.stdout)
        observation.chmod(0o600)
        invocation = next((owner / "invocations").glob("*/invocation.json"))
        row = {
            "graph_path": source.path,
            "request": str(request),
            "observation": str(observation),
            "invocation": str(invocation),
        }
        verify_member(row, schema_record=schema_record, version=version)
        rows.append(row)
    return rows


def verify_member(row, *, schema_record, version, transport="per_member"):
    """Reopen exact inputs, request, process environment and native output."""
    if transport == "batch.v1":
        from .original_schema_batch import verify_member as batch

        return batch(row, schema_record=schema_record, version=version)
    if transport != "per_member" or "transport" in row:
        raise ValueError("original defaults changed their explicitly selected observation transport")
    selection = _selection(Path(schema_record["selection_path"]).read_bytes())
    originals = {member["graph_path"]: member for member in schema_record["members"]}
    original = originals[row["graph_path"]]
    actual = I.require_environment(Path(row["invocation"]), environment=ENVIRONMENT)
    if (
        actual["argv"] != [selection["python"], "-I", str(observer()), row["request"]]
        or actual["stage"] != "native_original_schema_defaults"
    ):
        raise ValueError("original defaults lost their fixed native reader")
    inputs = {observer(), Path(row["request"]), Path(row["graph_path"]), Path(original["observation"])}
    if {pin["path"] for pin in actual["inputs"]} != {str(path.resolve()) for path in inputs}:
        raise ValueError("original defaults lost their complete original input membership")
    trace = json.loads(Path(row["graph_path"]).read_bytes())
    schemas = json.loads(Path(original["observation"]).read_bytes())
    if json.loads(Path(row["request"]).read_bytes()) != defaults_request(trace, schemas, version=version):
        raise ValueError("original default request differs from complete original schemas")
    observed = Path(row["observation"]).read_bytes()
    if observed != Path(actual["stdout"]["path"]).read_bytes():
        raise ValueError("original defaults differ from actual native output")
    return trace, schemas, json.loads(observed)


def verify_members(rows, *, schema_record, basis, destination, version, transport="per_member"):
    """Bind the selected transport and full original ordered graph denominator."""
    if transport == "batch.v1":
        from .original_schema_batch import verify_members as batch

        return batch(rows, schema_record=schema_record, basis=basis, destination=destination, version=version)
    if transport != "per_member" or [row["graph_path"] for row in rows] != [s.path for s in basis.graph_sources]:
        raise ValueError("original defaults lost their selected complete ordered source transport")
    for index, member in enumerate(rows):
        owner = Path(destination) / str(index)
        if (
            set(member) != {"graph_path", "request", "observation", "invocation"}
            or member["request"] != str(owner / "request.json")
            or member["observation"] != str(owner / "observation.json")
            or Path(member["invocation"]).parent.parent.parent != owner
        ):
            raise ValueError("original reference defaults lost their complete private owner")
        verify_member(member, schema_record=schema_record, version=version)

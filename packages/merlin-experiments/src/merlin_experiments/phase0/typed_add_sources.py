"""Actual public-schema defaults bound to original typed add source premises.

This observation extends an already issued original operator-schema intake;
it issues no software, numerical, effect or hardware capability.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.frontend_typed_add import add_forms, defaults_request

from .operator_schema_intake import _selection

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
SCHEMA = "merlin.original_typed_add_sources.v1"


def _observer():
    return module_source_path("merlin.targetgen.torch_schema_defaults_observer")


def observe(*, schema_record, basis, numerical_semantics, destination):
    """Run a fixed reader over every exact protected original schema member."""
    destination = Path(destination)
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("typed add observations need ordinary explicit output paths")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    selection = _selection(Path(schema_record["selection_path"]).read_bytes())
    members = {row["graph_path"]: row for row in schema_record["members"]}
    rows = []
    for ordinal, source in enumerate(basis.graph_sources):
        original = members[source.path]
        trace = json.loads(Path(source.path).read_bytes())
        schemas = json.loads(Path(original["observation"]).read_bytes())
        owner = destination / str(ordinal)
        owner.mkdir(mode=0o700)
        request = owner / "request.json"
        request.write_text(json.dumps(defaults_request(trace, schemas), sort_keys=True) + "\n")
        request.chmod(0o600)
        result = I.run(
            [selection["python"], "-I", str(_observer()), str(request)],
            directory=owner,
            stage="native_original_schema_defaults",
            inputs=(_observer(), request, Path(source.path), Path(original["observation"])),
            env=ENVIRONMENT,
            capture_output=True,
            timeout=60,
        )
        result.check_returncode()
        observation = owner / "observation.json"
        observation.write_bytes(result.stdout)
        observation.chmod(0o600)
        invocation = next((owner / "invocations").glob("*/invocation.json"))
        rows.append(
            {
                "graph_path": source.path,
                "request": str(request),
                "observation": str(observation),
                "invocation": str(invocation),
                "forms": add_forms(trace, schemas, json.loads(result.stdout), numerical_semantics=numerical_semantics),
            }
        )
    record = {"schema": SCHEMA, "members": rows}
    return verify(record, schema_record=schema_record, basis=basis, numerical_semantics=numerical_semantics)


def verify(record, *, schema_record, basis, numerical_semantics):
    """Recompute exact source forms from terminal native observations."""
    if not isinstance(record, dict) or set(record) != {"schema", "members"} or record["schema"] != SCHEMA:
        raise ValueError("typed add sources require their explicit closed observation version")
    selection = _selection(Path(schema_record["selection_path"]).read_bytes())
    originals = {row["graph_path"]: row for row in schema_record["members"]}
    if [row["graph_path"] for row in record["members"]] != [source.path for source in basis.graph_sources]:
        raise ValueError("typed add sources lost the complete original protected graph roster")
    for row in record["members"]:
        if set(row) != {"graph_path", "request", "observation", "invocation", "forms"}:
            raise ValueError("typed add source member observation fields changed")
        actual = I.require_environment(Path(row["invocation"]), environment=ENVIRONMENT)
        if (
            actual["argv"] != [selection["python"], "-I", str(_observer()), row["request"]]
            or actual["stage"] != "native_original_schema_defaults"
        ):
            raise ValueError("typed add defaults lost their fixed native reader")
        required_inputs = {
            _observer(),
            Path(row["request"]),
            Path(row["graph_path"]),
            Path(originals[row["graph_path"]]["observation"]),
        }
        if {pin["path"] for pin in actual["inputs"]} != {str(path.resolve()) for path in required_inputs}:
            raise ValueError("typed add defaults lost their actual complete original input membership")
        trace = json.loads(Path(row["graph_path"]).read_bytes())
        schemas = json.loads(Path(originals[row["graph_path"]]["observation"]).read_bytes())
        if json.loads(Path(row["request"]).read_bytes()) != defaults_request(trace, schemas):
            raise ValueError("typed add default request differs from complete original schemas")
        observed = Path(row["observation"]).read_bytes()
        if observed != Path(actual["stdout"]["path"]).read_bytes():
            raise ValueError("typed add defaults differ from actual native output")
        if row["forms"] != add_forms(trace, schemas, json.loads(observed), numerical_semantics=numerical_semantics):
            raise ValueError("typed add source premises differ from original typed SSA/default replay")
    return record


def forms(record, *, basis):
    return [
        (member["id"], row["forms"])
        for member, row in zip(json.loads(basis.declaration_json)["members"], record["members"], strict=True)
    ]

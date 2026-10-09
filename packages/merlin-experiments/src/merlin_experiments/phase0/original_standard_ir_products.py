"""Reopen every actual original-reference/upstream-IR product and invocation."""

import hashlib
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads

from . import original_reference_roster as R
from . import original_standard_ir_plan as P


def completed_process(path):
    """Retain a true completed nonzero refusal without calling it successful."""
    native = loads(R._plain(path).read_bytes())
    if native.get("returncode") == 0:
        return I.require_environment(path, environment=R.D.ENVIRONMENT)
    if (
        native.get("schema") != I.SCHEMA
        or native.get("kind") != "subprocess"
        or native.get("status") != "failed"
        or type(native.get("returncode")) is not int
        or native.get("environment") != I.environment_identity(R.D.ENVIRONMENT)
        or any(
            native.get(key) is not True
            for key in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")
        )
    ):
        raise ValueError("standard IR refusal lacks an actual unchanged completed native process")
    for pin in [
        native["executable"],
        native["stdout"],
        native["stderr"],
        *native["inputs"],
        *native["outputs"],
        *native["dependencies"],
    ]:
        if pin.get("sha256") is None and pin in native["outputs"] and not Path(pin["path"]).exists():
            continue
        if R._pin(pin["path"]) != pin:
            raise ValueError("standard IR refusal native source/product changed")
    return native


def parse_invocation(owner, paths, selected):
    candidates = tuple((owner / "parse/invocations").glob("*/invocation.json"))
    if len(candidates) != 1:
        raise ValueError("standard IR requires exactly one actual stock parsing invocation")
    invocation = R._pin(candidates[0])
    native = completed_process(candidates[0])
    if (
        native["argv"] != [selected["mlir_opt"], str(paths["source"]), "--verify-each", "-o", str(paths["verified"])]
        or native["stage"] != "original_standard_ir_native_parse"
        or native["inputs"] != [R._pin(paths["source"])]
        or native["outputs"] != ([R._pin(paths["verified"])] if paths["verified"].is_file() else [])
        or native["dependencies"] != [R._pin(selected["mlir_opt"])]
        or native["executable"]["sha256"] != R._pin(selected["mlir_opt"])["sha256"]
    ):
        raise ValueError("stock parsing invocation lost its exact complete source/tool/output join")
    if native["returncode"] == 0 and not paths["verified"].is_file():
        raise ValueError("successful stock parser omitted its actual complete output")
    return invocation, native["returncode"]


def verify(document, *, references, selection):
    from merlin.common.paths import module_source_path

    from . import original_reference_standard_ir as S
    from .operator_schema_intake import _selection

    record = P.required_members(references)
    selected = P.validate(loads(R._plain(selection).read_bytes()), references)
    capture = P.capture_sources(selected)
    pins = S._pin_sources(selection, capture)
    members, decisions, totals = P.preflight(record, selected)
    destination = Path(document["destination"])
    if (
        not destination.is_absolute()
        or not destination.is_dir()
        or any(path.is_symlink() for path in (destination, *destination.parents))
        or set(document)
        != {
            "schema",
            "scope",
            "reference_roster_sha256",
            "selection",
            "destination",
            "source_pins",
            "request",
            "invocation",
            "totals",
            "members",
            *({"observation"} if members else set()),
        }
        or document["schema"] != S.schemas(references)[0]
        or document["scope"] != S._SCOPE
        or document["reference_roster_sha256"] != references.sha256
        or document["selection"] != R._pin(selection)
        or document["source_pins"] != pins
        or canonical_json(document["totals"]) != canonical_json(totals)
        or len(document["members"]) != len(record["members"])
    ):
        raise ValueError("standard IR lost its exact source selection, complete roster or original scope")
    request = destination / "request.json"
    expected_request = {
        "schema": S.schemas(references)[1],
        "capture_sources": capture,
        "members": members,
        "budget": selected["budget"],
    }
    if document["request"] != R._pin(request) or canonical_json(loads(request.read_bytes())) != canonical_json(
        expected_request
    ):
        raise ValueError("standard IR request changed complete original input/source membership")
    returncode = None
    if members:
        schema = references.schema_intake.record()
        python = _selection(Path(schema["selection_path"]).read_bytes())["python"]
        observer = module_source_path(S._READERS[-1])
        reference_observer = R._observer(loads(references.selection.read_bytes()))
        invocation = document["invocation"]
        path = R._plain(invocation["path"])
        if invocation != R._pin(path) or path.parent.parent.parent != destination / "native":
            raise ValueError("standard IR native invocation changed its original private owner")
        inputs = [observer, request, reference_observer]
        outputs = [destination / "observation.json"]
        for member in members:
            inputs.extend(Path(member[key]) for key in ("source", "metadata", "inputs"))
            outputs.extend(
                path for key, path in S._paths(destination / str(member["index"])).items() if key != "verified"
            )
        argv = [
            python,
            "-I",
            "-B",
            str(observer),
            str(request),
            selected["capture_checkout"],
            str(reference_observer),
            str(destination),
        ]
        native = completed_process(path)
        if (
            native["argv"] != argv
            or {pin["path"] for pin in native["inputs"]} != {str(Path(path).resolve()) for path in inputs}
            or native["stage"] != "original_source_upstream_standard_ir"
            or native["outputs"] != [R._pin(path) for path in sorted(outputs) if path.is_file()]
        ):
            raise ValueError("standard IR native process changed its full produced product roster")
        expected_dependencies = {str(Path(pin["path"]).resolve()) for pin in pins} | {str(reference_observer.resolve())}
        if {pin["path"] for pin in native["dependencies"]} != expected_dependencies:
            raise ValueError("standard IR native process changed its exact source dependency membership")
        returncode = native["returncode"]
        observation = destination / "observation.json"
        if observation.stat().st_size > selected["budget"]["max_observation_bytes"]:
            raise ValueError("standard IR native frame exceeds its explicit predecode limit")
        if document["observation"] != R._pin(observation):
            raise ValueError("standard IR native frame changed")
        observed = loads(observation.read_bytes())
        if (
            set(observed) != {"schema", "rows", "dependencies"}
            or observed["schema"] != S.schemas(references)[2]
            or canonical_json([row.get("index") for row in observed["rows"]])
            != canonical_json([row["index"] for row in members])
            or any(
                row.get("status") not in {"converted", "unavailable"}
                or set(row) != ({"index", "status"} if row["status"] == "converted" else {"index", "status", "reason"})
                for row in observed["rows"]
            )
        ):
            raise ValueError("standard IR native rows are missing, reordered or duplicated")
        for row in observed["rows"]:
            paths = S._paths(destination / str(row["index"]))
            complete = all(paths[key].is_file() for key in ("source", "trace", "actual"))
            if complete != (row["status"] == "converted"):
                raise ValueError("standard IR native row status disagrees with actual complete products")
        for pin in observed["dependencies"]:
            if set(pin) != {"module", "path", "sha256"} or R._pin(pin["path"])["sha256"] != pin["sha256"]:
                raise ValueError("standard IR native observed runtime source changed")
    elif document["invocation"] is not None:
        raise ValueError("unavailable standard IR members cannot manufacture a native invocation")
    for original, decision, actual in zip(record["members"], decisions, document["members"], strict=True):
        expected = {
            "original": {
                key: original[key] for key in ("original_member_id", "graph_path", "node", "target", "cohort", "extent")
            },
            "reference_member_sha256": hashlib.sha256(R._json(original)).hexdigest(),
            "decision": decision,
            "state": "unavailable",
            "required_unknowns": [*original["required_unknowns"], *S._UNKNOWN],
        }
        if decision["state"] == "planned":
            expected.update(
                S._evaluate(
                    original,
                    selected,
                    destination / str(decision["index"]),
                    returncode,
                    references=references,
                    run_parse=False,
                )
            )
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("standard IR original ABI, full comparison or required unknowns differ from actual replay")
    return document

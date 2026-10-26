"""Strict live replay of conditional elision evidence, never saved JSON authority.

The Lean certificate checker is Elision.check with check_sound. This host protocol
only binds real inputs, code, reports and proof objects, and reruns that checker.
It neither proves Python correspondence nor approves an operation interpretation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from merlin.common.owned_process import run_owned

from . import lean_graph_state as stable
from . import lean_graph_state_elision as checker
from .lean_graph_state_elision_kernel import prepare

SCHEMA = "merlin.graph-state-elision.live-receipt.v1"
KEYS = {
    "schema",
    "version",
    "mode",
    "root",
    "originals",
    "contract",
    "snapshots",
    "report",
    "files",
    "dependencies",
    "claims",
}
FLAGS = (
    "public_acceptance_issued",
    "admission_approved",
    "native_fidelity_proved",
    "source_success_proved",
    "poison_freedom_proved",
    "numerical_gates_discharged",
    "full_model_claim",
)
ROOTS = {
    "Merlin.GraphState.Elision.check_sound",
    "Merlin.GraphState.Elision.context_preserved",
    "Merlin.GraphStateElisionInstance.accepted",
    "Merlin.GraphStateElisionInstance.actual_elision",
}


def closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("exact closed versioned receipt fields required")
    return value


def record(row):
    closed(row, {"path", "sha256", "bytes"})
    if type(row["path"]) is not str or not Path(row["path"]).is_absolute():
        raise ValueError("absolute exact file identity required")
    if type(row["bytes"]) is not int or row["bytes"] < 0:
        raise ValueError("exact integer file size required")
    if (
        type(row["sha256"]) is not str
        or len(row["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in row["sha256"])
    ):
        raise ValueError("exact SHA256 required")
    return row


def inventory(rows):
    if type(rows) is not list or not rows:
        raise ValueError("complete nonempty file inventory required")
    for row in rows:
        record(row)
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("duplicate inventory identity")


def claims(value):
    closed(value, {"status", *FLAGS})
    if value["status"] != "conditional_model_checked" or any(value[key] is not False for key in FLAGS):
        raise ValueError("conditional evidence cannot manufacture approval/success/native/full-model claims")


def audit(report):
    if report["status"] != "conditional_model_checked" or report["returncode"] != 0:
        raise ValueError("exact measured successful kernel outcome required")
    value = report["audit"]
    if value["status"] != "checked" or set(value["statements"]) != ROOTS:
        raise ValueError("exact full proof statement roster required")
    raw = value["raw"]
    if (
        raw["assurance"] != "lean_kernel_std3_no_native_reflection"
        or raw["violations"] != []
        or type(raw["theorems_audited"]) is not int
        or raw["theorems_audited"] <= 0
        or not set(raw["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    ):
        raise ValueError("clean real transitive standard-three audit required")
    actual = {row["path"]: row for row in value["imported_objects"]}
    if {row["olean"] for row in raw["imports"]} != set(actual):
        raise ValueError("exact complete imported object inventory required")
    return value


def create(root, destination):
    root = Path(root).resolve()
    report = json.loads((root / "report.json").read_text())
    kernel = json.loads((root / "kernel/report.json").read_text())
    audit(kernel)
    if report["status"] != "conditional_model_checked":
        raise ValueError("only measured conditional boundary evidence can be retained")
    value = {
        "schema": SCHEMA,
        "version": 1,
        "mode": "whole-call-elision-partial-full-state",
        "root": str(root),
        "originals": report["originals"],
        "contract": report["contract"],
        "snapshots": report["snapshots"],
        "report": stable.file_ref(root / "report.json"),
        "files": [stable.file_ref(path) for path in sorted(root.rglob("*")) if path.is_file()],
        "dependencies": sorted(
            {row["path"]: row for row in report["dependencies"]}.values(), key=lambda row: row["path"]
        ),
        "claims": {"status": "conditional_model_checked", **dict.fromkeys(FLAGS, False)},
    }
    validate(value)
    path = Path(destination)
    if path.exists() or path.resolve().is_relative_to(root):
        raise ValueError("receipt must be a new file outside its immutable evidence inventory")
    stable._write(path, value)
    return value


def validate(value):
    closed(value, KEYS)
    if value["schema"] != SCHEMA or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported exact receipt schema/version")
    if value["mode"] != "whole-call-elision-partial-full-state":
        raise ValueError("unsupported evidence mode")
    if type(value["root"]) is not str or not Path(value["root"]).is_absolute():
        raise ValueError("exact original evidence root required")
    for name in ("originals", "snapshots", "files", "dependencies"):
        inventory(value[name])
    if len(value["originals"]) != 2 or len(value["snapshots"]) != 3:
        raise ValueError("complete actual source/result/decoder binding required")
    record(value["contract"])
    record(value["report"])
    claims(value["claims"])
    return value


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate JSON receipt key")
        result[key] = value
    return result


def _noninteger(_):
    raise ValueError("receipt floating/nonfinite values refused")


def load(path):
    path = Path(path)
    data = path.read_bytes()
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("receipt explicit byte budget exceeded")
    return validate(json.loads(data, object_pairs_hook=_pairs, parse_float=_noninteger, parse_constant=_noninteger))


def replay(path, destination, *, originals, contract, original_contract_sha256, reference_root):
    value = load(path)
    root = Path(value["root"])
    if value["originals"] != originals or value["contract"] != stable.file_ref(contract):
        raise ValueError("receipt differs from independently selected original source/result/contract")
    if value["contract"]["sha256"] != original_contract_sha256:
        raise ValueError("original independent decoder selection differs")
    normative = [stable.file_ref(item) for item in sorted(root.rglob("*")) if item.is_file()]
    if normative != value["files"] or not stable.fresh(
        [*value["dependencies"], *originals, value["contract"], *value["snapshots"], value["report"]]
    ):
        raise ValueError("stale or incomplete actual evidence/dependency inventory")
    saved = json.loads((root / "report.json").read_text())
    if (
        saved["originals"] != originals
        or saved["snapshots"] != value["snapshots"]
        or saved["contract"] != value["contract"]
    ):
        raise ValueError("receipt/report/original association differs")
    expected_deps = sorted({row["path"]: row for row in saved["dependencies"]}.values(), key=lambda row: row["path"])
    if value["dependencies"] != expected_deps or saved["status"] != "conditional_model_checked":
        raise ValueError("exact measured dependency/report association differs")
    for flag in FLAGS:
        if flag in saved and saved[flag] is not False:
            raise ValueError("forged report approval flag")
    for original, snapshot in zip([*originals, value["contract"]], value["snapshots"], strict=True):
        if Path(original["path"]).read_bytes() != Path(snapshot["path"]).read_bytes():
            raise ValueError("original and checked snapshot bytes differ")
    old = json.loads((root / "kernel/report.json").read_text())
    oldaudit = audit(old)
    if (
        old["query"] != stable.file_ref(root / "kernel/GraphStateElisionInstance.lean")
        or old["profile"] != prepare()["manifest"]
    ):
        raise ValueError("saved query/profile association differs")
    if oldaudit["raw"] != json.loads((root / "kernel/audit.stdout").read_text()):
        raise ValueError("saved raw audit and retained actual stdout differ")
    if oldaudit["statements"] != json.loads((root / "kernel/statements.stdout").read_text())["statements"]:
        raise ValueError("saved statement association and actual retained stdout differ")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    fresh = checker.check_boundary(
        originals[0]["path"],
        originals[1]["path"],
        destination / "fresh",
        reference_root=reference_root,
        contract=contract,
        original_contract_sha256=original_contract_sha256,
        expected_originals=originals,
    )
    if fresh["status"] != "conditional_model_checked":
        raise ValueError("fresh original-bound kernel checking did not establish conditional correspondence")

    def external_dependencies(rows, folder):
        main = str(folder / "kernel/GraphStateElisionInstance.olean")
        return {row["path"]: row for row in rows if row["path"] != main}

    if external_dependencies(value["dependencies"], root) != external_dependencies(
        fresh["dependencies"], destination / "fresh"
    ):
        raise ValueError("saved complete dependency closure differs from live actual checker closure")
    fresh_projection = json.loads((destination / "fresh/projection/projection.json").read_text())
    source = root / "kernel/GraphStateElisionInstance.lean"
    if source.read_text() != checker.emit_query(fresh_projection):
        raise ValueError("saved proof does not prove this exact actual complete original/result projection")
    output = destination / "reproduced.olean"
    profile = prepare()
    with (destination / "reproduce.stdout").open("x") as out, (destination / "reproduce.stderr").open("x") as err:
        process = run_owned(
            [str(stable.LEAN), str(source), "-o", str(output)],
            cwd=source.parent,
            env={**os.environ, "LEAN_PATH": str(profile["lib"])},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    stable._write(destination / "reproduce.json", {"returncode": process.returncode})
    if process.returncode or output.read_bytes() != source.with_suffix(".olean").read_bytes():
        raise ValueError("saved actual proof object differs from exact original query reproduction")
    freshkernel = json.loads((destination / "fresh/kernel/report.json").read_text())
    newaudit = audit(freshkernel)
    if oldaudit["statements"] != newaudit["statements"]:
        raise ValueError("saved proof statement association differs from fresh actual theorem")
    for key in ("assurance", "theorems_audited", "axioms_used", "violations"):
        if oldaudit["raw"][key] != newaudit["raw"][key]:
            raise ValueError("saved raw proof audit differs from live actual proof")

    def imports(a):
        bypath = {row["path"]: row for row in a["imported_objects"]}
        return {
            row["module"]: (bypath[row["olean"]]["sha256"], bypath[row["olean"]]["bytes"])
            for row in a["raw"]["imports"]
            if row["module"] != "GraphStateElisionInstance"
        }

    if imports(oldaudit) != imports(newaudit):
        raise ValueError("saved proof import closure differs from fresh actual imports")
    if not stable.fresh([*value["files"], *value["dependencies"], *originals, value["contract"]]):
        raise ValueError("original evidence changed during live replay")
    result = {
        "schema": "merlin.graph-state-elision.live-replay.v1",
        "status": "conditional_model_checked",
        "receipt": stable.file_ref(path),
        "fresh_report": stable.file_ref(destination / "fresh/report.json"),
        "exact_old_object_reproduced": True,
        "claims": value["claims"],
        "host_protocol_proved_in_Lean": False,
    }
    stable._write(destination / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("receipt", "destination", "before", "after", "contract", "contract-sha256", "reference"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = replay(
        args.receipt,
        args.destination,
        originals=[stable.file_ref(args.before), stable.file_ref(args.after)],
        contract=args.contract,
        original_contract_sha256=args.contract_sha256,
        reference_root=args.reference,
    )
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()

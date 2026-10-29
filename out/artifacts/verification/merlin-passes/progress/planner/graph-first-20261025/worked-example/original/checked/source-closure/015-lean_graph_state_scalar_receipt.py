"""Closed untrusted scalar evidence transport with live original-bound replay.

Saved JSON grants no authority. Live replay reprojects complete originals,
checks the exact current theorem, reproduces the retained object from its exact
query path, and compares real transitive roots/imports/audits. Host projection,
transport and primitive/runtime interpretation remain explicit trust boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from . import lean_graph_state as stable
from . import lean_graph_state_scalar as checker
from .lean_graph_state_scalar_chain import emit_chain_query
from .lean_graph_state_scalar_kernel import invoke, prepare

SCHEMA = "merlin.graph-state-scalar.live-receipt.v1"
FLAGS = (
    "public_acceptance_issued",
    "review_approved",
    "native_fidelity_proved",
    "primitive_fidelity_proved",
    "source_success_proved",
    "poison_freedom_proved",
    "numerical_gates_discharged",
    "full_model_claim",
)
KEYS = {
    "schema",
    "version",
    "mode",
    "root",
    "originals",
    "snapshots",
    "report",
    "profile",
    "tool",
    "files",
    "dependencies",
    "claims",
}


def closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("exact closed versioned scalar receipt fields required")


def record(value):
    closed(value, {"path", "sha256", "bytes"})
    if type(value["path"]) is not str or not Path(value["path"]).is_absolute() or "\0" in value["path"]:
        raise ValueError("exact absolute file identity required")
    digest = value["sha256"]
    if type(digest) is not str or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("exact SHA256 identity required")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("exact nonnegative integer file size required")


def inventory(rows):
    if type(rows) is not list or not rows:
        raise ValueError("complete nonempty inventory required")
    for row in rows:
        record(row)
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("duplicate inventory file identity refused")


def validate(value):
    closed(value, KEYS)
    if value["schema"] != SCHEMA or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported exact scalar receipt version/schema")
    count = {"boundary": 2, "actual-two-stage-chain": 3}.get(value["mode"])
    if count is None or type(value["root"]) is not str or not Path(value["root"]).is_absolute():
        raise ValueError("exact supported mode and original evidence root required")
    for name in ("originals", "snapshots", "files", "dependencies"):
        inventory(value[name])
    if len(value["originals"]) != count or len(value["snapshots"]) != count:
        raise ValueError("complete original/snapshot endpoint and exact-middle roster required")
    for name in ("report", "profile", "tool"):
        record(value[name])
    closed(value["claims"], {"status", *FLAGS})
    if value["claims"]["status"] != "conditional_graph_checked" or any(
        value["claims"][flag] is not False for flag in FLAGS
    ):
        raise ValueError("receipt cannot manufacture approval/native/definedness/numerical/full-model claims")
    return value


def _pairs(rows):
    value = {}
    for key, item in rows:
        if key in value:
            raise ValueError("duplicate JSON key refused")
        value[key] = item
    return value


def _noninteger(_):
    raise ValueError("receipt floating/nonfinite fields refused")


def load(path):
    path = Path(path).resolve()
    first = path.stat()
    if not stat.S_ISREG(first.st_mode) or first.st_size > 64 * 1024 * 1024:
        raise ValueError("receipt needs a bounded regular input buffer")
    data = path.read_bytes()
    final = path.stat()
    if (first.st_mtime_ns, first.st_size) != (final.st_mtime_ns, final.st_size) or len(data) != final.st_size:
        raise ValueError("receipt changed while reading its parsed/hashed buffer")
    value = validate(json.loads(data, object_pairs_hook=_pairs, parse_float=_noninteger, parse_constant=_noninteger))
    return value, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def unique(rows):
    result = {}
    for row in rows:
        record(row)
        if row["path"] in result and result[row["path"]] != row:
            raise ValueError("contradictory dependency identities")
        result[row["path"]] = row
    return sorted(result.values(), key=lambda row: row["path"])


def measured(root):
    """Check complete associations; this is transport validation, not authority."""
    report = json.loads((root / "report.json").read_text())
    kernel = json.loads((root / "kernel/report.json").read_text())
    if (
        report["status"] != "conditional_graph_checked"
        or kernel["status"] != "conditional_graph_checked"
        or kernel["returncode"] != 0
    ):
        raise ValueError("exact measured successful graph/kernel status required")
    for row in (report, kernel):
        if any(flag in row and row[flag] is not False for flag in FLAGS):
            raise ValueError("forged elevated report claim")
    projection = json.loads((root / "projection/projection.json").read_text())
    if report["projection"] != stable.file_ref(root / "projection/projection.json") or report[
        "kernel"
    ] != stable.file_ref(root / "kernel/report.json"):
        raise ValueError("exact retained projection/kernel report association differs")
    if (
        type(report["genuine_changed_boundary"]) is not bool
        or report["genuine_changed_boundary"] != projection["genuine_changed_boundary"]
    ):
        raise ValueError("changed versus identity boundary association differs")
    if report["stage_changed"] != projection.get("stage_changed"):
        raise ValueError("changed stage labels differ from actual component projection")
    query, roots = (
        emit_chain_query(projection) if report["mode"] == "actual-two-stage-chain" else checker.emit_query(projection)
    )
    if (root / "kernel/GraphStateScalarInstance.lean").read_text() != query or report["exact_roots"] != roots:
        raise ValueError("saved actual projection/query/root association differs")
    if (
        kernel["query"] != stable.file_ref(root / "kernel/GraphStateScalarInstance.lean")
        or kernel["profile"] != prepare()["manifest"]
    ):
        raise ValueError("saved query/profile association differs from selected context")
    audit = kernel["audit"]
    if audit["status"] != "checked" or set(audit["statements"]) != set(roots):
        raise ValueError("complete actual transitive theorem/type roster required")
    raw = audit["raw"]
    if (
        raw["assurance"] != "lean_kernel_std3_no_native_reflection"
        or raw["violations"] != []
        or type(raw["theorems_audited"]) is not int
        or raw["theorems_audited"] <= 0
        or type(raw["axioms_used"]) is not list
        or not set(raw["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    ):
        raise ValueError("clean exact standard-three raw audit required")
    if (
        raw != json.loads((root / "kernel/audit.stdout").read_text())
        or audit["statements"] != json.loads((root / "kernel/statements.stdout").read_text())["statements"]
    ):
        raise ValueError("reported audit/types differ from retained actual buffers")
    imported = {row["path"]: row for row in audit["imported_objects"]}
    if len(imported) != len(audit["imported_objects"]) or {row["olean"] for row in raw["imports"]} != set(imported):
        raise ValueError("complete imported module/object association required")
    if len({row["module"] for row in raw["imports"]}) != len(raw["imports"]):
        raise ValueError("duplicate imported module identity")
    main = next((row for row in raw["imports"] if row["module"] == "GraphStateScalarInstance"), None)
    expected_main = stable.file_ref(root / "kernel/GraphStateScalarInstance.olean")
    if main is None or main["olean"] != expected_main["path"] or imported.get(main["olean"]) != expected_main:
        raise ValueError("actual instance object missing from complete proof import inventory")
    for pair in report["source_closure"]:
        if (
            not stable.fresh([pair["original"], pair["retained"]])
            or Path(pair["original"]["path"]).read_bytes() != Path(pair["retained"]["path"]).read_bytes()
        ):
            raise ValueError("checked original/retained source closure differs")
    return report, audit


def create(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    report, _ = measured(root)
    value = {
        "schema": SCHEMA,
        "version": 1,
        "mode": report["mode"],
        "root": str(root),
        "originals": report["originals"],
        "snapshots": report["snapshots"],
        "report": stable.file_ref(root / "report.json"),
        "profile": prepare()["manifest"],
        "tool": prepare()["lean"],
        "files": [stable.file_ref(path) for path in sorted(root.rglob("*")) if path.is_file()],
        "dependencies": unique(report["dependencies"]),
        "claims": {"status": "conditional_graph_checked", **dict.fromkeys(FLAGS, False)},
    }
    validate(value)
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("receipt must be a new transport outside its immutable evidence inventory")
    stable._write(destination, value)
    return value


def replay(path, destination, *, originals, reference_root):
    value, receipt_buffer = load(path)
    root = Path(value["root"])
    if value["originals"] != originals:
        raise ValueError("receipt differs from independent original source/middle/result selection")
    profile = prepare()
    if value["profile"] != profile["manifest"] or value["tool"] != profile["lean"]:
        raise ValueError("selected actual proof/tool profile differs")
    files = [stable.file_ref(item) for item in sorted(root.rglob("*")) if item.is_file()]
    if files != value["files"] or not stable.fresh(
        [*value["dependencies"], *originals, *value["snapshots"], value["report"]]
    ):
        raise ValueError("stale/incomplete actual evidence, source, tool or proof identity")
    report, oldaudit = measured(root)
    if (
        report["originals"] != originals
        or report["snapshots"] != value["snapshots"]
        or report["mode"] != value["mode"]
        or unique(report["dependencies"]) != value["dependencies"]
    ):
        raise ValueError("receipt/report/original/context dependency association differs")
    for original, snapshot in zip(originals, value["snapshots"], strict=True):
        if Path(original["path"]).read_bytes() != Path(snapshot["path"]).read_bytes():
            raise ValueError("actual original and checked raw snapshot bytes differ")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    selected = [row["path"] for row in originals]
    if value["mode"] == "boundary":
        fresh = checker.check_boundary(
            *selected, destination / "fresh", reference_root=reference_root, expected_originals=originals
        )
    else:
        fresh = checker.check_chain(
            *selected, destination / "fresh", reference_root=reference_root, expected_originals=originals
        )
    if fresh["status"] != "conditional_graph_checked":
        raise ValueError("fresh original-bound actual graph proof unavailable/refused")
    newreport, newaudit = measured(destination / "fresh")
    for field in (
        "mode",
        "genuine_changed_boundary",
        "stage_changed",
        "proof_routes",
        "exact_roots",
        "relation",
        "assumptions",
    ):
        if report[field] != newreport[field]:
            raise ValueError("saved graph claim/mode/assumption association differs from fresh actual result")
    source = root / "kernel/GraphStateScalarInstance.lean"
    if source.read_bytes() != (destination / "fresh/kernel/GraphStateScalarInstance.lean").read_bytes():
        raise ValueError("saved query differs from fresh complete original-bound theorem")

    def dependencies(rows, folder):
        main = str(folder / "kernel/GraphStateScalarInstance.olean")
        return unique([row for row in rows if row["path"] != main])

    if dependencies(value["dependencies"], root) != dependencies(newreport["dependencies"], destination / "fresh"):
        raise ValueError("saved complete dependency closure differs from live proof context")
    output = destination / "reproduced.olean"
    with (destination / "reproduce.stdout").open("x") as out, (destination / "reproduce.stderr").open("x") as err:
        result = invoke(
            [str(source), "-o", str(output)], source.parent, {**os.environ, "LEAN_PATH": str(profile["lib"])}, out, err
        )
    stable._write(
        destination / "reproduce.json", {"returncode": result.returncode, "original_query": stable.file_ref(source)}
    )
    if result.returncode or output.read_bytes() != source.with_suffix(".olean").read_bytes():
        raise ValueError("retained actual proof object differs from exact original-query reproduction")
    if oldaudit["statements"] != newaudit["statements"]:
        raise ValueError("retained proof theorem statements differ from fresh actual theorem")
    for key in ("assurance", "theorems_audited", "axioms_used", "violations"):
        if oldaudit["raw"][key] != newaudit["raw"][key]:
            raise ValueError("retained raw proof audit differs from fresh actual theorem")

    def imports(audit):
        bypath = {row["path"]: row for row in audit["imported_objects"]}
        return {
            row["module"]: (bypath[row["olean"]]["sha256"], bypath[row["olean"]]["bytes"])
            for row in audit["raw"]["imports"]
            if row["module"] != "GraphStateScalarInstance"
        }

    if imports(oldaudit) != imports(newaudit):
        raise ValueError("retained actual imported proof closure differs")
    if not stable.fresh([*value["files"], *value["dependencies"], *originals, receipt_buffer]):
        raise ValueError("actual retained context changed during live replay")
    result = {
        "schema": "merlin.graph-state-scalar.live-replay.v1",
        "status": "conditional_graph_checked",
        "receipt": receipt_buffer,
        "fresh_report": stable.file_ref(destination / "fresh/report.json"),
        "retained_object_reproduced": True,
        "host_transport_proved_in_Lean": False,
        "claims": value["claims"],
    }
    stable._write(destination / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--original", action="append", required=True)
    args = parser.parse_args()
    value = replay(
        args.receipt,
        args.destination,
        originals=[stable.file_ref(path) for path in args.original],
        reference_root=args.reference,
    )
    print(json.dumps({"status": value["status"]}))


if __name__ == "__main__":
    main()

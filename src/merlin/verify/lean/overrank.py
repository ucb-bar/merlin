"""Standalone, replayable verification of the invalid-overrank matmul intent repair.

The source is exported structurally with xDSL because upstream MLIR rejects it. The complete
output is exported and round-tripped through upstream MLIR. No literal source equivalence,
human intent approval, or whole-model execution is inferred from checker acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict
from functools import cache
from pathlib import Path

from merlin.common.paths import build_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.export import ExportError, export_file, export_text, round_trip_exported

MODULE = "MerlinVerify.Driver.Overrank"
THEOREMS = [
    "MerlinVerify.Check.Overrank.check_sound",
    "MerlinVerify.Check.Overrank.site_intent",
    "MerlinVerify.Check.Overrank.kernelCheck_sound",
    "MerlinVerify.Rules.OverrankOrder.trace_cell",
    "MerlinVerify.Rules.OverrankTensor.generic_sound",
    "MerlinVerify.Rules.OverrankDefined.successful_clean",
    "MerlinVerify.Rules.OverrankDefined.intent_clean",
    "MerlinVerify.Rules.OverrankIntent.domain_inhabited",
    "MerlinVerify.Rules.OverrankKernel.operations_sound",
    "MerlinVerify.Rules.OverrankKernel.program_defined",
    "MerlinVerify.Rules.OverrankIntake.i1_literal",
]


def run(command, *, timeout=600):
    env = dict(os.environ)
    env["PATH"] = str(Path.home() / ".elan/bin") + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        command, cwd=repo_root() / "merlin/lean", env=env, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}: {result.stdout[-4000:]} {result.stderr[-2000:]}")
    return result.stdout


@cache
def prepare():
    """Build this driver only; never relink the shared driver."""
    run(["lake", "build", MODULE])
    base = build_dir() / "overrank"
    base.mkdir(parents=True, exist_ok=True)
    launcher = base / "merlin-lean-overrank"
    lake = shutil.which("lake", path=str(Path.home() / ".elan/bin") + os.pathsep + os.environ.get("PATH", ""))
    if lake is None:
        raise RuntimeError("lake unavailable")
    directory = repo_root() / "merlin/lean"
    driver = directory / "MerlinVerify/Driver/Overrank.lean"
    launcher.write_text(
        f"#!/bin/sh\ncd {shlex.quote(str(directory))} || exit 2\n"
        f'exec {shlex.quote(lake)} env lean --run {shlex.quote(str(driver))} "$@"\n'
    )
    launcher.chmod(0o755)
    return launcher


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def source_export(text):
    """Canonical property placement matches MLIR's exporter; every key/value is retained.

    This is structural intake of an invalid operation, not a claimed MLIR interpretation.
    Target intake uses MLIR's exporter and round trip independently.
    """
    doc = export_text(text)

    def visit(op):
        if op["attrs"].keys() & op["props"].keys():
            raise ValueError("duplicate attributes/properties")
        if op["name"].partition(".")[0] in {"builtin", "func", "arith", "math", "tensor", "linalg", "scf", "cf"}:
            op["attrs"] = dict(sorted((op["attrs"] | op["props"]).items()))
            op["props"] = {}
        # Lean theorem OverrankIntake.i1_literal proves identical decoded one-bit values.
        if op["name"] == "arith.constant":
            literal = op["attrs"].get("value")
            if literal == {"kind": "int", "value": "-1", "type": {"kind": "int", "width": 1}}:
                literal["value"] = "1"
        for region in op["regions"]:
            for block in region["blocks"]:
                for child in block["ops"]:
                    visit(child)

    for op in doc["module"]["ops"]:
        visit(op)
    return doc


def run_checker(before, after, launcher=None):
    result = run([str(launcher or prepare()), str(before.resolve()), str(after.resolve())])
    return json.loads(result.strip().splitlines()[-1])


@cache
def audit():
    run(["lake", "build", "merlin-lean-audit"])
    report = json.loads(run(["lake", "exe", "merlin-lean-audit", MODULE]).strip().splitlines()[-1])
    if report["violations"]:
        raise RuntimeError(report)
    report["statements"] = json.loads(
        run(["lake", "exe", "merlin-lean-audit", "--statements", ",".join(THEOREMS), MODULE]).strip().splitlines()[-1]
    )["statements"]
    return report


def dependencies():
    root = repo_root() / "merlin/lean"
    todo, seen, paths = [MODULE], set(), set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        relative = Path(*module.split("."))
        source = (root / relative).with_suffix(".lean")
        paths.add(source)
        for line in source.read_text().splitlines():
            words = line.strip().split()
            if words and words[0] == "import":
                todo.extend(w for w in words[1:] if w.startswith("MerlinVerify."))
        compiled = build_dir() / "lean/build/lib/lean" / relative
        paths.update(
            p
            for suffix in (".olean", ".olean.private", ".olean.server", ".ir")
            if (p := compiled.with_suffix(suffix)).is_file()
        )
    paths.update(root / name for name in ("SPEC_LOCK", "lean-toolchain", "lake-manifest.json", "lakefile.toml"))
    paths.update(
        repo_root() / path
        for path in (
            "src/merlin/verify/lean/overrank.py",
            "src/merlin/verify/lean/export.py",
            "src/merlin/verify/lean/mlir_export.py",
            "src/merlin/llvmlower/passes_xdsl.py",
        )
    )
    paths.add(Path(run(["lake", "env", "which", "lean"]).strip()).resolve())
    return [_file_ref(p) for p in sorted(paths)]


@cache
def compiler_files(compiler_python):
    probe = subprocess.run(
        [
            str(compiler_python),
            "-c",
            "import json,sys; from torch_mlir import ir; from pathlib import Path; "
            "p=Path(ir.__file__).resolve(); "
            "print(json.dumps([sys.executable,str(p),*[str(q) for q in (p.parent/'_mlir_libs').glob('*.so')]]))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [_file_ref(Path(p).resolve()) for p in json.loads(probe.stdout)]


def actual_check(before, after):
    """Re-execute the production pass, not a hand-written replacement."""
    from io import StringIO

    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import collapse_overrank_matmul

    module = parse_mlir_text(before)
    count = collapse_overrank_matmul(module)
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    expected = source_export(stream.getvalue())["module"]
    return {"rewrites": count, "complete_actual_output_matches": expected == source_export(after)["module"]}


def export_before(source, output, compiler_python):
    """Prefer authoritative MLIR; retain invalid-source rejection and structural intake explicitly."""
    try:
        export_file(source, output, backend="mlir", m2m_python=compiler_python)
        return {"backend": "mlir", "invalid_source_diagnostic": None}
    except ExportError as error:
        write(output, source_export(source.read_text()))
        return {"backend": "xdsl_structural_invalid_source", "invalid_source_diagnostic": str(error)}


def check_boundary(before, after, destination, *, compiler_python):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    launcher = prepare()
    for label, path in (("before", before), ("after", after)):
        shutil.copyfile(path, destination / f"{label}.mlir")
    before, after = destination / "before.mlir", destination / "after.mlir"
    intake = export_before(before, destination / "before.json", compiler_python)
    run([str(launcher), "print", str(destination / "before.json"), str(destination / "source-roundtrip.mlir")])
    if intake["backend"] == "mlir":
        source_trip = round_trip_exported(
            before,
            destination / "before.json",
            destination / "source-roundtrip",
            lean_exe=launcher,
            backend="mlir",
            m2m_python=compiler_python,
        )
        if source_trip.status != "identical":
            raise ValueError(f"source round trip failed: {source_trip}")
    elif (
        source_export((destination / "source-roundtrip.mlir").read_text())["module"]
        != source_export(before.read_text())["module"]
    ):
        raise ValueError("invalid-source structural round trip differs")
    export_file(after, destination / "after.json", backend="mlir", m2m_python=compiler_python)
    report = run_checker(destination / "before.json", destination / "after.json", launcher)
    report["source_intake"] = intake
    report["actual_pass"] = actual_check(before.read_text(), after.read_text())
    if report["actual_pass"]["rewrites"] != report["covered_sites"]:
        raise ValueError("actual pass and Lean census disagree")
    trip = round_trip_exported(
        after,
        destination / "after.json",
        destination / "roundtrip",
        lean_exe=launcher,
        backend="mlir",
        m2m_python=compiler_python,
    )
    if trip.status != "identical":
        raise ValueError(f"target round trip failed: {trip}")
    report["target_round_trip"] = asdict(trip)
    report["proof_audit"] = audit()
    write(destination / "report.json", report)
    receipt = {
        "schema": "merlin.lean.overrank.v1",
        "intent_approved": False,
        "verdict": "abstracted" if report["checked"] else "rejected",
        "checked": report["checked"],
        "covered_sites": report["covered_sites"],
        "inputs": [_file_ref(before), _file_ref(after)],
        "exports": [_file_ref(destination / f"{label}.json") for label in ("before", "after")],
        "report": _file_ref(destination / "report.json"),
        "dependencies": dependencies(),
        "launcher": _file_ref(launcher),
        "compiler_files": compiler_files(compiler_python),
        "compiler_python": str(compiler_python.resolve()),
        "literal_source_equivalence_claimed": False,
    }
    if not report["actual_pass"]["complete_actual_output_matches"]:
        receipt.update(checked=False, verdict="rejected")
    write(destination / "receipt.json", receipt)
    return receipt


def replay(receipt_path, destination, *, compiler_python):
    """Re-export, re-run the actual pass, re-check and round-trip. Stale evidence fails closed."""
    receipt = json.loads(receipt_path.read_text())
    refs = [
        *receipt["dependencies"],
        *receipt.get("compiler_files", []),
        *receipt["inputs"],
        *receipt["exports"],
        receipt["report"],
        receipt["launcher"],
    ]
    stale = [
        r["path"]
        for r in refs
        if not Path(r["path"]).is_file() or hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() != r["sha256"]
    ]
    if receipt.get("compiler_files") != compiler_files(compiler_python):
        stale.append("compiler toolchain differs from receipt")
    if stale:
        return {"fresh": False, "matches_recorded": False, "stale": stale}
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source, target = [Path(r["path"]) for r in receipt["inputs"]]
    intake = export_before(source, destination / "before.json", compiler_python)
    export_file(target, destination / "after.json", backend="mlir", m2m_python=compiler_python)
    report = run_checker(destination / "before.json", destination / "after.json")
    actual = actual_check(source.read_text(), target.read_text())
    trip = round_trip_exported(
        target,
        destination / "after.json",
        destination / "roundtrip",
        lean_exe=prepare(),
        backend="mlir",
        m2m_python=compiler_python,
    )
    recorded = json.loads(Path(receipt["report"]["path"]).read_text())
    result = {
        "fresh": True,
        "matches_recorded": report
        == {
            k: v
            for k, v in recorded.items()
            if k not in {"actual_pass", "target_round_trip", "proof_audit", "source_intake"}
        },
        "checked": report["checked"] and trip.status == "identical" and actual["complete_actual_output_matches"],
        "actual_pass": actual,
        "target_round_trip": asdict(trip),
        "source_intake": intake,
    }
    write(destination / "replay.json", result)
    return result

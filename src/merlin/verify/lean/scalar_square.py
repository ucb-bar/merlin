"""Replayable provisional validation of prepare-scalar-squares, with an isolated Lean driver.

Acceptance proves complete output equality to a source-derived multiplication reference and
typed local equality conditional on ScalarSquareContract.Law. It does not discharge that law
or prove whole-capture source/target equivalence. Native counterexamples remain evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from merlin.common.paths import build_dir, repo_root
from merlin.verify.lean.check import _exporter_block, _file_ref
from merlin.verify.lean.export import export_file, round_trip_exported

MODULE = "MerlinVerify.Driver.ScalarSquare"
CHECKER = "MerlinVerify.Check.ScalarSquare"
CONTRACT = "MerlinVerify.Rules.ScalarSquareContract"
THEOREMS = [f"{CHECKER}.{name}" for name in ("check_sound", "complete_output_sound", "coverage", "Site.scalar_sound")]
THEOREMS += [
    f"{CONTRACT}.scalar_sound",
    f"{CONTRACT}.poison_sound",
    f"{CONTRACT}.law_nonempty",
    f"{CONTRACT}.law_not_universal",
    f"{CHECKER}.site_source",
    f"{CHECKER}.pairCheck_sound",
    f"{CHECKER}.accepted_pairs",
]


def _run(args: list[str], *, timeout: int = 600) -> str:
    env = dict(os.environ)
    env["PATH"] = str(Path.home() / ".elan/bin") + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        args, cwd=repo_root() / "merlin/lean", env=env, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {result.stdout[-3500:]} {result.stderr[-2000:]}")
    return result.stdout


def prepare() -> Path:
    """Build only this new driver; never relink the shared merlin-lean executable."""
    _run(["lake", "build", MODULE])
    directory = build_dir() / "scalar-square"
    directory.mkdir(parents=True, exist_ok=True)
    launcher = directory / "merlin-lean-scalar-square"
    lean = repo_root() / "merlin/lean"
    driver = lean / "MerlinVerify/Driver/ScalarSquare.lean"
    lake = shutil.which("lake", path=str(Path.home() / ".elan/bin") + os.pathsep + os.environ.get("PATH", ""))
    if lake is None:
        raise RuntimeError("lake unavailable")
    contents = (
        f"#!/bin/sh\ncd {shlex.quote(str(lean))} || exit 2\n"
        f'exec {shlex.quote(lake)} env lean --run {shlex.quote(str(driver))} "$@"\n'
    )
    if not launcher.exists() or launcher.read_text() != contents:
        launcher.write_text(contents)
        launcher.chmod(0o755)
    return launcher


def dependencies() -> list[dict]:
    """Bind this driver's source/import closure and compiled modules, excluding unrelated work."""
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
                todo.extend(word for word in words[1:] if word.startswith("MerlinVerify."))
        compiled = build_dir() / "lean/build/lib/lean" / relative
        paths.update(
            p
            for suffix in (".olean", ".olean.private", ".olean.server", ".ir")
            if (p := compiled.with_suffix(suffix)).is_file()
        )
    paths.update(root / name for name in ("SPEC_LOCK", "lean-toolchain", "lake-manifest.json", "lakefile.toml"))
    paths.add(Path(_run(["lake", "env", "which", "lean"]).strip()).resolve())
    paths.add(repo_root() / "src/merlin/verify/lean/scalar_square.py")
    return [_file_ref(path) for path in sorted(paths)]


def run_checker(before: Path, after: Path, launcher: Path) -> dict:
    return json.loads(_run([str(launcher), str(before.resolve()), str(after.resolve())]).strip().splitlines()[-1])


def audit() -> dict:
    _run(["lake", "build", "merlin-lean-audit"])
    report = json.loads(_run(["lake", "exe", "merlin-lean-audit", MODULE]).strip().splitlines()[-1])
    if report["violations"]:
        raise RuntimeError(f"Lean audit failed: {report}")
    statements = json.loads(
        _run(["lake", "exe", "merlin-lean-audit", "--statements", ",".join(THEOREMS), MODULE]).strip().splitlines()[-1]
    )["statements"]
    return {**report, "statements": statements}


def census(doc: dict) -> list[dict]:
    """Independent traversal of exported source, with lexical capture/shadowing and no target hints."""
    found = []

    def visit(ops, scope, path):
        scope = dict(scope)
        for index, op in enumerate(ops):
            here = [*path, index]
            if op["name"] == "math.powf" and len(op["results"]) == 1 and len(op["operands"]) == 2:
                ty = op["results"][0]["type"]
                fmt = ty.get("format") if ty["kind"] == "float" else None
                producer = scope.get(op["operands"][1])
                if fmt in {"f32", "f64"} and producer and producer["name"] == "arith.constant":
                    literal = {**producer["attrs"], **producer["props"]}.get("value", {})
                    if (
                        literal.get("kind") == "float"
                        and int(literal["bits"], 16) == {"f32": 0x40000000, "f64": 0x4000000000000000}[fmt]
                    ):
                        found.append({"path": here, "format": fmt})
            for ri, region in enumerate(op["regions"]):
                for bi, block in enumerate(region["blocks"]):
                    nested = {**scope, **{arg["id"]: None for arg in block["args"]}}
                    visit(block["ops"], nested, [*here, ri, bi])
            scope.update({result["id"]: op for result in op["results"]})

    visit(doc["module"]["ops"], {}, [])
    return found


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def check_stage(stage: Path, destination: Path, *, compiler_python: Path) -> dict:
    launcher = prepare()
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / "attempts" / uuid.uuid4().hex
    work.mkdir(parents=True)
    receipt_path = destination / "receipt.json"
    receipt = {"schema": "merlin.lean.scalar-square.v1", "verdict": "unmeasured", "stage": str(stage.resolve())}
    _write(receipt_path, receipt)
    try:
        meta = json.loads((stage / "meta.json").read_text())
        if meta.get("boundary") != "prepare-scalar-squares":
            raise ValueError("not the prepare-scalar-squares boundary")
        production = "merlin.llvmlower.scalar_square.canonicalize_scalar_squares"
        if meta.get("pass", production) != production:
            raise ValueError("stage names a different transformation")
        receipt["stage_metadata"] = _file_ref(stage / "meta.json")
        inputs, exports, trips = {}, {}, {}
        for name in ("before", "after"):
            original = stage / f"{name}.mlir"
            snapshot = work / f"{name}.mlir"
            shutil.copyfile(original, snapshot)
            reference = _file_ref(original)
            if meta.get(f"{name}_sha256", reference["sha256"]) != reference["sha256"]:
                raise ValueError("captured stage digest changed")
            if reference["sha256"] != _file_ref(snapshot)["sha256"]:
                raise ValueError("stage changed during snapshot")
            exported = work / f"{name}.json"
            export_file(snapshot, exported, backend="mlir", m2m_python=compiler_python)
            trip = round_trip_exported(
                snapshot,
                exported,
                work / f"roundtrip-{name}",
                lean_exe=launcher,
                backend="mlir",
                m2m_python=compiler_python,
            )
            if trip.status != "identical":
                raise ValueError(f"{name} round trip: {trip.status}: {trip.detail[:300]}")
            inputs[name] = {"original": reference, "snapshot": _file_ref(snapshot)}
            exports[name] = _file_ref(exported)
            trips[name] = trip.status
        bindings = dependencies()
        report = run_checker(Path(exports["before"]["path"]), Path(exports["after"]["path"]), launcher)
        independently_counted = census(json.loads(Path(exports["before"]["path"]).read_text()))
        if report["candidate_paths"] != [site["path"] for site in independently_counted]:
            raise ValueError("independent source census disagrees with Lean")
        report["independent_source_census"] = independently_counted
        report["proof_audit"] = audit()
        report_path = work / "report.json"
        _write(report_path, report)
        receipt.update(
            verdict="abstracted",
            conditional_match=report["checked"],
            inputs=inputs,
            exports=exports,
            report=_file_ref(report_path),
            dependencies=bindings,
            launcher=_file_ref(launcher),
            exporter=_exporter_block(compiler_python),
            round_trips=trips,
            numerical_law_status="provisional_not_discharged",
            reason="source-derived square policy checked; original pow/mul numerical equivalence is unproved",
        )
        if stale := freshness(receipt, compiler_python):
            raise ValueError(f"verification inputs changed: {stale}")
    except Exception as error:
        receipt.update(verdict="unmeasured", reason=f"{type(error).__name__}: {error}")
    _write(receipt_path, receipt)
    return receipt


def freshness(receipt: dict, compiler: Path) -> list[str]:
    refs = [
        *receipt["dependencies"],
        receipt["launcher"],
        receipt["report"],
        receipt["stage_metadata"],
        *receipt["exports"].values(),
    ]
    refs += [reference for pair in receipt["inputs"].values() for reference in pair.values()]
    changed = [ref["path"] for ref in refs if not Path(ref["path"]).is_file() or _file_ref(Path(ref["path"])) != ref]
    if receipt["exporter"] != _exporter_block(compiler):
        changed.append("exporter")
    return changed


def replay(path: Path, work: Path, *, compiler_python: Path) -> dict:
    receipt = json.loads(path.read_text())
    stale = freshness(receipt, compiler_python)
    if stale:
        return {"fresh": False, "stale": stale, "matches_recorded": False}
    work.mkdir(parents=True, exist_ok=False)
    recorded = json.loads(Path(receipt["report"]["path"]).read_text())
    launcher = Path(receipt["launcher"]["path"])
    for name in ("before", "after"):
        trip = round_trip_exported(
            Path(receipt["inputs"][name]["snapshot"]["path"]),
            Path(receipt["exports"][name]["path"]),
            work / name,
            lean_exe=launcher,
            backend="mlir",
            m2m_python=compiler_python,
        )
        if trip.status != "identical":
            raise ValueError(f"{name} replay round trip failed")
    actual = run_checker(
        Path(receipt["exports"]["before"]["path"]), Path(receipt["exports"]["after"]["path"]), launcher
    )
    differences = [key for key, value in actual.items() if value != recorded.get(key)]
    stale = freshness(receipt, compiler_python)
    result = {
        "fresh": not stale,
        "stale": stale,
        "matches_recorded": not differences,
        "differences": differences,
        "conditional_match": actual["checked"],
        "covered_sites": actual["covered_sites"],
        "numerical_law_status": actual["numerical_law_status"],
    }
    _write(work / "replay.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "replay"))
    parser.add_argument("input", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--compiler-python", type=Path, required=True)
    args = parser.parse_args()
    result = (
        check_stage(args.input, args.work, compiler_python=args.compiler_python)
        if args.command == "check"
        else replay(args.input, args.work, compiler_python=args.compiler_python)
    )
    print(json.dumps(result, indent=2))
    return (
        0
        if result.get("conditional_match")
        and (args.command == "check" or result.get("fresh") and result.get("matches_recorded"))
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

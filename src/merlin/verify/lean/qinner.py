"""Isolated quant-inner verification and replay with explicit runtime-binding evidence.

Receipts remain provisional. Neither annotation erasure nor the frozen poison-producing empty
semantics licenses treating archive bindings as original-source numerical equivalence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from functools import cache
from io import StringIO
from pathlib import Path

from merlin.common.paths import build_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.export import export_file, round_trip_exported

MODULE = "MerlinVerify.Driver.Qinner"
THEOREMS = [
    "MerlinVerify.Check.Qinner.check_sound",
    "MerlinVerify.Check.QinnerRuntime.check_sound",
    "MerlinVerify.Check.QinnerRuntime.values_equal",
    "MerlinVerify.Check.QinnerRuntime.program_equal",
    "MerlinVerify.Rules.QinnerExecution.accepted",
    "MerlinVerify.Rules.QinnerExecution.successful_clean",
    "MerlinVerify.Rules.QinnerExecution.domain_inhabited",
    "MerlinVerify.Check.Qinner.unchanged_sound",
    "MerlinVerify.Check.Qinner.slot_type_and_key",
    "MerlinVerify.Check.QinnerSSA.operations_sound",
    "MerlinVerify.Check.QinnerSSA.region_sound",
    "MerlinVerify.Check.QinnerProgram.program_equal",
    "MerlinVerify.Rules.QinnerABI.initial",
    "MerlinVerify.Check.QinnerPropagation.check_sound",
    "MerlinVerify.Check.QinnerPropagation.expected_output",
    "MerlinVerify.Rules.QinnerPropagation.complete_program",
]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


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
    run(["lake", "build", MODULE])
    root = repo_root() / "merlin/lean"
    base = build_dir() / "qinner"
    base.mkdir(parents=True, exist_ok=True)
    launcher = base / "merlin-lean-qinner"
    lake = shutil.which("lake", path=str(Path.home() / ".elan/bin") + os.pathsep + os.environ.get("PATH", ""))
    if not lake:
        raise RuntimeError("lake unavailable")
    content = (
        f"#!/bin/sh\ncd {shlex.quote(str(root))} || exit 2\n"
        f'exec {shlex.quote(lake)} env lean --run {shlex.quote(str(root / "MerlinVerify/Driver/Qinner.lean"))} "$@"\n'
    )
    if not launcher.is_file() or launcher.read_text() != content:
        import uuid

        temporary = base / f".launcher-{uuid.uuid4().hex}"
        temporary.write_text(content)
        temporary.chmod(0o755)
        temporary.replace(launcher)
    return launcher


def run_checker(source, target, *, mode="lift"):
    return json.loads(
        run([str(prepare()), mode, str(source.resolve()), str(target.resolve())]).strip().splitlines()[-1]
    )


def generic_text(module):
    from xdsl.printer import Printer

    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + "\n"


def actual(source):
    """Run the two production passes, preserving in-memory empty annotations in generic syntax."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import qinner
    from merlin.runtime.dispatch_runtime import _propagate_quant_inner

    module = parse_mlir_text(source)
    count = _propagate_quant_inner(module)
    propagated = generic_text(module)
    plan = qinner.plan(parse_mlir_text(source))
    appended = qinner.lift(module)
    module.verify()
    return {
        "propagated": propagated,
        "lifted": generic_text(module),
        "requests": count,
        "plan": [arg.as_json() for arg in plan],
        "appended": [arg.as_json() for arg in appended],
    }


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
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        relative = Path(*name.split("."))
        source = (root / relative).with_suffix(".lean")
        paths.add(source)
        for line in source.read_text().splitlines():
            words = line.strip().split()
            if words and words[0] == "import":
                todo.extend(word for word in words[1:] if word.startswith("MerlinVerify."))
        compiled = build_dir() / "lean/build/lib/lean" / relative
        paths.update(
            path
            for suffix in (".olean", ".olean.private", ".olean.server", ".ir")
            if (path := compiled.with_suffix(suffix)).is_file()
        )
    paths.update(root / name for name in ("SPEC_LOCK", "lean-toolchain", "lake-manifest.json", "lakefile.toml"))
    paths.update(
        repo_root() / name
        for name in (
            "src/merlin/verify/lean/qinner.py",
            "src/merlin/verify/lean/export.py",
            "src/merlin/verify/lean/mlir_export.py",
            "src/merlin/llvmlower/qinner.py",
            "src/merlin/runtime/dispatch_runtime.py",
            "src/merlin/llvmlower/c_runtime.py",
            "src/merlin/xdsl_dialects/lowering/outline.py",
        )
    )
    paths.add(Path(run(["lake", "env", "which", "lean"]).strip()).resolve())
    return [_file_ref(path) for path in sorted(paths)]


@cache
def compiler_files(compiler_python):
    result = subprocess.run(
        [
            str(compiler_python),
            "-c",
            "import json,sys; from torch_mlir import ir; "
            "from pathlib import Path; p=Path(ir.__file__).resolve(); "
            "print(json.dumps([sys.executable,str(p),*[str(q) for q in (p.parent/'_mlir_libs').glob('*.so')]]))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [_file_ref(Path(path).resolve()) for path in json.loads(result.stdout)]


def check_runtime(plan, extra, actual_arrays=None):
    """Compare every resolved byte in ABI order; a same-shape wrong tensor is an error.

    This checks concrete deliveries against the named archive, not the archive's relationship
    to the original torchao tensors or the behavior of a later target ABI implementation.
    """
    import numpy as np

    from merlin.llvmlower.qinner import QinnerArg, resolve

    args = [QinnerArg.from_json(item) for item in plan]
    keys = [arg.key for arg in args]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate quant-inner keys")
    expected = resolve(extra, args)
    arrays = expected if actual_arrays is None else actual_arrays
    if len(arrays) != len(expected):
        raise ValueError("missing or extra runtime arguments")
    records = []
    for index, (arg, want, got) in enumerate(zip(args, expected, arrays, strict=True)):
        got = np.ascontiguousarray(got)
        if got.shape != want.shape or got.dtype != want.dtype or got.tobytes() != want.tobytes():
            raise ValueError(f"incorrect runtime tensor binding at argument {index}: {arg.key}")
        records.append(
            {"index": index, **arg.as_json(), "bytes": got.nbytes, "sha256": hashlib.sha256(got.tobytes()).hexdigest()}
        )
    return records


def runtime_document(plan, extra, delivered=None):
    """Complete byte vectors for Lean's ordered ABI checker; preserve row-major little endian."""
    import numpy as np

    from merlin.llvmlower.qinner import QinnerArg, resolve

    args = [QinnerArg.from_json(row) for row in plan]
    expected = resolve(extra, args)
    actual_values = expected if delivered is None else delivered

    def blob(arg, value):
        array = np.ascontiguousarray(value)
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        if array.dtype.kind == "b":
            element = {"kind": "int", "width": 1}
        elif array.dtype.kind == "i":
            element = {"kind": "int", "width": array.dtype.itemsize * 8}
        elif array.dtype.kind == "f":
            element = {"kind": "float", "format": f"f{array.dtype.itemsize * 8}"}
        else:
            raise ValueError(f"unsupported runtime dtype {array.dtype}")
        return {
            "key": arg.key,
            "type": {"kind": "tensor", "shape": list(array.shape), "element": element, "encoding": None},
            "hex": array.tobytes().hex(),
        }

    return {
        "archive": [blob(arg, arr) for arg, arr in zip(args, expected, strict=True)],
        "delivered": [blob(arg, arr) for arg, arr in zip(args, actual_values, strict=True)],
    }


def run_binding(source, target, certificate):
    return json.loads(
        run([str(prepare()), "binding", str(source.resolve()), str(target.resolve()), str(certificate.resolve())])
        .strip()
        .splitlines()[-1]
    )


def check_boundary(before, after, destination, *, compiler_python, extra_path=None):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generated = actual(before.read_text())
    shutil.copyfile(before, destination / "before.mlir")
    (destination / "propagated.mlir").write_text(generated["propagated"])
    shutil.copyfile(after, destination / "after.mlir")
    write(destination / "plan.json", generated["plan"])
    for label in ("before", "propagated", "after"):
        export_file(
            destination / f"{label}.mlir", destination / f"{label}.json", backend="mlir", m2m_python=compiler_python
        )
    report = run_checker(destination / "propagated.json", destination / "after.json")
    report["propagation"] = run_checker(destination / "before.json", destination / "propagated.json", mode="propagate")
    expected_path = destination / "actual-after.mlir"
    expected_path.write_text(generated["lifted"])
    export_file(expected_path, destination / "actual-after.json", backend="mlir", m2m_python=compiler_python)
    expected = json.loads((destination / "actual-after.json").read_text())["module"]
    target = json.loads((destination / "after.json").read_text())["module"]
    report["actual_pass"] = {
        "complete_output_matches": expected == target,
        "plan_equals_appended": generated["plan"] == generated["appended"],
        "propagation_requests": generated["requests"],
    }
    report["round_trips"] = {}
    for label in ("before", "propagated", "after"):
        trip = round_trip_exported(
            destination / f"{label}.mlir",
            destination / f"{label}.json",
            destination / f"{label}-roundtrip",
            lean_exe=prepare(),
            backend="mlir",
            m2m_python=compiler_python,
        )
        report["round_trips"][label] = {"status": trip.status, "detail": trip.detail}
        if trip.status != "identical":
            raise ValueError(f"{label} round trip failed: {trip}")
    report["runtime_bindings"] = None
    extra_refs = []
    if extra_path is not None:
        import numpy as np

        shutil.copyfile(extra_path, destination / "extra.npz")
        with np.load(destination / "extra.npz", allow_pickle=False) as extra:
            report["runtime_bindings"] = check_runtime(generated["plan"], extra)
            write(destination / "runtime.json", runtime_document(generated["plan"], extra))
        report["runtime_checker"] = run_binding(
            destination / "propagated.json", destination / "after.json", destination / "runtime.json"
        )
        extra_refs = [_file_ref(destination / "extra.npz"), _file_ref(destination / "runtime.json")]
    report["proof_audit"] = audit()
    write(destination / "report.json", report)
    accepted = (
        report["checked"]
        and report["propagation"]["checked"]
        and all(report["actual_pass"][key] for key in ("complete_output_matches", "plan_equals_appended"))
    )
    accepted = accepted and report.get("runtime_checker", {"checked": True})["checked"]
    receipt = {
        "schema": "merlin.lean.qinner.v1",
        "verdict": "abstracted" if accepted else "rejected",
        "checked": accepted,
        "approved": False,
        "boundary_kind": report["boundary_kind"],
        "covered_sites": report["covered_sites"],
        "inputs": [_file_ref(destination / f"{label}.mlir") for label in ("before", "after")],
        "exports": [
            _file_ref(destination / f"{label}.json") for label in ("before", "propagated", "after", "actual-after")
        ],
        "plan": _file_ref(destination / "plan.json"),
        "extra": extra_refs,
        "report": _file_ref(destination / "report.json"),
        "dependencies": dependencies(),
        "launcher": _file_ref(prepare()),
        "compiler_files": compiler_files(compiler_python),
    }
    write(destination / "receipt.json", receipt)
    return receipt


def replay(receipt_path, destination, *, compiler_python):
    receipt = json.loads(receipt_path.read_text())
    refs = [
        *receipt["dependencies"],
        *receipt["compiler_files"],
        *receipt["inputs"],
        *receipt["exports"],
        *receipt["extra"],
        receipt["plan"],
        receipt["report"],
        receipt["launcher"],
    ]
    stale = [
        ref["path"]
        for ref in refs
        if not Path(ref["path"]).is_file()
        or hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]
    ]
    if receipt["compiler_files"] != compiler_files(compiler_python):
        stale.append("compiler toolchain differs")
    if stale:
        return {"fresh": False, "checked": False, "stale": stale}
    source, target = [Path(ref["path"]) for ref in receipt["inputs"]]
    result = check_boundary(
        source,
        target,
        destination,
        compiler_python=compiler_python,
        extra_path=Path(receipt["extra"][0]["path"]) if receipt["extra"] else None,
    )
    old = json.loads(Path(receipt["report"]["path"]).read_text())
    new = json.loads(Path(result["report"]["path"]).read_text())
    # Round-trip filenames change with the destination, statuses and all semantic fields do not.
    matches = {key: value for key, value in old.items() if key not in {"round_trips", "proof_audit"}} == {
        key: value for key, value in new.items() if key not in {"round_trips", "proof_audit"}
    }
    answer = {"fresh": True, "checked": result["checked"], "matches_recorded": matches}
    write(destination.resolve() / "replay.json", answer)
    return answer

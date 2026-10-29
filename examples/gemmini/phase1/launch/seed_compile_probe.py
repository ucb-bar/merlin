"""Seed preflight: the staged seed compiler imports and compiles one small public test case.

A fresh10 seed overlay kept only the members its author had edited, so it lost nine members the new
export's full compiler no longer stages (``targetgen/generate/*``, ``oot_starterkit/{iface,
transforms,verify}.py``), and the seeded run failed every compile with ``ModuleNotFoundError``.
Accepting the seed's bytes and staging placement did not notice.

Here the seed is staged exactly as the session stages it (``joint_seed.stage``, then the full
compiler staging), and the staged package's own entrypoints (``capsule_common.lower_interface``) run
on the smallest public interface MLIRs of the run's sealed corpus. Merlin modules are importable
only from what a candidate sandbox grants: the staged submission and the bundle's public common
sources, materialized as exact leaves. The export's source tree and any Merlin checkout installed
into the interpreter are not, so they cannot stand in for a member the staged compiler lacks.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

#: Smallest public cases tried; one compiled case passes. An intentional refusal of one shape
#: must not refuse the seed, an import error in every case must.
ATTEMPTS = 4
TIMEOUT_S = 300

#: Runs a Python entrypoint with every Merlin source root outside the granted ones removed from
#: ``sys.path``; argv: GRANTED_ROOT SCRIPT ARGS...
_BOOT = """\
import os, runpy, sys
granted, script = os.path.realpath(sys.argv[1]), sys.argv[2]
def foreign(entry):
    path = os.path.realpath(entry or os.curdir)
    return path != granted and os.path.isdir(os.path.join(path, "merlin"))
sys.path[:] = [os.path.dirname(os.path.abspath(script))] + [p for p in sys.path if not foreign(p)]
sys.argv = sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


class SeedCompileError(ValueError):
    pass


def granted_sources(bundle: dict, repo: Path, destination: Path) -> Path:
    """The bundle's public common sources at their package paths: the import root a sandbox grants."""
    from merlin.targetgen.sandbox import bwrap

    rows = (bundle.get("compiler_tool_profile") or {}).get("common_sources") or ()
    grants = [row["path"] for row in bundle.get("allowed", ()) if isinstance(row.get("path"), str)]
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        # The source leaf is the bundle's own grant for this module (as the sandbox mounts it).
        tail = row["path"].removeprefix("merlin/python/")
        matches = [grant for grant in grants if grant == row["path"] or grant.endswith("/" + tail)]
        if len(matches) != 1:
            raise SeedCompileError(f"public common source has no unique bundle grant: {row['path']}")
        data = bwrap.resolve_grant(matches[0], Path(repo)).read_bytes()
        if row.get("sha256") and hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SeedCompileError(f"public common source differs from the bundle pin: {row['path']}")
        target = destination / tail
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return destination


def _python_tool(tool: Path) -> bool:
    if tool.suffix == ".py":
        return True
    try:
        with tool.open("rb") as stream:
            first = stream.readline(256)
    except OSError:
        return False
    return first.startswith(b"#!") and b"python" in first


def invoker(granted: Path):
    """An entrypoint runner with the sandbox's Merlin import roots, for ``lower_interface(invoke=)``."""
    from merlin.targetgen import package_runtime as P

    def invoke(pkg, name, input_mlir, output_json=None, *, timeout=TIMEOUT_S, **_):
        argv = P._resolve_argv(pkg, name, Path(input_mlir), Path(output_json) if output_json else None)
        if argv and argv[0] == str(pkg.tool) and _python_tool(pkg.tool):
            argv = [sys.executable, "-c", _BOOT, str(granted), *argv]
        environment = dict(os.environ, PYTHONPATH=str(granted), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(argv, cwd=pkg.directory, env=environment, capture_output=True, text=True,
                              timeout=timeout)

    return invoke


def smallest_public_cases(corpus_roots, *, contract=None, count=ATTEMPTS) -> list[dict]:
    from merlin.targetgen import capsule_common as CC

    cases = []
    for capsule in CC.discover_capsules(list(corpus_roots), labels={"public"}, contract=contract):
        interface = Path(capsule["__dir__"]) / capsule.get("interface_mlir", "capsule.interface.mlir")
        if interface.is_file():
            cases.append((interface.stat().st_size, capsule["name"], interface))
    if not cases:
        raise SeedCompileError("the run's sealed corpus has no public interface MLIR to compile")
    return [dict(name=name, interface=interface) for _, name, interface in sorted(cases)[:count]]


def compile_one(submission: Path, cases, *, granted: Path, work: Path, contract=None) -> dict:
    """The staged package lowers at least one case; otherwise every case's first error is reported."""
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen.package_runtime import build_package, load_package

    package = load_package(submission, contract=contract)
    build_package(package)
    invoke, refused = invoker(granted), []
    for case in cases:
        generated = work / case["name"]
        generated.mkdir(parents=True, exist_ok=True)
        try:
            buffer, artifact = CC.lower_interface(package, case["interface"], generated, contract=contract,
                                                  timeout=TIMEOUT_S, invoke=invoke)
        except Exception as error:  # noqa: BLE001 - every failure is reported, none is a pass
            refused.append(f"{case['name']}: {type(error).__name__}: {str(error)[:400]}")
            continue
        if buffer and artifact.strip():
            return {"status": "compiled", "case": case["name"], "refused_first": refused}
        refused.append(f"{case['name']}: empty command buffer or target artifact")
    raise SeedCompileError("the staged seed compiler compiled none of the smallest public cases: "
                           + " | ".join(refused))

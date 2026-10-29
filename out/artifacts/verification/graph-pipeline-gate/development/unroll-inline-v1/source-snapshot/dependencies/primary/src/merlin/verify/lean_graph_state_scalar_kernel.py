"""Independent scalar graph pure-core profile, pinned tool, sequential admission.

No elision build/query is reopened. Any new native signal permanently stops this
scalar native context; saved proof/test success cannot substitute for that run.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import artifacts_dir, build_dir, repo_root

from . import lean_graph_state as stable

MODULES = (
    "GraphStateScalarContract",
    "GraphStateScalarNormalize",
    "GraphStateScalarSound",
    "GraphStateScalarTyping",
    "GraphStateScalarInitialized",
    "GraphStateScalar",
    "GraphStateScalarSteps",
)


def guard():
    blocked = build_dir() / "graph-verification/state-scalar/NATIVE_BLOCKER.json"
    if blocked.exists():
        raise RuntimeError("scalar native context blocked; no retries or promotion: " + str(blocked))
    unconfirmed = blocked.parent / "UNCONFIRMED_CONTEXT.json"
    if unconfirmed.exists():
        raise RuntimeError("unconfirmed scalar dependency context; source/binding work only")
    return blocked


def fixed_core():
    """Use existing exact pins from the one incident check, never new identities."""
    path = artifacts_dir() / "verification/graph-state-scalar/active/shared-toolchain-binding-check-v1.json"
    data = json.loads(path.read_text())
    if data["target_in_actual_proof_import_object_closure"] or not data["all_recorded_object_tool_bindings_exact"]:
        raise RuntimeError("shared dependency context affected/unconfirmed; source work only")
    if not stable.fresh([*data["fixed_standard_object_bindings"], *data["fixed_tool_bindings"]]):
        raise ValueError("fixed preexisting core object/tool pin changed; no rehash-to-promote")
    return {
        "selection": stable.file_ref(path),
        "standard_objects": data["fixed_standard_object_bindings"],
        "tool_bindings": data["fixed_tool_bindings"],
    }


def invoke(arguments, cwd, environment, out, err):
    blocked = guard()
    result = run_owned(
        [str(stable.LEAN), "-j1", *arguments], cwd=cwd, env=environment, stdout=out, stderr=err, timeout=60, check=False
    )
    if result.returncode < 0:
        stable._write(
            blocked,
            {
                "status": "UNQUALIFIED_NATIVE_SIGNAL",
                "returncode": result.returncode,
                "arguments": arguments,
                "cwd": str(cwd),
                "cause": "UNKNOWN",
                "tool": stable.file_ref(stable.LEAN),
                "promotion_permitted": False,
            },
        )
    return result


@cache
def prepare():
    guard()
    core = fixed_core()
    if stable.file_ref(stable.LEAN)["sha256"] != stable.LEAN_SHA256:
        raise ValueError("selected pure Lean executable differs from pinned intake")
    sources = [stable.file_ref(repo_root() / f"src/merlin/verify/lean/Merlin/{name}.lean") for name in MODULES]
    shared_library = stable.file_ref(stable.LEAN.parent.parent / "lib/lean/libleanshared.so")
    key = hashlib.sha256(
        json.dumps(
            {
                "sources": sources,
                "lean": stable.file_ref(stable.LEAN),
                "library": shared_library,
                "arguments": ["-j1"],
                "fixed_core": core,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    directory = build_dir() / "graph-verification/state-scalar/kernel" / key
    lib = directory / "lib/lean"
    marker = directory / "ready.json"
    if not marker.exists():
        (lib / "Merlin").mkdir(parents=True, exist_ok=True)
        for name, original in zip(MODULES, sources, strict=True):
            source = directory / f"sources/Merlin/{name}.lean"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(Path(original["path"]).read_bytes())
            with (directory / (name + ".stdout")).open("x") as out, (directory / (name + ".stderr")).open("x") as err:
                result = invoke(
                    [str(source), "-o", str(lib / f"Merlin/{name}.olean")],
                    directory / "sources",
                    {**os.environ, "LEAN_PATH": str(lib)},
                    out,
                    err,
                )
            if result.returncode:
                raise RuntimeError("new scalar profile unavailable; first buffers retained at " + str(directory))
        files = [stable.file_ref(path) for path in sorted(directory.rglob("*")) if path.is_file()]
        stable._write(
            marker,
            {
                "schema": "merlin.graph-state-scalar.kernel-profile.v1",
                "identity": key,
                "sources": sources,
                "lean": stable.file_ref(stable.LEAN),
                "shared_library": shared_library,
                "lean_arguments": ["-j1"],
                "fixed_core": core,
                "files": files,
            },
        )
    data = json.loads(marker.read_text())
    if (
        data["sources"] != sources
        or data["fixed_core"] != core
        or not stable.fresh(
            [
                *data["files"],
                *sources,
                data["lean"],
                data["shared_library"],
                *core["standard_objects"],
                core["selection"],
            ]
        )
    ):
        raise ValueError("selected scalar source/object/tool context changed")
    return {**data, "directory": directory, "lib": lib, "manifest": stable.file_ref(marker)}


def audit(destination, profile, module, roots):
    auditor = repo_root() / "src/merlin/verify/lean/Merlin/GraphStateAudit.lean"
    env = {
        **os.environ,
        "LEAN_PATH": os.pathsep.join((str(destination), str(profile["lib"]))),
        "PATH": str(stable.LEAN.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    measured = {}
    for name, arguments in (("audit", [module]), ("statements", ["--statements", ",".join(roots), module])):
        with (destination / (name + ".stdout")).open("x") as out, (destination / (name + ".stderr")).open("x") as err:
            result = invoke(["--run", str(auditor), *arguments], destination, env, out, err)
        if result.returncode:
            return {"status": "unavailable", "phase": name, "returncode": result.returncode}
        measured[name] = json.loads((destination / (name + ".stdout")).read_text())
    raw = measured["audit"]
    if (
        raw["assurance"] != "lean_kernel_std3_no_native_reflection"
        or raw["violations"]
        or not set(raw["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    ):
        raise ValueError("scalar actual imported proof violates standard-three policy")
    return {
        "status": "checked",
        "raw": raw,
        "statements": measured["statements"]["statements"],
        "auditor": stable.file_ref(auditor),
        "imported_objects": [stable.file_ref(row["olean"]) for row in raw["imports"]],
        "buffers": [
            stable.file_ref(destination / (name + "." + suffix))
            for name in ("audit", "statements")
            for suffix in ("stdout", "stderr")
        ],
    }


def check_query(text, destination, roots, *, module="GraphStateScalarInstance"):
    guard()
    profile = prepare()
    if not stable.fresh(
        [
            *profile["fixed_core"]["standard_objects"],
            *profile["fixed_core"]["tool_bindings"],
            profile["fixed_core"]["selection"],
        ]
    ):
        raise ValueError("fixed unaffected core binding changed before native check")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    query = destination / (module + ".lean")
    query.write_text(text)
    with (destination / "kernel.stdout").open("x") as out, (destination / "kernel.stderr").open("x") as err:
        result = invoke(
            [str(query), "-o", str(query.with_suffix(".olean"))],
            destination,
            {**os.environ, "LEAN_PATH": str(profile["lib"])},
            out,
            err,
        )
    checked_audit = audit(destination, profile, module, roots) if result.returncode == 0 else {"status": "not_run"}
    admitted = {row["path"]: row for row in profile["fixed_core"]["standard_objects"]}
    for imported in checked_audit.get("imported_objects", []):
        if (
            Path(imported["path"]).is_relative_to(stable.LEAN.parent.parent / "lib/lean")
            and admitted.get(imported["path"]) != imported
        ):
            stable._write(
                build_dir() / "graph-verification/state-scalar/UNCONFIRMED_CONTEXT.json",
                {
                    "status": "UNQUALIFIED",
                    "reason": "new/changed standard import outside fixed core",
                    "observed": imported,
                },
            )
            raise ValueError("unconfirmed standard import; no adoption or promotion")
    if not stable.fresh(
        [
            *profile["sources"],
            *profile["files"],
            profile["lean"],
            profile["shared_library"],
            profile["manifest"],
            *checked_audit.get("imported_objects", []),
            *profile["fixed_core"]["standard_objects"],
            *profile["fixed_core"]["tool_bindings"],
            profile["fixed_core"]["selection"],
        ]
    ):
        raise ValueError("scalar source/proof/tool/import changed during actual check")
    report = {
        "schema": "merlin.graph-state-scalar.kernel-check.v1",
        "query": stable.file_ref(query),
        "returncode": result.returncode,
        "profile": profile["manifest"],
        "audit": checked_audit,
        "status": "conditional_graph_checked"
        if result.returncode == 0 and checked_audit["status"] == "checked"
        else "unavailable",
        "public_acceptance_issued": False,
        "review_approved": False,
        "native_fidelity_proved": False,
        "source_success_proved": False,
        "poison_freedom_proved": False,
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
    }
    stable._write(destination / "report.json", report)
    return report

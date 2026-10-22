"""Replayable, provisional pass-local evidence; separate from the production CLI.

The status never represents approved full verification. Artifact identity and compiler
round trips are checked here; acceptance is recomputed by Lean, not a Python shape guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.verify.lean import check as infrastructure

SCHEMA = "merlin.local-passes.receipt.v1"
THEOREMS = {
    "bool-to-float": ("MerlinVerify.Rules.LocalPassesChain", "bool_sound"),
    "torchao-affine": ("MerlinVerify.Rules.LocalPassesChain", "torchao_sound"),
}
STATUSES = {
    "bool-to-float": "proved_policy_provisional",
    "torchao-affine": "proved_materialization_provisional",
}
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _ref(path: Path) -> dict[str, Any]:
    return infrastructure._file_ref(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _run(command: list[str], work: Path, label: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan/bin'}:{env.get('PATH', '')}"
    result = subprocess.run(
        command, cwd=infrastructure.lean_root(), env=env, capture_output=True, text=True, timeout=1200
    )
    (work / f"{label}.stdout").write_text(result.stdout)
    (work / f"{label}.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{label}: {(result.stdout + result.stderr)[-2000:]}")
    return result


def interpretation() -> dict[str, Any]:
    root, lean = repo_root(), infrastructure.lean_root()
    objects = root / "out/build/lean/build/lib/lean"
    sources = sorted((lean / "MerlinVerify").rglob("*.lean"))
    sources += [
        lean / name
        for name in (
            "LocalPassesCandidate.lean",
            "LocalPasses.lakefile.toml",
            "SPEC_LOCK",
            "lean-toolchain",
            "lake-manifest.json",
        )
    ]
    sources += [
        root / "src/merlin/verify/lean" / name
        for name in ("local_passes.py", "check.py", "export.py", "mlir_export.py")
    ]
    compiled = sorted((objects / "MerlinVerify").rglob("*.olean"))
    if not (objects / "MerlinVerify/Rules/BoolToFloatProgram.olean").is_file():
        raise ValueError("local pass checker has not been built")
    identity = {
        "sources": {str(p.relative_to(root)): _ref(p)["sha256"] for p in sources},
        "compiled": {str(p.relative_to(objects)): _ref(p)["sha256"] for p in compiled},
        "lean": infrastructure._lean_block(),
        "executable": _ref(root / "out/build/lean/build/bin/merlin-lean-local-passes"),
        "contract_review": "pending; not covered by prior float-to-bool approval",
    }
    return {**identity, "sha256": _digest(identity)}


def _proof(mode: str, lake: Path, work: Path) -> dict[str, Any]:
    module, short = THEOREMS[mode]
    theorem = f"{module}.{short}"
    result = _run([str(lake), "exe", "merlin-lean-audit", "--axioms", theorem, module], work, "axioms")
    axioms = json.loads(result.stdout.strip().splitlines()[-1])["axioms"]
    if not set(axioms) <= ALLOWED_AXIOMS:
        raise ValueError(f"unapproved theorem dependencies: {axioms}")
    result = _run([str(lake), "exe", "merlin-lean-audit", "--statements", theorem, module], work, "statement")
    statement = json.loads(result.stdout.strip().splitlines()[-1])["statements"][theorem]
    return {"theorem": theorem, "axioms": axioms, "statement": statement}


def verify(
    mode: str,
    before: Path,
    after: Path,
    destination: Path,
    *,
    compiler_python: Path,
    lake: Path | None = None,
    references: tuple[Path, ...] = (),
    suffix: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Build, round trip, check and bind exact artifacts. Every contract here is provisional."""
    if mode not in THEOREMS:
        raise ValueError(f"unsupported local pass: {mode}")
    work = infrastructure._work(destination)
    lake = lake or Path.home() / ".elan/bin/lake"
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "status": "unmeasured",
        "active": False,
        "approved_full_verification": False,
        "contract_review": "pending",
        "ordinary_source_equivalence": False,
        "whole_function_defined": False,
        "work": str(work.resolve()),
        "artifacts": [],
        "references": [_ref(path) for path in references],
        "source_call_refinement_proved": mode == "bool-to-float",
        "actual_capture_version_recipe_admitted": False,
    }
    try:
        _run([str(lake), "-f", "LocalPasses.lakefile.toml", "build", "merlin-lean-local-passes"], work, "build")
        identity = interpretation()
        receipt["interpretation"] = identity
        for index, source in enumerate((before, after, *suffix)):
            snapshot = work / f"stage-{index}.mlir"
            original = infrastructure._snapshot(source, snapshot)
            exported, error = infrastructure._export_checked(
                snapshot, work / f"round-trip-{index}", m2m_python=compiler_python
            )
            if error:
                raise ValueError(error)
            receipt["artifacts"].append({"original": original, "mlir": _ref(snapshot), "export": _ref(exported)})
        receipt["exporter"] = infrastructure._exporter_block(compiler_python)
        receipt["proof"] = _proof(mode, lake, work)
        paths = [a["export"]["path"] for a in receipt["artifacts"]]
        result = _run([identity["executable"]["path"], mode, *paths], work, "checker")
        answer = json.loads(result.stdout)
        if answer.get("theorem") != receipt["proof"]["theorem"]:
            raise ValueError("checker theorem identity mismatch")
        if identity != interpretation():
            raise ValueError("checker interpretation changed during execution")
        receipt["answer"] = answer
        receipt["result"] = _ref(work / "checker.stdout")
        receipt["status"] = STATUSES[mode] if answer.get("accepted") is True else "refused"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        receipt["reason"] = str(error)
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def replay(receipt: dict[str, Any], *, compiler_python: Path, lake: Path | None = None) -> None:
    """Re-export the bound MLIR, replay audits and run the actual Lean checker again."""
    expected = {
        "schema": SCHEMA,
        "active": False,
        "approved_full_verification": False,
        "contract_review": "pending",
        "ordinary_source_equivalence": False,
        "whole_function_defined": False,
        "actual_capture_version_recipe_admitted": False,
    }
    if any(receipt.get(k) != v or type(receipt.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("changed provisional claim or receipt status")
    mode = receipt.get("mode")
    if mode not in THEOREMS or receipt.get("interpretation") != interpretation():
        raise ValueError("stale or mixed interpretation")
    if receipt.get("status") != STATUSES[mode] or receipt.get("source_call_refinement_proved") is not (
        mode == "bool-to-float"
    ):
        raise ValueError("changed proof scope")
    for reference in receipt.get("references", []):
        if _ref(Path(reference["path"])) != reference:
            raise ValueError("reference bytes changed")
    if receipt.get("exporter") != infrastructure._exporter_block(compiler_python):
        raise ValueError("authoritative compiler/exporter changed")
    artifacts = receipt.get("artifacts", [])
    if len(artifacts) < 2:
        raise ValueError("complete before/after pair required")
    work = infrastructure._work(Path(receipt["work"]) / "replay")
    for index, artifact in enumerate(artifacts):
        for key in ("mlir", "export"):
            if _ref(Path(artifact[key]["path"])) != artifact[key]:
                raise ValueError(f"stale {key} artifact")
        source, exported = (Path(artifact[k]["path"]) for k in ("mlir", "export"))
        if not infrastructure._exported_from(exported, source):
            raise ValueError("export does not bind source bytes")
        trip = infrastructure.round_trip_exported(
            source,
            exported,
            work / f"round-trip-{index}",
            lean_exe=infrastructure.lean_exe(),
            backend="mlir",
            m2m_python=compiler_python,
        )
        if trip.status != "identical":
            raise ValueError(f"round-trip replay failed: {trip.detail}")
    if receipt.get("proof") != _proof(mode, lake or Path.home() / ".elan/bin/lake", work):
        raise ValueError("theorem audit changed")
    if _ref(Path(receipt["result"]["path"])) != receipt["result"]:
        raise ValueError("checker result changed")
    answer = json.loads(Path(receipt["result"]["path"]).read_text())
    result = _run(
        [receipt["interpretation"]["executable"]["path"], mode, *(a["export"]["path"] for a in artifacts)],
        work,
        "checker-replay",
    )
    if answer != receipt.get("answer") or answer.get("accepted") is not True or json.loads(result.stdout) != answer:
        raise ValueError("actual Lean checker replay differs")
    if interpretation() != receipt["interpretation"]:
        raise ValueError("interpretation changed during replay")

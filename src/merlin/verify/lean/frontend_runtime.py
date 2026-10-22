"""Inactive runtime-candidate verification, with byte-bound evidence and one interpretation.

This additive API does not issue frozen-model receipts or activate candidate semantics. Lean
checks the complete f32 repair boundary, actual initialized prefix/selection execution, and
every supplied suffix boundary. Optional actual source slices have explicit initialized
cut-input contracts, independent of the initialized initial-segment claim. Suffixes
currently require complete alpha correspondence.
The census enumerates remaining full-model obligations; it is not a definedness proof.
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

SCHEMA = "merlin.frontend.runtime.receipt.v1"
REQUEST = "merlin.frontend.runtime.request.v1"
MODULE = "MerlinVerify.Rules.FrontendRuntimeEvidence"
THEOREMS = (
    "MerlinVerify.Rules.FrontendRuntimeVerification.check_sound",
    "MerlinVerify.Rules.FrontendRuntimeVerification.full_policy",
    "MerlinVerify.Rules.FrontendRuntimeChain.compose",
    "MerlinVerify.Rules.FrontendRangeProjection.check_sound",
    "MerlinVerify.Rules.FrontendMaskRange.check_sound",
)
CONFIGURATION = {
    "base": "MerlinVerify.Rules.FrontendMaskRuntime.sem",
    "comparison": "MerlinVerify.Rules.FrontendF32Comparison.model(base)",
    "source_policy": "MerlinVerify.Rules.FrontendBoolSimulationWith.policyDenote",
    "poison_control": "stuck",
    "index_bits": 64,
    "limit": 1024,
}
_ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _ref(path: Path) -> dict[str, Any]:
    return infrastructure._file_ref(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def interpretation() -> dict[str, Any]:
    """Bind source bodies AND their local compiled objects; names alone are insufficient.

    Callers build first. Rechecking this identity after execution detects concurrent changes.
    Upstream Lean/Mathlib remain explicit T1 dependencies, with pinned revisions and Lean binary.
    """
    root = repo_root()
    lean = infrastructure.lean_root()
    obj = root / "out/build/lean/build/lib/lean"
    sources = sorted((lean / "MerlinVerify").rglob("*.lean"))
    sources += [
        lean / "FrontendRuntimeCandidate.lean",
        lean / "FrontendRuntime.lakefile.toml",
        lean / "SPEC_LOCK",
        lean / "lean-toolchain",
        lean / "lake-manifest.json",
    ]
    sources += [
        root / "src/merlin/verify/lean" / name
        for name in ("frontend_runtime.py", "check.py", "export.py", "mlir_export.py")
    ]
    files = {str(p.relative_to(root)): _ref(p)["sha256"] for p in sources}
    objects = sorted((obj / "MerlinVerify").rglob("*.olean"))
    if not objects or not (obj / "MerlinVerify/Rules/FrontendRuntimeObligations.olean").is_file():
        raise ValueError("candidate Lean objects have not been built")
    compiled = {str(p.relative_to(obj)): _ref(p)["sha256"] for p in objects}
    review = infrastructure._frontend_review_evidence()
    identity = {
        "configuration": dict(CONFIGURATION),
        "sources": files,
        "compiled": compiled,
        "lean": infrastructure._lean_block(),
        "candidate_executable": _ref(root / "out/build/lean/build/bin/merlin-lean-frontend-runtime"),
        "f32_policy_approval": review,
        "runtime_review": "pending; inactive candidate",
    }
    return {**identity, "sha256": _digest(identity)}


def _run(command: list[str], work: Path, name: str, *, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan/bin'}:{env.get('PATH', '')}"
    proc = subprocess.run(
        command, cwd=infrastructure.lean_root(), env=env, capture_output=True, text=True, timeout=timeout
    )
    (work / f"{name}.stdout").write_text(proc.stdout)
    (work / f"{name}.stderr").write_text(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"{name} failed ({proc.returncode}): {(proc.stdout + proc.stderr)[-1000:]}")
    return proc


def _proofs(work: Path, lake: Path) -> list[dict[str, Any]]:
    result = []
    for i, theorem in enumerate(THEOREMS):
        proc = _run([str(lake), "exe", "merlin-lean-audit", "--axioms", theorem, MODULE], work, f"axioms-{i}")
        axioms = json.loads(proc.stdout.strip().splitlines()[-1])["axioms"]
        if not set(axioms) <= _ALLOWED_AXIOMS:
            raise ValueError(f"unapproved axioms in {theorem}: {axioms}")
        result.append({"theorem": theorem, "axioms": axioms})
    return result


def verify_candidate(
    before: Path,
    after: Path,
    destination: Path,
    *,
    function: str,
    start: int,
    compiler_python: Path,
    suffix: tuple[Path, ...] = (),
    bounds_slices: tuple[tuple[int, int], ...] = (),
    lake: Path | None = None,
) -> dict[str, Any]:
    """Verify actual MLIR snapshots and all ordered suffix stages using the inactive candidate.

    Missing tools, rejected IR, round-trip failures and checker refusals are recorded explicitly.
    No receipt from another interpretation is admitted as a suffix proof. Artifacts are immutable
    snapshots in a fresh work directory, so existing evidence is never overwritten.
    """
    destination.mkdir(parents=True, exist_ok=True)
    work = infrastructure._work(destination)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "active": False,
        "status": "unmeasured",
        "claim_type": "frontend_policy_and_defined_checkpoint",
        "ordinary_source_equivalence": False,
        "whole_function_defined": False,
        "function": function,
        "start": start,
        "configuration": dict(CONFIGURATION),
        "work": str(work.resolve()),
        "artifacts": [],
        "bounds_slices": [{"start": begin, "count": count} for begin, count in bounds_slices],
    }
    lake = lake or Path.home() / ".elan/bin/lake"
    try:
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("checkpoint start must be a nonnegative integer")
        for begin, count in bounds_slices:
            if type(begin) is not int or type(count) is not int or begin < 0 or count <= 0:
                raise ValueError("bounds slices require nonnegative starts and positive counts")
        _run(
            [str(lake), "-f", "FrontendRuntime.lakefile.toml", "build", "merlin-lean-frontend-runtime"],
            work,
            "build",
        )
        identity = interpretation()
        receipt["interpretation"] = identity
        exported = []
        for i, path in enumerate((before, after, *suffix)):
            snapshot = work / f"stage-{i}.mlir"
            original = infrastructure._snapshot(path, snapshot)
            result, error = infrastructure._export_checked(
                snapshot, work / f"round-trip-{i}", m2m_python=compiler_python
            )
            if error:
                raise ValueError(error)
            exported.append(result)
            receipt["artifacts"].append({"original": original, "mlir": _ref(snapshot), "export": _ref(result)})
        receipt["exporter"] = infrastructure._exporter_block(compiler_python) | {"round_trip": "identical"}
        receipt["proofs"] = _proofs(work, lake)
        request = {
            "schema": REQUEST,
            "mode": "boundary",
            "function": function,
            "start": start,
            "limit": CONFIGURATION["limit"],
            "index_bits": CONFIGURATION["index_bits"],
            "before_export": str(exported[0]),
            "after_export": str(exported[1]),
            "suffix_exports": [str(p) for p in exported[2:]],
            "bounds_slices": receipt["bounds_slices"],
        }
        path = work / "request.json"
        path.write_text(json.dumps(request, indent=2) + "\n")
        executable = repo_root() / "out/build/lean/build/bin/merlin-lean-frontend-runtime"
        proc = _run([str(executable), str(path)], work, "candidate")
        answer = json.loads(proc.stdout)
        if answer.get("schema") != "merlin.frontend.runtime.result.v1" or answer.get("theorems") != list(THEOREMS):
            raise ValueError("unexpected candidate checker protocol")
        if interpretation() != identity:
            raise ValueError("interpretation changed during checking")
        for entry in receipt["artifacts"]:
            for key in ("mlir", "export"):
                if _ref(Path(entry[key]["path"])) != entry[key]:
                    raise ValueError("input snapshot changed during checking")
        receipt["request"] = _ref(path)
        receipt["result"] = _ref(work / "candidate.stdout")
        receipt["answer"] = answer
        receipt["status"] = "verified_candidate" if answer.get("accepted") is True else "abstracted"
        receipt["reason"] = (
            "proved under inactive interpretation; runtime admission awaits review"
            if answer.get("accepted") is True
            else "actual Lean checker refused repair, checkpoint, or supplied suffix"
        )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        receipt["reason"] = str(exc)
    receipt_path = work / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def validate_receipt(receipt: dict[str, Any], *, compiler_python: Path | None = None, lake: Path | None = None) -> None:
    """Reject stale source/config/artifacts and frozen or mixed-model records.

    This is evidence-integrity glue (T2), not a cryptographic signature or an additional theorem.
    The authoritative round trips and actual Lean checks are replayed. A result JSON, even
    with freshly recomputed hashes, never supplies an acceptance fact.
    """
    if receipt.get("schema") != SCHEMA or receipt.get("active") is not False:
        raise ValueError("receipt is not an inactive runtime-candidate record")
    if receipt.get("status") != "verified_candidate":
        raise ValueError("candidate was not verified")
    if receipt.get("configuration") != CONFIGURATION or receipt.get("interpretation") != interpretation():
        raise ValueError("stale or mixed interpretation/configuration")
    if (
        receipt.get("claim_type") != "frontend_policy_and_defined_checkpoint"
        or receipt.get("whole_function_defined") is not False
    ):
        raise ValueError("candidate claim was changed")
    if receipt.get("ordinary_source_equivalence") is not False:
        raise ValueError("candidate does not prove raw source equivalence")
    proofs = receipt.get("proofs", [])
    if [p.get("theorem") for p in proofs] != list(THEOREMS) or any(
        not set(p.get("axioms", [])) <= _ALLOWED_AXIOMS for p in proofs
    ):
        raise ValueError("invalid theorem dependencies")
    artifacts = receipt.get("artifacts", [])
    if len(artifacts) < 2:
        raise ValueError("before/after evidence missing")
    for entry in artifacts:
        for key in ("mlir", "export"):
            if _ref(Path(entry[key]["path"])) != entry[key]:
                raise ValueError(f"stale {key} artifact")
    for key in ("request", "result"):
        if _ref(Path(receipt[key]["path"])) != receipt[key]:
            raise ValueError(f"stale {key} artifact")
    request = json.loads(Path(receipt["request"]["path"]).read_text())
    expected = {
        "schema": REQUEST,
        "mode": "boundary",
        "function": receipt.get("function"),
        "start": receipt.get("start"),
        "limit": CONFIGURATION["limit"],
        "index_bits": CONFIGURATION["index_bits"],
        "before_export": artifacts[0]["export"]["path"],
        "after_export": artifacts[1]["export"]["path"],
        "suffix_exports": [entry["export"]["path"] for entry in artifacts[2:]],
        "bounds_slices": receipt.get("bounds_slices"),
    }
    if request != expected:
        raise ValueError("receipt order or checkpoint differs from actual checker request")
    answer = json.loads(Path(receipt["result"]["path"]).read_text())
    if answer != receipt.get("answer") or answer.get("accepted") is not True:
        raise ValueError("receipt differs from executed checker result")
    compiler = compiler_python or Path(os.environ.get("MERLIN_COMPILER_PYTHON", "/missing/compiler"))
    exporter = infrastructure._exporter_block(compiler) | {"round_trip": "identical"}
    if receipt.get("exporter") != exporter:
        raise ValueError("authoritative exporter changed")
    replay = infrastructure._work(Path(receipt["work"]) / "replay")
    for i, entry in enumerate(artifacts):
        source, exported = (Path(entry[key]["path"]) for key in ("mlir", "export"))
        if not infrastructure._exported_from(exported, source):
            raise ValueError("export does not bind the actual MLIR snapshot")
        trip = infrastructure.round_trip_exported(
            source,
            exported,
            replay / f"round-trip-{i}",
            lean_exe=infrastructure.lean_exe(),
            backend="mlir",
            m2m_python=compiler,
        )
        if trip.status != "identical":
            raise ValueError(f"authoritative round-trip replay failed: {trip.status}: {trip.detail}")
    if _proofs(replay, lake or Path.home() / ".elan/bin/lake") != proofs:
        raise ValueError("theorem dependency replay differs")
    executable = repo_root() / "out/build/lean/build/bin/merlin-lean-frontend-runtime"
    proc = _run([str(executable), receipt["request"]["path"]], replay, "candidate-replay")
    if json.loads(proc.stdout) != answer:
        raise ValueError("actual Lean checker replay differs from receipt")
    if interpretation() != receipt["interpretation"]:
        raise ValueError("interpretation changed during replay")
    for entry in artifacts:
        for key in ("mlir", "export"):
            if _ref(Path(entry[key]["path"])) != entry[key]:
                raise ValueError("input snapshot changed during replay")

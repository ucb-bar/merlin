"""Check one recorded compilation boundary and write its receipt (design D10, §8.3).

For a stage directory written by :mod:`.stages` (``before.mlir``, ``after.mlir``, ``meta.json``):

1. export both texts with the authoritative exporter (the pipeline's MLIR via ``torch_mlir``);
2. round-trip both through the Lean decoder/printer (design D9);
3. run ``merlin-lean check-module`` (theorem ``MerlinVerify.Check.moduleCheck_sound``);
4. write ``receipt.json`` (schema ``merlin.lean.receipt.v1``) binding the exact bytes, exporter, Lean
   toolchain, checker binary, ``SPEC_LOCK``, the theorem's audited axioms and the assumptions.

The verdict is ``verified`` only if both round trips are identical, every function verifies and
there is no unverified change outside function bodies. Anything that cannot run is ``unmeasured``
or ``abstracted`` with its reason, never ``verified``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from merlin.common.paths import out_dir, repo_root
from merlin.verify.lean.export import MLIR_EXPORT, export_file, round_trip

THEOREM = "MerlinVerify.Check.moduleCheck_sound"
CHECKER = "MerlinVerify.Check.moduleCheck"
LEDGER = ["A1", "A2", "A3", "A4", "A5", "A6", "T1", "T2", "T3", "T4", "T5", "T6"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def lean_root() -> Path:
    return repo_root() / "merlin" / "lean"


def lean_exe() -> Path:
    return out_dir() / "build" / "lean" / "build" / "bin" / "merlin-lean"


def _manifest_rev(name: str) -> str:
    manifest = json.loads((lean_root() / "lake-manifest.json").read_text())
    return next(p["rev"] for p in manifest["packages"] if p["name"] == name)


_AXIOMS: dict[tuple[str, str], list[str]] = {}


def theorem_axioms(theorem: str, module: str) -> list[str]:
    """``#print axioms`` of ``theorem`` via ``merlin-lean-audit`` (cached per checker binary)."""
    key = (theorem, _sha(lean_exe()))
    if key not in _AXIOMS:
        env = dict(os.environ)
        env["PATH"] = f"{Path.home() / '.elan' / 'bin'}:{env.get('PATH', '')}"
        proc = subprocess.run(["lake", "exe", "merlin-lean-audit", "--axioms", theorem, module],
                              cwd=lean_root(), env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"merlin-lean-audit failed: {proc.stderr.strip()[-500:]}")
        _AXIOMS[key] = json.loads(proc.stdout.strip().splitlines()[-1])["axioms"]
    return _AXIOMS[key]


def _mlir_revision(m2m_python: Path) -> str:
    proc = subprocess.run([str(m2m_python), "-c", "import importlib.metadata as m; print(m.version('torch-mlir'))"],
                          capture_output=True, text=True, check=False)
    return f"torch-mlir {proc.stdout.strip()}" if proc.returncode == 0 else "unknown"


def check_stage(stage: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Check one stage directory and write ``stage/receipt.json``; returns the receipt."""
    meta = json.loads((stage / "meta.json").read_text())
    before, after = stage / "before.mlir", stage / "after.mlir"
    receipt: dict[str, Any] = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": meta["boundary"],
        "claim_type": "refine",
        "checker": CHECKER,
        "theorem": THEOREM,
        "before": _file_ref(before),
        "after": _file_ref(after),
        "exporter": {
            "executable_sha256": _sha(MLIR_EXPORT),
            "mlir_revision": _mlir_revision(m2m_python),
            "round_trip": "not_run",
        },
        "lean": {
            "toolchain": (lean_root() / "lean-toolchain").read_text().strip(),
            "mathlib_rev": _manifest_rev("mathlib"),
            "checker_sha256": _sha(lean_exe()),
            "spec_lock_sha256": _sha(lean_root() / "SPEC_LOCK"),
        },
        "axioms": theorem_axioms(THEOREM, "MerlinVerify.Check.Module"),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    work = stage / "lean"
    trips = [round_trip(p, work / f"rt_{p.stem}", lean_exe=lean_exe(), backend="mlir", m2m_python=m2m_python)
             for p in (before, after)]
    bad = [t for t in trips if t.status != "identical"]
    receipt["exporter"]["round_trip"] = "identical" if not bad else "different"
    if bad:
        receipt.update(verdict="abstracted", reason=f"round trip: {bad[0].status}: {bad[0].detail[:300]}")
        return _write(stage, receipt)
    jb, ja = work / "before.json", work / "after.json"
    export_file(before, jb, backend="mlir", m2m_python=m2m_python)
    export_file(after, ja, backend="mlir", m2m_python=m2m_python)
    proc = subprocess.run([str(lean_exe()), "check-module", str(jb), str(ja)], capture_output=True, text=True,
                          check=False)
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(stage, receipt)
    result = json.loads(proc.stdout)
    receipt["verdict"] = result["verdict"]
    receipt["assumptions"] += [f"ABI change handled by trusted lowering (T4): {c}" for c in result["abi_changes"]]
    if result["verdict"] != "verified":
        reasons = list(result["unverified_changes"])
        if not result["functions_verified"]:
            reasons.insert(0, "a function body did not verify")
        receipt["reason"] = "; ".join(reasons)
    return _write(stage, receipt)


def _write(stage: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt

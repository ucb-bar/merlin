"""Check one recorded compilation boundary, or the support of a capture, and write the receipt
(design D10, §4.5, §8.3).

For a stage directory written by :mod:`.stages` (``before.mlir``, ``after.mlir``, ``meta.json``):

1. export both texts with the authoritative exporter (the pipeline's MLIR via ``torch_mlir``);
2. round-trip both through the Lean decoder/printer (design D9);
3. run ``merlin-lean check-module`` (theorem ``MerlinVerify.Check.moduleCheck_sound``);
4. write ``receipt.json`` (schema ``merlin.lean.receipt.v1``) binding the exact bytes, exporter, Lean
   toolchain, checker binary, ``SPEC_LOCK``, the theorem's audited axioms and the assumptions.

The verdict is ``verified`` only if both round trips are identical, every function verifies and
there is no unverified change outside function bodies. Anything that cannot run is ``unmeasured``
or ``abstracted`` with its reason, never ``verified``.

:func:`check_support` checks one program instead (normally the capture ``S0``): ``merlin-lean
check-support`` (theorem ``MerlinVerify.Check.supportCheck_sound``) proves the function is never
``stuck`` on inputs of its declared types, which turns the chain's conditional refinements into
strict ones (``MerlinVerify.Check.funcRefines_of_support``).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from merlin.common.paths import out_dir, repo_root
from merlin.verify.lean.export import MLIR_EXPORT, ExportError, export_file, round_trip_exported

THEOREM = "MerlinVerify.Check.moduleCheck_sound"
CHECKER = "MerlinVerify.Check.moduleCheck"
SUPPORT_THEOREM = "MerlinVerify.Check.supportCheck_sound"
SUPPORT_CHECKER = "MerlinVerify.Check.supportCheck"
LEDGER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "T1", "T2", "T3", "T4", "T5", "T6"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _snapshot(source: Path, target: Path) -> dict[str, Any]:
    data = source.read_bytes()
    target.write_bytes(data)
    return {"path": str(source.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _work(destination: Path) -> Path:
    work = destination / "lean" / uuid.uuid4().hex
    work.mkdir(parents=True)
    return work


def _export_checked(source: Path, work: Path, *, m2m_python: Path) -> tuple[Path, str | None]:
    exported = source.with_suffix(".json")
    try:
        export_file(source, exported, backend="mlir", m2m_python=m2m_python)
    except ExportError as exc:
        return exported, f"round trip: export_failed: {exc}"
    if not _exported_from(exported, source):
        return exported, "export does not name its source snapshot"
    digest = _sha(exported)
    trip = round_trip_exported(source, exported, work, lean_exe=lean_exe(), backend="mlir",
                               m2m_python=m2m_python)
    if _sha(exported) != digest:
        return exported, "export changed during its round trip"
    return exported, None if trip.status == "identical" else f"round trip: {trip.status}: {trip.detail[:300]}"


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


def _exporter_block(m2m_python: Path) -> dict[str, Any]:
    return {"executable_sha256": _sha(MLIR_EXPORT), "mlir_revision": _mlir_revision(m2m_python),
            "round_trip": "not_run"}


def _lean_block() -> dict[str, Any]:
    return {
        "toolchain": (lean_root() / "lean-toolchain").read_text().strip(),
        "mathlib_rev": _manifest_rev("mathlib"),
        "checker_sha256": _sha(lean_exe()),
        "spec_lock_sha256": _sha(lean_root() / "SPEC_LOCK"),
    }


def _attempt(destination: Path, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Invalidate a previous result before running; serialize attempts and publish failures too.

    An incomplete attempt deliberately is not a proof receipt. Readers must never mistake a
    previously successful receipt for the outcome of a crashed or interrupted rerun.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".receipt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = {"schema": "merlin.lean.attempt.v1", "attempt": uuid.uuid4().hex,
                   "verdict": "unmeasured", "reason": "verification attempt has not completed"}
        _write(destination, pending)
        try:
            return run()
        except Exception as exc:
            return _write(destination, {**pending, "reason": f"{type(exc).__name__}: {exc}"})


def check_stage(stage: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Check a stage, publishing an explicit unmeasured attempt on tool or input failure."""
    return _attempt(stage, lambda: _check_stage(stage, m2m_python=m2m_python))


def _check_stage(stage: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Check one stage directory and write ``stage/receipt.json``; returns the receipt."""
    meta = json.loads((stage / "meta.json").read_text())
    before, after = stage / "before.mlir", stage / "after.mlir"
    work = _work(stage)
    before_snapshot, after_snapshot = work / "before.mlir", work / "after.mlir"
    receipt: dict[str, Any] = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": meta["boundary"],
        # Conditional: the theorem says nothing where the source is unmodelled; strict refinement
        # comes only from `compose_chain` with the capture's support receipt.
        "claim_type": "refine_conditional",
        "checker": CHECKER,
        "theorem": THEOREM,
        "before": _snapshot(before, before_snapshot),
        "after": _snapshot(after, after_snapshot),
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(THEOREM, "MerlinVerify.Check.Module"),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    exports = [_export_checked(p, work / f"rt_{p.stem}", m2m_python=m2m_python)
               for p in (before_snapshot, after_snapshot)]
    bad = [reason for _, reason in exports if reason is not None]
    receipt["exporter"]["round_trip"] = "identical" if not bad else "different"
    if bad:
        receipt.update(verdict="abstracted", reason=bad[0])
        return _write(stage, receipt)
    jb, ja = (path for path, _ in exports)
    receipt["exports"] = {"before": _file_ref(jb), "after": _file_ref(ja)}
    proc = subprocess.run([str(lean_exe()), "check-module", str(jb), str(ja)], capture_output=True, text=True,
                          check=False)
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(stage, receipt)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    receipt["verdict"] = result["verdict"]
    receipt["assumptions"] += [f"ABI change handled by trusted lowering (T4): {c}" for c in result["abi_changes"]]
    if result["verdict"] != "verified":
        reasons = list(result["unverified_changes"])
        if not result["functions_verified"]:
            reasons.insert(0, "a function body did not verify")
        receipt["reason"] = "; ".join(reasons)
    return _finish(stage, receipt)


def check_support(program: Path, function: str, work: Path, *, m2m_python: Path,
                  index_bits: int = 64) -> dict[str, Any]:
    """Check source support; an unavailable check records unmeasured rather than an old verdict."""
    if type(index_bits) is not int or index_bits <= 0:
        _write(work, {"schema": "merlin.lean.attempt.v1", "verdict": "unmeasured",
                      "reason": "index_bits must be a positive integer"})
        raise ValueError("index_bits must be a positive integer")
    return _attempt(work, lambda: _check_support(program, function, work, m2m_python=m2m_python,
                                               index_bits=index_bits))


def _check_support(program: Path, function: str, work: Path, *, m2m_python: Path,
                   index_bits: int) -> dict[str, Any]:
    """Check that ``function`` of ``program`` is inside the proved fragment and write
    ``work/receipt.json``; returns the receipt.

    ``verified`` means the function is never ``stuck`` on inputs of its declared types in any
    context with ``index_bits``-bit ``index`` (``supportCheck_sound``); an op outside the proved
    fragment makes it ``abstracted``, with the first refused op as the reason."""
    if type(index_bits) is not int or index_bits <= 0:
        raise ValueError("index_bits must be a positive integer")
    workdir = _work(work)
    snapshot = workdir / "program.mlir"
    receipt: dict[str, Any] = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": f"support:{program.name}",
        "claim_type": "not_stuck",
        "checker": SUPPORT_CHECKER,
        "theorem": SUPPORT_THEOREM,
        "subject": _snapshot(program, snapshot),
        "function": function,
        "index_bits": index_bits,
        "domain": f"inputs of the declared argument types; contexts with {index_bits}-bit index",
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(SUPPORT_THEOREM, "MerlinVerify.Check.Rulebook"),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    exported, reason = _export_checked(snapshot, workdir / "rt", m2m_python=m2m_python)
    receipt["exporter"]["round_trip"] = "identical" if reason is None else "different"
    if reason is not None:
        receipt.update(verdict="abstracted", reason=reason)
        return _write(work, receipt)
    receipt["exports"] = {"subject": _file_ref(exported)}
    proc = subprocess.run([str(lean_exe()), "check-support", str(exported), function, str(index_bits)],
                          capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(work, receipt)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    if result.get("index_bits") != index_bits:
        receipt.update(verdict="unmeasured", reason="checker used a different index width")
        return _write(work, receipt)
    if result["verdict"] == "verified":
        receipt["verdict"] = "verified"
    else:
        receipt.update(verdict="abstracted", reason=f"outside the proved fragment: {result['first_refusal']}")
    return _finish(work, receipt)


def _exported_from(exported: Path, source: Path) -> bool:
    """The export's recorded source digest and size are those of ``source``'s bytes."""
    doc = json.loads(exported.read_text())
    return doc["source"]["sha256"] == _sha(source) and doc["source"]["bytes"] == source.stat().st_size


def _stale(receipt: dict[str, Any], what: str) -> list[str]:
    """Why a receipt no longer describes the current bytes, checker and lock (empty when fresh)."""
    reasons = []
    refs = {k: receipt[k] for k in ("before", "after", "subject") if k in receipt}
    refs.update({f"export {k}": v for k, v in receipt.get("exports", {}).items()})
    for key, ref in refs.items():
        if ref is None:
            continue
        path = Path(ref.get("path", ""))
        if not path.is_file() or _sha(path) != ref["sha256"]:
            reasons.append(f"{what}: {key} file changed or missing since it was checked")
    if receipt["lean"]["checker_sha256"] != _sha(lean_exe()):
        reasons.append(f"{what}: checked by a different merlin-lean build")
    if receipt["lean"].get("spec_lock_sha256") != _sha(lean_root() / "SPEC_LOCK"):
        reasons.append(f"{what}: checked under a different SPEC_LOCK")
    return reasons


def _finish(destination: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    reasons = _stale(receipt, "verification attempt")
    if receipt["exporter"]["executable_sha256"] != _sha(MLIR_EXPORT):
        reasons.append("exporter changed during verification")
    if reasons:
        receipt.update(verdict="unmeasured", reason="; ".join(reasons))
    return _write(destination, receipt)


CHAIN_THEOREM = "MerlinVerify.Check.chain_refines"


def compose_chain(stages: list[Path], support: Path, function: str, out: Path,
                  handoff: Path | None = None) -> dict[str, Any]:
    """Compose checked boundaries (``stage/receipt.json``, in order) and the capture's support
    receipt into ``out`` (schema ``merlin.lean.chain.v1``); returns the document.

    ``verified`` means: every boundary verified, each boundary's ``before`` is the previous
    ``after`` byte for byte, and the support receipt is about the first ``before`` -- the hypotheses
    of ``chain_refines`` -- and every receipt still describes the current bytes, checker build and
    ``SPEC_LOCK``. With ``handoff`` (the ``handoff.json`` the lowering recorded), the chain must also
    end at the exact bytes handed to the upstream lowering. Anything missing makes the chain
    ``abstracted`` with the reasons."""
    links, reasons, assumptions = [], [], []
    receipts = []
    for stage in stages:
        path = stage / "receipt.json"
        receipt = json.loads(path.read_text())
        if receipt.get("claim_type") != "refine_conditional":
            raise ValueError(f"{path} is not a boundary receipt (claim_type {receipt.get('claim_type')!r})")
        reasons += _stale(receipt, receipt["boundary"])
        receipts.append(receipt)
        links.append({"boundary": receipt["boundary"], "receipt_sha256": _sha(path),
                      "before_sha256": receipt["before"]["sha256"], "after_sha256": receipt["after"]["sha256"],
                      "verdict": receipt["verdict"]})
        if receipt["verdict"] != "verified":
            reasons.append(f"{receipt['boundary']}: {receipt['verdict']}: {receipt.get('reason', '')}")
        assumptions += [a for a in receipt.get("assumptions", []) if a not in assumptions]
    sup = json.loads(support.read_text())
    if sup.get("claim_type") != "not_stuck" or sup.get("function") != function:
        raise ValueError(f"{support} is not a not_stuck receipt for {function}")
    if sup["verdict"] != "verified":
        reasons.append(f"support: {sup['verdict']}: {sup.get('reason', '')}")
    reasons += _stale(sup, "support")
    assumptions += [a for a in sup.get("assumptions", []) if a not in assumptions]
    chained = bool(receipts) and sup["subject"]["sha256"] == receipts[0]["before"]["sha256"] and all(
        a["after"]["sha256"] == b["before"]["sha256"] for a, b in zip(receipts, receipts[1:]))
    handoff_sha = json.loads(handoff.read_text())["sha256"] if handoff is not None else None
    if receipts and handoff_sha is not None and receipts[-1]["after"]["sha256"] != handoff_sha:
        chained = False
        reasons.append("the chain does not end at the bytes handed to the upstream lowering")
    if not receipts:
        reasons.append("no boundaries")
    elif not chained:
        reasons.append("the boundaries do not form a chain from the support subject")
    doc: dict[str, Any] = {
        "schema": "merlin.lean.chain.v1",
        "theorem": CHAIN_THEOREM,
        "function": function,
        "links": links,
        "support": {"receipt_sha256": _sha(support), "subject_sha256": sup["subject"]["sha256"],
                    "verdict": sup["verdict"], "domain": sup["domain"]},
        "chained": chained,
        **({"handoff_sha256": handoff_sha} if handoff_sha is not None else {}),
        "verdict": "verified" if not reasons else "abstracted",
        "assumptions": assumptions,
    }
    if reasons:
        doc["reason"] = "; ".join(reasons)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return doc


def _check_exit(proc: subprocess.CompletedProcess, result: dict[str, Any]) -> None:
    if (proc.returncode, result.get("verdict")) not in {
        (0, "verified"), (1, "abstracted"), (1, "unsupported"),
    }:
        raise ValueError("checker exit status disagrees with its verdict")


def _write(stage: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=stage, prefix=".receipt-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True))
            stream.flush()
            os.replace(temporary, stage / "receipt.json")
        finally:
            temporary.unlink(missing_ok=True)
    return receipt

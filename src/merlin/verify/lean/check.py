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

:func:`check_definedness` proves successful, nonpoison results on explicitly typed, nonpoison
inputs for the supported straight-line fragment. With this separate receipt, :func:`compose_chain`
can establish defined execution as well as refinement of the recorded source. The claim covers
the recorded chain and stated assumptions, not capture, numerical policies, or downstream codegen.

:func:`check_typed_stage` checks a boundary on declared input types at an explicit index width.
Its separate theorem justifies type-dependent map expansion using checked source execution.
Replay preserves this domain; mixed chains require matching argument types, tensor shapes and
index widths. The universal :func:`check_stage` claim and theorem remain unchanged.

The recorded ``prepare-fix-bool-fptosi`` boundary routes to :func:`check_frontend_bool_stage`,
which issues a policy receipt bound to A2's f32 frontend specification and audits the accepted-site
evaluator theorem. A separate policy composition record proves conditional refinement against
Intent at the exact repair checkpoint. The ordinary source-refinement verdict stays abstracted.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.paths import out_dir, repo_root
from merlin.verify.lean.export import MLIR_EXPORT, ExportError, export_file, round_trip_exported

THEOREM = "MerlinVerify.Check.moduleCheck_sound"
CHECKER = "MerlinVerify.Check.moduleCheck"
SUPPORT_THEOREM = "MerlinVerify.Check.supportCheck_sound"
SUPPORT_CHECKER = "MerlinVerify.Check.supportCheck"
LEDGER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "T1", "T2", "T3", "T4", "T5", "T6"]

DEFINED_THEOREM = "MerlinVerify.Check.Definedness.definedCheck_sound"
DEFINED_CHECKER = "MerlinVerify.Check.Definedness.check"
EXTENDED_DEFINED_THEOREM = "MerlinVerify.Check.DefinedExtended.check_sound"
EXTENDED_DEFINED_CHECKER = "MerlinVerify.Check.DefinedExtended.check"


@dataclass(frozen=True)
class _Obligation:
    command: str
    checker: str
    theorem: str
    claim_type: str
    module: str
    defined_inputs: bool = False


_MODULE = _Obligation("check-module", CHECKER, THEOREM, "refine_conditional", "MerlinVerify.Check.Module")
_TYPED_MODULE = _Obligation(
    "check-module-typed",
    "MerlinVerify.Check.TypedModule.check",
    "MerlinVerify.Check.TypedModule.check_sound",
    "refine_conditional",
    "MerlinVerify.Check.TypedModule",
)
_FRONTEND_BOOL = _Obligation(
    "check-frontend-bool",
    "MerlinVerify.Rules.FrontendBoolSimulation.check",
    "MerlinVerify.Rules.FrontendBoolSimulation.check_sound",
    "policy",
    "MerlinVerify.Rules.FrontendBoolSimulation",
)
_FRONTEND_SITE_THEOREM = "MerlinVerify.Rules.FrontendBoolSimulation.ops_sound"
_SUPPORT = _Obligation("check-support", SUPPORT_CHECKER, SUPPORT_THEOREM, "not_stuck", "MerlinVerify.Check.Rulebook")
_DEFINED = _Obligation(
    "check-definedness",
    DEFINED_CHECKER,
    DEFINED_THEOREM,
    "defined",
    "MerlinVerify.Check.DefinedLinalg",
    defined_inputs=True,
)
_EXTENDED_DEFINED = _Obligation(
    "check-definedness-extended",
    EXTENDED_DEFINED_CHECKER,
    EXTENDED_DEFINED_THEOREM,
    "defined",
    "MerlinVerify.Check.DefinedExtended",
    defined_inputs=True,
)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _snapshot(source: Path, target: Path) -> dict[str, Any]:
    data = source.read_bytes()
    target.write_bytes(data)
    target.chmod(0o444)
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
    trip = round_trip_exported(source, exported, work, lean_exe=lean_exe(), backend="mlir", m2m_python=m2m_python)
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
        proc = subprocess.run(
            ["lake", "exe", "merlin-lean-audit", "--axioms", theorem, module],
            cwd=lean_root(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"merlin-lean-audit failed: {proc.stderr.strip()[-500:]}")
        _AXIOMS[key] = json.loads(proc.stdout.strip().splitlines()[-1])["axioms"]
    return _AXIOMS[key]


def _mlir_revision(m2m_python: Path) -> str:
    proc = subprocess.run(
        [str(m2m_python), "-c", "import importlib.metadata as m; print(m.version('torch-mlir'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("cannot identify the authoritative MLIR environment")
    return f"torch-mlir {proc.stdout.strip()}"


def _exporter_block(m2m_python: Path) -> dict[str, Any]:
    return {
        "executable_sha256": _sha(MLIR_EXPORT),
        "mlir_revision": _mlir_revision(m2m_python),
        "round_trip": "not_run",
    }


def _lean_block() -> dict[str, Any]:
    return {
        "toolchain": (lean_root() / "lean-toolchain").read_text().strip(),
        "mathlib_rev": _manifest_rev("mathlib"),
        "checker_sha256": _sha(lean_exe()),
        "spec_lock_sha256": _sha(lean_root() / "SPEC_LOCK"),
    }


def _frontend_review_evidence() -> dict[str, Any]:
    """Bind the user's scoped approval; refuse changed specification/interpretation bytes.

    This records the conversation decision, not SPEC_LOCK admission or approval of other A2
    primitives. The review record distinguishes semantic approval from source-line review.
    """
    path = repo_root() / "out/artifacts/verification/merlin-passes/frontend/f32-policy-approval.json"
    record = json.loads(path.read_text())
    if (
        record.get("schema") != "merlin.frontend.f32-policy-review.v1"
        or record.get("status") != "approved"
        or record.get("answer") != "Approve the documented f32 policy and direct IEEE instance"
    ):
        raise ValueError("documented f32 policy approval is missing")
    expected = {
        "merlin/lean/MerlinVerify/Spec/FrontendIntent.lean",
        *(
            f"merlin/lean/MerlinVerify/Rules/{name}.lean"
            for name in ("FrontendBoolPolicy", "FrontendBoolProgram", "FrontendF32Comparison")
        ),
    }
    if set(record.get("bindings", {})) != expected:
        raise ValueError("f32 policy approval has incomplete bindings")
    for relative, sha in record["bindings"].items():
        if _sha(repo_root() / relative) != sha:
            raise ValueError(f"approved f32 interpretation changed: {relative}")
    return _file_ref(path)


def _frontend_comparison_evidence() -> dict[str, Any]:
    """Bind the approved direct IEEE instance; its proof does not use the host bridge."""
    module = "MerlinVerify.Rules.FrontendF32Comparison"
    proofs = [
        {"theorem": f"{module}.{name}", "axioms": theorem_axioms(f"{module}.{name}", module)}
        for name in ("model_correct", "accepted_site_exec", "model_tensor")
    ]
    if any(not set(proof["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"} for proof in proofs):
        raise ValueError("f32 comparison proof has unapproved axioms")
    return {
        "instance": f"{module}.model(base)",
        "status": "approved f32 policy and direct IEEE instance",
        "proof_source": _file_ref(lean_root() / "MerlinVerify/Rules/FrontendF32Comparison.lean"),
        "proofs": proofs,
        "host_bridge": "unproved: Differential.hostModel uses opaque Float32.toFloat",
    }


def _frontend_program_evidence() -> dict[str, Any]:
    """Bind the source-only A2 interpreter and complete actual-program simulation proof."""
    module = "MerlinVerify.Rules.FrontendBoolSimulation"
    theorem = f"{module}.check_sound_ieee"
    axioms = theorem_axioms(theorem, module)
    if not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}:
        raise ValueError("frontend program proof has unapproved axioms")
    qualification_proofs = [
        {
            "theorem": f"MerlinVerify.Rules.FrontendBoolQualified.{name}",
            "axioms": theorem_axioms(
                f"MerlinVerify.Rules.FrontendBoolQualified.{name}", "MerlinVerify.Rules.FrontendBoolQualified"
            ),
        }
        for name in ("support_sound", "defined_sound")
    ]
    if any(not set(p["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"} for p in qualification_proofs):
        raise ValueError("frontend qualification proof has unapproved axioms")
    return {
        "reference": f"{module}.policyDenote(the exact before module)",
        "theorem": theorem,
        "axioms": axioms,
        "sources": [
            _file_ref(lean_root() / f"MerlinVerify/Rules/{name}.lean")
            for name in (
                "FrontendBoolSimulation",
                "FrontendBoolPolicy",
                "FrontendBoolContext",
                "FrontendBoolQualified",
                "FrontendBoolDefined",
                "FrontendBoolProgram",
            )
        ],
        "conclusion": "exact evaluator outcome equality under the approved direct IEEE f32 instance",
        "review": (
            "user approved documented f32 policy and direct IEEE instance; "
            "other A2 primitives and SPEC_LOCK admission excluded"
        ),
        "execution_support": "separate obligation; exact equality also preserves stuck outcomes",
        "qualification_proofs": qualification_proofs,
    }


def _attempt(destination: Path, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Invalidate a previous result before running; serialize attempts and publish failures too.

    An incomplete attempt deliberately is not a proof receipt. Readers must never mistake a
    previously successful receipt for the outcome of a crashed or interrupted rerun.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".receipt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = {
            "schema": "merlin.lean.attempt.v1",
            "attempt": uuid.uuid4().hex,
            "verdict": "unmeasured",
            "reason": "verification attempt has not completed",
        }
        _write(destination, pending)
        try:
            return run()
        except Exception as exc:
            return _write(destination, {**pending, "reason": f"{type(exc).__name__}: {exc}"})


def _is_frontend_bool_stage(stage: Path) -> bool:
    try:
        return json.loads((stage / "meta.json").read_text()).get("boundary") == "prepare-fix-bool-fptosi"
    except (OSError, ValueError, TypeError):
        return False


def check_stage(stage: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Check a stage, publishing an explicit unmeasured attempt on tool or input failure."""
    if _is_frontend_bool_stage(stage):
        return check_frontend_bool_stage(stage, m2m_python=m2m_python)
    return _attempt(stage, lambda: _check_stage(stage, m2m_python=m2m_python))


def check_typed_stage(stage: Path, *, m2m_python: Path, index_bits: int = 64) -> dict[str, Any]:
    """Check a boundary on declared input types at an explicit index width; preserve its domain."""
    if _is_frontend_bool_stage(stage):
        return check_frontend_bool_stage(stage, m2m_python=m2m_python)
    return _function_attempt(
        stage,
        index_bits,
        lambda: _check_stage(stage, m2m_python=m2m_python, obligation=_TYPED_MODULE, index_bits=index_bits),
    )


def check_frontend_bool_stage(stage: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Check the f32 bool repair against the direct A2 source-program interpretation.

    The general theorem names ``UneCorrect`` explicitly; the approved direct IEEE instance
    discharges it universally. Support/definedness are separate checked obligations.
    """
    return _attempt(stage, lambda: _check_stage(stage, m2m_python=m2m_python, obligation=_FRONTEND_BOOL))


def _check_stage(
    stage: Path, *, m2m_python: Path, obligation: _Obligation = _MODULE, index_bits: int | None = None
) -> dict[str, Any]:
    """Check one stage directory and write ``stage/receipt.json``; returns the receipt."""
    meta = json.loads((stage / "meta.json").read_text())
    before, after = stage / "before.mlir", stage / "after.mlir"
    work = _work(stage)
    before_snapshot, after_snapshot = work / "before.mlir", work / "after.mlir"
    receipt: dict[str, Any] = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": meta["boundary"],
        # The obligation determines the claim: ordinary boundaries are conditional refinements;
        # frontend repairs use a separate policy label and cannot enter the ordinary chain proof.
        "claim_type": obligation.claim_type,
        "checker": obligation.checker,
        "theorem": obligation.theorem,
        "before": _snapshot(before, before_snapshot),
        "after": _snapshot(after, after_snapshot),
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(obligation.theorem, obligation.module),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    if obligation is _FRONTEND_BOOL:
        receipt.update(
            policy="frontend-float-to-bool",
            policy_review=_frontend_review_evidence(),
            spec=_file_ref(lean_root() / "MerlinVerify/Spec/FrontendIntent.lean"),
            site_theorem=_FRONTEND_SITE_THEOREM,
            site_axioms=theorem_axioms(_FRONTEND_SITE_THEOREM, obligation.module),
            comparison_model=_frontend_comparison_evidence(),
            program_policy=_frontend_program_evidence(),
        )
        receipt["assumptions"] += [
            "approved direct IEEE f32 comparison instance discharges UneCorrect",
            "documented f32 source-policy approved by user; SPEC_LOCK admission excluded",
        ]
    if obligation is _TYPED_MODULE:
        receipt.update(index_bits=index_bits, domain=_domain(index_bits))
    exports = [
        _export_checked(p, work / f"rt_{p.stem}", m2m_python=m2m_python) for p in (before_snapshot, after_snapshot)
    ]
    bad = [reason for _, reason in exports if reason is not None]
    receipt["exporter"]["round_trip"] = "identical" if not bad else "different"
    if bad:
        receipt.update(verdict="abstracted", reason=bad[0])
        return _write(stage, receipt)
    jb, ja = (path for path, _ in exports)
    receipt["exports"] = {"before": _file_ref(jb), "after": _file_ref(ja)}
    command = [str(lean_exe()), obligation.command, str(jb), str(ja)]
    if obligation is _TYPED_MODULE:
        command.append(str(index_bits))
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(stage, receipt)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    _result_identity(result, obligation)
    if obligation is _TYPED_MODULE and result.get("index_bits") != index_bits:
        receipt.update(verdict="unmeasured", reason="checker used another typed boundary domain")
        return _write(stage, receipt)
    receipt["verdict"] = result["verdict"]
    if obligation is _FRONTEND_BOOL and result.get("policy") != receipt["policy"]:
        receipt.update(verdict="unmeasured", reason="checker returned another policy")
        return _write(stage, receipt)
    receipt["functions"] = result["functions"]
    if obligation is _FRONTEND_BOOL:
        receipt["execution_qualification"] = result["execution_qualification"]
    receipt["assumptions"] += [f"ABI change handled by trusted lowering (T4): {c}" for c in result["abi_changes"]]
    if result["verdict"] != "verified":
        reasons = list(result["unverified_changes"])
        if not result["functions_verified"]:
            reasons.insert(0, "a function body did not verify")
        receipt["reason"] = "; ".join(reasons)
    return _finish(stage, receipt)


def check_support(
    program: Path, function: str, work: Path, *, m2m_python: Path, index_bits: int = 64
) -> dict[str, Any]:
    """Check source support; an unavailable check records unmeasured rather than an old verdict."""
    return _function_attempt(
        work, index_bits, lambda: _check_support(program, function, work, m2m_python=m2m_python, index_bits=index_bits)
    )


def check_definedness(
    program: Path, function: str, work: Path, *, m2m_python: Path, index_bits: int = 64
) -> dict[str, Any]:
    """Prove successful nonpoison results on typed, nonpoison inputs, or explicitly abstain."""
    return _function_attempt(
        work,
        index_bits,
        lambda: _check_function(
            _EXTENDED_DEFINED, program, function, work, m2m_python=m2m_python, index_bits=index_bits
        ),
    )


def _function_attempt(work: Path, index_bits: int, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if type(index_bits) is not int or index_bits <= 0:
        _write(
            work,
            {
                "schema": "merlin.lean.attempt.v1",
                "verdict": "unmeasured",
                "reason": "index_bits must be a positive integer",
            },
        )
        raise ValueError("index_bits must be a positive integer")
    return _attempt(work, run)


def _check_support(program: Path, function: str, work: Path, *, m2m_python: Path, index_bits: int) -> dict[str, Any]:
    return _check_function(_SUPPORT, program, function, work, m2m_python=m2m_python, index_bits=index_bits)


def _check_function(
    obligation: _Obligation, program: Path, function: str, work: Path, *, m2m_python: Path, index_bits: int
) -> dict[str, Any]:
    """Check one function obligation on the exact export validated by the round trip."""
    workdir = _work(work)
    snapshot = workdir / "program.mlir"
    receipt: dict[str, Any] = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": f"{obligation.command.removeprefix('check-')}:{program.name}",
        "claim_type": obligation.claim_type,
        "checker": obligation.checker,
        "theorem": obligation.theorem,
        "subject": _snapshot(program, snapshot),
        "function": function,
        "index_bits": index_bits,
        "domain": _domain(index_bits, defined_inputs=obligation.defined_inputs),
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(obligation.theorem, obligation.module),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    exported, reason = _export_checked(snapshot, workdir / "rt", m2m_python=m2m_python)
    receipt["exporter"]["round_trip"] = "identical" if reason is None else "different"
    if reason is not None:
        receipt.update(verdict="abstracted", reason=reason)
        return _write(work, receipt)
    receipt["exports"] = {"subject": _file_ref(exported)}
    proc = subprocess.run(
        [str(lean_exe()), obligation.command, str(exported), function, str(index_bits)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(work, receipt)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    _result_identity(result, obligation)
    if result.get("index_bits") != index_bits or result.get("function") != function:
        receipt.update(verdict="unmeasured", reason="checker used a different function or index width")
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
        if not path.is_file() or _file_ref(path) != ref:
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
DEFINED_CHAIN_THEOREM = "MerlinVerify.Check.chain_defined_refines"
TYPED_CHAIN_THEOREM = "MerlinVerify.Check.chain_typed_refines"
TYPED_DEFINED_CHAIN_THEOREM = "MerlinVerify.Check.chain_typed_defined_refines"
EXTENDED_DEFINED_CHAIN_THEOREM = "MerlinVerify.Check.chain_extended_defined_refines"
TYPED_EXTENDED_DEFINED_CHAIN_THEOREM = "MerlinVerify.Check.chain_typed_extended_defined_refines"


def _result_identity(result: dict[str, Any], obligation: _Obligation) -> None:
    expected = (obligation.command.removeprefix("check-"), obligation.theorem, obligation.claim_type)
    actual = tuple(result.get(k) for k in ("checker", "theorem", "claim_type"))
    if actual != expected:
        raise ValueError("checker returned a different proof obligation")


def _validate_receipt(
    receipt: dict[str, Any], *, obligation: _Obligation, function: str, m2m_python: Path, work: Path
) -> list[str]:
    """Re-establish proof premises from the exact recorded exports, never from verdict strings."""
    import jsonschema

    schema = repo_root() / "merlin/contract/schemas/lean_receipt.schema.json"
    try:
        jsonschema.validate(receipt, json.loads(schema.read_text()))
    except jsonschema.ValidationError as exc:
        return [f"receipt does not match the replay schema; regenerate it: {exc.message}"]
    single = obligation in (_SUPPORT, _DEFINED, _EXTENDED_DEFINED)
    expected = (obligation.checker, obligation.theorem, obligation.claim_type)
    if tuple(receipt[k] for k in ("checker", "theorem", "claim_type")) != expected:
        return ["receipt does not establish the required proof obligation"]
    if obligation is _FRONTEND_BOOL:
        if receipt.get("policy") != "frontend-float-to-bool":
            return ["receipt has another frontend policy"]
        if receipt.get("spec") != _file_ref(lean_root() / "MerlinVerify/Spec/FrontendIntent.lean"):
            return ["A2 specification changed since the receipt was issued"]
        site_axioms = theorem_axioms(_FRONTEND_SITE_THEOREM, obligation.module)
        if (
            receipt.get("site_theorem") != _FRONTEND_SITE_THEOREM
            or receipt.get("site_axioms") != site_axioms
            or not set(site_axioms) <= {"propext", "Classical.choice", "Quot.sound"}
        ):
            return ["accepted-site theorem or axiom audit changed"]
        if receipt.get("policy_review") != _frontend_review_evidence():
            return ["f32 policy approval evidence changed"]
        if receipt.get("comparison_model") != _frontend_comparison_evidence():
            return ["f32 comparison evidence changed"]
        if receipt.get("program_policy") != _frontend_program_evidence():
            return ["source-policy program theorem evidence changed"]
    if receipt["verdict"] != "verified":
        return [f"receipt is {receipt['verdict']}: {receipt.get('reason', '')}"]
    reasons = _stale(receipt, "receipt")
    if receipt["lean"] != _lean_block():
        reasons.append("Lean provenance differs from the current checker")
    if receipt["exporter"] != {**_exporter_block(m2m_python), "round_trip": "identical"}:
        reasons.append("exporter identity or successful round trip does not match")
    axioms = theorem_axioms(obligation.theorem, obligation.module)
    if receipt["axioms"] != axioms or not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}:
        reasons.append("receipt axioms do not match the audited theorem")
    keys = ("subject",) if single else ("before", "after")
    if set(receipt.get("exports", {})) != set(keys):
        reasons.append("receipt does not bind all required exports")
    if single:
        domain = _domain(receipt["index_bits"], defined_inputs=obligation.defined_inputs)
        if receipt["function"] != function or receipt["domain"] != domain:
            reasons.append("function receipt has another function or input domain")
    elif obligation is _TYPED_MODULE:
        if receipt.get("domain") != _domain(receipt["index_bits"]):
            reasons.append("typed boundary receipt has another input domain")
    if reasons:
        return reasons
    work.mkdir()
    exports = []
    for key in keys:
        source, exported = work / f"{key}.mlir", work / f"{key}.json"
        for ref, snapshot in ((receipt[key], source), (receipt["exports"][key], exported)):
            if _snapshot(Path(ref["path"]), snapshot) != ref:
                return ["receipt files changed while replaying"]
        if not _exported_from(exported, source):
            return ["export does not name the bound source snapshot"]
        trip = round_trip_exported(
            source, exported, work / f"rt_{key}", lean_exe=lean_exe(), backend="mlir", m2m_python=m2m_python
        )
        if trip.status != "identical":
            return [f"receipt replay round trip: {trip.status}: {trip.detail}"]
        exports.append(exported)
    args = (
        [obligation.command, str(exports[0]), function, str(receipt["index_bits"])]
        if single
        else [obligation.command, *(str(p) for p in exports)]
    )
    if obligation is _TYPED_MODULE:
        args.append(str(receipt["index_bits"]))
    proc = subprocess.run([str(lean_exe()), *args], capture_output=True, text=True, check=False)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    _result_identity(result, obligation)
    if result["verdict"] != "verified":
        return [f"checker replay returned {result['verdict']}"]
    if obligation is _FRONTEND_BOOL and result.get("policy") != receipt["policy"]:
        return ["checker replay returned another frontend policy"]
    if obligation is _FRONTEND_BOOL and receipt.get("execution_qualification") != result.get("execution_qualification"):
        return ["frontend operational qualification does not match replay"]
    assumptions = list(LEDGER)
    if obligation is _FRONTEND_BOOL:
        assumptions += [
            "approved direct IEEE f32 comparison instance discharges UneCorrect",
            "documented f32 source-policy approved by user; SPEC_LOCK admission excluded",
        ]
    if single:
        if result.get("index_bits") != receipt["index_bits"] or result.get("function") != function:
            return ["checker replay has a different input domain"]
    else:
        if obligation is _TYPED_MODULE and result.get("index_bits") != receipt["index_bits"]:
            return ["checker replay has another typed boundary domain"]
        if (
            not result["functions_verified"]
            or result["unverified_changes"]
            or result["functions"] != receipt.get("functions")
            or function not in result["functions"]
        ):
            return ["module receipt does not cover the requested function and all changes"]
        assumptions += [f"ABI change handled by trusted lowering (T4): {c}" for c in result["abi_changes"]]
    if receipt.get("assumptions") != assumptions:
        return ["receipt assumptions do not match the replayed obligation"]
    return _stale(receipt, "replayed receipt")


def _replay(
    receipt: dict[str, Any],
    inputs: dict[str, Path],
    root: Path,
    *,
    obligation: _Obligation,
    function: str,
    m2m_python: Path,
) -> list[str]:
    """Bind the replayed claim to the actual pipeline stage, then recheck its exact exports."""
    for key, source in inputs.items():
        current = _file_ref(source)
        if any(current[k] != receipt[key][k] for k in ("sha256", "bytes")):
            return [f"{key} bytes do not match the receipt"]
    errors = _validate_receipt(receipt, obligation=obligation, function=function, m2m_python=m2m_python, work=root)
    return [f"receipt does not match the replay: {error}" for error in errors]


def _domain(index_bits: int, *, defined_inputs: bool = False) -> str:
    prefix = "nonpoison " if defined_inputs else ""
    return f"{prefix}inputs of the declared argument types; contexts with {index_bits}-bit index"


def _argument_types(export: dict[str, Any], function: str) -> list[dict[str, Any]]:
    """Read the declared domain from a byte-bound authoritative export, including tensor shapes."""
    matches = []
    for op in json.loads(Path(export["path"]).read_text())["module"]["ops"]:
        attrs = {**op["attrs"], **op["props"]}
        if op["name"] == "func.func" and attrs["sym_name"]["value"] == function:
            matches.append(attrs["function_type"]["value"]["inputs"])
    if len(matches) != 1:
        raise ValueError("typed chain requires one unambiguous function signature")
    return matches[0]


def _frontend_policy_composition(
    receipts: list[dict[str, Any]],
    function: str,
    index_bits: int | None,
    failures: list[str],
    handoff_sha: str | None,
) -> dict[str, Any] | None:
    """Compose the checked repair with its ordinary suffix at the exact policy checkpoint.

    Prefix refinements do not transport Intent back to the original capture. The ordinary
    chain verdict is kept separately; source support/definedness cannot replace intent support.
    """
    positions = [i for i, receipt in enumerate(receipts) if receipt.get("claim_type") == "policy"]
    if not positions:
        return None
    reasons = list(failures)
    if len(positions) != 1:
        reasons.append("composition of multiple repair checkpoints is unsupported")
    position = positions[0]
    repair = receipts[position]
    if repair.get("checker") != _FRONTEND_BOOL.checker:
        reasons.append("unsupported repair policy")
    if not all(a["after"]["sha256"] == b["before"]["sha256"] for a, b in zip(receipts, receipts[1:])):
        reasons.append("policy pipeline bytes do not form a contiguous chain")
    if handoff_sha is not None and receipts[-1]["after"]["sha256"] != handoff_sha:
        reasons.append("policy pipeline does not reach the recorded handoff")
    if type(index_bits) is not int or index_bits <= 0:
        reasons.append("invalid policy composition index width")
    if not reasons:
        argument_types = _argument_types(repair["exports"]["before"], function)
        for receipt in receipts[position:]:
            if receipt.get("checker") == _TYPED_MODULE.checker and receipt.get("index_bits") != index_bits:
                reasons.append("policy suffix has a different index width")
            for key in ("before", "after"):
                if _argument_types(receipt["exports"][key], function) != argument_types:
                    reasons.append("policy suffix changes the declared input domain")
    theorem = "MerlinVerify.Rules.FrontendBoolChain.chain_after_source_policy"
    axioms = theorem_axioms(theorem, "MerlinVerify.Rules.FrontendBoolChain")
    if not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}:
        reasons.append("policy composition theorem has unapproved axioms")
    return {
        "claim_type": "policy",
        "policy": "frontend-float-to-bool",
        "theorem": theorem,
        "axioms": axioms,
        "proof_source": _file_ref(lean_root() / "MerlinVerify/Rules/FrontendBoolChain.lean"),
        "anchor_before_sha256": repair["before"]["sha256"],
        "target_after_sha256": receipts[-1]["after"]["sha256"],
        "spec_sha256": repair["spec"]["sha256"],
        "reference": "FrontendBoolSimulation.policyDenote(the exact repair before module)",
        "domain": _domain(index_bits) if type(index_bits) is int and index_bits > 0 else "unavailable",
        **({"index_bits": index_bits} if type(index_bits) is int and index_bits > 0 else {}),
        "prefix_boundaries": position,
        "suffix_boundaries": len(receipts) - position - 1,
        "conclusion": "RefinesC(final, direct A2 source-policy(repair.before)); no original-source refinement",
        "a2_review": (
            f"user approved f32 policy and direct IEEE instance; record sha256 {repair['policy_review']['sha256']}"
        ),
        "a2_comparison": (
            "UneCorrect proved for approved direct IEEE f32 instance; host adapter bridge unused and unproved"
        ),
        "strict_support": "requires support of the direct A2 source-policy denotation, not the original capture",
        "verdict": "abstracted" if reasons else "verified",
        **({"reason": "; ".join(dict.fromkeys(reasons))} if reasons else {}),
    }


def compose_chain(
    stages: list[Path],
    support: Path,
    function: str,
    out: Path,
    handoff: Path | None = None,
    *,
    m2m_python: Path | None = None,
    definedness: Path | None = None,
) -> dict[str, Any]:
    """Replay the proof premises before composing; a receipt is an untrusted hint."""
    pending = {"schema": "merlin.lean.attempt.v1", "verdict": "unmeasured", "reason": "chain replay incomplete"}
    _atomic_json(out, pending)
    try:
        if m2m_python is None:
            from merlin.llvmlower.toolchain import compiler_python

            m2m_python = compiler_python()
        with tempfile.TemporaryDirectory(prefix=".lean-replay-", dir=out.parent) as scratch:
            return _compose_chain(stages, support, function, out, handoff, m2m_python, Path(scratch), definedness)
    except Exception as exc:
        result = {**pending, "reason": f"{type(exc).__name__}: {exc}"}
        _atomic_json(out, result)
        return result


def _compose_chain(
    stages: list[Path],
    support: Path,
    function: str,
    out: Path,
    handoff: Path | None,
    m2m_python: Path,
    work: Path,
    definedness: Path | None,
) -> dict[str, Any]:
    """Compose checked boundaries (``stage/receipt.json``, in order) and the capture's support
    receipt into ``out`` (schema ``merlin.lean.chain.v1``); returns the document.

    ``refinement_verdict == verified`` means: every boundary verified, each boundary's ``before``
    is the previous ``after`` byte for byte, and support covers the first ``before`` -- the hypotheses
    of ``chain_refines`` -- and every receipt still describes the current bytes, checker build and
    ``SPEC_LOCK``. With ``handoff`` (the ``handoff.json`` the lowering recorded), the chain must also
    end at the exact bytes handed to the upstream lowering. Anything missing makes the chain
    ``abstracted`` with the reasons. This refinement permits source UB and poison. The overall
    verdict additionally needs a replayed source-definedness receipt on the same source, function,
    and index width. Its domain explicitly requires typed, nonpoison inputs."""
    links, reasons, assumptions = [], [], []
    receipts = []
    witnesses = []
    live_inputs = []
    for index, stage in enumerate(stages):
        path = stage / "receipt.json"
        receipt, digest = _read_receipt(path)
        witnesses.append((path, digest, receipt))
        obligation = (
            _TYPED_MODULE
            if receipt.get("checker") == _TYPED_MODULE.checker
            else _FRONTEND_BOOL
            if receipt.get("checker") == _FRONTEND_BOOL.checker
            else _MODULE
        )
        inputs = {key: stage / f"{key}.mlir" for key in ("before", "after")}
        live_inputs.append((receipt, inputs))
        reasons += _replay(
            receipt, inputs, work / f"stage_{index}", obligation=obligation, function=function, m2m_python=m2m_python
        )
        receipts.append(receipt)
        links.append(
            {
                "boundary": receipt["boundary"],
                "receipt_sha256": digest,
                "before_sha256": receipt["before"]["sha256"],
                "after_sha256": receipt["after"]["sha256"],
                "verdict": receipt["verdict"],
                "claim_type": receipt["claim_type"],
                **({"policy": receipt["policy"]} if receipt.get("claim_type") == "policy" else {}),
            }
        )
        if receipt["verdict"] != "verified":
            reasons.append(f"{receipt['boundary']}: {receipt['verdict']}: {receipt.get('reason', '')}")
        assumptions += [a for a in receipt.get("assumptions", []) if a not in assumptions]
    boundary_failures = list(reasons)
    sup, support_digest = _read_receipt(support)
    witnesses.append((support, support_digest, sup))
    support_inputs = {"subject": stages[0] / "before.mlir"} if stages else {}
    reasons += _replay(
        sup, support_inputs, work / "support", obligation=_SUPPORT, function=function, m2m_python=m2m_python
    )
    if sup["verdict"] != "verified":
        reasons.append(f"support: {sup['verdict']}: {sup.get('reason', '')}")
    assumptions += [a for a in sup.get("assumptions", []) if a not in assumptions]
    typed = any(r.get("checker") == _TYPED_MODULE.checker for r in receipts)
    if typed and not reasons:
        domain_types = _argument_types(sup["exports"]["subject"], function)
        for receipt in receipts:
            if receipt.get("checker") == _TYPED_MODULE.checker and receipt.get("index_bits") != sup["index_bits"]:
                reasons.append("typed boundary has another index width than the source input domain")
            for key in ("before", "after"):
                if _argument_types(receipt["exports"][key], function) != domain_types:
                    reasons.append("typed chain changes the declared argument types or tensor shapes")
    chained = (
        bool(receipts)
        and sup["subject"]["sha256"] == receipts[0]["before"]["sha256"]
        and all(a["after"]["sha256"] == b["before"]["sha256"] for a, b in zip(receipts, receipts[1:]))
    )
    handoff_sha = None
    if handoff is not None:
        bound, handoff_digest = _read_receipt(handoff)
        handoff_sha = bound["sha256"]
    if receipts and handoff_sha is not None and receipts[-1]["after"]["sha256"] != handoff_sha:
        chained = False
        reasons.append("the chain does not end at the bytes handed to the upstream lowering")
    if not receipts:
        reasons.append("no boundaries")
    elif not chained:
        reasons.append("the boundaries do not form a chain from the support subject")
    if any(r.get("claim_type") == "policy" for r in receipts):
        reasons.append("policy repair cannot be composed by the ordinary source-refinement chain theorem")
    refinement_verdict = "verified" if not reasons else "abstracted"
    domain = (
        _domain(sup["index_bits"], defined_inputs=True)
        if type(sup.get("index_bits")) is int and sup["index_bits"] > 0
        else "unavailable; invalid or missing source input domain"
    )
    defined_info: dict[str, Any] = {
        "verdict": "unmeasured",
        "checker": DEFINED_CHECKER,
        "theorem": DEFINED_THEOREM,
        "domain": domain,
        "reason": "no source-definedness receipt was provided; NotStuck permits UB and poison",
    }
    if definedness is not None:
        defined_receipt, digest = _read_receipt(definedness)
        defined_obligation = (
            _EXTENDED_DEFINED if defined_receipt.get("checker") == EXTENDED_DEFINED_CHECKER else _DEFINED
        )
        failures = _validate_receipt(
            defined_receipt,
            obligation=defined_obligation,
            function=function,
            m2m_python=m2m_python,
            work=work / "definedness",
        )
        if (
            defined_receipt.get("subject", {}).get("sha256") != sup["subject"]["sha256"]
            or defined_receipt.get("index_bits") != sup["index_bits"]
        ):
            failures.append("definedness receipt has another source or index width")
        failures += _receipt_freshness(definedness, digest, defined_receipt)
        defined_info.update(
            verdict="abstracted" if failures else "verified",
            receipt_sha256=digest,
            reason="; ".join(failures),
            checker=defined_obligation.checker,
            theorem=defined_obligation.theorem,
        )
        assumptions += [a for a in defined_receipt.get("assumptions", []) if a not in assumptions]
    if defined_info["verdict"] != "verified":
        reasons.append(f"source definedness: {defined_info['reason']}")
    # The definedness replay may take time too: recheck earlier witnesses before publishing.
    policy_freshness = []
    for path, digest, receipt in witnesses:
        failures = _receipt_freshness(path, digest, receipt)
        if failures:
            refinement_verdict = "abstracted"
            reasons += failures
            policy_freshness += failures
    for receipt, inputs in live_inputs:
        for key, path in inputs.items():
            if (
                not path.is_file()
                or _sha(path) != receipt[key]["sha256"]
                or path.stat().st_size != receipt[key]["bytes"]
            ):
                reasons.append(f"{receipt['boundary']}: live {key} changed during composition")
                policy_freshness.append(f"{receipt['boundary']}: live {key} changed during composition")
                refinement_verdict = "abstracted"
    if handoff is not None:
        if (
            bound.get("schema") != "merlin.lean.handoff.v1"
            or _sha(handoff) != handoff_digest
            or not Path(bound["path"]).is_file()
            or _file_ref(Path(bound["path"])) != {k: bound[k] for k in ("path", "sha256", "bytes")}
        ):
            reasons.append("handoff file changed or is not a valid recorded handoff")
            policy_freshness.append("handoff file changed or is not a valid recorded handoff")
            refinement_verdict = "abstracted"
    if defined_info["checker"] == EXTENDED_DEFINED_CHECKER:
        chain_theorem = TYPED_EXTENDED_DEFINED_CHAIN_THEOREM if typed else EXTENDED_DEFINED_CHAIN_THEOREM
    else:
        chain_theorem = TYPED_DEFINED_CHAIN_THEOREM if typed else DEFINED_CHAIN_THEOREM
    doc: dict[str, Any] = {
        "schema": "merlin.lean.chain.v1",
        "theorem": chain_theorem,
        "refinement_theorem": TYPED_CHAIN_THEOREM if typed else CHAIN_THEOREM,
        **({"index_bits": sup["index_bits"]} if typed else {}),
        "domain": domain,
        "function": function,
        "links": links,
        "support": {
            "receipt_sha256": support_digest,
            "subject_sha256": sup["subject"]["sha256"],
            "verdict": sup["verdict"],
            "domain": sup["domain"],
        },
        "chained": chained,
        "replayed": refinement_verdict == "verified",
        "refinement_verdict": refinement_verdict,
        "source_definedness": defined_info,
        **({"handoff_sha256": handoff_sha} if handoff_sha is not None else {}),
        "verdict": "abstracted" if reasons else "verified",
        "assumptions": assumptions,
    }
    policy_composition = _frontend_policy_composition(
        receipts, function, sup.get("index_bits"), boundary_failures + policy_freshness, handoff_sha
    )
    if policy_composition is not None:
        doc["policy_composition"] = policy_composition
    if reasons:
        doc["reason"] = "; ".join(dict.fromkeys(reasons))
    _atomic_json(out, doc)
    return doc


def _read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _receipt_freshness(path: Path, digest: str, receipt: dict[str, Any]) -> list[str]:
    reasons = _stale(receipt, "chain receipt")
    if _sha(path) != digest:
        reasons.append("receipt changed during composition")
    return reasons


def _check_exit(proc: subprocess.CompletedProcess, result: dict[str, Any]) -> None:
    if (proc.returncode, result.get("verdict")) not in {
        (0, "verified"),
        (1, "abstracted"),
        (1, "unsupported"),
    }:
        raise ValueError("checker exit status disagrees with its verdict")


def _write(stage: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    _atomic_json(stage / "receipt.json", receipt)
    return receipt


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    stage = path.parent
    stage.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=stage, prefix=".receipt-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, sort_keys=True))
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

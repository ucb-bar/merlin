"""Live compiler conformance to the draft bf16/f16 f32-accumulation contract.

This explicitly selected check establishes equality to the source-derived draft
reference and strict qualification of every source contraction. Numerical policy
approval remains abstracted. A saved report cannot deliver compiler input: direct
delivery requires the exact completion issued by a fresh check in this process.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import bf16_policy_gate as policy_gate
from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

SELECTOR = "bf16_f32acc_draft_contract"
CHECK = "compiler_macc_contract"
THEOREM = "MerlinVerify.Check.Bf16F32AccBoundary.check_sound"
ROOTS = {**policy_gate.ROOTS, THEOREM: "MerlinVerify.Check.Bf16F32AccBoundary"}
SCHEMA = "merlin.macc.draft-boundary.v1"


@dataclass(frozen=True, eq=False)
class Completion:
    """Descriptive carrier; only the host's exact live object has authority."""

    report: Path


_LIVE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _params(params):
    if type(params) is not dict or set(params) != {"sdk_bin", "compiler_python", "index_bits"}:
        raise GateUnsupported("draft MACC requires explicit tools/index only; no approval or saved receipt")
    if type(params["index_bits"]) is not int or params["index_bits"] != 64:
        raise GateUnsupported("draft MACC supports index64 only")
    if any(type(params[k]) is not str or not params[k] for k in ("sdk_bin", "compiler_python")):
        raise GateUnsupported("explicit tool paths required")
    return dict(params)


def _check_boundary(before: Path, after: Path, destination: Path, params: dict) -> Completion:
    from .gate_checks import _stage
    from .lean import check as checker

    root = policy_gate.installed_root()
    originals = {"before": policy_gate._ref(before), "after": policy_gate._ref(after)}
    if originals["before"]["sha256"] == originals["after"]["sha256"]:
        raise GateUnsupported("identity pair; no positive draft MACC coverage")
    sources = policy_gate.source_bindings(root)
    tools = policy_gate._tools(params)
    destination.mkdir(parents=True, exist_ok=False)
    audited = policy_gate.audit_candidate(tools, destination, roots=ROOTS, roots_mode=True)

    def auditor(name, module):
        if ROOTS.get(name) != module:
            raise GateRejected("unexpected draft MACC theorem root")
        return audited["axioms"][name]

    stage = _stage(
        destination,
        "merlin.llvmlower.passes_xdsl." + policy_gate.PASS,
        before.read_text(),
        after.read_text(),
    )
    receipt = policy_gate.policy.check_stage(
        stage,
        destination / "receipt",
        m2m_python=Path(params["compiler_python"]),
        index_bits=64,
        theorem_auditor=auditor,
    )
    if receipt.get("verdict") == "unmeasured":
        raise GateUnavailable(receipt.get("reason", "draft MACC checker did not complete"))
    if (
        receipt.get("schema"),
        receipt.get("verdict"),
        receipt.get("checker"),
        receipt.get("theorem"),
        receipt.get("policy"),
        receipt.get("claim_type"),
    ) != (
        "merlin.lean.receipt.v1",
        "abstracted",
        policy_gate.policy.CHECKER,
        policy_gate.policy.THEOREM,
        policy_gate.policy.POLICY,
        "policy",
    ):
        raise GateUnavailable("draft MACC receipt identity/status differs")
    if receipt.get("exporter", {}).get("round_trip") != "identical":
        raise GateUnsupported("draft MACC authoritative export round trip refused")
    if receipt["lean"] != checker._lean_block() or receipt["exporter"] != {
        **checker._exporter_block(Path(params["compiler_python"])),
        "round_trip": "identical",
    }:
        raise GateRejected("draft MACC checker/exporter changed during checking")
    report = policy_gate._json(Path(receipt["cert"]["path"]))
    exports = report["program_exports"]
    if len(exports) != 2 or receipt["exports"] != dict(zip(("before", "after"), exports, strict=True)):
        raise GateRejected("draft MACC exports do not bind exactly one complete boundary")
    for name, original in originals.items():
        bound = receipt[name]
        if any(bound[k] != original[k] for k in ("sha256", "bytes")):
            raise GateRejected("draft MACC receipt differs from actual worker boundary")
        document = policy_gate._json(Path(receipt["exports"][name]["path"]))
        if any(document["source"][k] != original[k] for k in ("sha256", "bytes")):
            raise GateRejected("draft MACC export differs from actual source bytes")
    sites = policy_gate.validate_report(report, policy_gate._json(Path(exports[0]["path"])))
    local = report["pass_boundary"]
    if local.get("positive_boundary_theorem") != THEOREM or local.get("positive_boundary_checked") is not True:
        raise GateRejected("positive nonempty draft MACC boundary theorem did not accept")
    artifacts = [receipt["before"], receipt["after"], receipt["cert"], *exports, *audited["artifacts"]]
    artifacts.append(policy_gate._ref(destination / "receipt/receipt.json"))
    for ref in [*originals.values(), *tools.values(), *artifacts, *audited["project_objects"]]:
        policy_gate._bound(ref)
    if policy_gate.installed_root() != root or policy_gate.source_bindings(root) != sources:
        raise GateRejected("installed draft MACC sources changed during checking")
    if policy_gate._tools(params) != tools:
        raise GateRejected("draft MACC tools changed during checking")
    detail = {
        "schema": SCHEMA,
        "status": "checked_draft_contract",
        "claim": "complete output equals draft f32-accumulation reference under SSA/call laws; "
        "strict initialized-operand qualification of every source contraction",
        "claim_type": "draft_policy_conformance",
        "theorem": THEOREM,
        "theorems": list(ROOTS),
        "domain": policy_gate.DOMAIN,
        "policy_reviewed": False,
        "numerical_policy_verdict": "abstracted",
        "independent_review": "pending",
        "original_source_equivalence": False,
        "whole_model_definedness": False,
        "native_fidelity": False,
        "before": originals["before"],
        "after": originals["after"],
        "index_bits": 64,
        "sites": sites,
        "fusion_premises": report["fusion_premises"],
        "premises": list(policy_gate.policy.ASSUMPTIONS),
        "params": params,
        "tools": tools,
        "sources": sources,
        "artifacts": artifacts,
        "audit": audited,
        "audit_scope": audited["audit_scope"],
        "full_imported_theorem_audit": False,
        "receipt": str(destination / "receipt/receipt.json"),
    }
    path = destination / "draft-boundary.json"
    path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")
    completion = Completion(path.resolve())
    _LIVE[completion] = {
        "pid": os.getpid(),
        "thread": threading.get_ident(),
        "root": root,
        "detail": detail,
        "report": policy_gate._ref(path),
    }
    return completion


def check_boundary(
    before: Path,
    after: Path,
    destination: Path,
    *,
    sdk_bin: Path,
    compiler_python: Path,
    index_bits: int = 64,
) -> Completion:
    """Check the exact actual pair with fresh installed sources and selected tools."""
    params = _params({"sdk_bin": str(sdk_bin), "compiler_python": str(compiler_python), "index_bits": index_bits})
    try:
        return _check_boundary(Path(before), Path(after), Path(destination), params)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise GateUnavailable(f"draft MACC live check unavailable: {exc}") from exc


def validate_completion(completion: Completion) -> dict:
    """Recheck exact live issuance, bound bytes, installed sources and tool freshness."""
    issued = _LIVE.get(completion) if type(completion) is Completion else None
    if issued is None or (issued["pid"], issued["thread"]) != (os.getpid(), threading.get_ident()):
        raise GateRejected("draft MACC delivery requires this host's exact live completion")
    detail = issued["detail"]
    root = policy_gate.installed_root()
    if root != issued["root"] or policy_gate.source_bindings(root) != detail["sources"]:
        raise GateRejected("draft MACC completion has stale installed sources")
    if policy_gate._tools(detail["params"]) != detail["tools"]:
        raise GateRejected("draft MACC completion has stale tools")
    for ref in [issued["report"], detail["before"], detail["after"], *detail["artifacts"], *detail["tools"].values()]:
        policy_gate._bound(ref)
    if policy_gate.project_objects(root) != detail["audit"]["project_objects"]:
        raise GateRejected("draft MACC completion has stale imported project objects")
    return json.loads(json.dumps(detail))


def deliver(completion: Completion) -> str:
    """Release only the checked actual after-MLIR through a current live completion.

    Compiler object/product publication remains the ordinary speculative gate's
    separate custody operation. This function releases no downstream object claim.
    """
    detail = validate_completion(completion)
    after = Path(detail["after"]["path"]).read_text()
    if policy_gate._digest(after.encode()) != detail["after"]["sha256"]:
        raise GateRejected("draft MACC delivered source changed")
    validate_completion(completion)
    return after


def check(job: dict) -> dict:
    """Built-in worker: accepted draft conformance, with numerical approval abstracted."""
    params = _params(job.get("params"))
    completion = check_boundary(
        Path(job["before"]),
        Path(job["after"]),
        Path(job["directory"]) / "macc-contract",
        sdk_bin=Path(params["sdk_bin"]),
        compiler_python=Path(params["compiler_python"]),
        index_bits=params["index_bits"],
    )
    detail = validate_completion(completion)
    deliver(completion)
    return detail


def replay(report: Path, destination: Path, *, sdk_bin: Path, compiler_python: Path) -> Completion:
    """A saved report locates immutable source bytes; a new live check grants authority."""
    policy_gate.installed_root()
    saved = policy_gate._json(Path(report))
    if saved.get("schema") != SCHEMA:
        raise GateUnsupported("unsupported draft MACC replay report")
    for name in ("before", "after"):
        policy_gate._bound(saved[name])
    return check_boundary(
        Path(saved["before"]["path"]),
        Path(saved["after"]["path"]),
        destination,
        sdk_bin=sdk_bin,
        compiler_python=compiler_python,
        index_bits=64,
    )

"""Byte-bound W8A8 policy draft integration, with independent review kept as a hard refusal.

The Lean driver checks the entire actual output against its source-derived reference and
reports source definedness and reference support separately. A successful structural match
does not become a reviewed policy receipt. Ordinary refinement-chain replay must refuse it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from merlin.verify.lean.check import (
    LEDGER,
    _check_exit,
    _export_checked,
    _exporter_block,
    _file_ref,
    _finish,
    _function_attempt,
    _lean_block,
    _snapshot,
    _work,
    _write,
    lean_exe,
    theorem_axioms,
)

CHECKER = "MerlinVerify.Check.PolicyW8A8Module.qualify"
THEOREM = "MerlinVerify.Check.PolicyW8A8Module.qualify_sound"
MODULE = "MerlinVerify.Check.PolicyW8A8Module"


def check_w8a8_draft(stage: Path, *, m2m_python: Path, scope: str = "captured", index_bits: int = 64) -> dict[str, Any]:
    """Run complete draft-policy obligations on exact snapshots; never claim review occurred."""
    return _function_attempt(
        stage,
        index_bits,
        lambda: _check(stage, m2m_python=m2m_python, scope=scope, index_bits=index_bits),
    )


def _check(stage: Path, *, m2m_python: Path, scope: str, index_bits: int) -> dict[str, Any]:
    if scope not in {"all", "captured"}:
        raise ValueError("W8A8 scope must be all or captured")
    meta = json.loads((stage / "meta.json").read_text())
    work = _work(stage)
    before, after = work / "before.mlir", work / "after.mlir"
    receipt = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": meta["boundary"],
        "claim_type": "policy",
        "policy": f"w8a8-reference-draft:{scope}",
        "checker": CHECKER,
        "theorem": THEOREM,
        "index_bits": index_bits,
        "domain": "typed, nonpoison inputs; source definedness and reference support checked separately",
        "before": _snapshot(stage / "before.mlir", before),
        "after": _snapshot(stage / "after.mlir", after),
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(THEOREM, MODULE),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    exports = [_export_checked(p, work / f"rt_{p.stem}", m2m_python=m2m_python) for p in (before, after)]
    refusals = [why for _, why in exports if why is not None]
    receipt["exporter"]["round_trip"] = "different" if refusals else "identical"
    if refusals:
        receipt.update(verdict="abstracted", reason=refusals[0])
        return _write(stage, receipt)
    before_json, after_json = (p for p, _ in exports)
    receipt["exports"] = {"before": _file_ref(before_json), "after": _file_ref(after_json)}
    proc = subprocess.run(
        [str(lean_exe()), "check-w8a8-draft", str(before_json), str(after_json), str(index_bits), scope],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(stage, receipt)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    expected = ("w8a8-draft", THEOREM, "policy", "w8a8-reference-draft", scope, index_bits)
    actual = tuple(
        result.get(key) for key in ("checker", "theorem", "claim_type", "policy", "policy_scope", "index_bits")
    )
    if actual != expected:
        raise ValueError("checker returned a different policy or input domain")
    # Review status is not a candidate-supplied certificate bit. This API is intentionally
    # draft-only; a future reviewed policy needs its own locked declaration and replay path.
    if result.get("policy_reviewed") is not False or result.get("whole_pass_verified") is not False:
        raise ValueError("draft checker attempted to assert reviewed whole-pass verification")
    if result["verdict"] != "abstracted":
        raise ValueError("unreviewed policy cannot produce a verified receipt")
    report = work / "policy-report.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    receipt.update(verdict="abstracted", reason=result["reason"], functions=result["functions"], cert=_file_ref(report))
    receipt["assumptions"] += [f"ABI change handled by trusted lowering (T4): {c}" for c in result["abi_changes"]]
    return _finish(stage, receipt)

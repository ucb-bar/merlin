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

from merlin.verify.lean import policy_numerics
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
CHAIN_CHECKER = "MerlinVerify.Check.PolicyW8A8Chain.check"
CHAIN_THEOREM = "MerlinVerify.Check.PolicyW8A8Chain.check_sound"
CHAIN_MODULE = "MerlinVerify.Check.PolicyW8A8Chain"


def check_w8a8_draft(
    stage: Path, *, m2m_python: Path, scope: str = "captured", index_bits: int = 64, timeout_seconds: int = 300
) -> dict[str, Any]:
    """Run complete draft-policy obligations on exact snapshots; never claim review occurred."""
    return _function_attempt(
        stage,
        index_bits,
        lambda: _check_boundaries(
            stage, [stage], m2m_python=m2m_python, scope=scope, index_bits=index_bits, timeout_seconds=timeout_seconds
        ),
    )


def check_w8a8_chain_draft(
    stages: list[Path],
    destination: Path,
    *,
    m2m_python: Path,
    scope: str = "captured",
    index_bits: int = 64,
    handoff: Path | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Recheck actual policy and suffix boundaries; receipt verdict strings are never premises.

    The first boundary is W8A8. Every subsequent boundary must preserve its predecessor.
    The resulting claim is about P(first.before), with the same typed, nonpoison input domain.
    It remains a draft policy receipt, including for an empty downstream suffix.
    """
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check_boundaries(
            destination,
            stages,
            m2m_python=m2m_python,
            scope=scope,
            index_bits=index_bits,
            chain=True,
            handoff=handoff,
            timeout_seconds=timeout_seconds,
        ),
    )


def _check_boundaries(
    destination: Path,
    stages: list[Path],
    *,
    m2m_python: Path,
    scope: str,
    index_bits: int,
    chain: bool = False,
    handoff: Path | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    if scope not in {"all", "captured"}:
        raise ValueError("W8A8 scope must be all or captured")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("checker timeout must be a positive integer number of seconds")
    if not stages:
        raise ValueError("a W8A8 chain requires a policy boundary")
    if chain and destination.resolve() in {stage.resolve() for stage in stages}:
        raise ValueError("chain destination must differ from the input stage directories")
    checker, theorem, module = (CHAIN_CHECKER, CHAIN_THEOREM, CHAIN_MODULE) if chain else (CHECKER, THEOREM, MODULE)
    work = _work(destination)
    sources, snapshots, links = [], [], []
    for index, stage in enumerate(stages):
        meta = json.loads((stage / "meta.json").read_text())
        for key in ("before", "after"):
            snapshot = work / f"{index}_{key}.mlir"
            sources.append(_snapshot(stage / f"{key}.mlir", snapshot))
            snapshots.append(snapshot)
        links.append({"boundary": meta["boundary"], "before": sources[-2], "after": sources[-1]})
    for previous, following in zip(links, links[1:]):
        if any(previous["after"][key] != following["before"][key] for key in ("sha256", "bytes")):
            raise ValueError("the actual boundaries do not form a byte-identical policy/suffix chain")
    handoff_binding = None
    if handoff is not None:
        record = _snapshot(handoff, work / "handoff.json")
        bound = json.loads((work / "handoff.json").read_text())
        if bound.get("schema") != "merlin.lean.handoff.v1":
            raise ValueError("unsupported handoff record schema")
        subject = {key: bound[key] for key in ("path", "sha256", "bytes")}
        if _file_ref(Path(subject["path"])) != subject:
            raise ValueError("handoff subject changed or does not match its recorded bytes")
        if any(subject[key] != sources[-1][key] for key in ("sha256", "bytes")):
            raise ValueError("policy chain does not end at the actual compiler handoff bytes")
        handoff_binding = {"record": record, "subject": subject}
    # Duplicate joining snapshots are checked above and again for freshness before publishing;
    # the checker needs only each successive program, not duplicate exports of identical bytes.
    programs = [snapshots[0], *snapshots[1::2]]
    receipt = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": "w8a8-policy-suffix" if chain else links[0]["boundary"],
        "claim_type": "policy",
        "policy": f"w8a8-reference-draft:{scope}",
        "checker": checker,
        "theorem": theorem,
        "index_bits": index_bits,
        "domain": "typed, nonpoison inputs; source definedness and reference support checked separately",
        "before": sources[0],
        "after": sources[-1],
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(theorem, module),
        "verdict": "unmeasured",
        "assumptions": list(LEDGER),
    }
    exports = [_export_checked(p, work / f"rt_{p.stem}", m2m_python=m2m_python) for p in programs]
    refusals = [why for _, why in exports if why is not None]
    receipt["exporter"]["round_trip"] = "different" if refusals else "identical"
    if refusals:
        receipt.update(verdict="abstracted", reason=refusals[0])
        return _write(destination, receipt)
    before_json, after_json, *suffix = [p for p, _ in exports]
    export_refs = [_file_ref(p) for p, _ in exports]
    receipt["exports"] = {"before": export_refs[0], "after": export_refs[-1]}
    command = "check-w8a8-chain-draft" if chain else "check-w8a8-draft"
    try:
        proc = subprocess.run(
            [str(lean_exe()), command, str(before_json), str(after_json), str(index_bits), scope, *map(str, suffix)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"W8A8 checker timed out after {timeout_seconds} seconds; no claim established") from exc
    if proc.returncode not in (0, 1):
        receipt.update(verdict="unmeasured", reason=f"checker error: {proc.stderr.strip()[-300:]}")
        return _write(destination, receipt)
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    expected = (command.removeprefix("check-"), theorem, "policy", "w8a8-reference-draft", scope, index_bits)
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
    if chain and len(result["suffix_boundaries"]) != len(stages) - 1:
        raise ValueError("checker did not cover every downstream boundary")
    numerical, numerical_module = policy_numerics.validate(
        result, chain=chain, scope=scope, index_bits=index_bits, suffix_count=len(stages) - 1
    )
    numerical["axioms"] = theorem_axioms(numerical["theorem"], numerical_module)
    report = work / "policy-report.json"
    report.write_text(
        json.dumps(
            {
                **result,
                "boundary_links": links,
                "program_exports": export_refs,
                "handoff": handoff_binding,
                "checker_timeout_seconds": timeout_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    receipt.update(verdict="abstracted", reason=result["reason"], functions=result["functions"], cert=_file_ref(report))
    abi = result["abi_changes"] + result.get("suffix_abi_changes", [])
    receipt["assumptions"] += [f"ABI change handled by trusted lowering (T4): {c}" for c in abi]
    live = [*sources, *export_refs, receipt["cert"]]
    if handoff_binding is not None:
        live += list(handoff_binding.values())
    for ref in live:
        if _file_ref(Path(ref["path"])) != ref:
            raise ValueError("a policy/suffix input or bound report changed during verification")
    return _finish(destination, receipt)

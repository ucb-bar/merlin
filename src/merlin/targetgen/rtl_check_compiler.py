"""Compile endpoint-specific structural checks without assuming an accelerator protocol.

Selected OOT support owns RoCC TRACE assertions and their matching renderer. Shared
code retains taxonomy-driven kernel checks, endpoint routing and evidence projection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from merlin.common.facts_view import interface as _facts_interface

from . import rtl_checks as RC

RENDER_SCHEMA = "rtl-trace-render/v0"


def _endpoint_kind_for(target: str, facts_rec: dict) -> str | None:
    """The target's DERIVED codegen endpoint_kind, resolved the SAME way the capability layer derives it,
    so the FileCheck family routing matches the grader. Order: the residual+facts deriver
    (``manifest_for`` — works for every target that ships a residual, including the ones with no committed
    ``target_contract.yaml``), then the committed contract (``load_capability_manifest``), then straight
    from this run's facts record (``_endpoint_from_facts`` over scoped executable-interface evidence).
    None only when nothing grounds it (caller then treats it as non-RoCC, honestly)."""
    from .capability_manifests import _endpoint_from_facts, _facts_body, manifest_for

    try:
        ek = manifest_for(target).get("endpoint_kind")
        if ek:
            return ek
    except Exception:  # noqa: BLE001 — no residual/derivation for this target
        pass
    body = _facts_body(facts_rec)
    if (
        any(
            itf.get("name") in {"funct_decode_table", "self_hosted_isa"}
            for itf in (body.get("interfaces") or [])
            if isinstance(itf, dict)
        )
        and _endpoint_from_facts(body) is None
    ):
        return "unresolved"  # do not let a family default promote this field-local observation
    try:
        from .target_experiment import load_capability_manifest

        return load_capability_manifest(target).endpoint_kind
    except Exception:  # noqa: BLE001 — no committed contract either
        return _endpoint_from_facts(body)


def _is_rocc_target(target: str, facts_rec: dict) -> bool:
    """Is this a RoCC command-ISA target (the TRACE FileCheck applies) vs a self-hosted-ISA
    (external_backend, the KERNEL FileCheck) / other target? DERIVED, never a target-name test: routes on
    the DERIVED ``endpoint_kind`` (``inline_asm_insn`` == RoCC). We do NOT key on the presence of a
    ``funct_decode_table`` — the mlc icmp-fanout extractor synthesises one for ANY decoder (a self-hosted
    ISA gets a table too). A field-local comparison set never establishes an endpoint."""
    return _endpoint_kind_for(target, facts_rec) == "inline_asm_insn"


def compile_trace_checks(
    facts_rec: dict,
    capsule: dict,
    prefix: str = "TRACE",
    *,
    target: str,
    checks=None,
) -> str | None:
    """Compile assertions through the selected protocol, never by transport alone."""
    checks = RC.selected_checks(target) if checks is None else checks
    result = checks.compile_trace_checks(facts_rec, capsule, prefix)
    if result is not None and (not isinstance(result, str) or not result.strip()):
        raise RC.RtlChecksUnavailable("selected RTL check provider returned malformed trace assertions")
    return result


def _decode_table(facts_rec: dict) -> dict | None:
    """An observed decoder field, not proof of an executable endpoint or full ISA."""
    facts = facts_rec.get("facts", facts_rec)
    return _facts_interface(facts, "funct_decode_table")


def _provenance(facts_rec: dict) -> dict[str, Any]:
    """Per-check-family audit: the derivation source and whether it is genuinely DERIVED (vs a hand
    grouping / a fallback). This is how we answer "did we hand-pick this?" — every emitted check names
    its source, and a family with no resolvable source is reported unavailable, never guessed."""
    dt = _decode_table(facts_rec) or {}
    return {
        "isa_legality": {
            "source": dt.get("method", "funct_decode_table"),
            "derived": dt.get("complete_isa") is True,
            "evidence": dt.get("evidence"),
            "scope": dt.get("scope", "legacy_unverified"),
        },
        "abi_encoding": {
            "source": "funct_decode_table.custom_opcode/funct3",
            "derived": dt.get("custom_opcode") is not None,
        },
    }


def compile_kernel_checks(
    capsule: dict, prefix: str = "KERNEL", facts_rec: dict | None = None, target: str | None = None
) -> str | None:
    """FileCheck lines over a rendered self-hosted kernel instruction stream.

    Complete-word signatures can recognize known classes. Universal legality
    requires an explicitly complete taxonomy; neither an RTL field observation
    nor a partial taxonomy authorizes it. Returns None with no declared op.
    """
    op = (capsule.get("operation") or {}).get("op")
    op = op.lower() if isinstance(op, str) else None
    if op is None:
        return None
    required = list((capsule.get("expected") or {}).get("instruction_classes") or [])
    # Only complete-word decode signatures can recognize emitted words. A
    # field-local equality list cannot, even if every value in that field is
    # known; with no taxonomy the render emits '-' and this assertion is omitted.
    tax = {}
    if target:
        try:
            from . import isa_taxonomy as IT

            tax = IT.taxonomy_for_target(target)
        except Exception:  # noqa: BLE001 — unavailable taxonomy cannot establish universal legality
            tax = {}
    legality_determinable = bool(tax) and tax.get("complete_isa") is True
    L = [
        f"// Kernel checks (op={op}) — declared class coverage and ISA legality",
        f"// {prefix}-DAG: EMPTY_KERNEL no",
    ]  # the kernel must actually emit instructions
    if legality_determinable:
        L.insert(1, f"// {prefix}-DAG: ILLEGAL_OPCODE_COUNT 0{{{{$}}}}")  # every emitted opcode ∈ legal set
    # (1) CLASS COVERAGE: every instruction class the capsule requires (DERIVED per-target in
    # expected.instruction_classes) must actually be EMITTED — the render classifies each word via the
    # ISA-def decode signatures, so a matmul that emitted VADD instead of the MXU matmul fails here
    # (legality alone passes it). Literals are the capsule's own derived class names, not target data.
    for cls in required:
        L.append(f"// {prefix}-DAG: CLASS_PRESENT {cls}{{{{$}}}}")

    return "\n".join(L) + "\n"


def compile_checks(facts_rec: dict, capsule: dict, target: str, *, checks=None) -> dict[str, Any]:
    """Compile the check files for a capsule + a per-family PROVENANCE audit, ENDPOINT-aware and fully
    derived. A RoCC command-ISA target (endpoint ``inline_asm_insn``) gets the trace
    FileCheck over its decoded RoCC stream; a self-hosted-ISA target (``external_backend``)
    gets supported structural checks over its emitted `.word` stream. The RoCC vs self-hosted
    decision is the DERIVED ``endpoint_kind`` (never funct_decode_table presence — the mlc icmp-fanout
    extractor synthesises a table for a self-hosted decoder too, so that would mis-route). Both families
    check the target's actual emitted commands/instructions; neither depends on the agent's (per-run,
    un-derivable) dialect op mnemonics. We emit every check we can ground for the target's endpoint and
    drop the rest, never guessing."""
    endpoint = _endpoint_kind_for(target, facts_rec)
    is_rocc = endpoint == "inline_asm_insn"
    return {
        "schema": "rtl_checks_filecheck/v0",
        "capsule": capsule.get("name"),
        "target": target,
        "trace": compile_trace_checks(facts_rec, capsule, target=target, checks=checks) if is_rocc else None,
        "kernel": (
            compile_kernel_checks(capsule, facts_rec=facts_rec, target=target)
            if endpoint == "external_backend"
            else None
        ),
        "endpoint_status": "resolved" if endpoint in {"inline_asm_insn", "external_backend"} else "unverified",
        "provenance": _provenance(facts_rec),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="Compile RTL facts + capsule -> FileCheck assertion files.")
    ap.add_argument("capsule", help="path to capsule.yaml")
    from .rtl.facts import load_facts

    ap.add_argument("--target", required=True, help="target whose RTL facts + check family to compile")
    ap.add_argument("--facts", default=None, help="facts.json (default: regenerate the target from RTL)")
    a = ap.parse_args(argv)
    facts = json.loads(Path(a.facts).read_text()) if a.facts else load_facts(a.target)
    capsule = yaml.safe_load(Path(a.capsule).read_text())
    cc = compile_checks(facts, capsule, a.target)
    print(f"# capsule={cc['capsule']}\n# --- trace ---\n{cc['trace']}\n# --- kernel ---\n{cc['kernel']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

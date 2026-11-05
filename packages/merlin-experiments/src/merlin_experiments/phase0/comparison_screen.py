"""Withdraw a comparison family whose unfused member the selected declarations refuse.

A comparison group measures a fused member against the standalone parts it replaces. When one of
those parts cannot exist on the target -- the declarations admit its operation only fused onto
another -- the group has nothing to compare against, and writing the part produces a capsule that
verified generation must refuse. The family is withdrawn at derivation instead, with the refusal
recorded, rather than emitted as a group that can never be completed.

A part is screened as what the group declares it to be: one operation, standalone, so it is observed
as composed with nothing and carrying no epilogue. Only a definite refusal withdraws the family; an
unresolved constraint is left to the written program's own screen.
"""

from __future__ import annotations

import hashlib
from typing import Any


def refused_part(variants: list[dict], base: dict, *, software_spec: dict | None, binding) -> dict | None:
    """The first ``part`` member every matching declaration refuses, with the decisions, or ``None``."""
    from merlin.targetgen.semantic_families import from_op
    from merlin.targetgen.software_spec import admit_operation

    if not software_spec:
        return None
    for variant in variants:
        group = variant.get("comparison_group")
        role = group.get("role") if isinstance(group, dict) else None
        op = variant.get("op") or base.get("op")
        family = from_op(op) if op else None
        if role != "part" or family is None:
            continue
        signature: dict[str, Any] = {
            "family": family,
            "operand_dtype": variant.get("operand_dtype") or base.get("operand_dtype") or binding.operand_dtype,
            "accum_dtype": binding.accum_dtype,
            "composed_with": [],
            "epilogues": [],
        }
        if family == "elementwise_map":
            # The unfused part of an epilogue fusion runs in the accumulator domain it would be fused in.
            signature["operand_dtype"] = binding.accum_dtype
        decisions = []
        for declaration in software_spec.get("operations") or []:
            decision = admit_operation(
                {**software_spec, "operations": [declaration]}, str(op), signature, declaration["placement"]
            )
            if "declaration" in decision:
                decisions.append(decision)
        if decisions and all(decision["status"] == "unsupported" for decision in decisions):
            return {
                "member": op,
                "family": family,
                "reason": "; ".join(sorted({decision["reason"] for decision in decisions})),
                "decisions": decisions,
            }
    return None


def refused_emitted_part(entries: list[dict], *, binding, evidence) -> dict | None:
    """Definite refusal of any comparison-group member from its builder's typed interface.

    Every member of a comparison group is needed for the comparison -- a fused member and its parts,
    an island member and its matched no-island member -- so a member the selected declarations
    definitely refuse withdraws the family, whatever its role. This is a pre-write SW/host admission
    screen, not a numerical or execution proof. Unknown admissions remain for the written capsule's
    full screen.
    """
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.semantic_families import from_op

    from .program_admission import account_interface_text, summarize

    for entry in entries:
        group = entry.get("comparison_group")
        if not isinstance(group, dict) or not group.get("role"):
            continue
        if entry.get("source") not in (None, "direct"):
            continue  # A non-builder source has no interface to screen here.
        op = entry.get("op")
        if op not in CS.BUILDERS:
            continue
        try:
            _, selected_binding = CS.entry_binding(entry, binding)
            capsule, mlir = CS.build(entry, selected_binding)
            if "linalg_mlir" in capsule:
                # A builder that emits a source program is screened the way its written capsule
                # will be: per operation, as for a captured application.
                observed = _account_source_program(mlir, str(entry.get("name") or op), binding, evidence)
            else:
                observed = account_interface_text(mlir, target=binding.target, evidence=evidence)
        except (KeyError, TypeError, ValueError):
            # The ordinary writer reports a malformed or unavailable builder;
            # it must not become an inapplicability skip at this early screen.
            continue
        semantic = capsule.get("semantic") or {}
        host_only = semantic.get("must_accelerate") is False and semantic.get("eligible") is False
        decision = summarize(observed, host_only=host_only, scope="builder-emitted typed comparison part")
        if decision["status"] == "unsupported":
            return {
                "member": op,
                "name": entry.get("name"),
                "family": from_op(op),
                "reason": decision["reason"],
                "decisions": decision["decisions"],
                "interface_sha256": hashlib.sha256(mlir.encode()).hexdigest(),
                "basis": "builder_emitted_typed_interface",
            }
    return None


def _account_source_program(mlir: str, name: str, binding, evidence) -> list[dict]:
    import tempfile
    from pathlib import Path

    from .program_admission import account_program

    with tempfile.TemporaryDirectory(prefix="merlin-comparison-part-") as scratch:
        program = Path(scratch) / "capsule.linalg.mlir"
        program.write_text(mlir, encoding="utf-8")
        return account_program(program, name=name, target=binding.target, evidence=evidence)

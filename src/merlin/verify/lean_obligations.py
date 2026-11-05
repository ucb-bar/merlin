"""Mathematical projections after actual-IR structural correspondence has passed.

Totality/effects, SSA argument identity and exact body cloning are checked by
the frozen structural validator. These lemmas model functional observations,
not timing-sensitive environment interactions or physical target state.
"""

from __future__ import annotations

from merlin.transforms.scalar_witness import ScalarRewriteWitness
from merlin.transforms.unrolling import UnrollWitness


def extended_obligations(change) -> tuple[list[str], list[str]]:
    statements, proofs = [], []
    if change.kind in {"scalar_redundancy_elimination", "scalar_loop_invariant_motion"}:
        rules = set()
        for witness in change.certificate.witnesses:
            if not isinstance(witness, ScalarRewriteWitness) or not 1 <= witness.width <= 256:
                raise ValueError("scalar rewrite witness lacks its admitted actual integer operation")
            rules.add(witness.rule)
        for rule in sorted(rules):
            if rule == "cse":
                statements.append(
                    "∀ (α β σ : Type) (producer : α → β) (observe : β → β → σ) (argument : α), "
                    "(let first := producer argument; let second := producer argument; observe first second) = "
                    "(let first := producer argument; observe first first)"
                )
                proofs.append(
                    "by intro α β σ producer observe argument; "
                    "exact Merlin.eliminateDuplicate producer observe argument"
                )
            elif rule == "dce":
                statements.append(
                    "∀ (α β σ : Type) (producer : α → β) (argument : α) (result : σ), "
                    "(let _unused := producer argument; result) = result"
                )
                proofs.append(
                    "by intro α β σ producer argument result; exact Merlin.eliminateUnused producer argument result"
                )
            elif rule == "licm":
                statements.append(
                    "∀ (α β σ : Type) (producer : α → β) (argument : α) (body : β → σ → Nat → σ) "
                    "(iterations : List Nat) (initial : σ), "
                    "iterations.foldl (fun state index => body (producer argument) state index) initial = "
                    "(let invariant := producer argument; iterations.foldl (body invariant) initial)"
                )
                proofs.append(
                    "by intro α β σ producer argument body iterations initial; "
                    "exact Merlin.hoistInvariant producer argument body iterations initial"
                )
            else:
                raise ValueError("scalar rewrite has no admitted mathematical rule")
        return statements, proofs
    if change.kind != "ordered_loop_unrolling":
        raise ValueError("unknown extended mathematical domain")
    for witness in change.certificate.witnesses:
        if not isinstance(witness, UnrollWitness):
            raise ValueError("unrolling requires its complete source geometry")
        groups, factor, rest, count = witness.full_groups, witness.factor, witness.remainder, witness.iterations
        bulk = groups * factor
        statements.extend(
            [
                f"(0 < {factor}) ∧ ({bulk} + {rest} = {count}) ∧ ({rest} < {factor})",
                f"List.range' {witness.lower} {bulk} {witness.step} ++ "
                f"List.range' ({witness.lower} + {witness.step} * {bulk}) {rest} {witness.step} = "
                f"List.range' {witness.lower} {count} {witness.step}",
                f"∀ index : Nat, {factor} * (index / {factor}) + index % {factor} = index",
                "∀ (σ : Type) (body : σ → Nat → σ) (initial : σ) (chunks : List (List Nat)), "
                "chunks.foldl (fun state chunk => chunk.foldl body state) initial = chunks.flatten.foldl body initial",
                f"{witness.upper} + {witness.step} * {factor} ≤ {2 ** (witness.index_bits - 1) - 1}",
            ]
        )
        proofs.extend(
            [
                "by decide",
                f"by exact Merlin.rangeSplit {witness.lower} {bulk} {rest} {witness.step}",
                f"by intro index; exact Merlin.tileIndex index {factor}",
                "by intros; exact Merlin.orderedSegments _ _ _",
                "by decide",
            ]
        )
    return statements, proofs

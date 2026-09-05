"""Does the EMITTED PROGRAM contain the transformation whose cycles are being measured?

A performance member's whole verdict is a cycle count of the program the compiler emitted, and the
task contract the agent is handed states the consequence as a rule: *a lever that does not change the
emitted code is inert by definition*, and any permitted optimization must remain a general compiler
transformation. A member can therefore only be a lever for a transformation the emitted program
actually performs.

THE HOLE THIS CLOSES. The command-buffer ABI lets a buffer DECLARE that one of its tensors is derived
from another (``params.im2col_recipes``: "build ``X_im2col`` by gathering conv windows out of ``X``"),
and the harness then performs that gather in Python before the program runs. For a convolution
member, that hands the compiler a matrix already windowed: the windowing appears nowhere in the
emitted program, costs it no cycles, and by the contract's own rule im2col lowering becomes inert *by
definition* and can never be a lever. The agent would be right to report it as a null. Measured: the
frozen submission lowers every conv exactly this way -- it declares the recipe and emits
``RES_PACK / MATMUL_RESIDENT / COMMIT``, so all four conv performance members were measuring a plain
matmul over a matrix somebody else built.

WHAT THIS IS NOT. The recipe mechanism has a real job and it is not being taken away: a derived
activation must be materialized IDENTICALLY for the reference, the simulator and the device, or the
numeric comparison between the three compares three different stimuli. Stimulus materialization is
legitimate. Discharging the COMPILER's lowering obligation with it is not, and the two are separable
-- a program that emits its own gather still produces the same tensor, and every engine still reads
it from the same command.

THE TWO HONEST LOWERINGS, and the one that is not. The ABI already defines both alternatives: emit
the whole-operation opcode and let the datapath do the windowing, or emit an explicit gather command
that produces the column matrix and then contract over it. Either way the tensor the contraction
consumes is PRODUCED BY A COMMAND. Declaring a recipe and contracting over its result is the third
option, and it is the one that measures a program that never did the work.

So the obligation is stated structurally, and it names no operation, opcode, dtype or target:

    **No operand a performance member's program CONSUMES may be one the HARNESS materialized.**

That is decidable from the buffer alone -- a harness-derived tensor that some command reads and no
command writes -- which is why it can FAIL rather than being documented and hoped for. It is scoped
to members carrying a ``performance`` block; a functional conv capsule is answering a different
question (is the arithmetic right?), and its use of a recipe is untouched by this module.

FAIL CLOSED, twice over. The obligation is enforced for every performance member from the PRESENCE of
its performance block, not from a field a capsule edit could drop -- an obligation you can delete by
deleting its declaration is not an obligation. And a member that declares an obligation id this
module does not implement is ``UNENFORCEABLE``, never ``SATISFIED``: a contract nothing can evaluate
must not read as one that was met.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..runtime.commandbuffer import harness_derived_tensors, operand_flow

__all__ = ["OBLIGATION_ID", "DECLARATION_KEY", "SATISFIED", "VIOLATED", "NOT_APPLICABLE",
           "UNENFORCEABLE", "declared_obligation", "is_performance_member", "assess"]

#: The one obligation this module implements. A capsule may name it explicitly; a performance member
#: that names nothing is held to it anyway (see the module docstring).
OBLIGATION_ID = "operands_materialized_by_emitted_program"

#: Where a capsule states it: ``performance.lowering_obligation``. Beside ``gate``/``regime``, because
#: it is the same kind of thing -- a precondition of the member's claim that is frozen at authoring
#: time rather than decided per run.
DECLARATION_KEY = "lowering_obligation"

SATISFIED = "satisfied"
VIOLATED = "violated"
NOT_APPLICABLE = "not_applicable"
UNENFORCEABLE = "unenforceable"

#: The statuses a grading path must refuse. Exported so the caller cannot spell the pair itself and
#: drift from it (leaving out ``UNENFORCEABLE`` would restore exactly the silent-pass this closes).
REFUSING_STATUSES: frozenset[str] = frozenset({VIOLATED, UNENFORCEABLE})


def is_performance_member(capsule: Any) -> bool:
    """Is this capsule measured for CYCLES? The scope of the obligation, read off the capsule."""
    return isinstance(capsule, Mapping) and isinstance(capsule.get("performance"), Mapping)


def declared_obligation(capsule: Any) -> Mapping[str, Any] | None:
    """The capsule's own ``performance.lowering_obligation`` block, or None when it declares none."""
    if not isinstance(capsule, Mapping):
        return None
    performance = capsule.get("performance")
    if not isinstance(performance, Mapping):
        return None
    declaration = performance.get(DECLARATION_KEY)
    return declaration if isinstance(declaration, Mapping) else None


def _row(status: str, *, detail: str, **extra: Any) -> dict[str, Any]:
    return {"obligation": OBLIGATION_ID, "status": status, "detail": detail, **extra}


def assess(capsule: Any, command_buffer: Any) -> dict[str, Any]:
    """Decide one member's lowering obligation against the buffer its compiler emitted.

    Returns a row carrying the status, the reason in words, and the operands the verdict is about.
    Never raises on a mere violation -- a violation is data the caller records and gates on.
    """
    name = str((capsule or {}).get("name") or "<unnamed>") if isinstance(capsule, Mapping) else "<unnamed>"
    if not is_performance_member(capsule):
        return _row(NOT_APPLICABLE, capsule=name,
                    detail=("this capsule declares no performance block, so no cycle count is "
                            "attributed to its emitted program and it owes no lowering obligation"))

    declaration = declared_obligation(capsule)
    if declaration is not None:
        declared_id = declaration.get("id")
        if declared_id != OBLIGATION_ID:
            return _row(UNENFORCEABLE, capsule=name, declared_id=declared_id,
                        detail=(f"capsule {name!r} declares lowering obligation {declared_id!r}, "
                                f"which no procedure in this build can evaluate. The only obligation "
                                f"implemented is {OBLIGATION_ID!r}. An obligation nothing can decide "
                                f"is refused, never assumed satisfied"))

    if not isinstance(command_buffer, Mapping):
        return _row(UNENFORCEABLE, capsule=name,
                    detail=(f"capsule {name!r} is a performance member and no command buffer was "
                            f"supplied, so nothing could be read about what its program emits"))
    if command_buffer.get("declined"):
        return _row(NOT_APPLICABLE, capsule=name,
                    detail=(f"the backend DECLINED to compile {name!r}; there is no emitted program "
                            f"to hold to an obligation"))

    derived = harness_derived_tensors(command_buffer)
    written, referenced = operand_flow(command_buffer)
    # A recipe whose result NO command touches costs the program nothing and hides no lowering, so the
    # test is on what the program reads, not on the presence of a declaration.
    offenders = sorted(n for n in derived if n in referenced and n not in written)
    if offenders:
        where = ", ".join(f"{n!r} (declared by params.{derived[n]})" for n in offenders[:6])
        return _row(VIOLATED, capsule=name, operands=offenders,
                    recipe_keys=sorted({derived[n] for n in offenders}),
                    detail=(
                        f"performance member {name!r} contracts over operand(s) the HARNESS built for "
                        f"it: {where}. No command in the emitted program produces them, so the "
                        f"transformation that would -- for a windowed operation, the window gather -- "
                        f"appears nowhere in the program whose cycles are being measured, and no "
                        f"compiler change to it can move that number. Emit it: either the "
                        f"whole-operation opcode, whose datapath does the windowing, or an explicit "
                        f"gather command that PRODUCES the matrix the contraction then reads. The "
                        f"stimulus is unchanged either way -- the same tensor, materialized once, from "
                        f"a command every engine executes"))
    return _row(SATISFIED, capsule=name,
                harness_derived=sorted(derived),
                detail=(f"every operand {name!r}'s program consumes is either a declared leaf input or "
                        f"produced by one of its own commands"))

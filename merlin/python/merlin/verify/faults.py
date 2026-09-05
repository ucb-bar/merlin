"""A corpus of seeded compiler faults, and the mutation operators that inject them.

Why seeded faults: a verification layer is justified by what it DETECTS, not by how many tests it
contains. Each operator injects one defect into the ``interface`` module the real pass produced, so a
single mutation propagates consistently to every layer downstream — the structural checks, the SMT
refinement query, and the command buffer the simulator runs. That is the same trick the RTL
pre-screen's mutation demo uses, generalized from one screen to the whole ladder.

Each fault declares which layers *should* catch it. That declaration is an expectation the evaluation
tests, not a result it assumes: a fault nothing detects is reported as an uncovered class rather than
quietly dropped.

The operators mutate IR, never text, so they cannot accidentally produce something unparseable and
score a spurious "detection".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Fault:
    """One seeded defect."""
    name: str
    summary: str
    mutate: Callable[[Any], None]
    #: Layers expected to catch it. Purely documentary — the matrix reports what actually happened.
    expected: tuple[str, ...] = field(default=())


def _func_ops(module, name: str) -> list:
    func = next(o for o in module.walk() if o.name == "func.func")
    return [o for o in func.body.block.ops if o.name == name]


def _block(module):
    return next(o for o in module.walk() if o.name == "func.func").body.block


# --- numeric faults: the arithmetic changes, the op sequence does not -----------------------------

def _miswired_commit(module) -> None:
    commits = _func_ops(module, "interface.commit")
    matmuls = _func_ops(module, "interface.matmul")
    commits[1].operands[0] = matmuls[0].results[0]


def _swapped_operands(module) -> None:
    mm = _func_ops(module, "interface.matmul")[0]
    a, b = mm.operands[0], mm.operands[1]
    mm.operands[0], mm.operands[1] = b, a


def _dropped_activation(module) -> None:
    mm = _func_ops(module, "interface.matmul")
    mm[1].operands[0] = mm[0].operands[0]


# --- structural faults: the op sequence changes, the arithmetic may not ---------------------------

def _dropped_evict(module) -> None:
    for op in _func_ops(module, "interface.resident_evict"):
        _block(module).detach_op(op)
        op.erase()
        return


def _evict_before_last_use(module) -> None:
    blk = _block(module)
    ev = _func_ops(module, "interface.resident_evict")
    mm = _func_ops(module, "interface.matmul")
    if not ev or len(mm) < 2:
        return
    op = ev[0]
    blk.detach_op(op)
    blk.insert_op_before(op, mm[-1])


def _duplicate_pack(module) -> None:
    """Pack the same weight twice: residency was never actually established."""
    packs = _func_ops(module, "interface.resident_pack")
    if not packs:
        return
    clone = packs[0].clone()
    _block(module).insert_op_after(clone, packs[0])


def _duplicate_commit(module) -> None:
    """Commit the same accumulator twice: the commit-once invariant is violated.

    Added 2026-09-04. Every structural fault in this corpus attacked RESIDENCY, so the commit half
    of the structural layer -- the obligation `must_commit_accumulator_before_reuse` compiles to --
    had never been falsified by anything. A check no fault can make fail is not evidence, and the
    detection matrix was reporting a static layer whose commit half was untested.
    """
    commits = _func_ops(module, "interface.commit")
    if not commits:
        return
    clone = commits[0].clone()
    _block(module).insert_op_after(clone, commits[0])


def _commit_after_reuse(module) -> None:
    """Reuse the accumulator in a later matmul BEFORE its commit -- the literal obligation text.

    The op sequence becomes matmul, matmul, commit, commit rather than matmul, commit, matmul,
    commit: the accumulator is read again while still live, which is the defect the obligation
    `must_commit_accumulator_before_reuse` is named for.
    """
    blk = _block(module)
    commits = _func_ops(module, "interface.commit")
    matmuls = _func_ops(module, "interface.matmul")
    if len(commits) < 2 or len(matmuls) < 2:
        return
    # Sink the first commit past the second matmul, so the second use precedes the first commit.
    op = commits[0]
    blk.detach_op(op)
    blk.insert_op_after(op, matmuls[-1])


CORPUS: tuple[Fault, ...] = (
    Fault("miswired_commit",
          "the second commit reads the first accumulator (duplicated / mis-wired commit)",
          _miswired_commit, expected=("formal", "dynamic")),
    Fault("swapped_matmul_operands",
          "A @ W emitted as W @ A",
          _swapped_operands, expected=("formal", "dynamic")),
    Fault("dropped_activation",
          "the second matmul reuses the first activation (an input is dropped)",
          _dropped_activation, expected=("formal", "dynamic")),
    Fault("dropped_evict",
          "the resident weight is never evicted (leaked residency)",
          _dropped_evict, expected=("static",)),
    Fault("evict_before_last_use",
          "the weight is evicted while a later matmul still uses it (use-after-evict)",
          _evict_before_last_use, expected=("static",)),
    Fault("duplicate_pack",
          "the weight is packed twice, so the proven reuse never happens",
          _duplicate_pack, expected=("static",)),
    Fault("duplicate_commit",
          "the same accumulator is committed twice (commit-once violated)",
          _duplicate_commit, expected=("static",)),
    Fault("commit_after_reuse",
          "the accumulator is read by a later matmul before its commit (commit-after-reuse)",
          _commit_after_reuse, expected=("static",)),
)


# --- command-buffer faults ------------------------------------------------------------------------
# These mutate the ARTIFACT A BACKEND EMITS, not our own IR, so they are the negative controls for the
# layer that checks a compiler we did not write. They are kept separate from CORPUS because they take
# a command-buffer dict rather than an xDSL module, and because the layers they exercise differ.


def _cb_swap_matmul_operands(cb) -> None:
    """A @ W emitted as W @ A, at the command-buffer level."""
    ops = next(c for c in cb["commands"] if str(c["opcode"]).startswith("MATMUL"))["operands"]
    ops["lhs"], ops["rhs"] = ops["rhs"], ops["lhs"]


def _cb_crosswire_commit(cb) -> None:
    """The second commit reads the first accumulator: one result is silently duplicated."""
    commits = [c for c in cb["commands"] if c["opcode"] == "COMMIT"]
    commits[1]["operands"]["src"] = commits[0]["operands"]["src"]


def _cb_narrow_output(cb) -> None:
    """Narrow the readout the program did not ask to narrow.

    Worth its own fault because the reference's default is `i8` when `output_dtype` is ABSENT, so an
    emitter that simply forgets the attribute silently saturates every result. A structural check
    cannot see this at all; it is a pure value defect.
    """
    next(c for c in cb["commands"] if c["opcode"] == "COMMIT")["attributes"]["output_dtype"] = "i8"


def _cb_drop_requant(cb) -> None:
    """Drop a requant stage from the epilogue, leaving the result unscaled."""
    for c in cb["commands"]:
        if c["opcode"] == "COMMIT":
            stages = list(c.get("attributes", {}).get("epilogue") or [])
            if "requant" in stages:
                stages.remove("requant")
                c["attributes"]["epilogue"] = stages
                return
    # nothing to drop at this shape; the fault is inapplicable rather than undetectable
    raise ValueError("no requant stage present to drop")


#: Seeded faults over an emitted command buffer. Each takes the buffer dict and mutates it in place.
CB_CORPUS: tuple[Fault, ...] = (
    Fault("cb_swapped_matmul_operands",
          "the backend emitted W @ A where the program specified A @ W",
          _cb_swap_matmul_operands, expected=("compilation",)),
    Fault("cb_crosswire_commit",
          "the second commit reads the first accumulator (a duplicated result)",
          _cb_crosswire_commit, expected=("compilation",)),
    Fault("cb_narrow_output",
          "the readout saturates to i8 where the program declared a wider output",
          _cb_narrow_output, expected=("compilation",)),
)

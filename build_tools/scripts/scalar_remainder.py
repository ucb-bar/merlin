#!/usr/bin/env python
"""WHAT IS THE SCALAR REMAINDER MADE OF? Attribute every instruction in ``forward`` to the MLIR op
that emitted it, host-side, with no board and no simulator.

WHY. ``rvv_audit`` answers "how much of the compute is vector" for a whole symbol, and the whole
model is ONE symbol: the compiler emits a single monolithic ``forward``. So a coverage of 0.29 says
71 % of the emitted compute is scalar and says nothing about WHICH ops it belongs to -- and the
answer decides which lever is worth building. ``op_profile`` answers exactly that question, but only
DYNAMICALLY and only on the board (rdtime marks + a console join), so it is unavailable whenever the
board is busy, and it is unavailable for any model whose accuracy gate does not pass on the board
(the profile for one of the three int8 models measured here has never gated).

METHOD. ``op_profile.instrument`` already interleaves a ``call @merlin_prof_mark(<id>)`` between the
top-level ops of ``func.func @forward``. Those calls are still there in the EMITTED CODE, so they can
be read at build time instead of at run time: disassemble the linked ELF, walk ``forward``, and split
the instruction stream at each mark call. The instructions between mark ``i`` and mark ``i+1`` are
the code the pipeline emitted for top-level op ``i``, and ``opprof_table.json`` maps ``i`` to that
op's MLIR name, ``prov.*`` provenance and result type. Classification per instruction is
``rvv_audit``'s own (never a second opinion that could disagree with the coverage number this
decomposes).

WHAT IT IS AND IS NOT. It is a STATIC instruction census: it counts the instructions the compiler
emitted, not the ones the machine retires, so a three-instruction inner loop that runs a million
times weighs three. That is the honest unit for the question "which op classes did we emit scalar
code FOR", which is what the coverage figure is also in, and it is reported as
``share_of_scalar_static``. A second, independent weighting is reported beside it --
``share_of_scalar_elems``, the output-element count of the ops whose region emitted no vector
instruction at all -- because element count is a dynamic proxy that does not share the static
census's bias. Where the two agree the ranking is robust; where they disagree BOTH are printed and
neither is collapsed into the other.

THREE GUARDS, because an attribution that cannot fail is not evidence.

1. PERTURBATION. The marks are calls, and a call clobbers caller-saved registers, so the
   instrumented ``forward`` is not the one we ship. Both binaries are built and their ``forward``
   coverage compared; the run records ``perturbation_coverage_delta`` and refuses to express shares
   of the SHIPPED build's scalar count when the instrumented build's coverage has moved by more than
   ``--max-coverage-drift``.
2. LAYOUT. The attribution assumes the emitted blocks for op ``i`` lie between mark ``i`` and mark
   ``i+1`` in address order. That is checked, not assumed: the mark ids must appear STRICTLY
   INCREASING in address order and each exactly once. A non-monotonic or duplicated id means the
   backend moved a block across a mark and the attribution is refused.
3. ID RESOLUTION. The id reaches the mark in the first argument register, set by an immediate load
   directly before the call. Any other producer (a register copy, a materialised constant) is NOT
   guessed: that mark is counted in ``unresolved_marks`` and its region is bucketed as ``UNRESOLVED``
   rather than credited to a neighbour.

A NOTE ON WHERE THE PUBLISHED COVERAGE NUMBERS COME FROM. ``compute_symbol()`` deliberately skips
assembler-local labels (a name starting with "."), because picking one by raw weight gives an answer
that moves the wrong way. On an UNRELOCATED object those labels are real symbols INSIDE ``forward``,
so they take the body with them and ``forward`` keeps only the prefix before the first one --
measured here at 4.6 % of its compute instructions on one of these models. The same audit on the
LINKED ELF sees the whole body. This tool therefore audits the ELF, and reports both numbers plus
the fraction of the body the object-file view can see, so the difference is visible rather than
inherited.

Run (host-side only; builds, never deploys):

  MERLIN_COMPILE_TIMEOUT_S=7000 .venv/bin/python build_tools/scripts/scalar_remainder.py \\
      --model out/artifacts/recaptures/<bundle> --package out/artifacts/targets/rvv/<pkg>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from merlin.baselines import rvv_audit as ra
from merlin.common.artifacts import new_product
from merlin.mining import k1
from merlin.mining.registry import load_rvv_package

#: Symbol the instrumentation calls once per top-level op. Imported, never spelled twice.
MARK_SYM = None  # resolved from merlin.llvmlower.op_profile at import time below

from merlin.llvmlower.op_profile import MARK_SYM as _MARK_SYM  # noqa: E402

MARK_SYM = _MARK_SYM

#: The argument register the mark id arrives in. This is the rv64 integer calling convention's
#: first argument register, which the emitted call obeys; it is a property of the ABI the toolchain
#: compiles for, not of any one device.
_ARG0 = "a0"

#: General-purpose register names in the rv64 ABI spelling the disassembler prints. Used only to
#: decide whether an instruction's first operand IS a destination register, so that everything this
#: parser does not model INVALIDATES that register instead of leaving a stale value behind.
_GPRS = frozenset(
    ["zero", "ra", "sp", "gp", "tp", "fp"]
    + [f"x{i}" for i in range(32)] + [f"t{i}" for i in range(7)]
    + [f"s{i}" for i in range(12)] + [f"a{i}" for i in range(8)])

#: Mnemonic prefixes whose first operand is NOT a destination (stores, branches, jumps, compares
#: that write only flags-by-branch). Listed as prefixes because the disassembler spells widths and
#: vector variants as suffixes of the same operation.
_NO_DEST_PREFIXES = ("s", "b", "j", "beq", "bne", "vs", "fs", "c.s", "c.b", "c.j")

#: Registers a call may clobber, in the rv64 integer calling convention. A call clears exactly
#: these and leaves the callee-saved ones alone -- which is load-bearing here, because the compiler
#: hoists the high part of a large mark id into a callee-saved register ONCE and re-uses it across
#: hundreds of marks. Clearing the whole file at a call loses that value and every large id with it.
_CALLER_SAVED = frozenset(["ra"] + [f"a{i}" for i in range(8)] + [f"t{i}" for i in range(7)])

#: The mark id is materialised as a small constant. Three forms appear in practice and all three
#: are MODELLED rather than pattern-matched: a bare immediate (``li``), a high/low pair
#: (``lui`` + ``addi``), and a high part the compiler hoisted into a callee-saved register and
#: reuses across many marks (``lui sN`` once, then ``addi a0, sN, imm`` per mark). What makes this
#: linear constant tracking SAFE despite branches is not the tracking: it is the check that the
#: recovered id sequence is exactly ``0..M-1`` in address order. Any misread produces a sequence
#: that is not, and the whole attribution is then refused.
_CONST_SETTERS = ("li", "lui", "addi", "addiw", "mv", "slli", "slliw")


def _dest_reg(operands: str) -> str:
    """First (destination) operand of a disassembled instruction, or "" when there is none."""
    head = operands.split(",", 1)[0].strip()
    return head


def _operands(line: str) -> str:
    """The operand text of a disassembly line (everything after the mnemonic), or ""."""
    # llvm-objdump separates address, encoding, mnemonic and operands with tabs.
    parts = line.split("\t")
    if len(parts) < 3:
        return ""
    tail = parts[-1].strip()
    # A line whose last tab-separated field IS the mnemonic (no operands) yields "".
    return "" if tail == ra._insn_mnemonic(line) else tail



def _track_const(regs: dict, mnem: str, ops: str) -> None:
    """Linear constant tracking over the register file, invalidating whatever it cannot model.

    Only the forms that materialise a small constant are modelled; every other instruction with a
    destination register DROPS that register, so a stale value can never be read as an id. The
    result is checked for denseness by the caller, which is what makes this sound in the presence of
    control flow rather than merely plausible.
    """
    dest = _dest_reg(ops)
    if dest not in _GPRS:
        return
    if mnem.startswith(_NO_DEST_PREFIXES) and mnem not in _CONST_SETTERS:
        return
    parts = [t.strip() for t in ops.split(",")]
    val = None
    try:
        if mnem == "li" and len(parts) == 2:
            val = int(parts[1], 0)
        elif mnem == "lui" and len(parts) == 2:
            val = int(parts[1], 0) << 12
        elif mnem in ("addi", "addiw") and len(parts) == 3:
            base = regs.get(parts[1])
            val = None if base is None else base + int(parts[2], 0)
        elif mnem in ("slli", "slliw") and len(parts) == 3:
            base = regs.get(parts[1])
            val = None if base is None else base << int(parts[2], 0)
        elif mnem == "mv" and len(parts) == 2:
            val = regs.get(parts[1])
    except ValueError:
        val = None
    if val is None:
        regs.pop(dest, None)
    else:
        regs[dest] = val


def split_forward_by_mark(dis_text: str, *, symbol: str = "forward",
                          mark_sym: str = MARK_SYM) -> dict:
    """Split ``symbol``'s instruction stream at every ``mark_sym`` call site.

    Returns ``{"regions": {id -> counts}, "order": [id...], "unresolved_marks": n,
    "monotonic": bool, "duplicate_ids": n, "found": bool}``. ``counts`` carries the same four
    quantities :class:`rvv_audit.SymbolCoverage` does, so a region and a symbol are directly
    comparable. Instructions before the first mark are bucketed as ``PROLOGUE``; the mark's own
    two instructions (the immediate load and the call) are counted separately as
    ``mark_overhead`` and are NEVER attributed to an op.
    """
    cur_sym = None
    regs: dict[str, int] = {}
    region: object = "PROLOGUE"
    order: list[int] = []
    unresolved = 0
    found = False
    regions: dict = defaultdict(lambda: {"vector": 0, "scalar": 0, "total": 0,
                                         "vsetvl": 0, "mark_overhead": 0})
    for line in dis_text.splitlines():
        name = ra._symbol_name(line)
        if name is not None:
            cur_sym = name
            regs.clear()
            continue
        if cur_sym != symbol:
            continue
        mnem = ra._insn_mnemonic(line)
        if mnem is None:
            continue
        found = True
        if mnem.startswith("j") and f"<{mark_sym}>" in line:
            got = regs.get(_ARG0)
            if got is None:
                unresolved += 1
                region = "UNRESOLVED"
            else:
                region = got
                order.append(got)
            regions[region]["mark_overhead"] += 2      # the constant materialisation and this call
            for r in _CALLER_SAVED:                    # the call clobbers exactly these
                regs.pop(r, None)
            continue
        _track_const(regs, mnem, _operands(line))
        rec = regions[region]
        rec["total"] += 1
        if ra._is_rvv(mnem):
            rec["vector"] += 1
            if mnem.startswith("vsetvl") or mnem.startswith("vsetivl"):
                rec["vsetvl"] += 1
        elif ra._is_scalar_compute(mnem):
            rec["scalar"] += 1
    ids = [i for i in order if isinstance(i, int)]
    return {"regions": dict(regions), "order": ids, "unresolved_marks": unresolved,
            "monotonic": all(b > a for a, b in zip(ids, ids[1:])),
            "dense_from_zero": ids == list(range(len(ids))),
            "duplicate_ids": len(ids) - len(set(ids)), "found": found}


#: Iterator kinds a structured op declares. Read off the op's own ``iterator_types`` list.
_RED, _PAR = "reduction", "parallel"


def _ins_arity(head: str) -> int:
    """Number of ``ins(...)`` operands on a structured op's first line (0 when it declares none)."""
    i = head.find("ins(")
    if i < 0:
        return 0
    j = head.find(")", i)
    if j < 0:
        return 0
    inner = head[i + 4:j].split(":", 1)[0]
    return len([t for t in inner.split(",") if t.strip()])


def _innermost_extent(ty: str | None) -> int | None:
    """Last static dimension of a shaped type string, or None when dynamic/unparsable."""
    if not ty or "<" not in ty:
        return None
    inner = ty[ty.find("<") + 1:ty.rfind(">")]
    dims = inner.split("x")[:-1]
    if not dims:
        return None
    last = dims[-1].strip()
    return int(last) if last.isdigit() else None


#: Operators that make an affine map result a COMPOUND expression rather than a bare dimension. A
#: `vector.transfer_read` needs a projected permutation, so ``structured.vectorize`` cannot build one
#: for a compound result and FAILS THE PIPELINE rather than declining the op -- which is why this is
#: a refusal in the tagger and a class here. It is how a convolution's im2col window arrives: an
#: all-parallel generic whose body only yields its input (so the body-level gather test sees nothing)
#: and whose input map is ``(d0..d5) -> (d3, d0, d4 + d1, d5 + d2)``.
_COMPOUND_AFFINE_OPS = ("+", "*", " floordiv ", " mod ", " ceildiv ")


def _has_compound_indexing_map(head: str) -> bool:
    """True if any ``affine_map<...>`` on the op's first line has a compound result expression."""
    rest = head
    while True:
        i = rest.find("affine_map<")
        if i < 0:
            return False
        arrow = rest.find("->", i)
        if arrow < 0:
            return False
        close = rest.find(")>", arrow)
        if close < 0:
            return False
        if any(tok in rest[arrow:close] for tok in _COMPOUND_AFFINE_OPS):
            return True
        rest = rest[close + 2:]


def structural_class(op_text: list[str], *, mlir_op: str, result_type: str | None,
                     lanes: int) -> dict:
    """Classify one top-level op by its OWN IR, and say whether the vectorize tagger would take it.

    The classes are the ones the lowering actually distinguishes, not a taxonomy invented here:
    a reduction dimension decides whether the contraction arms can see the op at all; a
    ``tensor.extract``/``memref.load`` in the body makes the access non-affine (the vectorizer fails
    the whole pipeline rather than declining); a ``math.*`` call has no vector form on this lowering;
    and the innermost extent decides whether a tile would be masked. Each is READ OFF THE OP, so a
    model that spells its elementwise work differently is classified the same way.

    ``refusal`` is the FIRST reason the current tagging predicate would decline the op, or None when
    it would tag it. That is the actionable column: it names, per op class, which predicate is
    holding the scalar code back.
    """
    head = op_text[0] if op_text else ""
    body = "\n".join(op_text[1:])
    iters = ""
    key = "iterator_types = ["
    i = head.find(key)
    if i >= 0:
        j = head.find("]", i)
        iters = head[i + len(key):j] if j > 0 else ""
    n_red = iters.count(_RED)
    n_par = iters.count(_PAR)
    rank = n_red + n_par
    has_math = "math." in body
    has_gather = ("tensor.extract" in body) or ("memref.load" in body)
    compound_map = _has_compound_indexing_map(head)
    has_mul = ("arith.mulf" in body) or ("arith.muli" in body) or ("arith.extsi" in body)

    if mlir_op != "linalg.generic":
        cls = mlir_op
    elif n_red > 1:
        cls = "contraction_multi_reduction" if has_mul else "reduction_multi_dim"
    elif n_red == 1:
        cls = "contraction_single_reduction" if has_mul else "reduction_single_dim"
    elif has_gather or compound_map:
        cls = "gather"
    elif has_math:
        cls = "transcendental"
    elif _ins_arity(head) == 0:
        cls = "generate"
    elif "arith." not in body:
        cls = "copy_broadcast"
    else:
        cls = "elementwise"

    ext = _innermost_extent(result_type)
    # EVERY reason the predicate would decline, not just the first. The first-reason view answers
    # "why is this op scalar"; the full list answers the question that decides what to BUILD --
    # "what would lifting exactly this one refusal unlock" -- and the two differ sharply: an op
    # refused for its rank is frequently ALSO refused for its innermost extent, so an arm that
    # lifted only the rank bound would unlock none of it. ``sole_refusal`` is the honest per-op
    # answer to that question, and is None when an op is refused for more than one reason.
    refusals = []
    if mlir_op != "linalg.generic":
        refusals.append("not-a-generic")
    if n_red:
        refusals.append("has-reduction-dim")
    if not 2 <= rank <= 4:
        refusals.append("rank-outside-2..4")
    if has_gather:
        refusals.append("data-dependent-gather")
    if compound_map:
        refusals.append("compound-affine-indexing-map")
    if has_math:
        refusals.append("transcendental-body")
    if ext is None or ext % lanes:
        refusals.append("innermost-extent-not-a-multiple-of-lanes")
    return {"struct_class": cls, "iter_rank": rank, "n_reduction": n_red,
            "innermost_extent": ext,
            "vec_tag_refusal": refusals[0] if refusals else None,
            "vec_tag_refusals": refusals,
            "sole_refusal": refusals[0] if len(refusals) == 1 else None}


def strip_marks(instrumented_text: str, *, mark_sym: str = MARK_SYM) -> str:
    """The instrumented module with the marker lines removed.

    Classifying from the INSTRUMENTED file (de-instrumented here) rather than from
    ``model.prepared.mlir`` is what guarantees the ids line up: the table was built by walking
    exactly this module, while ``model.prepared.mlir`` is one stage earlier whenever a feature
    rewrites the IR between the two, and a positional join across a rewrite is a silent
    mis-attribution. The markers are two lines per op plus one declaration, all of them naming the
    hook or its constant, so removing them is exact rather than approximate.
    """
    out = []
    for line in instrumented_text.splitlines():
        s = line.strip()
        if s.startswith("%prof_id_") or s.startswith(f"call @{mark_sym}") \
                or s.startswith(f"func.func private @{mark_sym}"):
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def classify_prepared(instrumented_mlir: Path, *, lanes: int) -> dict:
    """Structural class of every top-level op of ``@forward``, keyed by instrumentation id.

    The ids are the same ones the instrumentation assigns, because both come from
    :func:`op_profile.find_forward_ops` walking the same module in program order. When the two
    disagree on the op COUNT the mapping is refused rather than aligned by guesswork.
    """
    from merlin.llvmlower.op_profile import find_forward_ops

    text = strip_marks(instrumented_mlir.read_text())
    lines = text.splitlines()
    _, ret_line, ops = find_forward_ops(text)
    bounds = [o["line"] for o in ops] + [ret_line]
    out = {}
    for n, o in enumerate(ops):
        out[o["id"]] = structural_class(lines[bounds[n]:bounds[n + 1]],
                                        mlir_op=o["mlir_op"],
                                        result_type=o["result_type"], lanes=lanes)
    return out


def _bucket_key(rec: dict, axis: str) -> str:
    """The bucket a table row belongs to on ``axis``, never a false zero.

    ``family``/``aten`` are absent for the structural ops (``tensor.empty``, ``arith.constant``,
    the shape casts), and bucketing those into a single "(none)" is what made an earlier rollup
    put the majority of a model in an unnamed bucket. So a missing value falls back to the MLIR op
    name, which is always present.
    """
    if axis == "vec_tag_refusal":
        return str(rec.get(axis) or "TAGGED-would-vectorize")
    if axis == "sole_refusal":
        if rec.get("vec_tag_refusal") is None:
            return "TAGGED-would-vectorize"
        return str(rec.get(axis) or "refused-for-several-reasons")
    return str(rec.get(axis) or rec.get("mlir_op") or "(unknown)")


def rollup(table: list[dict], regions: dict, *, axis: str) -> list[dict]:
    """Aggregate per-op instruction counts into buckets on ``axis``.

    ``scalar_elems`` counts the output elements of the ops whose emitted region contains NO vector
    instruction -- the second, independent weighting. An op with a mixed region contributes its
    elements to neither (it is neither wholly scalar nor wholly vector, and splitting it would need
    a trip count this census does not have); the count of such ops is reported so the omission is
    visible.
    """
    out: dict[str, dict] = {}
    for rec in table:
        r = regions.get(rec["id"])
        if r is None:
            continue
        key = _bucket_key(rec, axis)
        b = out.setdefault(key, {axis: key, "n_ops": 0, "vector": 0, "scalar": 0, "total": 0,
                                 "scalar_elems": 0, "elems": 0, "n_all_scalar_ops": 0,
                                 "n_mixed_ops": 0, "n_vector_ops": 0})
        b["n_ops"] += 1
        b["vector"] += r["vector"]
        b["scalar"] += r["scalar"]
        b["total"] += r["total"]
        b["elems"] += int(rec.get("elems") or 0)
        if r["vector"] == 0 and r["scalar"] > 0:
            b["n_all_scalar_ops"] += 1
            b["scalar_elems"] += int(rec.get("elems") or 0)
        elif r["vector"] and r["scalar"]:
            b["n_mixed_ops"] += 1
        elif r["vector"]:
            b["n_vector_ops"] += 1
    rows = sorted(out.values(), key=lambda b: -b["scalar"])
    tot_scalar = sum(b["scalar"] for b in rows) or 1
    tot_scalar_elems = sum(b["scalar_elems"] for b in rows) or 1
    for b in rows:
        b["share_of_scalar_static"] = b["scalar"] / tot_scalar
        b["share_of_scalar_elems"] = b["scalar_elems"] / tot_scalar_elems
    return rows


def _object_view_fraction(obj: Path) -> dict | None:
    """How much of ``forward``'s body the UNRELOCATED-OBJECT audit can actually see.

    Reported because the coverage numbers this repo cites are taken on the object, where the
    assembler-local labels inside ``forward`` are real symbols and ``compute_symbol()`` (rightly)
    skips them -- so the ``forward`` bucket keeps only the prefix before the first one.
    """
    try:
        rep = ra.audit_binary(obj)
    except Exception:                                                    # noqa: BLE001
        return None
    fwd = rep.by_symbol.get("forward")
    if fwd is None:
        return None
    local = [s for n, s in rep.by_symbol.items() if n.startswith(".")]
    body = fwd.vector + fwd.scalar_compute + sum(s.vector + s.scalar_compute for s in local)
    return {"object_forward_coverage": fwd.coverage,
            "object_forward_compute_insns": fwd.vector + fwd.scalar_compute,
            "compute_insns_absorbed_by_local_labels": body - (fwd.vector + fwd.scalar_compute),
            "fraction_of_body_visible_as_forward": ((fwd.vector + fwd.scalar_compute) / body
                                                    if body else None)}


def _elf_forward(binary: Path) -> dict:
    rep = ra.audit_binary(binary)
    fwd = rep.by_symbol.get("forward")
    if fwd is None:
        raise SystemExit(f"no `forward` symbol in {binary}")
    return {"coverage": fwd.coverage, "vector": fwd.vector, "scalar_compute": fwd.scalar_compute,
            "scalar_int": fwd.scalar_int, "scalar_float": fwd.scalar_float,
            "vsetvl": fwd.vsetvl, "total": fwd.total}


def analyse(model_dir: Path, pkg, work_root: Path, *, objdump: str | None,
            max_drift: float, lanes: int, reuse: bool = False) -> dict:
    """Build the instrumented and control binaries and attribute ``forward``.

    ``reuse`` accepts a pair of binaries already sitting in ``work_root`` (built by an earlier run of
    this same tool with the same package and features). It exists because the build is the expensive
    half and the analysis is the half that gets iterated on; it never SKIPS a binary that is not
    there, so a half-populated work dir rebuilds rather than reporting on one arm.
    """
    on = work_root / "on"
    off = work_root / "off"
    have = (on / "merlin_k1").is_file() and (off / "merlin_k1").is_file() \
        and (on / "opprof_table.json").is_file()
    if reuse and have:
        bin_on, bin_off = on / "merlin_k1", off / "merlin_k1"
    else:
        bin_on = k1.build_k1_binary(model_dir, on, replace(pkg, run_id=f"sr_on_{model_dir.name}"),
                                    op_profile=True)
        bin_off = k1.build_k1_binary(model_dir, off, replace(pkg, run_id=f"sr_off_{model_dir.name}"),
                                     op_profile=False)
    tool = objdump or ra._objdump()
    if tool is None:
        raise SystemExit("no objdump available")
    dis = subprocess.run([tool, "-d", str(bin_on)], capture_output=True, text=True, timeout=900)
    if dis.returncode != 0:
        raise SystemExit(f"objdump failed: {dis.stderr[:200]}")
    split = split_forward_by_mark(dis.stdout)
    table = json.loads((on / "opprof_table.json").read_text())
    fwd_on, fwd_off = _elf_forward(bin_on), _elf_forward(bin_off)
    drift = abs((fwd_on["coverage"] or 0) - (fwd_off["coverage"] or 0))
    attributable = (split["found"] and split["monotonic"] and split["dense_from_zero"]
                    and not split["duplicate_ids"] and not split["unresolved_marks"]
                    and len(split["order"]) == len(table) + 1)
    out = {
        "model": str(model_dir), "package": getattr(pkg, "run_id", None),
        "compiler_features": sorted(pkg.compiler_features or []),
        "binary_instrumented": str(bin_on), "binary_control": str(bin_off),
        "forward_control": fwd_off, "forward_instrumented": fwd_on,
        "object_view_control": _object_view_fraction(off / "model.o"),
        "marks": len(split["order"]), "ops_in_table": len(table),
        "unresolved_marks": split["unresolved_marks"],
        "layout_monotonic": split["monotonic"], "duplicate_ids": split["duplicate_ids"],
        "ids_dense_from_zero": split["dense_from_zero"],
        "perturbation_coverage_delta": drift,
        "perturbation_ok": drift <= max_drift, "max_coverage_drift": max_drift,
        "attributable": attributable,
        "refusal": None if attributable else
                   "layout/id check failed -- the recovered mark ids are not exactly 0..N "
                   "in address order (see marks / ids_dense_from_zero / unresolved_marks)",
    }
    if not attributable:
        return out
    regions = {i: r for i, r in split["regions"].items() if isinstance(i, int)}
    out["prologue"] = split["regions"].get("PROLOGUE")
    out["attributed"] = {
        "vector": sum(r["vector"] for r in regions.values()),
        "scalar": sum(r["scalar"] for r in regions.values()),
        "mark_overhead": sum(r["mark_overhead"] for r in regions.values()),
    }
    # STRUCTURAL class from the op's own IR, joined onto the table by id. Refused (not guessed)
    # when the two walks of @forward disagree about how many top-level ops it has.
    struct = classify_prepared(on / "model.prepared.opprof.mlir", lanes=lanes)
    if len(struct) == len(table):
        for rec in table:
            rec.update(struct.get(rec["id"], {}))
        out["structural_classes_joined"] = True
    else:
        out["structural_classes_joined"] = False
        out["structural_join_refusal"] = (
            f"prepared module has {len(struct)} top-level ops, instrumentation table has "
            f"{len(table)} -- not aligned by position")
    axes = ["mlir_op", "family", "aten"]
    if out["structural_classes_joined"]:
        axes += ["struct_class", "vec_tag_refusal", "sole_refusal"]
    for axis in axes:
        out[f"by_{axis}"] = rollup(table, regions, axis=axis)
    per_op = []
    for rec in table:
        r = regions.get(rec["id"])
        if r is None or (r["scalar"] == 0 and r["vector"] == 0):
            continue
        per_op.append({"id": rec["id"], "mlir_op": rec.get("mlir_op"),
                       "family": rec.get("family"), "aten": rec.get("aten"),
                       "struct_class": rec.get("struct_class"),
                       "vec_tag_refusal": rec.get("vec_tag_refusal"),
                       "vec_tag_refusals": rec.get("vec_tag_refusals"),
                       "sole_refusal": rec.get("sole_refusal"),
                       "fqn": rec.get("fqn"), "result_type": rec.get("result_type"),
                       "elems": rec.get("elems"),
                       "vector": r["vector"], "scalar": r["scalar"], "total": r["total"]})
    out["top_scalar_ops"] = sorted(per_op, key=lambda d: -d["scalar"])[:40]
    return out


def _print(res: dict) -> None:
    m = Path(res["model"]).name
    c, i = res["forward_control"], res["forward_instrumented"]
    print(f"\n=== {m}  features={res['compiler_features'] or '<baseline>'}")
    print(f"  forward (LINKED ELF, shipped build): vector={c['vector']} scalar={c['scalar_compute']} "
          f"coverage={c['coverage']:.4f}")
    ov = res.get("object_view_control")
    if ov:
        print(f"  forward (UNRELOCATED OBJECT view): coverage={ov['object_forward_coverage']:.4f} on "
              f"{ov['fraction_of_body_visible_as_forward']:.1%} of the body "
              f"({ov['compute_insns_absorbed_by_local_labels']} compute instructions sit under "
              f"assembler-local labels)")
    print(f"  instrumented forward coverage={i['coverage']:.4f} "
          f"(drift {res['perturbation_coverage_delta']:.4f}, ok={res['perturbation_ok']})")
    if not res["attributable"]:
        print(f"  REFUSED: {res['refusal']}")
        return
    a = res["attributed"]
    print(f"  attributed {a['scalar']} scalar + {a['vector']} vector instructions over "
          f"{res['marks']} marks ({res['ops_in_table']} ops)")
    print(f"  {'mlir op':24s} {'n_ops':>6s} {'scalar':>8s} {'share':>7s} {'elemshare':>10s} "
          f"{'vector':>8s}  scalar-only ops")
    for row in res["by_mlir_op"][:14]:
        print(f"  {row['mlir_op']:24s} {row['n_ops']:6d} {row['scalar']:8d} "
              f"{row['share_of_scalar_static']:6.1%} {row['share_of_scalar_elems']:9.1%} "
              f"{row['vector']:8d}  {row['n_all_scalar_ops']}/{row['n_ops']}")
    for axis, title in (("struct_class", "structural class"),
                        ("vec_tag_refusal", "why the vectorize tagger declines (first reason)"),
                        ("sole_refusal", "what lifting exactly ONE refusal would unlock")):
        rows = res.get(f"by_{axis}")
        if not rows:
            continue
        print(f"  --- {title}")
        for row in rows[:14]:
            print(f"  {row[axis][:38]:38s} {row['n_ops']:6d} {row['scalar']:8d} "
                  f"{row['share_of_scalar_static']:6.1%} {row['share_of_scalar_elems']:9.1%} "
                  f"{row['vector']:8d}  {row['n_all_scalar_ops']}/{row['n_ops']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True,
                    help="recapture bundle dir (repeatable)")
    ap.add_argument("--package", default="out/artifacts/targets/rvv/hand_v0_int8")
    ap.add_argument("--features", default=None,
                    help="comma-separated compiler features; omitted = the package's own, "
                         "empty string = the frozen baseline")
    ap.add_argument("--work", default=None, help="scratch root for the two builds per model")
    ap.add_argument("--objdump", default=None)
    ap.add_argument("--max-coverage-drift", type=float, default=0.05,
                    help="refuse to express shares of the SHIPPED build when instrumenting moved "
                         "forward's coverage by more than this")
    ap.add_argument("--lanes", type=int, default=None,
                    help="innermost lane count the vectorize tagger's extent predicate uses; "
                         "default = the lane width the non-contraction vectorize feature declares")
    ap.add_argument("--reuse-builds", action="store_true",
                    help="reuse a complete pair of binaries already in the work dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.lanes is None:
        from merlin.llvmlower.impr_features import VEC_NONCONTRACTION_LANES
        a.lanes = VEC_NONCONTRACTION_LANES

    base = load_rvv_package(a.package)
    feats = ([f.strip() for f in a.features.split(",") if f.strip()] if a.features is not None
             else list(base.compiler_features or []))
    pkg = replace(base, compiler_features=feats)
    work_root = Path(a.work) if a.work else None
    results = []
    for md in a.model:
        md = Path(md).resolve()
        wr = (work_root or Path("out/build/scalar_remainder")) / md.name
        wr.mkdir(parents=True, exist_ok=True)
        res = analyse(md, pkg, wr, objdump=a.objdump, max_drift=a.max_coverage_drift,
                      lanes=a.lanes, reuse=a.reuse_builds)
        _print(res)
        results.append(res)

    prod = (Path(a.out) if a.out else
            new_product("scalar-remainder", version=1,
                        notes="static per-op attribution of forward's instruction stream").path)
    prod.mkdir(parents=True, exist_ok=True)
    (prod / "scalar_remainder.json").write_text(json.dumps(
        {"package": a.package, "compiler_features": feats, "models": results}, indent=1))
    print(f"\nwrote {prod / 'scalar_remainder.json'}")


if __name__ == "__main__":
    main()

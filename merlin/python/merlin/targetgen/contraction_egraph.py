"""An e-class over a REAL contraction: the vector lowering and the matrix-unit call, side by side.

:mod:`persistent_equivalence` demonstrates the mechanism on synthetic ops — one ``test.TestOp`` per
candidate, carrying a unit name and a cost. That is enough to show that extraction reads a decision out of
an e-graph, and not enough to show it can decide anything a compiler would act on: nothing in that graph is
an implementation, so nothing can be *emitted* from the choice.

This builds the same e-class over the actual IR. Both alternatives are real operations with real types:

- the contraction itself — the ``linalg.generic`` the int8 rewrite produced, region and all, which is what
  the vector path lowers; and
- ``func.call @merlin_opu_gemm_i8_<n>`` — the call :mod:`llvmlower.passes_opu` emits, resolving to the
  certified microkernel.

Both yield the same ``tensor<MxNxi32>``, which is why an ``equivalence.class`` over them is type-correct,
and why the extracted function is *directly* the compile path's answer: whichever survives is the
implementation that runs.

**What this adds over the synthetic version, precisely.** The decision is now expressed in the same terms as
the code that will run it, so a selector can be READ OFF the extraction instead of being computed beside it
and hoped to agree. :func:`egraph_selector` returns exactly the ``select`` callable
``passes_opu.rewrite_contractions_to_opu`` takes, so the routing decision is made by minimisation over a
graph rather than by a threshold.

**What it still does not establish.** With one e-class and no rewrite rules, extraction is an argmin over
the costs it is given. That is a mechanism, not a win, and ``persistent_equivalence.HYPOTHESES`` keeps
saying so. What it does establish — and what the synthetic graph could not — is that the alternatives are
real enough to emit from, which is the precondition for any downstream pass changing the decision.

**A tie resolves to the first alternative in the class.** That is declaration order, and here the
contraction is added first, so a tie leaves the workload on the vector path. That direction is deliberate:
the vector path is the control, and a coin-flip should not move work onto a unit whose advantage is unproven.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ContractionChoice", "MATRIX", "VECTOR", "build_contraction_egraph", "egraph_selector",
           "extract_contraction_choice", "contraction_to_call_pattern", "for_rewrite", "measured_cost_of",
           "saturate_contraction"]

#: The two alternatives, named so a caller never compares against a spelling.
VECTOR = "vector"
MATRIX = "matrix"


@dataclass(frozen=True)
class ContractionChoice:
    """Which implementation of one contraction survived extraction, and what it cost to decide."""

    chosen: str | None
    costs: dict[str, int] = field(default_factory=dict)
    m: int = 0
    n: int = 0
    k: int = 0
    build_seconds: float = 0.0
    extract_seconds: float = 0.0
    gap: str | None = None

    @property
    def on_matrix_unit(self) -> bool:
        return self.chosen == MATRIX

    @property
    def total_seconds(self) -> float:
        return self.build_seconds + self.extract_seconds

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen, "costs": dict(self.costs),
                "m": self.m, "n": self.n, "k": self.k,
                "build_seconds": round(self.build_seconds, 6),
                "extract_seconds": round(self.extract_seconds, 6),
                "total_seconds": round(self.total_seconds, 6), "gap": self.gap}


def _context():
    """A context carrying every dialect the real IR uses.

    The synthetic graph needed only builtin/equivalence/test. Real contraction IR carries ``linalg``,
    ``arith``, ``tensor`` and ``func``, and a context missing one of them fails to verify the module rather
    than mis-extracting from it — loud, but only if the dialects are loaded in the first place.
    """
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, equivalence, func, linalg, tensor

    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, equivalence.Equivalence, func.Func, linalg.Linalg,
                    arith.Arith, tensor.Tensor):
        ctx.load_dialect(dialect)
    return ctx


def build_contraction_egraph(op, *, symbol: str, costs: Mapping[str, int]):
    """``(module, seconds)`` — a self-contained function holding both implementations of ``op``.

    ``op`` is a routable contraction as :func:`llvmlower.passes_opu.routable_contractions` returns it, and
    it is CLONED rather than moved: the module it came from must stay intact, because the decision made here
    is applied to it afterwards. ``costs`` maps :data:`VECTOR` / :data:`MATRIX` to already-scaled integer
    costs; an alternative with no cost is still present and simply not ranked, so extraction cannot prefer
    it for lack of data.

    The function's arguments are the contraction's operands, which is what makes the region self-contained:
    the eqsat passes require every operand of an alternative to be defined inside the graph, and a clone
    still referring to the original module's values is not.
    """
    from xdsl.dialects import equivalence as E, func
    from xdsl.dialects.builtin import IntAttr, ModuleOp
    from xdsl.ir import Block, Region

    operands = list(op.operands)
    if len(operands) != 3 or len(op.results) != 1:
        raise ValueError(f"expected a 3-operand, 1-result contraction, got {len(operands)} and "
                         f"{len(op.results)}; an e-class over a differently-shaped op would extract to "
                         "something the compile path cannot emit")

    started = time.perf_counter()
    arg_types = [o.type for o in operands]
    out_type = op.results[0].type
    block = Block(arg_types=arg_types)

    # The vector path: the contraction itself, operands remapped onto the block arguments.
    vector = op.clone(value_mapper=dict(zip(operands, block.args, strict=True)))
    if VECTOR in costs:
        vector.attributes[E.EQSAT_COST_LABEL] = IntAttr(int(costs[VECTOR]))
    block.add_op(vector)

    # The matrix path: exactly the call the rewrite emits, so what is costed is what would run.
    matrix = func.CallOp(symbol, list(block.args), [out_type])
    if MATRIX in costs:
        matrix.attributes[E.EQSAT_COST_LABEL] = IntAttr(int(costs[MATRIX]))
    block.add_op(matrix)

    # Order matters: a tie resolves to the first operand, and the vector path is the control.
    cls = E.ClassOp(vector.results[0], matrix.results[0])
    block.add_op(cls)
    block.add_op(func.ReturnOp(cls.results[0]))
    fn = func.FuncOp("contraction", ((*arg_types,), (out_type,)), Region([block]))
    # The callee has to be DECLARED or the module does not verify ("could not be found in symbol table").
    # The eqsat passes do not verify, so a graph without it extracts perfectly well and is quietly
    # malformed -- which is the kind of thing that only surfaces once something downstream does verify.
    decl = func.FuncOp(symbol, ((*arg_types,), (out_type,)), Region(), visibility="private")
    return ModuleOp([fn, decl]), time.perf_counter() - started


def extract_contraction_choice(op, *, symbol: str, costs: Mapping[str, int],
                               shape=None) -> ContractionChoice:
    """Build the e-graph for ``op`` and report which implementation survived extraction.

    The winner is READ BACK from the extracted IR — whether a ``func.call`` or a ``linalg.generic`` remains
    — rather than computed alongside it. That distinction is the whole point: a choice computed here would
    be a threshold with extra steps and would demonstrate nothing about deciding from the graph.

    With no cost for either alternative, extraction has nothing to minimise; this reports a gap and leaves
    ``chosen`` as None so the caller falls back rather than routing on no information.
    """
    from xdsl.dialects import func as _func
    from xdsl.dialects.linalg.ops import GenericOp as _GenericOp
    from xdsl.transforms import eqsat_add_costs, eqsat_extract

    m, n, k = 0, 0, 0
    if shape is not None:
        m, n = int(shape.parallel[0]), int(shape.parallel[1])
        k = int(shape.reduction[0])

    module, build_s = build_contraction_egraph(op, symbol=symbol, costs=costs)
    ranked = {key: int(costs[key]) for key in (VECTOR, MATRIX) if key in costs}
    if not ranked:
        return ContractionChoice(
            chosen=None, costs=ranked, m=m, n=n, k=k, build_seconds=build_s,
            gap=("neither alternative carries a cost, so extraction has nothing to minimise; the caller "
                 "must fall back rather than route on no information"))

    ctx = _context()
    started = time.perf_counter()
    eqsat_add_costs.EqsatAddCostsPass(default=None).apply(ctx, module)
    eqsat_extract.EqsatExtractPass().apply(ctx, module)
    extract_s = time.perf_counter() - started

    survivors = {op_.name for op_ in module.walk()}
    has_call = any(isinstance(o, _func.CallOp) for o in module.walk())
    has_generic = any(isinstance(o, _GenericOp) or o.name == "linalg.generic"
                      for o in module.walk())
    chosen: str | None
    if has_call and not has_generic:
        chosen = MATRIX
    elif has_generic and not has_call:
        chosen = VECTOR
    else:
        # Both or neither survived: extraction did not resolve the class, and guessing which the caller
        # meant would silently commit to one.
        chosen = None
    gap = None if chosen else (f"extraction left {'both' if has_call and has_generic else 'neither'} "
                               f"alternative in the IR (ops: {sorted(survivors)})")
    return ContractionChoice(chosen=chosen, costs=ranked, m=m, n=n, k=k,
                             build_seconds=build_s, extract_seconds=extract_s, gap=gap)


def measured_cost_of(cost_model, *, vector_unit: str, matrix_unit: str,
                     op: str = "matmul", in_fmt: str = "int8", weight_fmt: str = "int8"
                     ) -> Callable[[Any, str], "int | None"]:
    """A ``cost_of`` backed by :class:`routing.MeasuredCost`, which is the cost model that means something.

    The crude alternative — MACs for the vector unit, tile occupancy for the matrix unit, with a rate picked
    to look reasonable — decides confidently and wrongly. Measured on the real prepared spectformer: it
    routed 89 of 90 contractions to the matrix unit *including* the 48 FFT-family shapes whose N is 8 or 14,
    because a generous matrix rate swamps the occupancy penalty on a shape that fills one thirtieth of a
    tile. The e-graph minimised exactly what it was told to; the cost was the defect.

    :class:`routing.MeasuredCost` is the honest source: it charges tile occupancy, dispatch and the K-major
    pack, and it DECLINES a unit absent from its measurement table rather than scoring it optimistically. So
    with no measured matrix-unit throughput this returns None for the matrix path, every contraction stays on
    the vector path, and the thing standing between here and a real routing decision is a MEASUREMENT — not
    more machinery.
    """
    from . import routing as _routing

    def cost_of(shape, which: str) -> "int | None":
        demand = _routing.OpDemand(op=op, in_fmt=in_fmt, weight_fmt=weight_fmt, site="egraph",
                                   m=int(shape.parallel[0]), n=int(shape.parallel[1]),
                                   k=int(shape.reduction[0]))
        unit = vector_unit if which == VECTOR else matrix_unit
        kind = "vector" if which == VECTOR else "matrix"
        got = cost_model(demand, _routing.Candidate(unit=unit, kind=kind, acc="int32",
                                                    exposure="derived"))
        if got is None:
            return None
        from .persistent_equivalence import COST_SCALE
        return int(round(float(got) * COST_SCALE))

    return cost_of


def egraph_selector(cost_of: Callable[[Any, str], "int | None"], *,
                    symbol: str = "merlin_opu_gemm_i8",
                    record: "list[ContractionChoice] | None" = None
                    ) -> Callable[[Any, Any], bool]:
    """A ``select`` callable for the rewrite, backed by extraction from an e-graph.

    ``cost_of(shape, which)`` returns the scaled integer cost of implementing ``shape`` on ``which``
    (:data:`VECTOR` or :data:`MATRIX`), or None to decline — the same "decline rather than guess" contract
    :class:`routing.MeasuredCost` uses, and for the same reason: an unmeasured implementation that scored
    well would win decisions on the strength of having no data.

    Returns a two-argument callable ``(op, shape) -> bool``. The rewrite's own ``select`` takes only the
    shape, so :func:`for_rewrite` adapts it; the op is needed here because the e-graph is built over the
    real operation.

    ``record`` collects every :class:`ContractionChoice`, which is how a report can state what the decision
    cost in compile time instead of leaving it unmeasured.
    """
    def select(op, shape) -> bool:
        costs = {}
        for which in (VECTOR, MATRIX):
            got = cost_of(shape, which)
            if got is not None:
                costs[which] = int(got)
        got = extract_contraction_choice(op, symbol=symbol, costs=costs, shape=shape)
        if record is not None:
            record.append(got)
        return got.on_matrix_unit

    return select


def for_rewrite(op_shape_select: Callable[[Any, Any], bool],
                candidates: Sequence[tuple[Any, Any]]) -> Callable[[Any], bool]:
    """Adapt a ``(op, shape)`` decision into the shape-only ``select`` the rewrite takes.

    The rewrite deliberately passes only the shape, so that it cannot be handed a decision procedure that
    depends on IR it might have already changed. Deciding EVERY candidate up front, here, keeps that true:
    the extraction runs against the module as it was enumerated, and the rewrite then applies a decision
    that is already made.

    Shapes are matched by identity of the enumerated pair, so two contractions with identical extents get
    their own decisions rather than sharing one.
    """
    decided = {id(shape): op_shape_select(op, shape) for op, shape in candidates}

    def select(shape) -> bool:
        return bool(decided.get(id(shape), False))

    return select


# ==================================================================================================
# Saturation: let a RULE derive the alternative, instead of Python constructing it.
# ==================================================================================================
#
# Everything above builds the second alternative by hand: Python creates the call and puts it in the class.
# That is enough to decide, and it means the set of alternatives is whatever this file happens to construct
# — so a new capability of the target is a code change here.
#
# A PDL rewrite rule inverts that. Under `apply-eqsat-pdl` a `pdl.replace` does not delete the matched
# operation; it ADDS the replacement to the matched value's e-class. So the rule "a rank-2 int8 contraction
# is also computable by the microkernel" GROWS the graph, and adding a second way to compute a contraction
# becomes a new rule rather than new Python.
#
# MEASURED on the real IR: the e-class goes from `equivalence.class %generic` to
# `equivalence.class %generic, %call`, with the call created by the rewriter the pattern compiled into.
#
# **The rule cannot carry the legality conditions, and must not be trusted to.** PDL matches an operation by
# name, operand and result types; it cannot express "the element types are (i8, i8, i32)", "the iterator
# types are (parallel, parallel, reduction)" or "the accumulator init is a zero fill" without native
# constraints. A pattern matching `linalg.generic` therefore matches EVERY generic, including ones the unit
# cannot compute. So legality stays where it can actually be decided — `passes_opu.routable_contractions` —
# and saturation is only ever run on a region built from a contraction already known legal. Letting this
# pattern loose on a whole function would produce calls for things the microkernel does not implement.

#: Set when the PDL→PDL-interp conversion needs an external mlir-opt. The eqsat PDL pass shells out to it.
_MLIR_OPT_ENV = "XDSL_MLIR_OPT"


def contraction_to_call_pattern(symbol: str, *, benefit: int = 1) -> str:
    """The PDL rule "this contraction is also computable by the microkernel", for one callee.

    GENERATED per symbol rather than shipped as a file, because the callee is one of several monomorphic
    symbols (one per distinct signature) and a static pattern could only ever name one of them. Generating
    it also keeps the rule beside the code that explains what it may and may not be trusted to express.
    """
    return f"""\
// GENERATED by merlin.targetgen.contraction_egraph — do not edit.
//
// Under apply-eqsat-pdl this ADDS the call to the contraction's e-class rather than replacing it, so the
// graph grows by a rule. It matches `linalg.generic` by NAME and cannot express legality (element types,
// iterator types, a zero accumulator init) — see the module docstring. Apply it only to a region built
// from a contraction already established as legal.
pdl.pattern : benefit({int(benefit)}) {{
  %a = pdl.operand
  %b = pdl.operand
  %c = pdl.operand
  %t = pdl.type
  %mm = pdl.operation "linalg.generic" (%a, %b, %c : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
  pdl.rewrite %mm {{
    %callee = pdl.attribute = @{symbol}
    %call = pdl.operation "func.call" (%a, %b, %c : !pdl.value, !pdl.value, !pdl.value) {{"callee" = %callee}} -> (%t : !pdl.type)
    pdl.replace %mm with %call
  }}
}}
"""


def _resolve_mlir_opt() -> str:
    """The mlir-opt the PDL conversion will use, or a clear error naming what to set.

    ``xdsl.transforms.mlir_opt`` reads its executable from the environment **at import time** and stores it
    as a dataclass field default. So setting the variable is only enough if that module has not been
    imported yet, and it usually has been by the time anything here runs — which surfaced as
    ``mlir-opt is not available`` from three frames inside a pass. Both the environment variable and the
    already-captured default are therefore updated, and the update is verified rather than assumed.

    The pinned toolchain ships mlir-opt beside clang, so the common case needs no configuration at all.
    """
    import os
    import shutil

    got = os.environ.get(_MLIR_OPT_ENV)
    if not got or not shutil.which(got):
        from ..llvmlower import toolchain
        candidate = Path(toolchain.clang()).with_name("mlir-opt") if toolchain.available() else None
        if candidate is not None and candidate.is_file():
            got = str(candidate)
    if not got or not shutil.which(got):
        raise FileNotFoundError(
            f"saturation needs mlir-opt (the PDL to PDL-interp conversion shells out to it). Set "
            f"${_MLIR_OPT_ENV} to one, or make the pinned toolchain available")
    os.environ[_MLIR_OPT_ENV] = got

    from xdsl.transforms import mlir_opt as _mlir_opt
    _mlir_opt.DEFAULT_MLIR_OPT_EXECUTABLE = got
    field = _mlir_opt.MLIROptPass.__dataclass_fields__.get("executable")
    if field is not None:
        field.default = got
    probe = _mlir_opt.MLIROptPass(arguments=())
    if not shutil.which(probe.executable):
        raise FileNotFoundError(
            f"the eqsat PDL pass still resolves mlir-opt to {probe.executable!r}, which is not runnable; "
            f"set ${_MLIR_OPT_ENV} before importing xdsl")
    return got


def saturate_contraction(op, *, symbol: str, max_iterations: int = 4):
    """``(module, n_alternatives, seconds)`` — grow one contraction's e-class by applying the rule.

    Builds the contraction as a self-contained region in e-graph form (a class around every operand and
    around the result, which is what the eqsat passes consume), then applies the PDL rule. The result is a
    module whose e-class holds both the contraction and the call — with the call created by the RULE.

    ``n_alternatives`` is read from the e-class in the produced IR, so "the graph grew" is a measurement of
    the output rather than an assumption about what the pass does.
    """
    from xdsl.dialects import equivalence as E, func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.parser import Parser

    # BEFORE importing the pass: it captures its mlir-opt at import time.
    _resolve_mlir_opt()
    from xdsl.transforms import apply_eqsat_pdl
    operands = list(op.operands)
    if len(operands) != 3 or len(op.results) != 1:
        raise ValueError(f"expected a 3-operand, 1-result contraction, got {len(operands)} and "
                         f"{len(op.results)}")

    started = time.perf_counter()
    arg_types = [o.type for o in operands]
    out_type = op.results[0].type
    block = Block(arg_types=arg_types)
    # A class around each operand: the eqsat interpreter reasons over e-classes, and an operand that is a
    # bare block argument is not one.
    classes = [E.ClassOp(arg) for arg in block.args]
    for cls in classes:
        block.add_op(cls)
    body = op.clone(value_mapper=dict(zip(operands, [c.results[0] for c in classes], strict=True)))
    block.add_op(body)
    result_class = E.ClassOp(body.results[0])
    block.add_op(result_class)
    block.add_op(func.ReturnOp(result_class.results[0]))
    fn = func.FuncOp("contraction", ((*arg_types,), (out_type,)), Region([block]))
    decl = func.FuncOp(symbol, ((*arg_types,), (out_type,)), Region(), visibility="private")

    # The conversion round-trips through mlir-opt, whose PDL-interp output this has to PARSE back. So the
    # context needs the dialects that output uses, not only the ones the input has: without pdl_interp the
    # parse fails with "builtin.unregistered does not have a custom format", several frames from the cause.
    # `xdsl-opt` loads every dialect it knows, which is why the same pipeline works there and not here.
    ctx = _context()
    from xdsl.dialects import eqsat_pdl_interp, pdl, pdl_interp
    for dialect in (pdl.PDL, pdl_interp.PDLInterp, eqsat_pdl_interp.EqSatPDLInterp):
        ctx.load_dialect(dialect)
    pattern = Parser(ctx, contraction_to_call_pattern(symbol)).parse_module()
    module = ModuleOp([fn, decl, *[o.clone() for o in pattern.body.block.ops]])
    apply_eqsat_pdl.ApplyEqsatPDLPass(max_iterations=int(max_iterations)).apply(ctx, module)

    grown = max((len(c.operands) for c in module.walk() if isinstance(c, E.AnyClassOp)), default=0)
    return module, grown, time.perf_counter() - started

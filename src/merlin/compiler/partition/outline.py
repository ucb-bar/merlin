"""``merlin-outline-dispatches`` — split one ``func @forward`` into per-dispatch kernels.

This is the *unification core* of the pipeline: it takes a linalg-on-tensors module
(model2MLIR output, one monolithic ``func.func @forward``) and rewrites it into

  - one ``func.func @<forward>$kernel_<i>`` per compute dispatch (each owning its linalg
    payload plus the cheap producers it consumes), and
  - a thin driver ``func.func @<forward>`` that computes the view/glue ops inline and
    invokes each kernel in order via ``func.call``.

It is the MLIR analogue of IREE's two-phase dispatch formation
(``FormDispatchRegions`` → ``CloneProducersIntoDispatchRegions``), but Merlin-owned and
operating directly on linalg-on-tensors so we keep full control and never depend on
IREE's ``flow``/``stream``/``hal`` dialects.

Outlining makes the model a **dispatch table** the ``interface``/``runtime`` dialects can
reference by kernel symbol, makes each kernel independently compilable + checkable
(per-kernel host == torch bisection), and is the prerequisite for multicore dispatch and
bounded-memory whole-model runs.

Strategy (single-op-per-dispatch baseline):

- **Roots** = the heavy compute ops (every ``linalg.*`` op other than ``linalg.fill`` /
  ``linalg.yield``). Each becomes its own kernel.
- **Cloned producers** = the cheap, pure producers a root consumes (``arith.constant``,
  ``tensor.empty``, ``linalg.fill``) are *cloned into* the kernel — IREE's
  clone-producers-into-dispatch — so the kernel allocates and zeroes its own
  accumulator and is self-contained for the per-kernel backend (bufferization can fuse
  the fill into the contraction). They are left in the driver as well; if a clone is dead
  there it is harmless and a later DCE removes it.
- **Glue ops** (``tensor.extract_slice``/``expand_shape``/``cast``, scalar ``arith``,
  ``tensor.insert_slice``, …) stay inline in the driver and feed kernels as call operands.

The rewrite is purely structural and value-preserving: inlining every kernel call back
into the driver reproduces the original op set and dataflow (see
``test_outline.py::test_outline_is_value_preserving``).
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from merlin.compiler.dialects.xdsl_context import HAS_XDSL
from merlin.compiler.ir import mlir_query as mq
from merlin.compiler.ir.boundary import free_values

# ---- outline -----------------------------------------------------------------------------------

# Historical branch helper; use the same canonical nested-capture algorithm.
_free_values = free_values


class OutlineError(RuntimeError):
    pass


# Pure, cheap producers cloned into each consuming kernel instead of being outlined on
# their own (matches IREE CloneProducersIntoDispatchRegions). Keeping the fill inside the
# kernel lets per-kernel bufferization fold the zero-init into the contraction.
CLONE_INTO_KERNEL = ("arith.constant", "tensor.empty", "linalg.fill")


def _is_root(op) -> bool:
    """A heavy compute op that becomes its own kernel."""
    return op.name.startswith("linalg.") and op.name not in ("linalg.fill", "linalg.yield", "linalg.index")


def _is_block_arg(value) -> bool:
    from xdsl.ir import BlockArgument

    return isinstance(value, BlockArgument)


@dataclass
class DispatchInfo:
    """Bookkeeping for one outlined dispatch (consumed by interface/runtime lowering)."""

    index: int
    symbol: str
    root_op: str  # e.g. "linalg.matmul"
    n_operands: int
    result_types: list[str]
    prov: dict[str, str] = field(default_factory=dict)
    #: Set when the dispatch is a whole compute group rather than one op: the group's index, where
    #: it was placed, and its stages in program order. ``None`` on a single-op dispatch.
    group: int | None = None
    placement: str | None = None
    stages: list[str] = field(default_factory=list)


@dataclass
class OutlineResult:
    module: Any
    dispatches: list[DispatchInfo] = field(default_factory=list)
    # Permission to retain typed declarations, not evidence of executable implementations.
    # The caller must close these symbols over its compiled catalog before execution.
    external_symbols: tuple[str, ...] = ()

    @property
    def n_kernels(self) -> int:
        return len(self.dispatches)


@dataclass(frozen=True)
class ExternalizedDispatches:
    """Outlined driver with selected kernel bodies replaced by typed declarations.

    The caller must link an implementation for every selected symbol. This stage
    makes no claim that the implementation preserves the removed body's semantics.
    """

    module: Any
    symbols: tuple[str, ...]


def externalize_dispatches(outlined: OutlineResult, symbols: Iterable[str]) -> ExternalizedDispatches:
    """Expose exact outlined symbols to a separately compiled dispatch provider.

    Selection is by the symbols emitted by :func:`outline_dispatches`, never by
    a model name or by a guessed position in the module. The source module is
    left intact so an independent host/reference build can use its bodies.
    """
    from xdsl.dialects import func

    selected = tuple(symbols)
    if any(not isinstance(symbol, str) or not symbol for symbol in selected):
        raise OutlineError("external dispatch symbols must be nonempty strings")
    if len(set(selected)) != len(selected):
        raise OutlineError("external dispatch symbols must be unique")
    known = {dispatch.symbol for dispatch in outlined.dispatches}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise OutlineError("not an outlined dispatch: " + ", ".join(f"@{symbol}" for symbol in unknown))

    module = outlined.module.clone()
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    for symbol in selected:
        original = functions.get(symbol)
        if original is None or not original.body.blocks or original.parent is not module.body.block:
            raise OutlineError(f"outlined dispatch @{symbol} must have one top-level definition")
        signature = original.function_type
        declaration = func.FuncOp.external(symbol, list(signature.inputs.data), list(signature.outputs.data))
        for key, value in original.attributes.items():
            if key not in {"sym_name", "function_type", "sym_visibility"}:
                declaration.attributes[key] = value
        module.body.block.insert_op_before(declaration, original)
        original.detach()
    module.verify()
    return ExternalizedDispatches(module=module, symbols=selected)


def _prov(op) -> dict[str, str]:
    from xdsl.dialects.builtin import StringAttr

    out = {}
    for table in (op.attributes, getattr(op, "properties", {}) or {}):
        for key, val in table.items():
            if key.startswith("prov.") and isinstance(val, StringAttr):
                out[key] = val.data
    return out


# The token that separates the region-id provenance suffix from the ``$kernel_<idx>`` core in a
# dispatch symbol (``forward$kernel_3__rmatmul_0``). Kept here so the emitter and any reader share
# ONE definition (structured split, never a regex — repo lint gate).
REGION_SYMBOL_SEP = "__r"


def _sanitize_symbol(text: str) -> str:
    """Coerce a region_id to a C-symbol-safe token (alnum + underscore). region_ids are already
    ``<opkind>_<n>`` today; this just hardens against an unexpected char without a regex."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in text)


#: The infix every outlined dispatch symbol carries (``forward$kernel_3``). Required before the
#: ``__r`` suffix is read as provenance — see :func:`region_id_of_symbol`.
KERNEL_SYMBOL_INFIX = "$kernel_"


def region_id_of_symbol(symbol: str) -> str | None:
    """Recover the ``prov.region_id`` a dispatch symbol was tagged with (``None`` if untagged).

    Inverse of the suffix the outliner appends. Used to attribute an emitted kernel / ELF symbol
    back to its model region — the asm-side of the provenance join.

    ⚠️ REQUIRES the ``$kernel_`` infix, and this is not belt-and-braces. ``__r`` alone appears inside
    symbols nobody here emitted: XNNPACK names its vector kernels
    ``xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x4v__rvv``, which split on the separator alone yields a
    confident region id of ``"vv"``. That is the same ``__rvv`` collision that already had to be
    fixed once in corpus ingest, and here it is worse than a mislabel: ``section_mlir`` selects which
    regions to splice into a section build by this function, so a false positive silently builds a
    slice of the wrong model.
    """
    core, sep, rid = symbol.partition(REGION_SYMBOL_SEP)
    if not sep or not rid:
        return None
    if KERNEL_SYMBOL_INFIX not in core:
        return None  # not a symbol this outliner emitted; claim nothing about it
    return rid


def _cloneable(owner) -> bool:
    """A producer is cloned into the consuming kernel when it's a cheap pure op -- UNLESS it
    is a ``tensor.empty`` tagged ``prov.quant_inner`` (an elided torchao int_data/scale that
    the driver binds to real data): cloning it would give the kernel its own *uninitialized*
    copy, so it must stay in the driver and be passed in as a kernel argument instead."""
    from xdsl.ir import Operation

    if not isinstance(owner, Operation) or owner.name not in CLONE_INTO_KERNEL:
        return False
    if owner.name == "tensor.empty" and "prov.quant_inner" in owner.attributes:
        return False
    return True


def _producer_closure(root):
    """Root + the cheap producers feeding it, in clone (topological) order.

    Walks operands transitively through ``CLONE_INTO_KERNEL`` ops only, so e.g. a
    ``linalg.fill`` pulls in its ``tensor.empty`` init and ``arith.constant`` value.
    """
    ops: list = []
    seen: set[int] = set()

    def visit(op):
        if id(op) in seen:
            return
        seen.add(id(op))
        for operand in op.operands:
            if _is_block_arg(operand):
                continue
            if _cloneable(operand.owner):
                visit(operand.owner)
        ops.append(op)

    # Pull in the root's cloneable producers first, then the root itself last.
    for operand in root.operands:
        if _is_block_arg(operand):
            continue
        if _cloneable(operand.owner) or (
            root.name in {"linalg.matmul", "linalg.generic"}
            and hasattr(root, "outputs")
            and operand in root.outputs
            and operand.owner.name == "tensor.splat"
        ):
            visit(operand.owner)
    ops.append(root)
    return ops


def _group_closure(members):
    """A group's members plus the cheap producers each consumes, in clone order."""
    ops: list = []
    seen: set[int] = set()
    for member in members:
        for op in _producer_closure(member):
            if id(op) not in seen:
                seen.add(id(op))
                ops.append(op)
    return ops


def _external_declarations(module, external_symbols):
    """Resolve an explicit permission set to actual bodyless module declarations."""
    if isinstance(external_symbols, str):
        raise OutlineError("external_symbols must contain symbol names, not a string")
    names = tuple(external_symbols)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise OutlineError("external_symbols must contain unique nonempty symbol names")
    functions = {}
    for op in module.body.block.ops:
        if op.name == "func.func":
            name = op.sym_name.data
            if name in functions:
                raise OutlineError(f"duplicate function symbol @{name}")
            functions[name] = op
    declarations = {}
    for name in names:
        function = functions.get(name)
        if function is None or function.body.blocks:
            raise OutlineError(f"permitted external symbol @{name} must have a bodyless func declaration")
        declarations[name] = function
    return declarations


def outline_dispatches(
    module, forward: str | None = None, *, groups=None, external_symbols: Iterable[str] = ()
) -> OutlineResult:
    """Outline each compute dispatch of ``func @forward`` into its own kernel func.

    Returns the rewritten module (driver + kernels) and the per-dispatch table.

    ``groups`` (from :mod:`.compute_groups`, formed over THIS module) makes a dispatch a whole
    compute group: a contraction with the stages its unit absorbed, or a fused host region. The
    kernel is emitted where the group's LAST member stood. That is always legal, because a group
    is joined only along values nothing outside it reads: every operand a member takes from
    outside was defined before that member, and nothing outside waits on a member before the last.
    With no groups the behaviour is the single-op baseline, byte for byte.

    ``external_symbols`` explicitly permits retaining existing typed declarations. It does not
    supply their implementations: the caller must verify compiled object closure before execution.
    Missing declarations and all unpermitted external calls still refuse.
    """
    if not HAS_XDSL:
        return OutlineResult(module=module)
    from xdsl.dialects.builtin import FunctionType, IntegerAttr, IntegerType, ModuleOp, StringAttr
    from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    declarations = _external_declarations(module, external_symbols)
    # DECLARATIONS ARE NOT DEFINITIONS. A capture that leaves an operation to an external symbol
    # (a `func.func private @…` with no body) prints that declaration FIRST, so picking `fns[0]`
    # blind used to select it and die on `fn.body.blocks[0]` with a bare IndexError -- naming
    # neither the module nor the symbol. Filtering by "has a body" is what makes the driver the
    # function that HAS one; a symbol that stays undefined is named as such below.
    fns = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    if not fns:
        raise OutlineError("no func.func with a body in module (only external declarations)")
    if forward is not None:
        fns = [f for f in fns if f.sym_name.data == forward]
        if not fns:
            raise OutlineError(f"func @{forward} not found")
    fn = fns[0]
    fname = fn.sym_name.data
    if len(fn.body.blocks) != 1:
        raise OutlineError("outlining requires a single-block model driver")
    src = fn.body.blocks[0]

    arg_types = [a.type for a in src.args]
    driver = Block(arg_types=arg_types)
    dmap = dict(zip(src.args, driver.args))  # original value -> driver value

    kernels: list = []
    dispatches: list[DispatchInfo] = []

    # A group is outlined when it holds a compute op; a group of shape metadata alone stays glue.
    group_of: dict[int, Any] = {}
    last_member: dict[int, Any] = {}
    order = {id(op): index for index, op in enumerate(src.ops)}
    for grp in groups or ():
        members = sorted((m for m in grp.members if id(m) in order), key=lambda m: order[id(m)])
        if len(members) != len(grp.members):
            raise OutlineError(
                f"compute group {grp.index} names an op that is not in @{fname}; "
                f"groups must be formed over the module being outlined"
            )
        if not any(_is_root(m) or m.name == "builtin.unregistered" for m in members):
            continue
        for member in members:
            group_of[id(member)] = (grp, members)
        last_member[id(members[-1])] = (grp, members)

    for op in list(src.ops):
        if op.name == "func.return":
            driver.add_op(ReturnOp(*[dmap[o] for o in op.operands]))
            continue
        if id(op) in group_of and id(op) not in last_member:
            continue  # emitted with its group, where the group's last op is
        if id(op) not in last_member and not _is_root(op):
            cloned = op.clone(value_mapper=dmap)
            driver.add_op(cloned)
            for old, new in zip(op.results, cloned.results):
                dmap[old] = new
            continue

        # --- a compute dispatch: build its kernel ---
        idx = len(kernels)
        # Encode the model-layer provenance (prov.region_id) INTO the symbol so it is the one
        # thread that survives to the emitted ELF (the asm-join key for the cross-compiler compare
        # + the section slicer). Keep the ``$kernel_<idx>`` core intact — it is the load-bearing
        # driver-vs-kernel marker (``"$kernel_" in sym_name``) and the kernel-dedup key is the BODY
        # text, not the symbol, so the suffix neither breaks detection nor defeats dedup. Absent a
        # region_id (pre-provenance capture) the symbol is byte-identical to before (back-compat).
        grp, members = last_member.get(id(op), (None, [op]))
        # A group is named for its root (the contraction), a host region for its first compute op.
        named = op if grp is None else (grp.root or next((m for m in members if _is_root(m)), members[0]))
        prov = _prov(named)
        rid = prov.get("prov.region_id")
        suffix = f"__r{_sanitize_symbol(rid)}" if rid else ""
        symbol = f"{fname}$kernel_{idx}{suffix}"
        kernel_ops = _producer_closure(op) if grp is None else _group_closure(members)
        params = free_values(kernel_ops)
        result_types = [r.type for r in op.results]

        kblock = Block(arg_types=[p.type for p in params])
        kmap = dict(zip(params, kblock.args))
        for kop in kernel_ops:
            c = kop.clone(value_mapper=kmap)
            kblock.add_op(c)
            for old, new in zip(kop.results, c.results):
                kmap[old] = new
        kblock.add_op(ReturnOp(*[kmap[r] for r in op.results]))
        kfn = FuncOp(symbol, FunctionType.from_lists([p.type for p in params], result_types), Region([kblock]))
        kfn.sym_visibility = StringAttr("private")
        if grp is not None:
            # On the function, not on its ops: a function attribute survives every later lowering,
            # which is what lets emitted code be attributed back to a placement decision.
            kfn.attributes["merlin.group"] = IntegerAttr(grp.index, IntegerType(64))
            kfn.attributes["merlin.placement"] = StringAttr(str(grp.placement))
        kernels.append(kfn)

        # --- the driver-side call, mapping external operands to driver values ---
        call = CallOp(symbol, [dmap[p] for p in params], result_types)
        driver.add_op(call)
        for old, new in zip(op.results, call.results):
            dmap[old] = new

        dispatches.append(
            DispatchInfo(
                index=idx,
                symbol=symbol,
                root_op=named.name,
                n_operands=len(params),
                result_types=[str(t) for t in result_types],
                prov=prov,
                group=None if grp is None else grp.index,
                placement=None if grp is None else str(grp.placement),
                stages=[] if grp is None else list(grp.stages),
            )
        )

    new_fn = FuncOp(fname, FunctionType.from_lists(arg_types, list(fn.function_type.outputs.data)), Region([driver]))
    # Keep any function-level attributes (e.g. llvm.emit_c_interface) on the driver.
    for key, val in fn.attributes.items():
        if key not in ("sym_name", "function_type", "sym_visibility"):
            new_fn.attributes[key] = val

    # Ordinary captures still need bodies for every operation. A catalog caller may explicitly
    # retain declarations, with the same ABI and attributes, and separately prove object closure.
    defined = {k.sym_name.data for k in kernels} | {fname} | set(declarations)
    undefined: list[str] = []
    for function in (new_fn, *kernels):
        for op in function.walk():
            if op.name != "func.call":
                continue
            callee = op.callee.string_value()
            if callee not in defined and callee not in undefined:
                undefined.append(callee)
    if undefined:
        raise OutlineError(
            f"@{fname} calls {len(undefined)} symbol(s) this module never defines: "
            + ", ".join(f"@{sym}" for sym in undefined)
            + ". The capture left these operations to an external implementation, so there is "
            "nothing to outline, compile or run for them -- they must be decomposed into linalg "
            "at capture time (or defined here) before this model can execute."
        )

    out = ModuleOp([new_fn, *kernels, *(declaration.clone() for declaration in declarations.values())])
    out.verify()
    return OutlineResult(module=out, dispatches=dispatches, external_symbols=tuple(declarations))


# ---- contraction_coverage ----------------------------------------------------------------------
# Which contractions did the matcher miss, and what do they cost.
#
# A contraction that stays a `linalg.generic` never becomes `linalg.matmul`, so it never reaches a
# matrix unit AND it never appears in the denominator when we report "what fraction of contraction
# work is routed". Both halves of that are silent. This module makes the miss countable.
#
# MEASURED, and the reason this exists. On `spectformer_int8_full` 16 generics are contractions the
# matcher did not recognise -- 8 with loops `(4, 196, 196, reduce 64)` and 8 with
# `(4, 196, 64, reduce 196)`, i.e. attention's Q.K^T and scores.V, the canonical matrix-unit workload.
# They are 157.4 MMAC against 1702.2 MMAC of `linalg.matmul` work, so **8.5% of the model's contraction
# MACs were invisible**, and the published "88.3% of contraction cycles are routed" and "the unit is at
# 96.6% of its Amdahl ceiling" were both computed against the under-count.
#
# THE COST IS SHAPE-DEPENDENT, so do not carry 8.5% around as a constant. The same miss happens on
# `gemma2_2b_int8_full_seq8` -- 26 pairs, one per layer -- but at seq=8 the scores are 8x8 and it is
# 0.03% of that model's MACs. The gap is systematic; whether it matters is set by sequence length.
#
# CLASSIFY STRUCTURALLY, NEVER BY TAG. 751 of that model's 843 generics carry no `prov.family` /
# `role` / `aten` / `module` at all, and the tags that do exist mislead in four documented ways. What a
# loop nest computes is legible from `iterator_types` (is any dimension reduced) plus the body's arith
# ops (is the accumulation a multiply-add).
#
# GET EXTENTS FROM THE INDEXING MAPS. The tempting shortcut -- "the reduced extent is the one in the
# inputs that is missing from the output" -- is wrong whenever a contracted extent also appears in the
# output. `scores.V` has output `4x196x64` and contracts over a *second* 196, so the shortcut finds
# nothing and undercounts by 196x (it reported 79.1 MMAC instead of 157.4). MACs are the product of ALL
# loop extents, parallel and reduction alike, each read off an operand that carries that iteration dim.

_MUL = {"arith.mulf", "arith.muli"}
_ADD = {"arith.addf", "arith.addi"}
_MAX = {"arith.maxf", "arith.maximumf", "arith.maxsi", "arith.maxnumf"}
_ABS = {"math.absf", "math.fabs"}

#: op names whose MACs we can price directly, mapped to nothing -- membership is the point.
MATMUL_OPS = ("linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul")


@dataclass(frozen=True)
class UnloweredContraction:
    """A generic that computes a contraction the matcher did not recognise."""

    result_type: str
    loop_extents: tuple[tuple[int, int], ...]
    macs: int


@dataclass
class CoverageReport:
    lowered_macs: int = 0
    unlowered: list[UnloweredContraction] = field(default_factory=list)
    #: generics whose extents could not be derived. Counted and surfaced, never dropped -- an
    #: unpriceable op is a hole in the denominator and must not read as a zero.
    unpriceable: list[str] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def unlowered_macs(self) -> int:
        return sum(u.macs for u in self.unlowered)

    @property
    def total_macs(self) -> int:
        return self.lowered_macs + self.unlowered_macs

    @property
    def unlowered_share(self) -> float:
        """Fraction of all contraction MACs the matcher never saw. 0.0 when there is no work."""
        return (self.unlowered_macs / self.total_macs) if self.total_macs else 0.0


def _body_op_names(op) -> set[str]:
    if not op.regions or not op.regions[0].blocks:
        return set()
    return {mq.op_name(o) for o in op.regions[0].blocks[0].ops}


def _iterator_text(op) -> str:
    return str(op.properties.get("iterator_types") or op.attributes.get("iterator_types") or "")


#: ``arith.cmpf`` predicates: unordered (either operand NaN) and ordered greater-than.
_CMPF_UNO, _CMPF_OGT = 14, 2

#: Whether a NaN-propagating select-max body classifies as a max reduction. Only a caller whose
#: consumer cannot produce a NaN (an integer device readout) may say so: see :func:`select_max_is_max`.
_SELECT_MAX_IS_MAX: ContextVar[bool] = ContextVar("select_max_is_max", default=False)


@contextmanager
def select_max_is_max(enabled: bool = True):
    """Within this block, classify ``select(isnan(x) or x > acc, x, acc)`` reductions as max.

    Exact for a consumer whose values cannot be NaN -- the integer readout of a device group, whose
    pool takes the quantized accumulator -- and therefore scoped to the caller that routes such groups
    rather than changed for every reader of a capture."""
    token = _SELECT_MAX_IS_MAX.set(bool(enabled))
    try:
        yield
    finally:
        _SELECT_MAX_IS_MAX.reset(token)


def _is_select_max(op) -> bool:
    """A reduction body ``select(isnan(x) or x > acc, x, acc)``: max that propagates NaN.

    That is how a framework's ``max_pool2d`` is spelled when it must return NaN for a window holding
    one. On any NaN-free input it IS the maximum, element for element; a reduction of it over a window
    is a max-pool. Recognized by exact structure only -- the operands, the predicates and the order."""
    if not op.regions or not op.regions[0].blocks:
        return False
    block = op.regions[0].blocks[0]
    if len(block.args) < 2:
        return False
    x, acc = block.args[0], block.args[-1]
    ops = list(block.ops)
    if [mq.op_name(o) for o in ops] != ["arith.cmpf", "arith.cmpf", "arith.ori", "arith.select", "linalg.yield"]:
        return False
    nan, greater, either, pick, done = ops

    def predicate(cmp) -> int | None:
        raw = cmp.properties.get("predicate") or cmp.attributes.get("predicate")
        value = getattr(raw, "value", None)
        return int(value.data) if value is not None else None

    return (
        list(nan.operands) == [x, x]
        and predicate(nan) == _CMPF_UNO
        and list(greater.operands) == [x, acc]
        and predicate(greater) == _CMPF_OGT
        and set(either.operands) == {nan.results[0], greater.results[0]}
        and list(pick.operands) == [either.results[0], x, acc]
        and list(done.operands) == [pick.results[0]]
    )


def classify_generic(op) -> str:
    """What this `linalg.generic` computes, by structure alone."""
    reduces = "reduction" in _iterator_text(op)
    body = _body_op_names(op)
    if reduces and _SELECT_MAX_IS_MAX.get() and _is_select_max(op):
        return "max-reduction"
    if reduces and (body & _MUL) and (body & _ADD):
        return "contraction"
    if reduces and (body & _MAX) and (body & _ABS):
        return "absmax"
    if reduces and (body & _MAX):
        return "max-reduction"
    if reduces and (body & _ADD):
        return "sum-reduction"
    if reduces:
        return "other-reduction"
    if not body - {"linalg.yield"}:
        return "movement"  # a copy or broadcast: no arithmetic at all
    return "elementwise"


def loop_extents(op) -> dict[int, int] | None:
    """Iteration-dim index -> extent, read off the operands through their indexing maps.

    Returns None when the maps are absent or carry an expression this does not understand, so the
    caller can record the op as unpriceable rather than silently treating it as zero work.
    """
    maps = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    if maps is None:
        return None
    operands = list(op.operands)
    extents: dict[int, int] = {}
    for i, mattr in enumerate(maps):
        if i >= len(operands):
            break
        shape, _ = mq.type_shape_dtype(operands[i].type)
        for j, expr in enumerate(getattr(mattr.data, "results", ())):
            pos = getattr(expr, "position", None)  # a plain dim expr; anything else is skipped
            if pos is not None and j < len(shape):
                extents.setdefault(int(pos), int(shape[j]))
    return extents or None


def _product(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def contraction_coverage(module: Any) -> CoverageReport:
    """Price every contraction in `module`, split by whether the matcher recognised it."""
    rep = CoverageReport()

    for g in mq.walk(module, "linalg.generic"):
        label = classify_generic(g)
        rep.labels[label] = rep.labels.get(label, 0) + 1
        if label != "contraction":
            continue
        ext = loop_extents(g)
        rtype = str(g.results[0].type) if g.results else "<no result>"
        if ext is None:
            rep.unpriceable.append(rtype)
            continue
        rep.unlowered.append(
            UnloweredContraction(
                result_type=rtype,
                loop_extents=tuple(sorted(ext.items())),
                macs=_product(ext.values()),
            )
        )

    for name in MATMUL_OPS:
        for op in mq.walk(module, name):
            if not op.results:
                continue
            out_shape, _ = mq.type_shape_dtype(op.results[0].type)
            ins = [mq.type_shape_dtype(o.type)[0] for o in op.operands]
            # A matmul contracts the last dimension of its first operand; that is the one extent the
            # result type does not carry.
            k = ins[0][-1] if ins and ins[0] else 1
            rep.lowered_macs += _product(out_shape) * int(k)

    return rep

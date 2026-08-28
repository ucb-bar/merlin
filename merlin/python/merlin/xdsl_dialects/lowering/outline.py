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

from dataclasses import dataclass, field
from typing import Any

from .._common import HAS_XDSL


class OutlineError(RuntimeError):
    pass


# Pure, cheap producers cloned into each consuming kernel instead of being outlined on
# their own (matches IREE CloneProducersIntoDispatchRegions). Keeping the fill inside the
# kernel lets per-kernel bufferization fold the zero-init into the contraction.
CLONE_INTO_KERNEL = ("arith.constant", "tensor.empty", "linalg.fill")


def _is_root(op) -> bool:
    """A heavy compute op that becomes its own kernel."""
    return op.name.startswith("linalg.") and op.name not in (
        "linalg.fill", "linalg.yield", "linalg.index")


def _is_block_arg(value) -> bool:
    from xdsl.ir import BlockArgument

    return isinstance(value, BlockArgument)


@dataclass
class DispatchInfo:
    """Bookkeeping for one outlined dispatch (consumed by interface/runtime lowering)."""

    index: int
    symbol: str
    root_op: str                       # e.g. "linalg.matmul"
    n_operands: int
    result_types: list[str]
    prov: dict[str, str] = field(default_factory=dict)


@dataclass
class OutlineResult:
    module: Any
    dispatches: list[DispatchInfo] = field(default_factory=list)

    @property
    def n_kernels(self) -> int:
        return len(self.dispatches)


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
        return None            # not a symbol this outliner emitted; claim nothing about it
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
        if _cloneable(operand.owner):
            visit(operand.owner)
    ops.append(root)
    return ops


def _free_values(kernel_ops):
    """Ordered, de-duplicated values the kernel must receive as arguments.

    A value is *free* (a parameter) when it is used anywhere in the kernel — including
    deep inside an op's regions — but is defined outside it. This must recurse into
    regions: model2MLIR ``linalg.generic`` bodies *capture* outer tensors (e.g. a
    ``tensor.extract`` gather for RoPE/embeddings), and those captures are not visible
    as top-level operands. Missing them yields an ``IsolatedFromAbove`` violation when
    the op is cloned into the kernel func.

    Values defined *inside* the kernel — results of any kernel op (or nested op) and the
    block arguments of the ops' own regions (e.g. a generic's ``^bb0`` element args) —
    are not parameters.
    """
    internal: set[int] = set()
    all_ops: list = []
    for op in kernel_ops:
        for sub in op.walk():
            all_ops.append(sub)
            for res in sub.results:
                internal.add(id(res))
            for region in sub.regions:
                for block in region.blocks:
                    for arg in block.args:
                        internal.add(id(arg))

    params: list = []
    seen: set[int] = set()
    for sub in all_ops:
        for operand in sub.operands:
            if id(operand) in internal or id(operand) in seen:
                continue
            seen.add(id(operand))
            params.append(operand)
    return params


def outline_dispatches(module, forward: str | None = None) -> OutlineResult:
    """Outline each compute dispatch of ``func @forward`` into its own kernel func.

    Returns the rewritten module (driver + kernels) and the per-dispatch table.
    """
    if not HAS_XDSL:
        return OutlineResult(module=module)
    from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
    from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise OutlineError("no func.func in module")
    if forward is not None:
        fns = [f for f in fns if f.sym_name.data == forward]
        if not fns:
            raise OutlineError(f"func @{forward} not found")
    fn = fns[0]
    fname = fn.sym_name.data
    src = fn.body.blocks[0]

    arg_types = [a.type for a in src.args]
    driver = Block(arg_types=arg_types)
    dmap = dict(zip(src.args, driver.args))  # original value -> driver value

    kernels: list = []
    dispatches: list[DispatchInfo] = []

    for op in list(src.ops):
        if op.name == "func.return":
            driver.add_op(ReturnOp(*[dmap[o] for o in op.operands]))
            continue
        if not _is_root(op):
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
        prov = _prov(op)
        rid = prov.get("prov.region_id")
        suffix = f"__r{_sanitize_symbol(rid)}" if rid else ""
        symbol = f"{fname}$kernel_{idx}{suffix}"
        kernel_ops = _producer_closure(op)
        params = _free_values(kernel_ops)
        result_types = [r.type for r in op.results]

        kblock = Block(arg_types=[p.type for p in params])
        kmap = dict(zip(params, kblock.args))
        for kop in kernel_ops:
            c = kop.clone(value_mapper=kmap)
            kblock.add_op(c)
            for old, new in zip(kop.results, c.results):
                kmap[old] = new
        kblock.add_op(ReturnOp(*[kmap[r] for r in op.results]))
        kfn = FuncOp(symbol, FunctionType.from_lists([p.type for p in params],
                                                     result_types),
                     Region([kblock]))
        kfn.sym_visibility = StringAttr("private")
        kernels.append(kfn)

        # --- the driver-side call, mapping external operands to driver values ---
        call = CallOp(symbol, [dmap[p] for p in params], result_types)
        driver.add_op(call)
        for old, new in zip(op.results, call.results):
            dmap[old] = new

        dispatches.append(DispatchInfo(
            index=idx, symbol=symbol, root_op=op.name, n_operands=len(params),
            result_types=[str(t) for t in result_types], prov=prov))

    new_fn = FuncOp(fname, FunctionType.from_lists(
        arg_types, list(fn.function_type.outputs.data)), Region([driver]))
    # Keep any function-level attributes (e.g. llvm.emit_c_interface) on the driver.
    for key, val in fn.attributes.items():
        if key not in ("sym_name", "function_type", "sym_visibility"):
            new_fn.attributes[key] = val

    out = ModuleOp([new_fn, *kernels])
    out.verify()
    return OutlineResult(module=out, dispatches=dispatches)

"""``linalg-on-tensors`` interface grammar: a structural reader for the *second* frozen input
format the experiment ABI hands an out-of-tree backend package.

Two input grammars ship over the same ``capsule.interface.mlir`` slot:

- ``merlin_iface`` v0.1 (see :mod:`.interface_emit`) — a tiny residency command list
  (``merlin_iface.tensor/matmul/commit``). Matmul-family only.
- ``linalg-on-tensors`` (this module) — a ``func.func @forward`` whose body is standard MLIR
  ``linalg``/``tensor``/``math`` ops, tagged with ``prov.op``/``prov.family`` provenance. This is
  what model-slice capsules carry for the op classes that have no residency mnemonic
  (softmax, rmsnorm, layernorm, gelu, rope, elementwise …) plus matmul expressed as
  ``linalg.matmul``/``linalg.generic``.

This reader is the front-door **parse** for that grammar: it returns a structural inventory of the
workload (entry signature + one record per payload op, with provenance, operand/result shapes and
dtypes, matmul extents, and — for ``linalg.generic``/``reduce`` — the inner arithmetic op names that
name the elementwise/reduction semantics). It is the exact analogue of
:func:`.interface_emit.parse_interface_mlir`: a **reader**, not a lowering. A backend package walks
this inventory and authors its own lowering to the target command buffer.

Design constraints (mirror the contract surface rules):
- Target-agnostic: the grammar is fixed by the contract, identical for every accelerator. No target
  semantics, no target-name literals.
- Structural, NOT textual: parse the real IR with xDSL (``linalg``/``tensor``/``math`` dialects) and
  read fields off the typed ops. No regex — xDSL 0.68 parses every shipped capsule with a plain
  permissive context (no textual pre-normalization required).
"""
from __future__ import annotations

from typing import Any

LEVEL = "linalg-on-tensors"

# linalg named ops whose semantics are a matmul-family contraction (M,K)x(K,N)->(M,N), possibly
# batched. A contraction may ALSO arrive as a `linalg.generic` with a reduction iterator — in that
# case the record's ``prov_family == "contraction"`` still marks it, and ``reduction_dims`` is set.
_MATMUL_KINDS = ("linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul", "linalg.matmul_transpose_b")


def make_linalg_context():
    """A permissive xDSL context that loads exactly the dialects the linalg-on-tensors grammar uses.

    ``allow_unregistered=True`` tolerates the ``prov.*`` attribute namespace the capsules carry.
    """
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.cf import Cf
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor as TensorDialect

    ctx = Context(allow_unregistered=True)
    for d in (Builtin, Func, Arith, Linalg, TensorDialect, Scf, Math, Cf):
        ctx.load_dialect(d)
    return ctx


def is_linalg_on_tensors(text: str) -> bool:
    """Cheap discriminator: does this interface MLIR carry the linalg-on-tensors provenance level?

    Lets a package's ``parse`` route between the two grammars without committing to a full parse.
    """
    return f'prov.level = "{LEVEL}"' in text


# --------------------------------------------------------------------------- structural field reads


def _shape(t) -> list[int]:
    from xdsl.dialects.builtin import TensorType

    return [int(d) for d in t.get_shape()] if isinstance(t, TensorType) else []


def _dtype(t) -> str:
    from xdsl.dialects.builtin import TensorType

    return str(t.element_type) if isinstance(t, TensorType) else str(t)


def _prov(op) -> dict[str, str]:
    from xdsl.dialects.builtin import StringAttr

    out: dict[str, str] = {}
    for table in (op.attributes, getattr(op, "properties", {}) or {}):
        for key, val in table.items():
            if key.startswith("prov.") and isinstance(val, StringAttr):
                out[key[len("prov."):]] = val.data
    return out


def _ins_outs(op) -> tuple[list, list]:
    """The linalg operand split (ins= data operands, outs= init/destination operands).

    Named + generic linalg ops expose ``.inputs``/``.outputs``; anything else falls back to all
    operands as inputs and none as outputs.
    """
    ins = list(getattr(op, "inputs", []) or [])
    outs = list(getattr(op, "outputs", []) or [])
    if not ins and not outs:
        ins = list(op.operands)
    return ins, outs


def _body_op_names(op) -> list[str]:
    """Inner arithmetic op names of a ``linalg.generic``/``reduce``/``map`` body (``arith.addf``,
    ``arith.maximumf``, ``math.exp`` …), in order, minus the terminator. These NAME the elementwise
    / reduction semantics that the op family alone does not (add vs mul vs sub vs div; max vs sum;
    exp/rsqrt/tanh/sin/cos)."""
    names: list[str] = []
    for region in getattr(op, "regions", []) or []:
        for block in region.blocks:
            for inner in block.ops:
                if inner.name in ("linalg.yield", "func.return"):
                    continue
                names.append(inner.name)
    return names


def _reduction_dims(op) -> list[int]:
    """The reduction iterator positions of a linalg op (empty for pure-parallel elementwise).

    Read structurally from ``iterator_types`` when present; ``linalg.reduce`` carries explicit
    ``dimensions``.
    """
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

    attrs = {**op.attributes, **(getattr(op, "properties", {}) or {})}
    dims = attrs.get("dimensions")
    if dims is not None and hasattr(dims, "get_values"):   # DenseArrayBase (linalg.reduce dimensions)
        return [int(v) for v in dims.get_values()]
    if isinstance(dims, (list, tuple)):
        return [int(d.value.data) if isinstance(d, IntegerAttr) else int(d) for d in dims]
    if isinstance(dims, ArrayAttr):
        return [int(d.value.data) if isinstance(d, IntegerAttr) else int(d) for d in dims.data]
    its = attrs.get("iterator_types")
    out: list[int] = []
    if isinstance(its, ArrayAttr):
        for i, it in enumerate(its.data):
            s = it.data.data if hasattr(it, "data") and hasattr(it.data, "data") else str(it)
            if "reduction" in str(s):
                out.append(i)
    return out


def _matmul_extents(ins: list, outs: list) -> dict[str, int]:
    """(M,K,N) for a 2-input contraction, derived from operand shapes. Batched forms prepend batch
    dims; take the trailing 2 of each. Returns {} when the shapes are not a clean contraction."""
    if len(ins) < 2:
        return {}
    a, b = _shape(ins[0].type), _shape(ins[1].type)
    if len(a) < 2 or len(b) < 2:
        return {}
    m, k = a[-2], a[-1]
    k2, n = b[-2], b[-1]
    ext = {"m": int(m), "k": int(k), "n": int(n)}
    if len(a) >= 3 or len(b) >= 3:
        ext["batch"] = int(a[0] if len(a) >= 3 else b[0])
    if k2 != k:  # e.g. transpose_b (K,N) stored as (N,K); surface both so the lowering can adapt
        ext["k_rhs"] = int(k2)
    return ext


# --------------------------------------------------------------------------- parse


def parse_linalg_mlir(text: str, *, ctx=None) -> dict[str, Any]:
    """Parse ``linalg-on-tensors`` interface MLIR into a structural workload inventory.

    Returns::

        {
          "level": "linalg-on-tensors",
          "entry": "forward",
          "args":   [{"index": 0, "shape": [16,16], "dtype": "bf16"}, ...],   # @forward operands
          "results":[{"shape": [16,16], "dtype": "bf16"}, ...],               # @forward results
          "ops": [                                                            # payload ops, in order
            {"id": 0, "kind": "linalg.matmul", "op": "matmul", "family": "contraction",
             "prov": {...}, "ins": [{"source": ("arg", 0), "shape": [16,16], "dtype": "bf16"}, ...],
             "outs": [{"source": ("op", -1) | ("init", "fill"), "shape": [...], "dtype": "..."}],
             "results": [{"shape": [...], "dtype": "..."}], "extents": {"m":16,"k":16,"n":16},
             "body_ops": [], "reduction_dims": []},
            ...
          ],
        }

    ``source`` is the structural dataflow edge for each operand: ``("arg", i)`` a ``@forward`` operand,
    ``("op", j)`` the result of payload op ``j``, ``("init", kind)`` a ``tensor.empty``/``linalg.fill``
    destination, or ``("const", None)`` an ``arith.constant``. This is the DAG a backend lowers.
    """
    from xdsl.ir import BlockArgument

    try:
        from ...common.ir_lock import IR_LOCK
    except ImportError:  # sandbox: staged flat on sys.path (no parent package) — a per-process lock is
        import threading  # semantically sufficient (each entrypoint parses single-process)
        IR_LOCK = threading.Lock()
    from xdsl.parser import Parser

    with IR_LOCK:
        module = Parser(ctx or make_linalg_context(), text).parse_module()

    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise ValueError("linalg-on-tensors interface has no func.func entry")
    fn = fns[0]
    entry = _fn_name(fn)
    block = fn.body.blocks[0]
    func_args = list(block.args)

    # The payload ops we surface are the DIRECT children of the @forward entry block (never the
    # arithmetic ops nested inside a linalg.generic/reduce region body — those are captured per op in
    # ``body_ops``, and walking them as top-level ops would double-count e.g. a softmax's inner
    # ``math.exp``). Structural init ops (tensor.empty / arith.constant / linalg.fill / tensor.splat)
    # are destinations/constants: referenced as operand ``source``s but not lowered as commands.
    _INIT = ("tensor.empty", "arith.constant", "linalg.fill", "tensor.splat")
    _SKIP = _INIT + ("func.return", "linalg.yield")
    payload = [op for op in block.ops if op.name not in _SKIP]

    # map every SSA result value -> the payload-op id that produced it (for dataflow edges)
    result_owner: dict[Any, int] = {}
    for i, op in enumerate(payload):
        for res in op.results:
            result_owner[res] = i

    def _source(value):
        if isinstance(value, BlockArgument):
            return ("arg", func_args.index(value)) if value in func_args else ("arg", None)
        owner = value.owner
        oname = getattr(owner, "name", "")
        if value in result_owner:
            return ("op", result_owner[value])
        if oname in ("tensor.empty", "linalg.fill", "tensor.splat"):
            return ("init", oname.split(".")[-1])
        if oname == "arith.constant":
            return ("const", None)
        return ("other", oname)

    def _operand_rec(value):
        return {"source": _source(value), "shape": _shape(value.type), "dtype": _dtype(value.type)}

    ops_out: list[dict[str, Any]] = []
    for i, op in enumerate(payload):
        ins, outs = _ins_outs(op)
        prov = _prov(op)
        rec: dict[str, Any] = {
            "id": i,
            "kind": op.name,
            "op": prov.get("op", op.name.split(".")[-1]),
            "family": prov.get("family", ""),
            "prov": prov,
            "ins": [_operand_rec(v) for v in ins],
            "outs": [_operand_rec(v) for v in outs],
            "results": [{"shape": _shape(r.type), "dtype": _dtype(r.type)} for r in op.results],
            "body_ops": _body_op_names(op),
            "reduction_dims": _reduction_dims(op),
        }
        ext = {}
        if op.name in _MATMUL_KINDS or prov.get("family") == "contraction":
            ext = _matmul_extents(ins, outs)
        if ext:
            rec["extents"] = ext
        ops_out.append(rec)

    return {
        "level": LEVEL,
        "entry": entry,
        "args": [{"index": i, "shape": _shape(a.type), "dtype": _dtype(a.type)}
                 for i, a in enumerate(func_args)],
        "results": [{"shape": _shape(t), "dtype": _dtype(t)} for t in _fn_result_types(fn)],
        "ops": ops_out,
    }


def _fn_name(fn) -> str:
    from xdsl.dialects.builtin import StringAttr

    for table in (getattr(fn, "properties", {}) or {}, fn.attributes):
        v = table.get("sym_name")
        if isinstance(v, StringAttr):
            return v.data
    return "forward"


def _fn_result_types(fn) -> list:
    """The @forward result types, read from the function type."""
    ft = None
    for table in (getattr(fn, "properties", {}) or {}, fn.attributes):
        if "function_type" in table:
            ft = table["function_type"]
            break
    if ft is not None and hasattr(ft, "outputs"):
        return list(ft.outputs.data)
    # fallback: the terminator's operand types
    block = fn.body.blocks[0]
    term = block.ops.last
    return [o.type for o in term.operands] if term is not None else []


def matmul_records(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Convenience view: just the contraction (matmul-family) op records, with extents.

    A package that only lowers the matmul family (reusing the residency command path) filters here.
    """
    return [o for o in parsed.get("ops", [])
            if o["kind"] in _MATMUL_KINDS or o.get("family") == "contraction"]

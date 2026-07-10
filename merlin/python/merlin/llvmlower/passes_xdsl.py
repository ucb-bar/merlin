"""Merlin-authored xDSL passes that make the model module pure-upstream MLIR.

These run before the upstream pipeline:

- :func:`lower_quant_ext` — rewrite model2MLIR's ``quant_ext.dequantize_per_channel``
  (i8 weights + per-channel f32 scales + i32 zero points) into ``linalg.generic``:
  ``out[i,j] = (sitofp(w[i,j]) - sitofp(zp[j])) * scale[j]`` (axis-broadcast). After
  this, the module contains only upstream linalg/arith/tensor/scf/func ops.
- :func:`add_c_interface` — attach ``llvm.emit_c_interface`` so each public func gets
  a `_mlir_ciface_<name>` wrapper taking one pointer per memref argument.
"""
from __future__ import annotations

import re

from ..frontends.linalg_mlir import make_context, parse_mlir_text  # noqa: F401


def collapse_overrank_matmul(module) -> int:
    """Fix `linalg.matmul` ops with a >2-D LHS + 2-D RHS (model2MLIR emits `aten.linear` on a
    3-D activation as a 2-D-map matmul over 3-D operands, which is invalid linalg). Rebuild
    each as a `linalg.generic` batched matmul with a broadcast RHS, using proper-rank maps:

        out[b.., n] = sum_k lhs[b.., k] * rhs[k, n]

    Iteration dims = (R-1 leading parallel) + N(parallel) + K(reduction). Returns count.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr,
                                       IntegerType, TensorType)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION
    targets = [op for op in module.walk()
               if op.name == "linalg.matmul" and isinstance(op.results[0].type, TensorType)
               and len(op.results[0].type.get_shape()) > 2
               and len(op.inputs[1].type.get_shape()) == 2]
    for op in targets:
        lhs, rhs = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        et = out_t.element_type
        R = len(out_t.get_shape())                 # out rank (== lhs rank)
        ndim = R + 1                                # leading(R-1) + N + K
        idm = AffineMap.identity(ndim)
        dims = idm.results
        lead = dims[:R - 1]
        N, K = dims[R - 1], dims[R]
        lhs_map = AffineMap(ndim, 0, (*lead, K))
        rhs_map = AffineMap(ndim, 0, (K, N))
        out_map = AffineMap(ndim, 0, (*lead, N))
        iters = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1)
                          + [L.IteratorTypeAttr(par), L.IteratorTypeAttr(red)])
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        zero = arith.ConstantOp(IntegerAttr(0, et) if isinstance(et, IntegerType)
                                else FloatAttr(0.0, et))
        fill = L.FillOp(inputs=[zero.results[0]], outputs=[empty.results[0]], res=[out_t])
        body = Block(arg_types=[et, et, et])
        a, b, acc = body.args
        prod = arith.MulfOp(a, b)
        summ = arith.AddfOp(acc, prod.result)
        body.add_ops([prod, summ, L.YieldOp(summ.result)])
        gen = L.GenericOp(
            inputs=(lhs, rhs), outputs=(fill.results[0],), body=Region(body),
            indexing_maps=ArrayAttr([AffineMapAttr(lhs_map), AffineMapAttr(rhs_map),
                                     AffineMapAttr(out_map)]),
            iterator_types=iters, result_types=(out_t,))
        for key, val in op.attributes.items():
            if key.startswith("prov."):
                gen.attributes[key] = val
        for new_op in (empty, zero, fill, gen):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(gen.results[0])
        block.detach_op(op)
    return len(targets)


def lower_quant_ext(module) -> int:
    """Rewrite all dequantize_per_channel ops; returns the number rewritten."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, IntegerType,
                                       StringAttr, TensorType)
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.dialects.builtin import f32
    from xdsl.ir import Attribute, Block, Region
    from xdsl.utils.hints import isa
    from xdsl.ir.affine import AffineMap

    rewrites = []
    for op in module.walk():
        # quant_ext is not a registered dialect; ops parse as builtin.unregistered.
        name = getattr(op, "op_name", None)
        if op.name == "builtin.unregistered" and name is not None \
                and name.data == "quant_ext.dequantize_per_channel":
            rewrites.append(op)

    for op in rewrites:
        w, scale, zp = op.operands
        out_t = op.results[0].type
        rank = len(out_t.get_shape())
        axis = int(op.properties["axis"].value.data)
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        identity = AffineMap.identity(rank)
        axis_proj = AffineMap(rank, 0, (identity.results[axis],))
        maps = ArrayAttr([AffineMapAttr(identity), AffineMapAttr(axis_proj),
                          AffineMapAttr(axis_proj), AffineMapAttr(identity)])
        iters = ArrayAttr([linalg_ops.IteratorTypeAttr(linalg_ops.IteratorType.PARALLEL)
                           for _ in range(rank)])

        elem = out_t.element_type  # f32 or bf16 — arithmetic in the output type
        body = Block(arg_types=[w.type.element_type, scale.type.element_type,
                                zp.type.element_type, elem])
        wv, sv, zv, _ = body.args
        wf = arith.SIToFPOp(wv, elem)
        zf = arith.SIToFPOp(zv, elem)
        sub = arith.SubfOp(wf.result, zf.result)
        mul = arith.MulfOp(sub.result, sv)
        body.add_ops([wf, zf, sub, mul, linalg_ops.YieldOp(mul.result)])

        generic = linalg_ops.GenericOp(
            inputs=(w, scale, zp),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=maps,
            iterator_types=iters,
            result_types=(out_t,),
        )
        # Preserve provenance for downstream tooling.
        for key, val in op.attributes.items():
            if key.startswith("prov."):
                generic.attributes[key] = val

        block.insert_op_before(empty, op)
        block.insert_op_before(generic, op)
        op.results[0].replace_all_uses_with(generic.results[0])
        block.detach_op(op)
    return len(rewrites)


def lower_bf16_matmul_f32acc(module) -> int:
    """Rewrite every bf16 ``linalg.matmul`` to accumulate in f32; returns the count.

    A bf16 ``linalg.matmul`` accumulates in its bf16 output type, rounding every partial
    sum to 8-bit mantissa — lossy over long contractions. Hardware/torch instead accumulate
    in f32 and round only the final result. This rewrites

        C_bf16 = matmul(A_bf16, B_bf16)              # bf16 accumulate

    into a ``linalg.generic`` matmul whose body extends each operand to f32, multiplies and
    accumulates in f32, followed by a ``truncf`` back to bf16 — matching the reference
    numerics. (``extf``/``truncf`` here are scalar ops inside the generic body, so the
    earlier "extf is tensor-rank-polymorphic" limitation does not apply.)

    Also handles bf16 ``linalg.generic`` contractions (e.g. ``batch_matmul``, ``aten.bmm``)
    whose body is the standard ``mulf``/``addf`` reduction; these otherwise accumulate in
    bf16 too (attention scores/context), the dominant precision loss on VLAs. Their own
    indexing maps + iterator types are reused; only the accumulation dtype is promoted.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (AffineMapAttr, ArrayAttr, BFloat16Type, FloatAttr,
                                       TensorType, f32)
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    bf16 = BFloat16Type()
    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def is_bf16_tensor(t):
        return isinstance(t, TensorType) and isinstance(t.element_type, BFloat16Type)

    def is_mul_add_contraction(op):
        """A 2-input linalg.generic with a reduction whose body is mulf+addf+yield."""
        if op.name != "linalg.generic" or len(op.inputs) != 2:
            return False
        iters = [a for a in op.iterator_types]
        if not any(getattr(a, "data", a) == L.IteratorType.REDUCTION for a in iters):
            return False
        names = [b.name for b in op.body.blocks[0].ops]
        return names == ["arith.mulf", "arith.addf", "linalg.yield"]

    targets = []
    for op in module.walk():
        if op.name == "linalg.matmul" and is_bf16_tensor(op.results[0].type):
            targets.append(("matmul", op))
        elif is_mul_add_contraction(op) and is_bf16_tensor(op.results[0].type) \
                and all(is_bf16_tensor(i.type) for i in op.inputs):
            targets.append(("generic", op))
    if not targets:
        return 0

    idm = AffineMap.identity(3)
    d0, d1, d2 = idm.results
    mm_maps = ArrayAttr([AffineMapAttr(AffineMap(3, 0, (d0, d2))),
                         AffineMapAttr(AffineMap(3, 0, (d2, d1))),
                         AffineMapAttr(AffineMap(3, 0, (d0, d1)))])
    mm_iters = ArrayAttr([L.IteratorTypeAttr(par), L.IteratorTypeAttr(par),
                          L.IteratorTypeAttr(red)])

    for kind, op in targets:
        a, b = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        f32_t = TensorType(f32, out_t.get_shape())
        block = op.parent_block()
        maps = mm_maps if kind == "matmul" else op.indexing_maps
        iters = mm_iters if kind == "matmul" else op.iterator_types

        empty_f = tensor.EmptyOp((), f32_t)
        zero = arith.ConstantOp(FloatAttr(0.0, f32))
        fill_f = L.FillOp(inputs=[zero.results[0]], outputs=[empty_f.results[0]], res=[f32_t])
        body = Block(arg_types=[bf16, bf16, f32])
        x, y, acc = body.args
        xf, yf = arith.ExtFOp(x, f32), arith.ExtFOp(y, f32)
        prod = arith.MulfOp(xf.result, yf.result)
        summ = arith.AddfOp(acc, prod.result)
        body.add_ops([xf, yf, prod, summ, L.YieldOp(summ.result)])
        acc_f32 = L.GenericOp(inputs=(a, b), outputs=(fill_f.results[0],),
                              body=Region(body), indexing_maps=maps, iterator_types=iters,
                              result_types=(f32_t,))

        # truncf the f32 accumulator back to bf16 (identity map of the result rank)
        rank = len(out_t.get_shape())
        idr = AffineMap.identity(rank)
        empty_b = tensor.EmptyOp((), out_t)
        body2 = Block(arg_types=[f32, bf16])
        trf = arith.TruncFOp(body2.args[0], bf16)
        body2.add_ops([trf, L.YieldOp(trf.result)])
        trunc = L.GenericOp(
            inputs=(acc_f32.results[0],), outputs=(empty_b.results[0],), body=Region(body2),
            indexing_maps=ArrayAttr([AffineMapAttr(idr), AffineMapAttr(idr)]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * rank),
            result_types=(out_t,))

        for key, val in op.attributes.items():
            if key.startswith("prov."):
                acc_f32.attributes[key] = val
        for new_op in (empty_f, zero, fill_f, acc_f32, empty_b, trunc):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(trunc.results[0])
        block.detach_op(op)
    return len(targets)


def fix_bool_sitofp(module) -> int:
    """Rewrite ``arith.sitofp`` of an ``i1`` (a bool) to ``arith.uitofp``; returns the count.

    PyTorch casts a bool tensor to float as ``True -> 1.0`` (unsigned). model2MLIR, however,
    emits a *signed* ``arith.sitofp %b : i1 to fN`` for these casts. ``sitofp`` interprets the
    i1 as a signed 1-bit integer, whose ``true`` bit pattern is the value ``-1`` (two's
    complement), so the compiled kernel produces ``-1.0`` instead of ``+1.0``. This is silent
    on the driver's scalar interpreter (Python ``float(True) == 1.0``) but wrong in every
    *compiled* kernel (host clang and RVV alike).

    The canonical place it bites: the eager-attention causal mask, built as
    ``mask_f32 * sitofp(bool)``; the sign flip turns the ``-FLT_MAX`` fill into ``+FLT_MAX``,
    so softmax attends to *future* tokens (molmoact decoder cos 0.10). Rewriting to
    ``uitofp`` (``true -> 1.0``) matches torch. ``module.walk()`` recurses into linalg.generic
    bodies, so casts inside outlined kernels are covered. Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name != "arith.sitofp":
            continue
        src = op.operands[0]
        st = src.type
        if not (isinstance(st, IntegerType) and st.width.data == 1):
            continue
        new = arith.UIToFPOp(src, op.results[0].type)
        for key, val in op.attributes.items():
            if key.startswith("prov."):
                new.attributes[key] = val
        op.parent_block().insert_op_before(new, op)
        op.results[0].replace_all_uses_with(new.results[0])
        op.parent_block().detach_op(op)
        n += 1
    return n


def add_c_interface(module) -> int:
    """Mark public funcs with llvm.emit_c_interface; returns count marked."""
    from xdsl.dialects.builtin import UnitAttr

    n = 0
    for op in module.walk():
        if op.name == "func.func":
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
            n += 1
    return n


def preprocess_text(mlir_text: str) -> tuple[str, dict]:
    """Run all Merlin xDSL passes on linalg-on-tensors MLIR text -> upstream text."""
    from ..frontends.linalg_mlir import parse_mlir_text
    from ..xdsl_dialects._common import text as module_to_text

    module = parse_mlir_text(mlir_text)
    stats = {
        "dequantize_lowered": lower_quant_ext(module),
        "c_interface_funcs": add_c_interface(module),
    }
    return module_to_text(module), stats


# --- textual variant -----------------------------------------------------------
#
# xDSL 0.65 re-prints rank-reducing tensor.extract_slice with truncated
# static_sizes, so round-tripping the full smolVLA module through xDSL produces
# invalid IR. The model artifacts keep each `quant_ext.dequantize_per_channel` on a
# single line, so the same rewrite is done textually for whole-model lowering.

_DEQUANT_RE = re.compile(
    r"^(?P<ind>\s*)(?P<res>%\S+) = \"quant_ext\.dequantize_per_channel\""
    r"\((?P<w>%\S+), (?P<s>%\S+), (?P<z>%\S+)\) <\{axis = (?P<axis>\d+) : i64[^}]*\}>"
    r".* : \((?P<wty>[^,]+), (?P<sty>[^,]+), (?P<zty>[^)]+)\) -> (?P<outty>.+)$")


def _dequant_to_generic(m: "re.Match[str]") -> str:
    ind, res = m["ind"], m["res"]
    init = f"%dq_init_{res[1:]}"   # %123 -> %dq_init_123 (suffixing %123 is invalid)
    wty, sty, zty, outty = (m[k].strip() for k in ("wty", "sty", "zty", "outty"))
    axis = int(m["axis"])
    shape = outty[len("tensor<"):-1].split("x")
    elem = shape[-1]
    rank = len(shape) - 1
    dims = ", ".join(f"d{i}" for i in range(rank))
    ident = f"affine_map<({dims}) -> ({dims})>"
    proj = f"affine_map<({dims}) -> (d{axis})>"
    iters = ", ".join(['"parallel"'] * rank)
    welem = wty[len("tensor<"):-1].split("x")[-1]
    zelem = zty[len("tensor<"):-1].split("x")[-1]
    return (
        f"{ind}{init} = tensor.empty() : {outty}\n"
        f"{ind}{res} = linalg.generic {{indexing_maps = [{ident}, {proj}, {proj}, "
        f"{ident}], iterator_types = [{iters}]}} "
        f"ins({m['w']}, {m['s']}, {m['z']} : {wty}, {sty}, {zty}) "
        f"outs({init} : {outty}) {{\n"
        f"{ind}^bb0(%w_el: {welem}, %s_el: {elem}, %z_el: {zelem}, %o_el: {elem}):\n"
        f"{ind}  %dq_wf = arith.sitofp %w_el : {welem} to {elem}\n"
        f"{ind}  %dq_zf = arith.sitofp %z_el : {zelem} to {elem}\n"
        f"{ind}  %dq_df = arith.subf %dq_wf, %dq_zf : {elem}\n"
        f"{ind}  %dq_r = arith.mulf %dq_df, %s_el : {elem}\n"
        f"{ind}  linalg.yield %dq_r : {elem}\n"
        f"{ind}}} -> {outty}")


_FUNC_RE = re.compile(r"(func\.func @\w+\([^{]*?\))\s*(->\s*[^{]*?)?\s*\{")

# model2MLIR emits rank-reduced tensor.extract_slice with ONLY the result dims in
# static_sizes (e.g. sizes [32] for source tensor<1x32x32xi1>). Upstream MLIR
# requires offsets/sizes/strides to match the SOURCE rank — left-pad sizes with 1s.
_EXTRACT_SLICE_RE = re.compile(
    r"\"tensor\.extract_slice\"\((?P<args>[^)]+)\) "
    r"<\{(?P<props>[^}]*static_sizes = array<i64: (?P<sizes>[^>]+)>[^}]*)\}>"
    r"(?P<attrs>[^:]*): \(tensor<(?P<src>[^>]+)>\)")


# Pre-fix model2MLIR artifacts that predate the slice_scatter step fix
# (m2m/ir/decompositions.py read `step` from the `end` arg slot, so insert_slice got
# strides[dim]=end instead of step). `step` is almost always 1; reset any stride that
# overruns the destination back to 1. (New captures don't hit this.) Matches the full
# attribute dict so trailing operandSegmentSizes after static_strides is fine.
_INSERT_SLICE_RE = re.compile(
    r"\"tensor\.insert_slice\"\((?P<args>[^)]+)\) "
    r"<\{(?P<props>[^}]*static_sizes = array<i64: (?P<sizes>[^>]+)>"
    r"(?P<mid>[^}]*?)static_strides = array<i64: (?P<strides>[^>]+)>"
    r"(?P<post>[^}]*))\}>"
    r"(?P<attrs>[^:]*): \(tensor<(?P<src>[^>]+)>, tensor<(?P<dst>[^>]+)>\)")


def _fix_insert_slice(m: "re.Match[str]") -> str:
    sizes = [int(s) for s in m["sizes"].split(",")]
    strides = [int(s) for s in m["strides"].split(",")]
    dst_dims = [int(d) for d in m["dst"].split("x")[:-1]]
    fixed = [1 if sz * st > dst else st
             for sz, st, dst in zip(sizes, strides, dst_dims)]
    if fixed == strides:
        return m.group(0)
    props = m["props"].replace(
        f"static_strides = array<i64: {m['strides']}>",
        "static_strides = array<i64: " + ", ".join(str(s) for s in fixed) + ">", 1)
    return (f"\"tensor.insert_slice\"({m['args']}) <{{{props}}}>"
            f"{m['attrs']}: (tensor<{m['src']}>, tensor<{m['dst']}>)")


def _fix_extract_slice(m: "re.Match[str]") -> str:
    src_dims = m["src"].split("x")[:-1]
    sizes = [s.strip() for s in m["sizes"].split(",")]
    if len(sizes) >= len(src_dims):
        return m.group(0)
    padded = ["1"] * (len(src_dims) - len(sizes)) + sizes
    props = m["props"].replace(
        f"static_sizes = array<i64: {m['sizes']}>",
        "static_sizes = array<i64: " + ", ".join(padded) + ">")
    return (f"\"tensor.extract_slice\"({m['args']}) <{{{props}}}>"
            f"{m['attrs']}: (tensor<{m['src']}>)")


def preprocess_text_textual(mlir_text: str) -> tuple[str, dict]:
    """Whole-model variant of :func:`preprocess_text` (pure text; no xDSL roundtrip)."""
    out_lines = []
    n = 0
    for line in mlir_text.splitlines():
        m = _DEQUANT_RE.match(line)
        if m:
            out_lines.append(_dequant_to_generic(m))
            n += 1
        else:
            line = _EXTRACT_SLICE_RE.sub(_fix_extract_slice, line)
            line = _INSERT_SLICE_RE.sub(_fix_insert_slice, line)
            out_lines.append(line)
    text = "\n".join(out_lines)

    n_funcs = 0

    def _attach(m: "re.Match[str]") -> str:
        nonlocal n_funcs
        n_funcs += 1
        ret = m.group(2) or ""
        return f"{m.group(1)} {ret} attributes {{llvm.emit_c_interface}} {{"

    text = _FUNC_RE.sub(_attach, text, count=1)
    return text, {"dequantize_lowered": n, "c_interface_funcs": n_funcs}

"""Gemmini Relax BYOC and replayable TensorIR schedules for TVM v0.19.

Install in tvm.relax.backend.contrib. Pattern checks are conservative: this
backend preserves signed INT8 x INT8 -> INT32 semantics, never quantizes floats.
The C1 call_tir deployment preserves the existing static export contract.
"""

from dataclasses import dataclass

import tvm
from tvm import relax, te, tir
from tvm.script import tir as T
from tvm.relax.dpl import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.backend.pattern_registry import register_patterns, get_patterns_with_prefix
from tvm.relax.backend.utils import has_leaking_intermediate_variables
from tvm.relax.backend.contrib.gemmini import _eligible, _inline_cpu_producers


@dataclass
class ScheduledMatmul:
    """Keep the untiled computation and replayable scheduling decisions."""

    semantic_mod: tvm.IRModule
    scheduled_mod: tvm.IRModule
    trace: object
    shape: tuple


def _tile_intrinsic():
    # Symbolic strides allow TVM allocation compaction and compute_at to choose
    # the host staging layout; device addresses are fixed inside one primitive.
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle):
        sa, sb, sc = T.int64(), T.int64(), T.int64()
        A = T.match_buffer(a, (16, 16), "int8", strides=[sa, 1], offset_factor=1)
        B = T.match_buffer(b, (16, 8), "int8", strides=[sb, 1], offset_factor=1)
        C = T.match_buffer(c, (16, 8), "int32", strides=[sc, 1], offset_factor=1)
        with T.block("root"):
            T.reads(C[0:16, 0:8], A[0:16, 0:16], B[0:16, 0:8])
            T.writes(C[0:16, 0:8])
            for i, j, k in T.grid(16, 8, 16):
                with T.block("update"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    C[vi, vj] += T.Cast("int32", A[vi, vk]) * T.Cast("int32", B[vk, vj])

    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle):
        sa, sb, sc = T.int64(), T.int64(), T.int64()
        A = T.match_buffer(a, (16, 16), "int8", strides=[sa, 1], offset_factor=1)
        B = T.match_buffer(b, (16, 8), "int8", strides=[sb, 1], offset_factor=1)
        C = T.match_buffer(c, (16, 8), "int32", strides=[sc, 1], offset_factor=1)
        with T.block("root"):
            T.reads(C[0:16, 0:8], A[0:16, 0:16], B[0:16, 0:8])
            T.writes(C[0:16, 0:8])
            T.evaluate(T.call_extern("void", "tvm_gemmini_tile_update_i8_i32",
                                    A.access_ptr("r"), B.access_ptr("r"), C.access_ptr("rw"), sa, sb, sc))

    name = "gemmini.byoc.tile_update_16x8x16"
    tir.TensorIntrin.register(name, desc, impl, override=True)
    return name


def schedule_matmul(m, n, k, *, order="ijk", seed=1):
    """Tile canonical TE mathematics using only public TVM Schedule primitives.

    pad_einsum supplies exact zero padding for tails. compute_at and
    reverse_compute_at bound staging to the current tile. INT32 initialization
    remains explicit; each device tile implements C += A @ B, including arbitrary
    incoming C. K chunks of 16 respect the pinned signed 20-bit PE output.
    """
    if any(type(x) is not int or x <= 0 for x in (m, n, k)) or k > 131071:
        raise ValueError("requires positive static shapes with K <= 131071")
    if order not in ("ijk", "jik", "sample"):
        raise ValueError("order must be ijk, jik or sample")
    if max(m * k, k * n, m * n * 4) > (1 << 63) - 1:
        raise ValueError("tensor span exceeds RV64")
    if max(k, n * 4) > (1 << 32) - 1:
        raise ValueError("DMA byte stride exceeds Gemmini configuration fields")
    A, B = te.placeholder((m, k), "int8", "A"), te.placeholder((k, n), "int8", "B")
    rk = te.reduce_axis((0, k), "k")
    C = te.compute((m, n), lambda i, j: te.sum(A[i, rk].astype("int32") * B[rk, j].astype("int32"), axis=rk), name="matmul")
    semantic = tvm.IRModule({"main": te.create_prim_func([A, B, C])})
    return schedule_primfunc(semantic["main"], order=order, seed=seed)


def schedule_primfunc(func, *, order="ijk", seed=1):
    """Schedule a TVM-legalized/fused matmul and its pointwise epilogues.

    Fusion and buffer indexing come from LegalizeOps/FuseTIR. Scheduling inlines
    internal pointwise producers and places the remaining epilogue beside its
    output tile, so the complete product need not be materialized in DRAM.
    """
    if order not in ("ijk", "jik", "sample"):
        raise ValueError("order must be ijk, jik or sample")
    semantic = tvm.IRModule({"main": func.with_attr("global_symbol", "main")})
    sch = tir.Schedule(semantic, debug_mask="all", seed=seed)
    if order == "sample":
        decision = sch.sample_categorical([0, 1], [0.5, 0.5])
        order = "ijk" if int(sch.get(decision)) == 0 else "jik"
    external = {buffer.data for buffer in func.buffer_map.values()}
    for child in sch.get_child_blocks(sch.get_block("root")):
        node = sch.get(child)
        if node.init is None and len(node.writes) == 1 and node.writes[0].buffer.data not in external:
            sch.compute_inline(child)
    block = sch.get_block("matmul")
    node = sch.get(block)
    shape = tuple(int(axis.dom.extent) for axis in node.iter_vars)
    m, n, k = shape
    if min(shape) <= 0 or k > 131071 or max(k, n * 4) > (1 << 32) - 1:
        raise ValueError("Fused contraction exceeds the declared Gemmini arithmetic/stride contract")
    read_names = [region.buffer.name for region in node.reads]
    sch.pad_einsum(block, [16, 8, 16])
    # pad_einsum may introduce a copy into the original internal matmul buffer.
    # Inline that copy when it feeds an epilogue, through TVM's legality checks.
    for consumer in sch.get_consumers(block):
        node = sch.get(consumer)
        if node.writes[0].buffer.data not in external:
            sch.compute_inline(consumer)
    i, j, r = sch.get_loops(block)
    io, ii = sch.split(i, factors=[None, 16])
    jo, ji = sch.split(j, factors=[None, 8])
    ko, ki = sch.split(r, factors=[None, 16])
    sch.reorder(*([io, jo] if order == "ijk" else [jo, io]), ko, ii, ji, ki)
    sch.decompose_reduction(block, ko)
    # Padding blocks only exist when the corresponding shape is unaligned.
    for name in (name + "_pad" for name in read_names):
        try:
            padded = sch.get_block(name)
        except tvm.TVMError:
            continue
        sch.compute_at(padded, ko)
    for output in sch.get_consumers(block):
        sch.reverse_compute_at(output, jo if order == "ijk" else io)
    sch.tensorize(ii, _tile_intrinsic())
    return ScheduledMatmul(semantic, sch.mod, sch.trace, shape)


def _check(context):
    if has_leaking_intermediate_variables(context):
        return False
    shape = _eligible(context.annotated_expr["matmul"])
    return shape is not None and max(shape[2], shape[1] * 4) <= (1 << 32) - 1


def _patterns():
    result = []
    # The registry gives later entries higher priority.
    for bias, relu in ((False, False), (False, True), (True, False), (True, True)):
        a, b = wildcard(), wildcard()
        mm = is_op("relax.matmul")(a, b)
        out = is_op("relax.add")(mm, wildcard()) if bias else mm
        if relu:
            out = is_op("relax.nn.relu")(out)
        name = "gemmini.matmul" + ("_bias" if bias else "") + ("_relu" if relu else "")
        result.append((name, out, {"matmul": mm}, _check))
    return result


register_patterns(_patterns())


def partition_for_gemmini(mod):
    """Partition exact static INT8 contractions through the standard BYOC API.

    Constants remain arguments so static export and runtime-module codegen have
    the same explicit ABI. Epilogues retain ordinary Relax/TVM semantics inside
    a composite; INT32 full-width readout does not implement device ReLU.
    """
    return relax.transform.FuseOpsByPattern(
        get_patterns_with_prefix("gemmini."), bind_constants=False, annotate_codegen=True
    )(mod)


@mutator
class _ImportRegion(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.replacements = {}

    def visit_global_var_(self, var):
        return self.replacements.get(var, var)


@mutator
class _DecomposeBatch(PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.matmul":
            return call
        from tvm.relax.backend.contrib.gemmini import _shape

        a = _shape(call.args[0].struct_info, "int8", 3)
        b = _shape(call.args[1].struct_info, "int8", 3)
        out = _shape(call.struct_info, "int32", 3)
        if a is None or b is None or out is None:
            return call
        batch, m, k = a
        if batch > 64 or b[0] != batch or b[1] != k or out != (batch, m, b[2]) or k > 131071:
            return call
        n = b[2]
        products = []
        for index in range(batch):
            operands = []
            for value, shape in zip(call.args, ((m, k), (k, n))):
                sliced = self.builder_.emit(relax.op.strided_slice(value, [0], [index], [index + 1]))
                operands.append(self.builder_.emit(relax.op.reshape(sliced, shape)))
            product = self.builder_.emit(relax.op.matmul(*operands, out_dtype="int32"))
            products.append(self.builder_.emit(relax.op.reshape(product, (1, m, n))))
        return self.builder_.normalize(relax.op.concat(products, axis=0))


@mutator
class _ExpandComposite(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.outlined = {}

    def visit_function_(self, func):
        if func.attrs and "Codegen" in func.attrs and func.attrs["Codegen"] != "gemmini":
            return func
        return super().visit_function_(func)

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        func = self.lookup_binding(call.op) if isinstance(call.op, relax.Var) else call.op
        if isinstance(func, relax.Function) and func.attrs and str(func.attrs.get("Composite", "")).startswith("gemmini."):
            func = func.without_attr("Composite")
            gv = self.builder_.add_func(func, "gemmini_region")
            self.outlined[gv] = func
            return self.builder_.normalize(relax.Call(gv, call.args))
        return call


def _expand_composites(mod, selected):
    expand = _ExpandComposite(mod)
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            expand.builder_.update_func(gv, expand.visit_expr(func))
    expanded = expand.builder_.get()
    replacements = {**expand.outlined, **{gv: expanded[gv] for gv in selected}}
    for gv, func in list(expanded.functions_items()):
        if gv not in replacements and isinstance(func, relax.Function) and not (func.attrs and "Codegen" in func.attrs):
            expanded.update_func(gv, func.inline_functions(replacements))
    return relax.transform.DeadCodeElimination()(expanded)


@tvm.transform.module_pass(opt_level=0, name="LowerGemminiBYOC")
class LowerGemminiBYOC:
    """C1 deployment: lower BYOC regions to call_tir without changing exporter.

    TVM expands the selected region, performs ordinary CPU legalization and
    fusion, and schedules its contractions. No handwritten operator C codegen
    or private exporter exceptions are used.
    """

    def __init__(self, order="ijk", records=None):
        self.order = order
        self.records = records if records is not None else []

    def transform_module(self, mod, _ctx):
        selected = [gv for gv, f in mod.functions_items() if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "gemmini"]
        for gv in selected:
            func = mod[gv].without_attr("Codegen").with_attr("global_symbol", "main")
            region = _expand_composites(tvm.IRModule({"main": func}), [])
            region = tvm.transform.Sequential([
                relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(), relax.transform.FuseTIR(),
            ])(region)
            importer = _ImportRegion(mod)
            for local_gv, primitive in region.functions_items():
                if not isinstance(primitive, tir.PrimFunc):
                    continue
                scheduled = schedule_primfunc(primitive, order=self.order)
                body = scheduled.scheduled_mod["main"].without_attr("global_symbol").with_attr("op_pattern", 8)
                new_gv = importer.builder_.add_func(body, "gemmini_byoc_region")
                importer.replacements[local_gv] = new_gv
                self.records.append((new_gv.name_hint, scheduled))
            func = importer.visit_expr(region["main"]).without_attr("global_symbol")
            importer.builder_.update_func(gv, func)
            mod = importer.builder_.get()
        if not selected:
            return mod
        return _expand_composites(mod, selected)


def prepare_gemmini_byoc(mod, *, order="ijk", records=None):
    """Optimize, partition, schedule and legalize for static baremetal export."""
    from tvm.relax.backend.contrib.gemmini import DecomposeGemminiLinear
    from tvm.relax.backend.contrib.gemmini_conv import DecomposeGemminiConv2D

    batch = _DecomposeBatch(mod)
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            batch.builder_.update_func(gv, batch.visit_expr(func))
    mod = DecomposeGemminiLinear()(DecomposeGemminiConv2D()(batch.builder_.get()))
    mod = relax.transform.FoldConstant()(mod)
    mod = partition_for_gemmini(mod)
    mod = LowerGemminiBYOC(order, records)(mod)
    mod = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR(),
    ])(mod)
    return _inline_cpu_producers(mod)

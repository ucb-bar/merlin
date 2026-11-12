"""Lower ``math.roundeven`` to LLVM's exact round-even intrinsic after loop formation.

Dynamic activation quantization contains one round-to-nearest-even operation per element.  Leaving
it for ``convert-math-to-libm`` creates a scalar ``roundevenf`` call that blocks LLVM loop
vectorization.  Expanding it into ordinary arithmetic removes that barrier, but substantially
inflates large whole-model LLVM IR and compile time.  LLVM already has the operation we mean:
``llvm.intr.roundeven``.  This default-off lowering changes only the representation of the same
specified rounding operation and runs after linalg has become loops, where an LLVM-dialect scalar
intrinsic is legal and LLVM's loop vectorizer can form its vector counterpart.

The rewrite is deliberately structure-only: every scalar/vector ``math.roundeven`` is replaced,
without inspecting a model name, tensor shape, dtype recipe, or target.  Other math operations are
left for the existing libm path.  It is an alternative to ``fuse_quantize_round_convert``; enabling
both would erase the round before this stage and is rejected by the lowering entry point.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

FEATURE = "lower_roundeven_to_intrinsic"
MARKER = "__merlin_lower_roundeven_to_intrinsic__"
REPORT_TOKEN = "OK roundeven_intrinsic"

#: The same lowering for EVERY exactly specified rounding op, not round-even alone. Each of these is a
#: correctly rounded (indeed exact) IEEE operation, so the LLVM intrinsic and the libm call return the
#: same bits for every non-NaN input; for a NaN the libm functions return ``x + x`` (on RISC-V the
#: canonical NaN) where the bare intrinsic keeps the input's payload, so the exact feature returns
#: ``x + x`` for a NaN too (``absf`` needs nothing: both only clear the sign bit). Checked against
#: newlib on every one of the 2**32 f32 inputs, scalar and RVV. What changes is that a
#: call in a loop body blocks LLVM's vectorizer and
#: costs a libm call per element, while the intrinsic is a few instructions scalar and a vector op
#: under RVV. Measured on an integer-nonlinear capture: 115 ``math.floor`` sites (the integer floor
#: division and shifts are spelled with a float floor). Transcendentals (exp, log, pow, tanh, erf)
#: are NOT exact operations, so they keep the libm path and their bits -- with one exactly specified
#: case: ``pow(2.0, sitofp(i))``, a power of two whose correctly rounded value is 2**i (or its
#: overflow/underflow), which the integer nonlinears use for every per-element variable shift. It is
#: built from the exponent bits as ``2**e1 * 2**e2`` with each part clamped to the normal exponent
#: range, so overflow to infinity and subnormal or zero results round exactly as the libm value does.
EXACT_FEATURE = "lower_exact_math_inline"
EXACT_MARKER = "__merlin_lower_exact_math_inline__"
EXACT_REPORT_TOKEN = "OK exact_math_inline"
EXACT_ROUNDING = {
    "math.floor": "llvm.intr.floor",
    "math.ceil": "llvm.intr.ceil",
    "math.trunc": "llvm.intr.trunc",
    "math.round": "llvm.intr.round",
    "math.roundeven": "llvm.intr.roundeven",
    "math.absf": "llvm.intr.fabs",
}

#: The other exactly specified per-element call: rounding an f32 to bf16. A target without bf16
#: arithmetic has LLVM compute every bf16 op in f32 and round the f32 result to bf16 through a
#: ``__truncsfbf2`` call, one per element and per op -- 53,426 call sites in an int8-full SmolVLA
#: host object, whose requant casts (``sitofp`` to bf16, a bf16 scale multiply, round-even, clamp)
#: each paid one call per step and none of whose loops could vectorize. The rewrite spells the same
#: promotion out before LLVM sees it: each bf16 operand widened (a 16-bit shift, exact), the op in
#: f32, the result rounded to bf16 in integer arithmetic. That rounding is the runtime helper's own
#: (``src/merlin/runtime/native/abi/mlir_runtime.c``): round half to even on the bits, infinity kept, a NaN's
#: payload truncated with its quiet bit set -- independent of the floating-point rounding mode, as
#: the helper is. Negation flips the sign bit, as LLVM's own bf16 negation does. Ops LLVM does not
#: compute this way (remainder, transcendentals) keep their path.
BF16_PROMOTED = (
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.maximumf",
    "arith.minimumf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.sitofp",
    "arith.uitofp",
    "math.floor",
    "math.ceil",
    "math.trunc",
    "math.round",
    "math.roundeven",
)


def _after_loops(passes: list[str], marker: str, feature: str) -> list[str]:
    """Insert ``marker`` after linalg-to-loop conversion."""
    matches = [
        i for i, p in enumerate(passes) if "convert-linalg-to-loops" in p or "convert-linalg-to-parallel-loops" in p
    ]
    if not matches:
        raise ValueError(f"{feature} requires a linalg-to-loops lowering stage")
    i = max(matches) + 1
    return [*passes[:i], marker, *passes[i:]]


def _edit_pipeline(passes: list[str]) -> list[str]:
    """Insert the runner-owned marker after linalg-to-loop conversion."""
    return _after_loops(passes, MARKER, FEATURE)


def _edit_pipeline_exact(passes: list[str]) -> list[str]:
    return _after_loops(passes, EXACT_MARKER, EXACT_FEATURE)


def ensure_registered() -> str:
    from merlin.compiler.selection.shared import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Replace math.roundeven with the semantically identical llvm.intr.roundeven after "
                    "linalg-to-loop conversion. This prevents convert-math-to-libm from emitting a "
                    "scalar roundevenf call while avoiding the large ordinary-arithmetic expansion of "
                    "fuse_quantize_round_convert. Structure-only, target/model/shape independent, and "
                    "default-off; mutually exclusive with fuse_quantize_round_convert."
                ),
                edit_pipeline=_edit_pipeline,
            )
        )
    if EXACT_FEATURE not in known():
        register(
            ImprFeature(
                name=EXACT_FEATURE,
                action_class="PASS",
                description=(
                    "Replace every exactly specified math op after linalg-to-loop conversion -- floor, "
                    "ceil, trunc, round, roundeven and absf by their LLVM intrinsics, pow(2.0, "
                    "sitofp(i)) by a power of two built from exponent bits, and every bf16 op by the "
                    "f32 op with the runtime's f32-to-bf16 rounding in integer arithmetic -- so no "
                    "per-element libm or __truncsfbf2 call is left and LLVM can vectorize the loop. "
                    "Bit-identical: each op is exact or the same promotion LLVM performs. Other "
                    "transcendentals keep the libm path. Structure-only; on for open-model host code."
                ),
                edit_pipeline=_edit_pipeline_exact,
            )
        )
    return FEATURE


RUNNER_PRELUDE = r'''
def _lower_math_intrinsics(ctx, module, mapping, libm_nan=False):
    """Replace every op named in ``mapping`` with its LLVM intrinsic; return the exact count.

    ``libm_nan`` returns ``x + x`` for a NaN ``x`` from every rounding op, as the libm functions do
    (newlib's ``floorf``/``ceilf``/``truncf``/``roundf``/``rintf`` and their doubles), where the bare
    intrinsic returns the NaN unchanged: the two then agree on the bits of every input, NaN payloads
    and signs included. ``fabs`` is the same sign-bit clear either way and keeps the intrinsic alone."""
    from torch_mlir import ir as _ri_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name in mapping:
                        todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx, _ri_ir.Location.unknown():
        for old in todo:
            ty, x = old.results[0].type, old.operands[0]
            with _ri_ir.InsertionPoint(old):
                value = _ri_ir.Operation.create(
                    mapping[old.operation.name], results=[ty], operands=[x]).results[0]
                if libm_nan and old.operation.name != "math.absf":
                    i1 = _ri_ir.IntegerType.get_signless(1)
                    flag = (_ri_ir.VectorType.get(_ri_ir.VectorType(ty).shape, i1)
                            if str(ty).startswith("vector<") else i1)
                    unordered = _ri_ir.Operation.create(
                        "arith.cmpf", results=[flag], operands=[x, x],
                        attributes={"predicate": _ri_ir.IntegerAttr.get(
                            _ri_ir.IntegerType.get_signless(64), 14)}).results[0]  # uno: x is NaN
                    doubled = _ri_ir.Operation.create("arith.addf", results=[ty], operands=[x, x]).results[0]
                    value = _ri_ir.Operation.create(
                        "arith.select", results=[ty], operands=[unordered, doubled, value]).results[0]
            old.results[0].replace_all_uses_with(value)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


def _lower_roundeven_intrinsics(ctx, module):
    """Replace every math.roundeven with llvm.intr.roundeven; return the exact count."""
    return _lower_math_intrinsics(ctx, module, {"math.roundeven": "llvm.intr.roundeven"})


_EXACT_ROUNDING = __EXACT_ROUNDING__

#: (exponent bias, smallest normal exponent, largest finite exponent, mantissa bits) per float width.
_POW2_FORMAT = {64: (1023, -1022, 1023, 52), 32: (127, -126, 127, 23)}


def _lower_pow2_of_integer(ctx, module):
    """Replace every ``math.powf(2.0, sitofp(i))`` by ``2**e1 * 2**e2`` built from exponent bits.

    ``e1 = clamp(i, lo, hi)`` and ``e2 = clamp(i - e1, lo, hi)``: in range the second factor is 1; past
    the top the product overflows to infinity; below the normal range it is the exact subnormal, or
    rounds to zero where 2**i is below the smallest one -- the correctly rounded value of 2**i in
    every case, which is what the libm call returns for a power of two. Returns the count."""
    from torch_mlir import ir as _ri_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.powf":
                        todo.append(inner)
                    walk(inner.operation)

    def width_of(t):
        text = str(t)
        return {"f64": 64, "f32": 32}.get(text)

    def int_width(t):
        text = str(t)
        return int(text[1:]) if text.startswith("i") and text[1:].isdigit() else None

    def is_two(value):
        owner = getattr(value, "owner", None)
        if owner is None or getattr(owner, "name", "") != "arith.constant":
            return False
        attrs = owner.attributes
        if "value" not in attrs:
            return False
        try:
            return _ri_ir.FloatAttr(attrs["value"]).value == 2.0
        except Exception:
            return False

    walk(module.operation)
    count = 0
    with ctx, _ri_ir.Location.unknown():
        i64 = _ri_ir.IntegerType.get_signless(64)
        for op in todo:
            width = width_of(op.results[0].type)
            base, exponent = op.operands[0], op.operands[1]
            source = getattr(exponent, "owner", None)
            if width is None or not is_two(base) or getattr(source, "name", "") != "arith.sitofp":
                continue
            integer = source.operands[0]
            bits = int_width(integer.type)
            if bits is None or bits > 64:
                continue
            bias, lo, hi, mantissa = _POW2_FORMAT[width]
            with _ri_ir.InsertionPoint(op):
                def make(name, operands, result=i64, **attrs):
                    return _ri_ir.Operation.create(name, results=[result], operands=operands,
                                                    attributes=attrs).results[0]

                def const(value):
                    return make("arith.constant", [], value=_ri_ir.IntegerAttr.get(i64, value))

                wide = integer if bits == 64 else make("arith.extsi", [integer])
                low, high = const(lo), const(hi)

                def clamp(v):
                    return make("arith.maxsi", [make("arith.minsi", [v, high]), low])

                def power(e):
                    field = make("arith.shli", [make("arith.addi", [e, const(bias)]), const(mantissa)])
                    if width == 32:
                        field = make("arith.trunci", [field], result=_ri_ir.IntegerType.get_signless(32))
                    return make("arith.bitcast", [field], result=op.results[0].type)

                e1 = clamp(wide)
                e2 = clamp(make("arith.subi", [wide, e1]))
                value = make("arith.mulf", [power(e1), power(e2)], result=op.results[0].type)
            op.results[0].replace_all_uses_with(value)
            op.operation.erase()
            count += 1
    module.operation.verify()
    return count


_BF16_PROMOTED = __BF16_PROMOTED__


def _inline_bf16_rounding(ctx, module):
    """Compute every scalar bf16 op in f32 and round to bf16 in integer arithmetic; return the count.

    The same promotion LLVM applies on a target without bf16 arithmetic, with the f32 -> bf16 rounding
    spelled as the runtime helper ``__truncsfbf2`` computes it, so the result bits are the helper's and
    no call is left in the loop."""
    from torch_mlir import ir as _ri_ir
    names = set(_BF16_PROMOTED) | {"arith.truncf", "arith.extf", "arith.fptosi", "arith.fptoui",
                                     "arith.cmpf", "arith.negf"}
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name in names:
                        todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    count = 0
    with ctx, _ri_ir.Location.unknown():
        bf16, f32 = _ri_ir.BF16Type.get(), _ri_ir.F32Type.get()
        i16, i32 = _ri_ir.IntegerType.get_signless(16), _ri_ir.IntegerType.get_signless(32)

        def is_bf16(t):
            return str(t) == "bf16"

        def is_f32(t):
            return str(t) == "f32"

        for op in todo:
            name = op.operation.name
            operands = list(op.operands)
            result = op.results[0]
            touches = any(is_bf16(v.type) for v in operands) or is_bf16(result.type)
            if not touches:
                continue
            with _ri_ir.InsertionPoint(op):
                def make(kind, args, rtype, **attrs):
                    return _ri_ir.Operation.create(kind, results=[rtype], operands=args,
                                                    attributes=attrs).results[0]

                def const(value, itype=i32):
                    return make("arith.constant", [], itype, value=_ri_ir.IntegerAttr.get(itype, value))

                def widen(v):
                    """bf16 -> f32: the bf16 bits are the f32's top half (exact)."""
                    bits = make("arith.extui", [make("arith.bitcast", [v], i16)], i32)
                    return make("arith.bitcast", [make("arith.shli", [bits, const(16)], i32)], f32)

                def narrow(v):
                    """f32 -> bf16 exactly as the runtime's ``__truncsfbf2``."""
                    bits = make("arith.bitcast", [v], i32)
                    top = make("arith.shrui", [bits, const(16)], i32)
                    special = make("arith.cmpi", [make("arith.andi", [bits, const(0x7F800000)], i32),
                                                  const(0x7F800000)], _ri_ir.IntegerType.get_signless(1),
                                   predicate=_ri_ir.IntegerAttr.get(_ri_ir.IntegerType.get_signless(64), 0))
                    payload = make("arith.cmpi", [make("arith.andi", [bits, const(0x007FFFFF)], i32), const(0)],
                                   _ri_ir.IntegerType.get_signless(1),
                                   predicate=_ri_ir.IntegerAttr.get(_ri_ir.IntegerType.get_signless(64), 1))
                    quiet = make("arith.select", [payload, const(0x40), const(0)], i32)
                    nan = make("arith.ori", [top, quiet], i32)
                    bias = make("arith.addi", [make("arith.andi", [top, const(1)], i32), const(0x7FFF)], i32)
                    rounded = make("arith.shrui", [make("arith.addi", [bits, bias], i32), const(16)], i32)
                    chosen = make("arith.select", [special, nan, rounded], i32)
                    return make("arith.bitcast", [make("arith.trunci", [chosen], i16)], bf16)

                held = op.operation.attributes  # iterates names or NamedAttributes by binding version
                attrs = {a: held[a] for a in held} if all(isinstance(a, str) for a in held) else {
                    a.name: a.attr for a in held}
                if name == "arith.extf":
                    if not (is_bf16(operands[0].type) and is_f32(result.type)):
                        continue
                    value = widen(operands[0])
                elif name == "arith.truncf":
                    if not (is_f32(operands[0].type) and is_bf16(result.type)):
                        continue
                    value = narrow(operands[0])
                elif name == "arith.negf":
                    if not is_bf16(result.type):
                        continue
                    bits = make("arith.bitcast", [operands[0]], i16)
                    value = make("arith.bitcast", [make("arith.xori", [bits, const(-0x8000, i16)], i16)], bf16)
                elif name in ("arith.fptosi", "arith.fptoui", "arith.cmpf"):
                    if not all(is_bf16(v.type) for v in operands):
                        continue
                    value = make(name, [widen(v) for v in operands], result.type, **attrs)
                else:
                    # An integer-to-bf16 conversion converts to f32 and rounds; every other op here
                    # takes bf16 operands only.
                    converts = name in ("arith.sitofp", "arith.uitofp")
                    if not is_bf16(result.type) or not (converts or all(is_bf16(v.type) for v in operands)):
                        continue
                    args = operands if converts else [widen(v) for v in operands]
                    value = narrow(make(name, args, f32, **attrs))
            result.replace_all_uses_with(value)
            op.operation.erase()
            count += 1
    module.operation.verify()
    return count


_RI_MARKER = "__merlin_lower_roundeven_to_intrinsic__"
_RI_EXACT_MARKER = "__merlin_lower_exact_math_inline__"
_RI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    markers = [p for p in passes if p in (_RI_MARKER, _RI_EXACT_MARKER)]
    if not markers:
        return _RI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp,
                                   pre_generalize)
    # Both markers sit right after linalg-to-loops: run the head, every requested rewrite, then the
    # tail. mid/pre-generalize belong to the head, while parallel grain/team/coarsening stages still
    # belong to the tail; keeping that routing explicit avoids silently dropping another requested
    # rewrite when this lever is composed with it.
    i = min(passes.index(m) for m in markers)
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    if _RI_MARKER in markers:
        print('OK roundeven_intrinsic', _lower_roundeven_intrinsics(ctx, module))
    if _RI_EXACT_MARKER in markers:
        # bf16 first: a bf16 round-even becomes an f32 one, which the intrinsic mapping then takes.
        bf16 = _inline_bf16_rounding(ctx, module)
        print('OK exact_math_inline', _lower_math_intrinsics(ctx, module, _EXACT_ROUNDING, libm_nan=True),
              _lower_pow2_of_integer(ctx, module), bf16)
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(p for p in passes[i + 1:] if p not in markers), 0, (),
                        late, post_openmp, ())
'''.replace("__EXACT_ROUNDING__", repr(EXACT_ROUNDING)).replace("__BF16_PROMOTED__", repr(BF16_PROMOTED))


def apply_for_test(mlir_text: str) -> tuple[str, int]:
    """Apply the shipped rewrite source to a module in the toolchain-owning Python."""
    from merlin.compiler.lower.model.toolchain import m2m_python

    work = Path(tempfile.mkdtemp(prefix="merlin_roundeven_intrinsic_"))
    src, script = work / "in.mlir", work / "run.py"
    src.write_text(mlir_text, encoding="utf-8")
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n" + RUNNER_PRELUDE.split("_RI_MARKER =", 1)[0] + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f: module = ir.Module.parse(f.read(), ctx)\n"
        "n = _lower_roundeven_intrinsics(ctx, module)\n"
        "print('COUNT', n)\nprint('MODULE_BEGIN')\nprint(module.operation)\n",
        encoding="utf-8",
    )
    proc = subprocess.run([str(m2m_python()), str(script), str(src)], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"roundeven intrinsic rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    count = int(next(line.split()[1] for line in proc.stdout.splitlines() if line.startswith("COUNT ")))
    return proc.stdout.split("MODULE_BEGIN\n", 1)[1], count

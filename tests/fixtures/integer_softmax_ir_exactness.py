"""The int_softmax_table rewrite is the captured program, bit for bit (runs in the compiler's Python).

usage: integer_softmax_ir_exactness.py <rewrite source> <data dir> <targetgen dir> <runtime .so> <section>

Each section executes the captured module and the rewritten one on the host (the MLIR execution
engine, the same upstream passes for both) and compares the outputs' BITS:

* ``table``: the table the rewrite evaluates from the IR equals the capture's own per-element definition
  (``_integer_nonlinear.iexp_int`` and its floor division, in torch) at every grid index;
* ``softmax``: the integer softmax alone, on rows that reach every grid index, on -inf masks, ties,
  constant rows and a lone huge value;
* ``attention``: two whole int8 attentions (bf16 with a mask, f32 over 1024 keys) where the rewrite also
  quantizes the probabilities once per row and moves the score scale -- including flat rows, whose
  quantization scale clamps to its eps so the int8 operand is NOT the softmax numerator;
* ``refusal``: a module whose clamp's upper bound binds is left exactly as it was, and one whose row
  maximum may not be an element keeps its probability quantization while the rest is rewritten.
"""

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_mlir import ir
from torch_mlir.execution_engine import ExecutionEngine
from torch_mlir.passmanager import PassManager
from torch_mlir.runtime import get_ranked_memref_descriptor

REWRITE, DATA, TARGETGEN, RUNTIME, SECTION = sys.argv[1:6]
namespace: dict = {}
exec(compile(Path(REWRITE).read_text(encoding="utf-8"), REWRITE, "exec"), namespace)
sys.path.insert(0, TARGETGEN)
import _integer_nonlinear as NL  # noqa: E402

LOWER = (
    "builtin.module(canonicalize,cse,func.func(linalg-fuse-elementwise-ops),"
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
    "buffer-results-to-out-params{modify-public-functions},func.func(convert-linalg-to-loops),convert-scf-to-cf,"
    "expand-strided-metadata,lower-affine,convert-math-to-llvm,convert-math-to-libm,convert-index-to-llvm,"
    "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)
NUMPY = {"f32": np.float32, "bf16": np.uint16, "i64": np.int64, "i32": np.int32, "i8": np.int8}


def text(name):
    return (Path(DATA) / name).read_text(encoding="utf-8")


def rewrite(source):
    ctx = ir.Context()
    module = ir.Module.parse(source, ctx)
    report = namespace["_int_softmax_table"](ctx, module)
    return str(module), report, module


def run(source, inputs):
    """``@forward`` of ``source`` on ``inputs``; its outputs as numpy arrays."""
    ctx = ir.Context()
    module = ir.Module.parse(source, ctx)
    forward = next(op for op in module.body.operations if op.operation.name == "func.func")
    results = ir.FunctionType(ir.TypeAttr(forward.operation.attributes["function_type"]).value).results
    forward.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(ctx)
    PassManager.parse(LOWER, ctx).run(module.operation)
    engine = ExecutionEngine(module, opt_level=0, shared_libs=[RUNTIME])
    outputs = [np.zeros(ir.RankedTensorType(t).shape, NUMPY[str(ir.RankedTensorType(t).element_type)]) for t in results]
    descriptors = [get_ranked_memref_descriptor(np.ascontiguousarray(a)) for a in [*inputs, *outputs]]
    engine.invoke("forward", *[ctypes.pointer(ctypes.pointer(d)) for d in descriptors])
    return outputs


def bits(a):
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def bf16(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).bfloat16().view(torch.uint16).numpy()


def same(source, rewritten, inputs, what):
    a, b = run(source, inputs), run(rewritten, inputs)
    assert len(a) == len(b) and all(np.array_equal(bits(x), bits(y)) for x, y in zip(a, b)), what


def section_table():
    _, report, module = rewrite(text("softmax_wide.mlir"))
    assert report["softmax"] == 1, report
    tables = []
    for op in namespace["_ist_walk"](module.operation):
        if op.name == "arith.constant" and str(op.results[0].type).startswith("tensor<11081x"):
            tables.append(np.array(ir.DenseIntElementsAttr(op.attributes["value"])))
    (table,) = tables
    q = torch.arange(-NL.IEXP_QMAX, 1, dtype=torch.int64)  # the table is indexed by q - lo
    e0 = NL.iexp_zero()
    reference = NL._floordiv(NL.iexp_int(q) * NL.Q8 + e0 // 2, e0).numpy()
    assert len(table) == NL.IEXP_QMAX + 1 and np.array_equal(table.astype(np.int64), reference)
    print("ok table", len(table))


def section_softmax():
    source = text("softmax_wide.mlir")
    rewritten, report, _ = rewrite(source)
    assert (report["softmax"], report["int32_sums"]) == (1, 1), report
    step = np.float32(NL.IEXP_S)
    row = (-np.arange(11200, dtype=np.float32) * step).astype(np.float32)
    grid = np.clip(np.rint((row - row.max()) / step), -NL.IEXP_QMAX, 0)
    assert len(np.unique(grid)) == NL.IEXP_QMAX + 1  # one row reaches every grid index
    rng = np.random.default_rng(0)
    cases = [np.stack([row, row * np.float32(0.5) + 3, rng.permutation(row)])]
    for scale in (1e-3, 1.0, 30.0, 1e4):
        x = (rng.standard_normal((3, 11200)) * scale).astype(np.float32)
        x[:, 2::3] = -np.inf  # masked keys
        x[:, 1] = x[:, 0]  # a tie at the maximum
        cases.append(x)
    flat = np.zeros((3, 11200), np.float32)
    flat[1, 7] = 1e30
    cases.append(flat)
    for i, x in enumerate(cases):
        same(source, rewritten, [x], f"softmax case {i}")
    print("ok softmax", len(cases), NL.IEXP_QMAX + 1)


def section_attention():
    rng = np.random.default_rng(1)
    count = 0
    for name, convert, shapes, masked in (
        ("attention_bf16_masked.mlir", bf16, [(1, 2, 20, 32), (1, 2, 24, 32), (1, 2, 24, 32)], True),
        ("attention_long_f32.mlir", f32, [(1, 1, 6, 32), (1, 1, 1024, 32), (1, 1, 1024, 32)], False),
    ):
        source = text(name)
        rewritten, report, _ = rewrite(source)
        expected = {"softmax": 1, "int32_sums": 1, "row_quantizations": 1, "scales_moved": 1, "refused": []}
        assert {k: report[k] for k in expected} == expected, (name, report)
        for scale in (0.0, 1e-3, 0.05, 1.0, 4.0, 30.0):
            for flat_row in (False, True):
                q, k, v = (rng.standard_normal(s) for s in shapes)
                q = q * scale
                if flat_row:
                    q[..., 0, :] = 0.0  # every score equal: p = 127 everywhere, P = 1/n
                inputs = [convert(q), convert(k), convert(v)]
                if masked:
                    mask = np.zeros((1, 1, 20, 24), np.float32)
                    mask[..., (count % 7) :: 7] = -np.inf
                    inputs.append(convert(mask))
                same(source, rewritten, inputs, f"{name} scale {scale} flat {flat_row}")
                count += 1
    # A flat row of 1024 keys has P = 1/1024 everywhere, so its symmetric scale (P_max / 127.5) is below
    # the activation eps and clamps to it: the int8 operand is round(P / eps), not the numerator 127.
    assert 1.0 / 1024 / 127.5 < 1e-5
    print("ok attention", count)


def section_refusal():
    source = text("softmax_wide.mlir")
    binding = source.replace("arith.constant 0.000000e+00 : f32\n", "arith.constant -1.000000e+00 : f32\n", 1)
    assert binding != source
    rewritten, report, _ = rewrite(binding)
    assert report["softmax"] == 0 and any("clamp" in r for r in report["refused"]), report
    ctx = ir.Context()
    assert rewritten == str(ir.Module.parse(binding, ctx))  # untouched
    # The softmax's row maximum starts from 0 instead of -inf: still an upper bound of every element (the
    # clamp stays provably one-sided), but no longer necessarily one of them.
    lines = text("attention_long_f32.mlir").splitlines(keepends=True)
    splat = next(
        line
        for line in lines
        if "tensor.splat" in line and "aten.amax.default" in line and line.rstrip().endswith(": tensor<1x1x6xf32>")
    )
    name = splat.split("tensor.splat", 1)[1].split()[0]
    (at,) = [i for i, line in enumerate(lines) if line.lstrip().startswith(name + " = arith.constant")]
    assert "0xff800000 : f32" in lines[at]
    lines[at] = lines[at].replace("0xff800000 : f32", "0.000000e+00 : f32")
    unattained = "".join(lines)
    rewritten, report, _ = rewrite(unattained)
    assert report["softmax"] == 1 and report["row_quantizations"] == 0, report
    assert any("row maximum" in r for r in report["refused"]), report
    rng = np.random.default_rng(2)
    inputs = [rng.standard_normal(s).astype(np.float32) for s in [(1, 1, 6, 32), (1, 1, 1024, 32), (1, 1, 1024, 32)]]
    same(unattained, rewritten, inputs, "attention with a non-attained row maximum")
    print("ok refusal", json.dumps(report["refused"]))


def section_mutant():
    """The comparison can fail: one table entry off by one (the row maximum's) is caught."""
    original = namespace["_ist_run"]

    def off_by_one(steps, size, q):
        values = original(steps, size, q)
        return [v - 1 if q == 0 and i == len(values) - 1 else v for i, v in enumerate(values)]

    namespace["_ist_run"] = off_by_one
    try:
        source = text("softmax_wide.mlir")
        rewritten, report, _ = rewrite(source)
    finally:
        namespace["_ist_run"] = original
    assert report["softmax"] == 1, report
    x = np.random.default_rng(3).standard_normal((3, 11200)).astype(np.float32)
    try:
        same(source, rewritten, [x], "mutant")
    except AssertionError:
        print("ok mutant")
        return
    raise SystemExit("a rewrite with a wrong table entry compared equal")


{
    "mutant": section_mutant,
    "table": section_table,
    "softmax": section_softmax,
    "attention": section_attention,
    "refusal": section_refusal,
}[SECTION]()

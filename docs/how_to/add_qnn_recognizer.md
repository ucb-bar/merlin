# How to add a QNN recognizer

A *recognizer* is the function that converts one shape of MLIR (a
`linalg.X` op or chain) into a `QnnGraphDesc` — the IR layer that
the emitter renders to a `.qnn.cpp` source file. Adding a new
recognizer extends the v2 emitter's coverage to a new MLIR pattern.

This guide walks through the steps using a concrete worked example:
adding an `tensor.expand_shape` recognizer (a pure layout op, easy to
land in <30 minutes).

## Prerequisites

  - The QNN test suite is green in your tree:
    `pytest kernels/qnn/tests/` should pass before you start.
    (Historical context: a v2 bindings-based emitter was archived to
    `tools/archive/qnn_v2/` — the active recognizers live in
    `kernels/qnn/recognizers/` and the live test suite at
    `kernels/qnn/tests/` covers them.)
  - You know the MLIR pattern you want to recognize (paste the
    relevant `linalg.generic` body or named-op signature here).
  - You know the QNN op shape that this should lower to (consult
    Qualcomm's `QnnOpDef.h` — `share/QNN/converter/jni/QnnOpDef.h`
    in the QAIRT SDK).

## Step 1 — pick a fixture file

Drop a minimal MLIR fixture under
`benchmarks/QRB5165/mlir/<recognizer_name>_smoke.mlir`. It should be
a single `func.func` whose body is the pattern you want to recognize.
Keep it small (≤2 KB); fixtures get re-parsed on every test run.

Example for `tensor.expand_shape`:

```mlir
module {
  func.func @expand_smoke(%input: tensor<1x256xi8>)
      -> tensor<1x16x16xi8> {
    %out = tensor.expand_shape %input [[0], [1, 2]]
        : tensor<1x256xi8> into tensor<1x16x16xi8>
    return %out : tensor<1x16x16xi8>
  }
}
```

## Step 2 — write the recognizer

Create
`kernels/qnn/recognizers/<recognizer_name>.py` with the
canonical contract:

```python
from typing import Any
from .base import find_func, find_named_op, func_arg_values, shape_of

NAME = "expand_shape_smoke"

def try_recognize(
    module: Any, *, fp_dtype: str = "float32", **_: object
) -> Any | None:
    func = find_func(module)
    if func is None:
        return None
    anchor = find_named_op(func, "tensor.expand_shape")
    if anchor is None:
        return None

    # Validate signature.
    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_shape = shape_of(args[0])
    out_val = func.regions[0].blocks[0].operations[-1].operands[0]
    out_shape = shape_of(out_val)

    # Build a QnnGraphDesc with a single Reshape node.
    from qnn_ir import (
        QnnGraphDesc,
        QuantParams,
        TensorDesc,
        reshape_node,
    )
    qp = QuantParams(scale=1.0, offset=0)
    return QnnGraphDesc(
        name="expand_smoke",
        tensors=(
            TensorDesc("input", tuple(in_shape), "int8", "input", quant=qp),
            TensorDesc("output", tuple(out_shape), "int8", "output", quant=qp),
        ),
        nodes=(reshape_node("op", "input", "output"),),
    )
```

**Strict rule:** no regex. Use `iree.compiler.ir` bindings
(`find_func`, `walk_inner_ops`, `find_named_op`, `dense_to_bytes`,
`parse_dense_2d_attr`, `parse_qparams_attr`, …) from
`kernels/qnn/recognizers/base.py`. The dispatcher's
test suite enforces this with a `grep -rE "import re|re\..."`
audit.

## Step 3 — register in REGISTRY

Edit `kernels/qnn/recognizers/__init__.py`:

```python
from . import (
    ...,
    expand_shape,  # add the import
    ...,
)

REGISTRY = (
    ...,
    expand_shape,  # most-specific first; this is fairly specific
                   # because tensor.expand_shape is a unique anchor
    ...,
)
```

## Step 4 — write a parity / coverage test

Append a parametric entry to the relevant test file under
`kernels/qnn/tests/` (e.g. `test_qnn_phase5_gates.py` for new conv-shape
coverage, or to a recognizer-specific
test file otherwise. The test should:

  1. Parse the fixture.
  2. Run `qnn_emit_v2.parse_mlir`.
  3. Assert the resulting QnnGraphDesc has the right node sequence
     and tensor metadata (shapes / dtypes / quant params).

```python
def test_expand_shape_recognizer() -> None:
    from qnn_emit_v2 import parse_mlir
    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/expand_smoke.mlir"
    graph = parse_mlir(fixture.read_text())
    assert graph is not None
    assert [n.op_type for n in graph.nodes] == ["Reshape"]
```

## Step 5 — register the anchor in the partitioner

If your recognizer anchors on a named op (like `tensor.expand_shape`),
add it to `_ANCHOR_OP_TO_RECOGNIZER` in
`kernels/qnn/partition.py`:

```python
_ANCHOR_OP_TO_RECOGNIZER = (
    ...,
    ("tensor.expand_shape", "expand_shape", "qnn-gpu"),
    ...,
)
```

This lets the partitioner identify your op as an island anchor when
it appears in real model IR.

## Step 6 — run the gates

```sh
conda run -n merlin-dev uv run pytest kernels/qnn/tests/ -v
```

You should see your new test pass alongside all the existing ones.
If your recognizer changes the partitioning of any model in
`kernels/qnn/tests/test_qnn_phase6_multimodel.py`, update the
expected counts there.

## Step 7 — (optional) on-board build round-trip

If you have QAIRT SDK access:

```sh
MERLIN_QNN_BUILD=1 conda run -n merlin-dev uv run pytest \
    kernels/qnn/tests/test_qnn_emit_v2_yolov8_build.py
```

This compiles the emitter's output to a `.qnn-ctx` via
`qnn-context-binary-generator` and confirms the QNN OpDef validator
accepts the structure. CPU-backend validator gaps (e.g. int8
Transpose) are recorded as `xfail` automatically — your recognizer
shouldn't introduce a *new* unrecognized failure mode.

## Reference table

| You want to extract | Use |
|---|---|
| Func name | `func_name(func)` |
| Block-arg shape | `shape_of(func_arg_values(func)[i])` |
| Element dtype | `elem_dtype_of(value)` |
| Op attribute (int) | `integer_attr_value(op, "name")` |
| Op attribute (float) | `ir.FloatAttr(op.attributes["name"]).value` |
| `dense<X>` splat scalar | `splat_constant_value(op)` |
| `dense<[h,w]>` 2D | `parse_dense_2d_attr(op, "name")` |
| Per-element bytes from `dense<...>` | `dense_to_bytes(op, "i8" / "f32" / "i32")` |
| `merlin.qnn.<x>_qparams` dict | `parse_qparams_attr(func, "x_qparams")` |
| Walk all ops in func | `walk_inner_ops(func)` |
| Find first op of a name | `find_named_op(func, "linalg.foo")` |

## Common pitfalls

  - **`id(op)` is not stable.** The bindings return new Python
    proxies on each access. Use sequential indices from
    `_build_op_index` (the partitioner does this) or SSA names
    (`value.get_name()`) as keys.
  - **Block-argument names are canonical.** `%arg0`, `%arg1` —
    not `%input` from the source text. If your recognizer needs
    source-level names (concat does), thread `mlir_text` through and
    parse the func signature with simple string ops (no regex; see
    `concat.py`'s structured parsing for an example).
  - **Float-precision divergence.** `FloatAttr.value` returns an
    f32-precise float64 (e.g. `0.05000000074505806`). Use
    `qnn_ir.py`'s `:.7g` scale formatter to collapse to the
    canonical short representation. Both forms compile to the same
    `.qnn-ctx`.

## Worked examples

  - `kernels/qnn/recognizers/f32_conv2d_relu.py` — single-anchor + tensor-constant extraction
  - `kernels/qnn/recognizers/nchw_int8_conv.py` — multi-op DAG with optional fused-activation chain
  - `kernels/qnn/recognizers/nchw_int8_concat.py` — multi-input pattern
  - `kernels/qnn/recognizers/nchw_int8_pool.py` — anchor + sibling dequant chain

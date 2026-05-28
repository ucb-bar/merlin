# `qnn` dialect — in-compiler QNN code generation

A finite, op-set MLIR dialect mirroring the subset of Qualcomm's
`QnnOpDef.h` we currently codegen for. Every op corresponds 1:1 to a
QNN_OP_* constant; attributes mirror the QNN parameter set byte-for-byte.

This dialect replaces the Python-orchestrated kernel pipeline
(`kernels/qnn/recognizers/*.py` + `qnn_ir.py` + `precompile.py`).
That pipeline was a Python-level conversion + lowering pass; the dialect
moves it into the compiler proper as a real IREE backend.

See: `~/.claude/plans/i-want-to-enable-rosy-sundae.md` for the multi-phase
migration plan.

## Layout

```
QNN/
├── IR/
│   ├── QNNDialect.{td,h,cpp}      Dialect registration
│   ├── QNNOps.td                  Op definitions (Conv2d, ElementWiseNeuron, …)
│   ├── QNNAttrs.td                Attribute definitions (quant params)
│   └── CMakeLists.txt
├── Transforms/
│   ├── ConvertLinalgToQNN.cpp     Pattern-match linalg → qnn (Phase 2)
│   ├── LegalizeLayoutToNHWC.cpp   NCHW → NHWC rewrite (Phase 3)
│   ├── Passes.{h,cpp}             Pass registration
│   └── CMakeLists.txt
├── test/                          FileCheck round-trip + verifier tests
└── README.md                      this file
```

The actual `qnn → serialized graph` codegen and SSH-on-board build path live
under `compiler/plugins/target/QNN/Codegen/` (out-of-tree relative to this
dialect because they're plugin-private).

## Op coverage today

The 14 ops we register correspond 1:1 to today's Python recognizers under
`kernels/qnn/recognizers/`:

| MLIR op | QNN_OP_* | Replaces Python recognizer |
|---|---|---|
| `qnn.conv2d` | `QNN_OP_CONV_2D` | `nhwc_int8_conv`, `nchw_int8_conv` |
| `qnn.depthwise_conv2d` | `QNN_OP_DEPTH_WISE_CONV_2D` | `depthwise_conv` |
| `qnn.fully_connected` | `QNN_OP_FULLY_CONNECTED` | (new) |
| `qnn.matmul` | `QNN_OP_MAT_MUL` | (new) |
| `qnn.element_wise_neuron` | `QNN_OP_ELEMENT_WISE_NEURON` | `elementwise_unary` |
| `qnn.element_wise_binary` | `QNN_OP_ELEMENT_WISE_BINARY` | `elementwise_binary` |
| `qnn.pool_max2d` | `QNN_OP_POOL_MAX_2D` | `nchw_int8_pool`, `maxpool` |
| `qnn.pool_avg2d` | `QNN_OP_POOL_AVG_2D` | (new) |
| `qnn.concat` | `QNN_OP_CONCAT` | `nchw_int8_concat`, `concat` |
| `qnn.reshape` | `QNN_OP_RESHAPE` | `nchw_int8_reshape`, `reshape` |
| `qnn.transpose` | `QNN_OP_TRANSPOSE` | `nchw_int8_transpose` |
| `qnn.quantize` | `QNN_OP_QUANTIZE` | (new) |
| `qnn.dequantize` | `QNN_OP_DEQUANTIZE` | `f32_conv2d_relu` (dequant chain) |

Adding a new op:

1. Add the op definition in `IR/QNNOps.td` mirroring the QNN_OP_* parameter set.
2. Add a verifier in `IR/QNNDialect.cpp` if the op needs layout/dtype invariants.
3. Add a conversion pattern in `Transforms/ConvertLinalgToQNN.cpp` that matches
   the equivalent linalg pattern and emits the qnn op.
4. Add a writer in `compiler/plugins/target/QNN/Codegen/SerializeGraph.cpp`
   that emits the binary node record.
5. Add a parser/builder branch in
   `runtime/src/iree/hal/drivers/qnn/qnn_graph_builder.c` that calls
   `Qnn_Graph_addNode` with the right Qnn_OpConfig_t.
6. Add a round-trip test in `test/round_trip.mlir`.

## Backend → dtype constraints (QAIRT 2.45)

The verifier today accepts any rank-4-NHWC tensor; backend-specific dtype
acceptance is enforced at codegen time (compiler/plugins/target/QNN/Codegen):

| Backend | Conv2d dtype |
|---|---|
| HTA | UFIXED_POINT_8 (in/weight/out), UFIXED_POINT_8 or SFIXED_POINT_32 (bias) |
| GPU (Adreno) | FLOAT_32 / FLOAT_16 only — no quantized Conv2d on QAIRT 2.45 |
| HTP (Hexagon v68+) | superset of HTA |

When the codegen sees a Conv2d targeting GPU with int8 inputs, it inserts
a `qnn.dequantize` ahead of the conv and a `qnn.quantize` after. Layout
legalization (`LegalizeLayoutToNHWC`) handles NCHW → NHWC.

## See also

- `compiler/plugins/target/QNN/QNNTarget.cpp` — the HAL target backend.
- `compiler/plugins/target/QNN/Codegen/SerializeGraph.{h,cpp}` — qnn → flat binary.
- `runtime/src/iree/hal/drivers/qnn/qnn_graph_builder.{h,c}` — runtime parser.
- `kernels/qnn/recognizers/` — the Python pipeline this dialect replaces.

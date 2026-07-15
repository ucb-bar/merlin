# bareMetalC uplift report (capsule_bench_v0)

## What was uplifted

12 bareMetalC behavioral specimens were uplifted into **compiler-input capsules** (see
`bareMetalC_uplift_plan.md` for the specimen→capsule table). Each capsule is an interface-MLIR
program + a numpy/`Tensor`-engine golden + an expected-instruction-coverage spec, compiled through
`agent_spec_v1_mlir_oot` and verified end-to-end. No bareMetalC C was copied, no bareMetalC kernel
was called, and no high-level Gemmini C kernel is invoked as the implementation — the device code is
the package's MLIR-lowered RoCC, confirmed by the decoded instruction trace (`rocc_decode`).

## Verification (per the recorded artifacts)

Pass/fail, numeric exactness, instruction coverage, and oracle tier reached for every uplifted
capsule are in `capsule_bench_v0_report.md` (generated from `capsule_result.json`),
`numerical_correctness_report.md`, and `isa_coverage_report.md`. Summary: the uplifted ISA + layer
capsules certify three-way bit-exact (golden == reference == simulate == oracle) on spike (L2) and
verilator RTL (L3), cycle-accurate; conv2d is compiler-lowered to im2col + matmul (runner-side im2col
materialization shared by golden/reference/simulator/harness); movement is a pure MVIN→MVOUT path
with no compute (trace_check enforces the absence of PRELOAD/COMPUTE).

## Explicitly NOT uplifted in v0 (out of scope; honest)

These bareMetalC families exercise features the v1 backend does not model and are recorded as
"not covered" rows in `isa_coverage_report.md` rather than silently omitted:

- output-stationary dataflow (`matmul_os.c`, `tiled_matmul_os.c`)
- hardware loop instructions LOOP_WS / LOOP_CONV_WS (`tiled_matmul_ws.c` hw-loop form, `conv*.c`)
- igelu / layernorm / softmax activations (`tiled_matmul_ws_{igelu,layernorm,softmax}.c`)
- depthwise / dilated / transposed / strided / rot180 conv variants (`conv_dw.c`, `conv_stride.c`,
  `conv_trans_*.c`, `conv_with_*dilation*.c`)
- residual-add / pooling / transpose / global-average (`resadd*.c`, `conv_with_pool.c` pool stage,
  `transpose*.c`, `global_average.c`)
- full-i64 C readout, low-D bias (`tiled_matmul_ws_full_C.c`, `tiled_matmul_ws_low_D.c`)

The v0 uplift covers the weight-stationary i8×i8→i32 matmul core + acc_scale/relu epilogues +
K-accumulation + resident reuse + edge padding + im2col conv + movement — the instruction classes the
backend emits. Broadening to the above is future work, not a v0 claim.

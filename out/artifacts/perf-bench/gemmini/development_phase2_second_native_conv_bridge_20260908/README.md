# Second native-convolution bridge: rejected on hardware evidence

This isolated artifact was opened to extend canonical ResNet-50 native `LOOP_CONV` coverage from
the dense stem to task 3 (`1x1`, `64 -> 64`, `56x56`).  It does not modify the sealed
`development_phase2_compute_only_loop_conv_20260908` checkpoint.

The experiment stopped before a whole-model build.  Queue job q545 established that the exact
compute-only mechanism is **slower on FireSim even for the stem**, where it needs no activation
layout copy: 1,383,906,735 cycles versus q535's 1,316,619,699, a 67,287,036-cycle or 5.1106%
regression.  Adding task 3's padded-NCHW bridge to that mechanism therefore fails the predeclared
benefit gate.  Coverage remains **1/53 native `LOOP_CONV` and 52/53 accelerated streamed-row
im2col + `LOOP_WS`**; the latter are accelerator kernels, not CPU fallbacks.

## Why q545 loses

The hardware counters and generated command stream agree on the mechanism:

| metric | q535 | q545 | delta |
|---|---:|---:|---:|
| measured cycles | 1,316,619,699 | 1,383,906,735 | +67,287,036 (+5.1106%) |
| retired instructions | 506,265,232 | 496,240,008 | -10,025,224 (-1.9802%) |
| main execute cycles | 3,564,267 | 42,665,060 | +39,100,793 (+1097.0%) |
| execute-active cycles | 31,255,069 | 67,561,554 | +36,306,485 (+116.2%) |
| loop-matmul-active cycles | 53,520,245 | 57,948,257 | +4,428,012 (+8.27%) |
| reservation-station-active cycles | 53,934,929 | 93,972,533 | +40,037,604 (+74.23%) |
| RDMA bytes | 1,142,808,576 | 1,145,952,896 | +3,144,320 (+0.275%) |
| WDMA bytes | 54,136,736 | 54,136,736 | 0 |

The issue is not CPU instruction count or output write volume.  q545 retires fewer instructions
and writes exactly the same number of DMA bytes.  Its extra main-execute time alone is 58.1% of
the end-to-end regression.

At task 1, q535 issues 112 `LOOP_WS` blocks (789 inline accelerator/fence commands).  q545 issues
120 `LOOP_CONV` descriptors but also 4,480 external accumulator-zero `MVIN3` commands and 4,480
full-width `MVOUT` commands, for 9,925 inline accelerator/fence commands.  The added RDMA volume is
0.979x one full i32 stem tensor, matching the explicit zero stream.

More importantly, the current RTL makes batch-one NCHW intrinsically inefficient.  With
`trans_input_3120`, `LoopConvExecute` sets systolic `I` to `min(batch, DIM)`, which is 1, and
advances output columns one at a time.  The safe `max_pixels_per_row=1` additionally serialises
the seven stem kernel columns.  The q535 im2col path presents spatial pixels as matrix rows and
uses the 16-row array.  Task 3's `1x1` avoids the kernel-column penalty but still exposes only one
row per execute, so copying padded NCHW to dense NCHW would not repair the utilization loss.
The RTL address equation can consume task 3's padded NCHW directly by putting its physical pitch
(64) in `in_col_dim`; this artifact implements and tests that exact addressing.  It deliberately
does not enable it automatically because legality does not reverse q545's hardware ranking.

Run `python analyze_q545.py` to reproduce the counter and command differential directly from the
sealed q535 receipt, q535 generated MLIR, q545 UART, and q545 generated MLIR.

## Viable route

The next attempt must not extend q545's contract.  It needs all three properties below:

1. Propagate physical NHWC through the producer/consumer chain so `LOOP_CONV` runs with
   `trans_input_3120=false` and exposes up to 16 spatial rows to the array.  For task 3, the host
   segment producing `t3` should write physical NHWC directly; a separate dense-NCHW copy is the
   wrong bridge.
2. Add an exact full-width `LOOP_CONV` store contract.  Current `LoopConvSt` hardcodes
   `read_full=false`, so it cannot directly produce the canonical i32 accumulator tensor.
3. Add no-bias overwrite semantics on the first reduction.  Current `LoopConvExecute` hardcodes
   accumulator `accumulate=true`; that forced q545's explicit zero stream.  On the first
   `(krow,kcol,kch)` iteration with `no_bias`, the controller should overwrite, then accumulate on
   later reductions.

The production compiler now rejects the q545 path with
`hardware_cost_guard_transposed_nchw_underfills_systolic_rows`.  The explicit
`--diagnostic-transposed-nchw-loop-conv` switch retains it for exact protocol work.  A canonical
production compile emits 0 native / 53 streamed convolutions and is byte-identical to q535 target
MLIR; diagnostic mode emits 1 native / 52 streamed and is byte-identical to q545 target MLIR.

The existing narrow fused store cannot substitute for (2): every captured convolution uses
genuinely per-channel f32 weight scale and non-integral f32 bias, while this target exposes one
scalar store scale.  Re-associating those stages is not bit-exact.

Until the controller/store contract changes, q535's streamed-row im2col + `LOOP_WS` is the correct
hardware-ranked implementation for all 53 convolutions.  A compiler cost model should reject
batch-one `trans_input_3120` compute-only selection rather than trusting Spike's -1.98% estimate.

## Executable whole-graph layout plan

`compiler/mlir_oot/lowering/physical_layout.py` now implements the missing bounded planning layer
without target assumptions.  It joins producer/consumer values across layout-polymorphic operators,
makes residual-branch agreement a hard constraint, maximizes accelerated operators in their preferred
layout, and places conversions only on non-polymorphic ports.  The operation vocabulary and supported
layouts are caller-provided data.

`compiler/mlir_oot/frontend/capture_layout.py` constructs that graph structurally from xDSL IR and
fails closed on unattributed activation operations.  Running `analyze_physical_layout.py` on the
canonical per-tensor ResNet capture produces the payload-free graph and census under
`validation/canonical_resnet50/`:

| structural result | count |
|---|---:|
| convolutions selected in preferred NHWC | 53 / 53 |
| residual merges with branch agreement enforced | 16 / 16 |
| rank-4 quantize/dequantize/ReLU regions propagated | 151 |
| activation values | 223 |
| joined layout components | 3 |
| required current-codegen boundaries | 4 |
| source-IR bytes at those boundaries | 5,017,600 |

The four conversions are exact and explainable: canonical ABI input to NHWC, NHWC to/from the
currently NCHW-only maxpool lowering, and NHWC to canonical layout before the final average-pool
reduction.  The latter `linalg.reduce` has no provenance annotation in the capture, so the extractor
correctly records it as an unattributed boundary.  The terminal rank-changing view also fails closed,
but it follows a canonical-layout reduction and therefore adds no conversion.

This is a **structural plan, not a runtime result**.  Enabling it requires backend implementations for
NHWC buffer allocation/index maps and the four conversion sites.  Making maxpool layout-polymorphic
would remove two of those sites; restoring provenance and adding a physical-axis contract to the
average-pool reduction is the next graph-level blocker.  The verifier re-solves the packaged graph and
checks all 53 convolutions and 16 residual constraints, so the census cannot silently drift.

At whole-graph level, the current command buffer still exposes 119 host regions: 50
`quantize_per_tensor`, 49 min/max, 16 residual adds, one maxpool, one avgpool, one view, and one
miscellaneous region.  The highest-value automatic path is therefore NHWC propagation plus an
exact per-channel-capable fused narrow store (removing quantize/minmax), Gemmini residual-add
lowering where legal, and fused maxpool.  The current one-scalar-scale narrow store is not yet
that contract.

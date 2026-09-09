# Atlas SmolVLA command-image integration recovery v3

This versioned artifact advances the committed compact-loop backend from
**compile/IMEM-fit only** to **representative RTL numeric**.  It does not claim a
whole SmolVLA image.

## Outcome

The first dependent multi-kernel interface failed before command-buffer
construction because `frontend.py` named only leaf tensors.  A `commit` result
consumed by a later matmul raised a lookup error containing
`OpResult[tensor<4x8xbf16>]`.  The isolated package now registers the committed
SSA value under its output name.  The resulting command buffer names `Y0` as the
second matmul's lhs and assigns non-overlapping storage to `W0`, `A0`, `W1`,
`Y0`, and `Y1`.

Six deterministic programs execute bit-exact on elaborated RTL through GSIM:

| case | purpose | words | commands | cycles | checked |
|---|---:|---:|---:|---:|---:|
| `bf16_movement_single` | minimal oracle control | 391 | 1 | 1,075 | 32/32 |
| `bf16_movements` | two outputs | 671 | 2 | 1,877 | 53/53 |
| `independent` | two FP8 matmuls | 2,359 | 8 | 6,010 | 53/53 |
| `chained` | FP8 matmul -> BF16 matmul | 1,882 | 8 | 7,314 | 52/52 |
| `smolvla_tail_50_720_32` | exact SmolVLA contraction shape | 12,605 | 4 | 300,068 | 1,600/1,600 |
| `smolvla_state_proj_1_32_960` | exact state projection + BF16 bias | 1,509 | 4 | 154,458 | 960/960 |

The 50x720x32 case selects the compact-loop emitter (46 logical tile
iterations), executes the K=16 tail, and checks every BF16 output bit-exact.
It is one of the saved 28 unique SmolVLA contraction shapes.  Its raw GSIM spec,
stdout, 3,200-byte readback, and hashes are retained under its case directory;
the spec contains only words, preloads, read ranges, and a cycle limit.  No
expected output is passed to GSIM.  An ECALL-only negative control returns zero
instead of the 43 nonzero expected values, independently ruling out expected
value echo/substitution.

## Functional-model contradiction

The fast Python functional result is not accepted as evidence for these emitted
programs.  The exact same 391-word single-BF16-movement program is bit-exact on
GSIM but all-zero in the functional model.  The model's own raw log reports the
eight emitted DMA config words as unsupported.

The decoder discrepancy is definitive:

- RTL `Instructions.scala` defines `DMA_CONFIG_ANY` with funct7=0 and
  `DMA_WAIT_ANY` with funct7=1.
- `submission/mlir_oot/encoder.py` emits those same values.
- `npu_model/configs/isa_definition.py` assigns funct7=1 to both config and wait.

Exact paths, source hashes, and excerpts are in
`evidence/functional_decoder_mismatch.txt`.  This establishes that the
functional decoder is incompatible with the RTL/compiler encoding; it does not
assert that this is its only defect.

## Honest coverage and remaining boundary

The new hybrid runtime slice turns the whole-capture manifests into a
deterministic 6,104-event host/device schedule and assigns all 391 accelerator
outputs to a 32-byte-aligned, lifetime-reused activation arena. The symbolic
arena peaks at 29,884,416 bytes instead of 550,404,032 bytes with dedicated
storage. This is allocation planning, not a measured runtime result.

Importantly, the inventory's 2,033 layout-bridge candidates are not all aliases.
The executable proof classifies 1,675 `view`/`unsqueeze` regions as metadata
aliases. It now also classifies and executes the other 358 as host
materializations: 246 `expand` and 112 `copy` regions, divided into 291 identity
copies and 67 constant-zero-axis broadcasts across 25 exact shape classes.
Admission proves the complete affine map and scalar value-copy body; execution
produces distinct C-contiguous storage. Eleven fresh real-capture witnesses
cover every semantic/dtype/map/rank topology class against independent NumPy
oracles. This does not implement zero-stride device descriptors or physical
DMA. There are separately 2,430 semantic
host-required regions. The fail-closed generic host lane now accepts all 2,430
of them from their complete extracted pointwise/cast/scalar-DAG signature,
including exact affine constant-zero broadcasting. The second tranche adds all
353 captured `pow`/`rsqrt`/trigonometric/sigmoid/GELU/reciprocal/clamp regions,
with declared intermediate dtype conversion and BF16 round-to-nearest-even.
The constructor tranche adds all 63 `arange` and 50 `fill` regions from their
exact scalar DAGs, constants, dtypes, and static shapes. It
also admits all 150 captured reduction/composite regions: mean, stable softmax,
two-pass layer norm, arg-min, cumsum, and boolean-to-integer sum. Their complete
iterator/reduction topology, scalar SSA dataflow, constants, comparison
predicates, and per-step dtype rules are signed and executed directly. It
distinguishes 45 conditional `aten.where` selects from two same-named
slice/reshape regions. A movement tranche then admits those two exact static
select chains together with all 129 static slices, 112 static slice scatters,
95 concats, 56 splits, 16 boolean bitwise regions, and two bucketize
reductions—412 regions total. It validates static bounds, concat axes and
shapes, reshape element counts, generic scalar dataflow, and declared dtypes;
malformed shapes or operation bodies fail closed. Relative to the prior
scoped-bridge baseline of 2,408 missing host regions, the exact missing count is
now zero, a reduction of 2,408. The final six qualified regions are two exact
embedding gathers, one two-index boolean gather, a linked mask-gather/index-put
pair, and the 16x16-stride patch-embedding im2col convolution. Their complete
affine/index/loop/reshape dataflow is signed; runtime indices and dynamic mask
update counts are checked. Fresh real capture-shape execution matches
independent numeric oracles, including the full 512x512 convolution input. The
existing real p0243-to-p0244 bridge still replays 27 regions and exactly
reproduces the saved p0244 activation hash, but the schedule no longer treats
scoped provenance membership alone as executable semantic evidence. The
neighboring device evidence is retained, assertion-clean RTL evidence and was
not rerun by this builder.

The builder also finds, rather than names, a 16-region consecutive real capture
chain spanning arange/compare/cast/arithmetic/select/pow/reciprocal, seeds one
fresh input, executes all 17 SSA dependencies, and obtains identical per-region
hashes on a second run. Additional signature-selected real witnesses exercise
cumsum-to-mean normalization, masked softmax, arg-min/reduction-sum successors,
sigmoid gating, trigonometric fan-out, constructor-fed arange/fill chains, and
static split/slice, slice-scatter, concat, bitwise, bucketize, and static-select
chains. Standalone layer norm and GELU instances are also exercised where
accelerator boundaries prevent a host dependency. This is fresh host numeric
evidence only.

Host semantic and layout-bridge coverage are now complete, but the hybrid
schedule remains explicitly `e2e_blocked_fail_closed`: 387 accelerator
partitions and the physical event/DMA runtime remain unresolved. See
`HYBRID_RUNTIME_REPORT.md` and
`whole_capture_plan/hybrid_schedule_summary.json`. No E2E or performance claim
is added.

The accelerator side is now split into 31 exact kernel/source/origin classes.
Static source, ABI, allocation, command-chain, and image-receipt agreement is
proven for 391/391 partitions: 208 BF16 no-bias rank-2 matmuls, 18 F32 no-bias
rank-2 matmuls, 77 F32 bias-fused rank-2 matmuls, 24 BF16 batched matmuls, and
64 F32 batched matmuls. Their 1,250/1,250 conversion boundaries have concrete
FP8/BF16 host conversion semantics, including all 88 direct device
requantizations and scaled BF16 output publication. Five real-shape software
witnesses match independent oracles. These events remain physically
non-executable unless their capture-bound RTL receipt exists, so physical
coverage is 4/391 and 15/1,250.

The batched command defect is repaired fail-closed: each buffer now records
`RES_PACK W -> W_resident`, consumes `W_resident` in `BATCHED_MATMUL`, and
evicts the handle. Mutating the consumer back to raw `W` or deleting the evict
is rejected. Fresh assertion-enabled GSIM preserves the exact
`15x50x64x113` result at 4,114,764 cycles with 0/84,750 mismatches; this is a
shape-level control, not physical capture qualification. The patch convolution
now has an exact fail-closed source contract for NCHW-to-im2col materialization,
kernel reshape, matrix ABI, NCHW result reshape, and per-channel bias. Altering
its affine stride or ABI origin chain is rejected. Its full `768x768x1024`
command was assertion-clean through a bounded 1,000,000-cycle GSIM run but did
not halt at 20,000,000 cycles, so it receives no numeric or physical promotion.
The retained cap receipt is under `patch_im2col_gsim_probe_v1/`.

The deterministic full-capture planner identifies 391 structural contraction
partitions, 28 kernel variants, 88 direct accelerator dependencies, and 303
maximal accelerator islands (239 singletons, 40 pairs, and 24 triples). Stable
dependency, lifetime, and ABI manifests are under `whole_capture_plan/`. All 28
variants and therefore all 391 structural occurrences compile within the
32,768-word IMEM; the maximum is 32,458 words. This is structural/IMEM coverage,
not executable f32-capture coverage: every partition still requires calibrated
f32-to-FP8 input/weight conversion and BF16-to-f32 output conversion, so the
base planner manifest explicitly reports zero capture-semantics-executable
partitions because that snapshot describes the structural emitter alone.  The
bounded calibrated bridges subsequently qualify exactly **4/391** real capture
partitions: `atlas_p0098`, `model.state_proj` (`matmul_97` + `add_99`);
`atlas_p0102`, the first text-layer attention QK contraction (`matmul_101`);
`atlas_p0243`, `model.action_in_proj` (`matmul_242` + `add_197`); and
`atlas_p0244`, `model.action_time_mlp_in` (`matmul_243` + `add_199`).

For `atlas_p0102`, the publication boundary includes its sole immediate graph
frontier, the captured multiply by 0.125. The exact `15x113x64x113` operands
run on assertion-enabled GSIM in 8,700,444 cycles. The result passes the fixed
source gate with 0.103215 max absolute error and 0.999278 cosine similarity;
the independent quantized-domain comparison has 0.006253 max absolute error
and 0.999998 cosine similarity. Raw-`W`, missing-evict, and perturbed-source
controls fail closed. This qualifies one physical partition, not the later
mask/softmax chain or whole-model execution.

For the state-projection partition the bridge loads the original `state` input and
`model.state_proj.{weight,bias}` tensors, applies the recorded weight transpose,
and uses symmetric per-tensor E4M3FN calibration:

```
qA = E4M3FN_RNE(A / sA)
qW = E4M3FN_RNE(W / sW)
qB = BF16_RNE(B / (sA * sW))
Y_f32 = f32(Y_bf16_device) * (sA * sW)
```

Bias is therefore added in the device quantization domain before output
dequantization; it is not compared as an unscaled BF16 value.  The fixed,
predeclared gate in `calibration_contract.json` is max absolute error <= 0.125
and cosine similarity >= 0.995.  On elaborated RTL the real partition passes at
0.070929 max absolute error, 0.017297 mean absolute error, 0.021717 RMSE, and
0.999455 cosine similarity versus the independent f32 source reference.  The
stabilized max relative error is 16.8589 with denominator floor 0.001 and is
reported, not gated, because near-zero reference elements dominate it.  Against
an independently reconstructed quantized-domain reference, max absolute error
is 0.012992 and cosine similarity is 0.999997.  Folding the bias contributes at
most 0.000459 absolute error after rescaling.  The unchanged 1,509-word image
halts in 154,458 GSIM cycles.

The second bridge binds `noise` directly from captured `inputs.npz:in5`, checks
its exact `tensor.expand_shape`/`tensor.collapse_shape` view chain, and binds the
original `model.action_in_proj.{weight,bias}` tensors. The resulting real
50x32x720 projection executes the planned 15,175-word image on assertion-enabled
GSIM in 573,715 cycles. It passes the same fixed gate with 0.107365 max absolute
error, 0.017213 mean absolute error, 0.021912 RMSE, and 0.999329 cosine
similarity versus the independently recomputed f32 source result. Against the
quantized-domain reference, max absolute error is 0.019068 and cosine similarity
is 0.999997. Its dispatch manifest publishes at capture op 8287 and retains the
output through its last consumer frontier at op 8357.

The third bridge is the first small real chain. It consumes the qualified
`atlas_p0243` device result, reproduces the capture's f64-to-f32 sinusoidal time
embedding on the host, concatenates those two 50x720 tensors, and binds the
original `model.action_time_mlp_in.{weight,bias}`. A single unsliced command
buffer is invalid in the adopted simulator: the harness masks DRAM addresses to
a 1 MiB byte array, while that image spans 1,182,240 bytes (133,664 bytes too
large). A fail-closed preflight now rejects such images. The bridge instead
executes three compiler-generated contiguous N slices (256, 256, and 208
columns), each independently alias-free and assertion-clean, and concatenates
their outputs into the capture's 50x720 result. Calibration uses an explicit
E4M3 code cap of 16 per slice; the scale equation is unchanged, but this
headroom prevents unsupported quant-domain accumulator growth. The combined
result passes the fixed source gate with 0.066370 max absolute error and
0.999033 cosine similarity; its three dispatches total 15,211,545 GSIM cycles.
Against the independently reconstructed quantized-domain reference, max
absolute error is 0.032333 and cosine similarity is 0.999969. This counts as one
capture partition and three RTL dispatches, not three partitions or E2E.
The earlier unsliced trace is retained under
`capture_semantics_action_time_mlp_in_diagnostic/`, explicitly marked invalid
due to harness aliasing; its numerical differences are not compiler evidence.

The preferred real 50x720x32 `model.action_out_proj` partition
(`atlas_p0390`) is not counted as qualified. Its A0 originates in host-required
`dtype_cast_471`, behind `view_1321`, so the captured inputs do not contain the
partition activation. Substituting the model's final output would invalidate
the test. This partition remains blocked until the preceding host prefix is
executed and its value is handed to the device ABI.

`capture_semantics_state_proj/dispatch_manifest.json` records bind, conversion,
launch, publication at op 2827, and release after the final frontier consumer at
op 2834.  An assertion-enabled replay first exposed and rejected an illegal
`VLI_ALL 63`: the destination encoded odd bank 63, but pair writers require an
even base. Atlas is not using RV32's five-bit `rd` here: RTL `ScalarDecoder`
extracts six-bit VR/VI fields (`vd=instr[12:7]`, `vs1=instr[18:13]`, and
`vs2=instr[24:19]`), and `VectorEngineTop` enforces even primary/secondary pair
reads and pair writes. The emitter now reserves `VLI_ALL 62/63`. A static
encoded-word validator mirrors those VPU rules plus MXU BF16-pop pair writes,
while allowing legal odd single-bank loads/stores.

`raw_gsim_spec.json`, stdout, stderr, and
`raw_gsim_receipt.json` retain the exact submitted words/preloads, no-golden raw
result page, halt/cycle/read/write counters, engine hash, and explicit absence
of a final-PC field. The retained run uses `atlas_gsim_sim_assert`, returns zero,
halts, and has empty stdout/stderr assertion diagnostics (stdout contains only
the final JSON result page).
`device_output.bf16.bin` retains those device words. Validation verifies all
receipt hashes, reloads the original source tensors and device words, and
recomputes both comparisons; it also verifies that a deliberate output
perturbation fails the fixed gate. The parallel
`capture_semantics_action_in_proj/` and
`capture_semantics_action_time_mlp_in/` directories retain equivalent
no-golden raw simulator evidence, independently recomputed references, and
separate negative controls. Together with the batched QK result, this is
approximate FP8 qualification of four partitions, not source bit-exactness or
whole-model execution.

RTL numeric coverage is currently 4/28 unique capture-bound contraction shapes;
this counts distinct kernel IDs among the four directly qualified physical
partitions and excludes non-transitive shape-only controls. Real
capture-semantic coverage is exactly 4/391 physical contraction instances,
plus the small dependency/ABI controls above. The new state-projection case
caught and fixed a real defect: compact FP8 matmul had
silently omitted `bias_add`; a first fix used `VREDSUM` as a broadcast and
permuted lanes on RTL. The backend now materializes the exact two-register row
layout, adds bias before ReLU, and rejects compact scale/ReLU combinations until
their dynamic epilogues exist. Inputs intentionally use one exactly
representable nonzero product per dot product, so arbitrary FP8 accumulation
remains unproven.

The command list is generated from the same workload and its tensor ABI drives
RTL preload/readback, but Atlas executes the emitted kernel words; the JSON
command list itself is descriptive and is not interpreted by the RTL harness.

The unchanged 4.35-MB full capture now parses and verifies in about 7.7 seconds.
The apparent first `tensor.expand_shape` failure was actually an xDSL
multi-result `linalg.generic` parser/printer inconsistency: this xDSL revision
prints `} -> (tensor<T0>, tensor<T1>)` but parses only the equivalent unwrapped
result list.  `normalize_xdsl_parser_compat` removes exactly eight such wrappers
in memory; it does not modify the capture, its types, attributes, or SSA graph.

The probe now reaches the real full-graph boundary.  It writes a 514-tensor
command-buffer schema with an explicit zero-command decline, then fails target
emission instead of manufacturing an ECALL-only image.  The verified inventory
contains 4,930 provenance regions and 391 physical contractions (303 rank-2 and
88 batched).  Exactly 2,131 regions have no isolated semantic emitter.  A
further 299 vector-semantic regions have dtypes, ranks, or extents outside the
existing emitter, making the current effective host-required total 2,430.

One concrete partition is retained under `partitions/first_addmm_matmul_0`.
It fuses capture regions `matmul_0` and `add_3` (the first bias-fused rank-2
physical contraction after the patch convolution), names all three
inputs and its output at the host/device boundary, folds the bias into commit,
and compiles to 8,037 words: 24.5% of the 32,768-word IMEM. This proves a real
partition can be selected and emitted from the full-capture inventory.  It is
compile-only: FP8/BF16 calibration and numeric comparison to the f32 capture
remain required.

The 28 unique contraction programs do not fit one resident image. A real
end-to-end SmolVLA result still requires:

1. capture-specific calibration and numeric qualification for the remaining 387
   accelerator partitions and their 1,235 physically unqualified conversion
   events; the static conversion contracts now cover 1,250/1,250 boundaries;
2. a physical runtime for the already ordered 6,104 host, bridge, conversion,
   and device events, including binding the interval-planned device arena;
3. one fresh full-input execution and source-model output comparison.

No whole-model image, whole-model numeric result, performance result, or new
capsule score is claimed.

## Reproduce

From this artifact directory:

```bash
MERLIN_REPO_PYTHON=../../../../../.venv/bin/python
$MERLIN_REPO_PYTHON -m pytest -q test_*.py
$MERLIN_REPO_PYTHON validate_recovery.py
MERLIN_ATLAS_GSIM_DIR=/path/to/atlas-gsim MERLIN_MLIR_INSTALL=/path/to/llvm-install $MERLIN_REPO_PYTHON run_capture_partition.py state_proj
MERLIN_ATLAS_GSIM_DIR=/path/to/atlas-gsim MERLIN_MLIR_INSTALL=/path/to/llvm-install $MERLIN_REPO_PYTHON run_capture_partition.py action_in_proj
MERLIN_ATLAS_GSIM_DIR=/path/to/atlas-gsim MERLIN_MLIR_INSTALL=/path/to/llvm-install $MERLIN_REPO_PYTHON run_capture_partition.py action_time_mlp_in
$MERLIN_REPO_PYTHON run_integration.py chained --engine gsim --max-cycles 200000
$MERLIN_REPO_PYTHON run_integration.py smolvla_tail_50_720_32 --engine gsim --max-cycles 50000000
$MERLIN_REPO_PYTHON run_integration.py smolvla_state_proj_1_32_960 --engine gsim --max-cycles 1000000
$MERLIN_REPO_PYTHON run_raw_gsim.py
$MERLIN_REPO_PYTHON run_negative_control.py
$MERLIN_REPO_PYTHON probe_full_capture.py
$MERLIN_REPO_PYTHON inventory_full_capture.py
$MERLIN_REPO_PYTHON inventory_capture_bindings.py --check
$MERLIN_REPO_PYTHON build_first_partition.py
$MERLIN_REPO_PYTHON build_partition_plan.py
$MERLIN_REPO_PYTHON build_hybrid_schedule.py
```

`validation.json` and `receipt.json` are the machine-readable summary.  The
backend lives only in this recovery artifact and does not overwrite the
committed compact-loop package.

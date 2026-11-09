# rvv-mlir

## Explicit integer file preparation

`backend.integer_file_api()` exposes default-off preparation for a native file
workflow. `prepare_file(source, work, target, windows=False,
contractions=False, blocks=False)` uses Merlin's existing contraction normal
form for closed signed-I8, zero-seeded I32 MACs. It preserves the public output
layout and rejects unknown scalar wiring, unsigned extensions and overflow
flags. The selected provider owns canonical gathering and register scheduling;
the caller still runs ordinary native lowering and complete numerical checks.

With `blocks=True`, exact power-of-two M/N divisors are bounded by the supplied
`PanelTarget`: VLEN, SEW, LMUL, architectural register budget and explicit
workspace/cache budgets remain separate. This avoids arithmetic on padded
output tiles; prime dimensions may conservatively use one-element tiles. It is
a legalization/scheduling option, not a measured optimal kernel or a hardware
utilization claim. `lowering_schedule(report['register_blocks']['table'])`
uses the existing per-op schedule machinery. Combine it with the public
streaming API's `lowering_pipeline(schedule_path)` to retain its structural
tags. No existing pipeline or default feature is changed.

The **base RISC-V Vector (RVV) layer** for [Merlin](https://github.com/ucb-bar/merlin).

The `handwritten-implementation` branch carries executable RVV pipeline construction in `src/rvv/backend.py`
and checked schedule payloads in `schedules/`. Merlin loads `plugin.rvv_codegen` from the pinned provider
root through the small `backend.py` compatibility entrypoint and checks `API_VERSION` before selecting the vector path. A scalar build can run without this
provider. The pipeline uses upstream MLIR passes; this repository does not define a custom RVV dialect.

Everything in this repository is plain RVV. It contains no matrix-unit or OPU code and no
chip-specific facts. Chip packages build on it and add only what their chip has:

```
rvv-mlir      base: the RVV target contract (vector unit, dtypes, requant), RVV schedules
 ├── kodiak-mlir   Kodiak tapeout: Saturn + outer-product unit (OPU, BME encodings), its boards
 └── iris-mlir     Iris: 2× Shuttle+Saturn(VME) + Rocket(DMA vector), its boards and VME extension
```

A chip package's `provider.yaml` says `extends: {target: rvv, commit: <sha of this repo>}`, and its
`contracts/overlay.yaml` holds only the delta. `merlin-target-compose` turns base + overlay into the
effective contract and records the exact base revision, so a chip package inherits *all* of RVV and
notices when this base moves.

## Compiler layout

- `src/rvv/backend.py` builds the RVV pass pipeline and exposes schedule and preparation APIs.
  Root `backend.py` forwards the pinned plugin entrypoint to that package.
- `src/rvv/features/` registers named tuning choices; `src/rvv/transforms/` contains
  per-operation blocking, vector tagging, reduction and requantization rewrites, and runner source.
- `src/rvv/packing/` implements im2col and weight panel layouts; `src/rvv/policy/`
  derives legal blocks, LMUL, and workload shape adaptations.
- `schedules/` contains Transform dialect programs and knob data. Upstream MLIR supplies the passes
  those programs invoke; this repository does not ship a custom dialect or pass binary.
- Merlin owns target-neutral capture, quantization semantics, and scalar compilation. Kodiak and
  Iris overlays own their OPU and VME extensions.

## What is on which branch

| branch | what it is |
|---|---|
| `handwritten-implementation` (this branch) | the RVV support provider and executable pipeline selected by Merlin: contract, `src/rvv/`, schedules, tests |
| `main` | historical landing page; the repository default branch changes after integration gates pass |
| `stable/impr_tuned_wholemodel_vf_int8` | int8 W8A8 whole-model schedule (`vwmacc`), K1-verified; the RVV schedule the Kodiak builds use |
| `stable/sol_k1_int8` | int8 speed-of-light schedule, K1-verified, with its analysis |
| `baseline-int8_w8a8` | frozen, hand-written, unoptimized int8 control (the before/after reference) |
| `stable/rvv_tuned_v1_d1_vfmacc_outerproduct` | fp32 schedule, spike-verified |

> **Note on `rvv_tuned_v1_d1_vfmacc_outerproduct`.** "outerproduct" is the name of MLIR's
> `vector.contract` lowering strategy (`lower_contraction lowering_strategy = "outerproduct"`). It is
> **not** the Saturn outer-product unit. That package emits ordinary RVV (`vfmul.vv` + `vfadd.vv`). The
> OPU kernels are in `kodiak-mlir`.

Each schedule branch is a standalone package: `payload/schedule.mlir` (transform-dialect tiling and
vectorization of the contractions), `payload/knobs.yaml` (cflags, dtype strategy, tile sizes, LMUL policy,
expected instructions) and `.merlin/` provenance. Merlin applies one with
`merlin.mining.apply.apply_rvv_package(...)`; only the schedule and cflags change.
Chip delivery packages and board receipts, including Kodiak's prepack W8A8 image, live in the
corresponding chip repository. The reusable `prepack_weight_layout` feature remains here.

## Using it

```sh
git clone https://github.com/ucb-bar/merlin.git && cd merlin
uv sync --extra dev
# The Merlin checkout vendors this provider at examples/rvv/support and selects it by default.
# To use a clone of this repository instead:
export MERLIN_TARGET_PATH=/path/to/rvv-mlir
merlin-compile --workload <model> --dtype int8 --target rvv --board <board> --run spike --verify
```

For a specific chip, fetch its package. `merlin-target-fetch` follows `extends` and brings this base too:

```sh
merlin-target-fetch kodiak      # clones kodiak-mlir, then rvv-mlir at the commit kodiak pins
```

## Adding a chip on top of RVV

1. Create `<chip>-mlir` with a `provider.yaml`:
   ```yaml
   schema: merlin.provider.v2
   id: <chip>-support
   target: <chip>
   role: support
   extends:
     target: rvv
     repo: https://github.com/ucb-bar/rvv-mlir.git
     commit: <full sha from rvv-mlir handwritten-implementation>
   ```
2. Put only the delta in `contracts/overlay.yaml`: the RTL config (`rtl_source`), extra compute units
   (a matrix unit), `matrix_contract`, and your own `plugin` entries. The base plugin block is never
   inherited, so executable support always lives in the package that declares it. Use `remove:` and
   `replace:` (lists of dotted paths) for anything the base says that is not true of your chip.
3. Put your boards in `boards.yaml` (Merlin's board schema; select with
   `MERLIN_BOARD_CATALOG=<chip>-mlir`).
4. Run `merlin-target-compose <chip>-mlir`. Commit the generated `contracts/target_contract.yaml`
   and `contracts/composed.lock.yaml`. CI runs `merlin-target-compose --check`.

Hardware revisions (RTL commits) are pinned in Merlin's single registry,
`merlin/contract/hardware_pins.yaml`, and a chip package refers to them by pin id.

## Checks

```sh
python -m pytest             # with Merlin installed; pyproject.toml selects tests/
ruff check .                 # pyproject.toml configures local linting
```

`provider.yaml` and the target contract remain the runtime metadata for Merlin. Merlin does not
consume a `backend.toml`; the schedule and flags are selected by the versioned `src/rvv/backend.py`
API rather than a second configuration file.
`provenance.json` records the content hash of the contract. Nothing here claims hardware execution or
candidate certification.

## Opt-in streaming im2col preparation API

The default pipeline is unchanged. The provider now exposes
`backend.im2col_stream_api()` for callers that explicitly select bounded,
sequential producer/consumer panels. This is a preparation API; the standard
`merlin-compile` CLI does not automatically select it.

```python
api = backend.im2col_stream_api()
target = api.PanelTarget(
    vlen_bits=target_vlen, elen_bits=target_elen,
    operand_bits=8, accumulator_bits=32, operand_lmul=1,
    workspace_bytes=activation_panel_budget,
    # cache_budget_bytes=explicit_panel_cache_budget,  # optional, never inferred
)
prepared, report = api.rewrite_prepared_file(
    original_prepared_ir, target, private_build_directory, enabled=True
)
if report.streamed:
    schedule.write_text(api.lowering_schedule(report))
    pipeline = api.lowering_pipeline(schedule)
    # Normal upstream compiler lowering:
    # mlir-opt prepared --pass-pipeline="builtin.module(<pipeline>)" -o lowered.mlir
    # mlir-translate --mlir-to-llvmir lowered.mlir -o model.ll
    # clang -target riscv64 -march=rv64gcv_zvl<TARGET_VLEN>b ... -c model.ll
```

The built-in adapter recognizes a static NCHW copy-gather followed by the
canonical `weight[F,K] × activation[K,M] -> output[F,M]` contraction. It
admits only signed i8 operands, i32 modular accumulation, an explicit zero
fill, the closed extend/multiply/add body, and default overflow flags.
Shared gathers, unsupported layouts/bodies/types, unknown dimensions,
insufficient workspace and windows outside the input are refused with named
reasons. Padding must already be explicit; the pass does not invent or move
padding or quantization.

Each SCF iteration gathers a local receptive field, contracts it and inserts
only its output slice. Full panels and row tails have separate static shapes.
Their workspaces are outside their loops and fully overwritten. The summed
activation payload is bounded by the requested budget; output storage,
allocator alignment overhead and vector lowering scratch are separate
requirements. The output accumulator and each element's increasing-K
reduction order are preserved. Panel loops are sequential; this API does not
authorize sharing scratch across concurrent iterations.

VLEN controls architectural register capacity. Operand SEW, accumulator SEW,
ELEN, LMUL and the register budget are separate inputs. Spatial panel width is
limited by explicit workspace/cache budgets and K, independently of register
block NR; it is not set equal to VLEN. DLEN and physical utilization are not
inferred. The geometry is an admissible starting point, not a measured tuning
winner.

Extension providers can use `WindowSpec` and
`build_stream(input, accumulator, spec, T, consumer, panel_layout="mk",
output_axis=0)`. The callback receives a panel, existing accumulator slice
and static width and returns operations plus the resulting slice.
`WindowSpec(input_layout="NHWC", reduction_order=("kh","kw","c"), ...)`
preserves the NHWC flattened reduction order; the default is NCHW with
`("c","kh","kw")`. The extension owns arithmetic, call alias contracts and
admission. The built-in RVV matching adapter currently supports NCHW only.

Run the bounded functional/toolchain proof explicitly:

```sh
PYTHONPATH=/path/to/merlin/src python tests/prove_im2col_stream.py \
  --llvm-bin /path/to/llvm/bin --linker /path/to/riscv64-unknown-elf-ld \
  --spike /path/to/spike --output /new/proof-directory
```

The current proof covers strides, dilation, batch/row tails and both selected
LMUL budgets at architectural VLEN 128/256/512/1024, with full-output and guard
checks on host and RVV Spike execution. It checks actual guest `vlenb`,
emitted vector arithmetic and allocation placement. It establishes no speedup
for this pass, no physical lane utilization, and no timing accuracy. FPGA
panel-locality observations that motivated the optimization are separate
experiments; performance at other VLEN values has not been measured.

## Opt-in static strided-copy lowering

Use `backend.copy_loops_api().lower_file(..., enabled=True)` to replace eligible
integer `memref.copy` operations with lexicographic sequential SCF load/store
loops after bufferization. The normal pipeline remains the default:

```python
copies = backend.copy_loops_api()
result = copies.lower_file(
    prepared_ir,
    pipeline=stream_api.lowering_pipeline(schedule),  # complete provider pipeline
    mlir_opt=selected_mlir_opt,
    compiler_python=selected_python_with_torch_mlir,
    work=new_private_directory,
    enabled=True,
)
# result["llvm_dialect"] -> ordinary mlir-translate -> LLVM code generation
```

The selected native Python bindings are an explicit compiler tool; the
rewriter imports no Merlin code and records their file digests. The API
preserves the selected virtual-environment interpreter path. Existing output
directories are refused.

The pass accepts ranked static i8/i16/i32/i64 copies in CPU memory with
strided layouts, up to the explicit rank budget (default 4). Dynamic offsets
and strides are retained in the original descriptors. The loops read an
element before storing that element and preserve lexicographic order for
overlapping views, including negative and zero strides. Loads/stores specify
alignment 1; no alignment, disjointness, noalias or parallel-execution
assumptions are added. No buffer allocation, tile layout, arithmetic,
quantization domain or reduction order changes. Both-contiguous copies keep
the upstream fast path. Dynamic shapes, unranked buffers, noninteger data,
other memory spaces and excessive rank retain their original transport with
named refusal reasons. This is cheaper copy transport, not copy elimination.

VLEN is not hardcoded by this pass. The caller's original vector schedule
and register geometry remain in force. The guarded convolution corpus passes
at VLEN 128/256/512/1024; the separate alias corpus checks full storage for 80
before/after host and native cases across all four integer widths. These
functional checks do not establish timing accuracy or physical utilization.

A fresh relink of the original tiny compiler-generated convolution keeps the
driver, runtime, VME controls and independent goldens unchanged. The first
functional comparison reduces RVV instruction work substantially; the release
receipts distinguish instruction counts from hardware cycles. No automatic
CLI routing, whole-model coverage or FPGA speedup is claimed.

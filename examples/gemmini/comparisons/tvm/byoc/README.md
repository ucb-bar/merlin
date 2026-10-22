> Legacy BYOC snapshot pinned to the bringup compiler. The current TensorIR and
> standard MetaSchedule path is documented in [../README.md](../README.md).
> This directory remains a regression reference and is not installed into the new path.

# Gemmini BYOC with TVM-owned schedules

This implements **C1 from the requested outline**: TVM's BYOC pattern registry
and `FuseOpsByPattern` select matmul/bias/ReLU composites; `LegalizeOps`,
`FuseOps` and `FuseTIR` create their semantic TensorIR; a public `tir.Schedule`
tiles and tensorizes that IR; the existing baremetal exporter consumes ordinary
`call_tir`. The complete product buffer is not exported between a selected
matmul and its epilogue. TVM owns CPU arithmetic, padding, allocation compaction,
loop generation and LLVM code generation.

The implementation is [gemmini_byoc.py](gemmini_byoc.py). All 51 requested
operation names are covered by [SUPPORT.md](SUPPORT.md), including supported
integer subsets, CPU work, frontend gaps and hardware-specific reasons for
limits. Full generic ATen coverage and full-model qualification are not claimed.

This is intentionally the partitioning variant of BYOC, **not C2**: there is
no `relax.ext.gemmini`/`RunCodegen` runtime-module integration. Apache's
[BYOC architecture](https://tvm.apache.org/docs/arch/external_library_dispatch.html)
describes that additional runtime-module route. Here a backend-specific lowering
pass converts the standard selected regions to TVM-generated TensorIR, preserving
the existing static exporter ABI. No fake runtime module, exporter exception,
handwritten graph operator or TVM core/RTL patch is used. The pinned v0.19
[pattern API](https://github.com/apache/tvm/blob/v0.19.0/python/tvm/relax/transform/transform.py)
and [TensorIR Schedule API](https://github.com/apache/tvm/blob/v0.19.0/python/tvm/tir/schedule/schedule.py)
are the actual implementation interfaces.

## Scheduling and numerical contract

Computation is an untiled ordinary integer reduction. Its schedule records
pointwise inlining, `pad_einsum`, `split`, `reorder`, `decompose_reduction`,
`compute_at`, `reverse_compute_at` and `tensorize`. Both output-tile orders are
supported; `schedule_matmul(..., order="sample", seed=1)` uses TVM's
`sample_categorical` and records its decision. Traces are replayed under
`debug_mask="all"`. These are native TVM scheduling decisions and can be used
as search candidates. No measured tuning winner or general MetaSchedule search
space is claimed.

The primitive is a mathematical **16×8×16 C += A×B** operation implemented in
[byoc_primitives.c](byoc_primitives.c). It reloads incoming C into accumulator
SRAM, computes a fresh bounded OS reduction, reads full INT32 output and fences.
Its semantics do not depend on hidden device state across calls. This permits
schedule reordering at the cost of extra DMA/configuration/fences. K chunks of
16 protect the pinned 20-bit PE output; full K≤131071 protects signed INT32
contractions for every INT8 input.

The N tile is 8 so compact INT32 staging is 512 bytes. TVM's default CPU lowering
places constant global allocations **strictly smaller than 1024 bytes** on the
stack. A 16×16 staging tile would instead require its workspace allocator.
The implementation uses ordinary scheduling/compaction and leaves that rule and
the exporter's allocator rejection intact. The selected small guests retain a
16-KiB stack and check its bottom canary. This does not establish stack capacity
for arbitrary future workloads.

Bias/ReLU are ordinary CPU epilogues inside a fused, tiled function. Full-width
readout on this RTL bypasses narrow activation/scaling, so enabling hardware
ReLU would not implement the declared INT32 result. Convolution retains explicit
CPU packing/restoration; BMM uses per-batch contractions. Full im2col and all
surrounding CPU work count toward the workload.

## Install and use

From the Merlin worktree, with the existing pinned tools/build:

```bash
base=examples/gemmini/comparisons/tvm/byoc
python3 "$base/install_backend.py" \
  --tvm-source out/build/baselines/tvm-gemmini/tvm-source
```

The installer checks TVM `6b376cc8895a83387ec38cc06ef8f7b09827b071` and copies
only the new contrib module and primitive source. `--check` verifies exact
installed bytes. No C++ rebuild or dependency installation is needed. The tracked
copies here are authoritative; the generated TVM checkout is gitignored in
Merlin. Development branches are `comparisons/tvm-gemmini-byoc` in Merlin and
`gemmini/byoc` in that TVM checkout.

```python
from tvm.relax.backend.contrib.gemmini_byoc import prepare_gemmini_byoc

records = []
prepared = prepare_gemmini_byoc(semantic_relax_module, records=records)
# Existing apps/gemmini/baremetal.export_graph(prepared, ...) remains unchanged.
# Link the new byoc_primitives.c against the selected pinned Gemmini headers.
for symbol, schedule in records:
    print(symbol, schedule.shape, schedule.trace)
```

`partition_for_gemmini` and `LowerGemminiBYOC` are also exposed separately for
inspection or pass-pipeline composition. Unsupported calls keep ordinary Relax
semantics. Other backends' Codegen functions are preserved.

## Bounded reproduction

[run_local.sh](run_local.sh) uses one service at a time with **1 GiB memory,
zero swap, one CPU, 64 tasks and a 60-second service deadline**. Guest execution
has a separate 45-second deadline; Verilator has a 200,000-cycle limit. It uses
the previously built simulator and Spike plugin. It does not build hardware,
import models, download weights or run a full model. The wrapper's workspace
defaults can be overridden with `TVM_ROOT`, `TVM_BUILD`, `BYOC_PYTHON` and
`CHIPYARD_ROOT`. It expects the retained exact-header adapter and host dependency
directory described by the parent comparison README.

Use a fresh output root:

```bash
base=examples/gemmini/comparisons/tvm/byoc
run=out/artifacts/verification/gemmini/tvm/byoc-new
bash "$base/run_local.sh" host "$run/host"
for case in tails bmm conv; do
  bash "$base/run_local.sh" prepare "$case" "$run/$case"
  bash "$base/run_local.sh" spike "$run/$case" "$run/$case/spike.json"
  bash "$base/run_local.sh" rtl "$run/$case" "$run/$case/rtl.json"
done
bash "$base/run_local.sh" spike "$run/tails" "$run/tails/spike-negative.json" --bad-oracle
bash "$base/run_local.sh" rtl "$run/tails" "$run/tails/rtl-negative.json" --bad-oracle
bash "$base/run_local.sh" spike "$run/tails" "$run/tails/spike-bad-input.json" --bad-input
bash "$base/run_local.sh" rtl "$run/tails" "$run/tails/rtl-bad-input.json" --bad-input
```

The guest checks two exact output comparisons, device call counts, output and
workspace overrun guards, input preservation, rejected unaligned/null inputs and
the stack canary. The tails case also runs the existing vendor OS adapter as a
separate **Gemmini practice** check against an independent INT64-derived oracle.
Final ELFs have no unresolved symbols and are audited against **CUSTOM_3 opcode
0x7b**, derived from the pinned header. An injected forbidden instruction proves
the audit is active. Wrong-oracle runs require the numerical-failure marker and
guest 22; on this Verilator harness that means decoded TSI guest 22 plus direct
host status 255. Positive RTL runs require the application PASS and `$finish`.

Input preservation uses volatile reads of a writable input fixture and an
independently host-computed scalar digest. The `--bad-input` control changes one
byte after numerical checks and must report guest 26 with the preservation
failure marker. This replaces the earlier constant-array comparison, whose
reference copy could be pooled onto the input itself by the C compiler.

Host regressions use a labelled scalar stand-in for the intrinsic to test compiler
transformations; hardware evidence comes from the real primitive in Spike/RTL.
They cover canonical and fused trace replay, both loop orders, sampled decisions,
tails, maximum K, leaked intermediates, dtype/dynamic-shape rejection, isolation
from other backends, BMM, convolution and CPU scalar/reduction fallback.

The [validation report](VALIDATION.md) records actual results and retained receipts.
Spike cycles are functional instruction-model counts. Local RTL uses the existing
256-MiB magic-memory model; its cycle counts are small-run diagnostics, not
FireSim measurements, model-quality evidence or a performance advantage claim.

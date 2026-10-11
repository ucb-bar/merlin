# Gemmini verification walkthrough

These retained probes are historical inspection tools. Fresh Phase 1 and Phase 2
require independently issued hardware and runtime authorities; running a probe
with a previously written compiler does not issue those authorities. The
handwritten support provider and copied kernel headers have been removed.

### Qualify a matching gSIM build

[qualify_gsim.py](qualify_gsim.py) exercises an existing generated compiler on
existing capsule artifacts. It requires a complete gSIM build receipt and the
exact same selected FIRRTL bytes for its Verilator reference. It does not
regenerate the corpus, modify the compiler, or substitute a different RTL
configuration. For each requested capsule it retains the command buffer,
lowered LLVM MLIR, same ELF, observed consoles, wall times and numerical capture.

```sh
python examples/gemmini/verification/qualify_gsim.py \
  --build-root /generated/gsim-build \
  --verilator /selected/simulator-chipyard.harness-GemminiRocketConfig \
  --verilator-firrtl /selected/TestHarness.fir \
  --package /generated/phase1/compiler \
  --capsule /generated/phase0/corpus/layers/example/capsule.yaml \
  --out /generated/gsim-qualification --jobs 3 --serial-timing
```

The optional serial timing check also requires exact output and kernel-cycle
agreement with the same serial TSI loading policy. The normal fast gSIM path
uses backdoor loading, which can leave different cache state from Verilator's
serial loader; its cycles are not directly comparable. Numerical captures are
reusable for identical pinned ELF/engine/input bytes, while timing requires a
separately observed loading/warmup policy. These finite checks qualify only the
tested workloads, not every model or all RTL configurations. At most four
qualification jobs run together; experiment gSIM concurrency remains capped
at five.

The [software spec](../target/README.md) is authored software-visible semantics.
[hardware.yaml](../platform/hardware.yaml) selects the deterministic production
policy and direct audit questions. [host-capabilities.yaml](../platform/host-capabilities.yaml)
is a separate declaration bound to immutable
host compiler bytes; its precision strategy is not an operation support claim.

The software spec is a small authored input: selected arithmetic/overflow and
readout semantics, typed operation placement constraints, quantization parameters,
and explicit transfer candidates. It contains no backend, ISA table, mesh size,
calibration history or generated qualification records. Those come from the
explicitly selected OOT provider and generated RTL/characterization artifacts.
The twenty-bit partial-sum bound is a numerical contract, not thirty-two-bit SRAM
geometry. Review/unknown states prevent those choices being mistaken for working
compiler support. Missing host support is not silently filled in.

Name an operation once and write its typed constraints directly underneath it.
For unchanged lane crossings, `copy: {dtype: int8, layout: row_major_contiguous}`
states both endpoint types/layouts and bit-preserving semantics once. The generator
expands these into the full consumer contract in artifacts; no target-specific
defaults or review claims are added. See the [authoring guide](../../../docs/guides/phase0_specification.md).

Merlin's v1 reader also accepts old inline `capability_contract` documents, but
new minimal specs require a separately selected same-target backend contract.
Loading the YAML never discovers one or imports its runtime. A selected provider's
broader legacy operation list cannot override the spec's typed SW admission screen.

[`descriptor.yaml`](../target/descriptor.yaml) owns experiment resources;
[`evidence_concepts.yaml`](../target/evidence_concepts.yaml) names evidence concepts.
Runtime backends, reference-program helpers and calibration live in the selected
OOT support package, not these example inputs. Select it and the extraction tools
explicitly before the commands below:

```sh
export MERLIN_MLC_DIR=/path/to/ModelIR
```

No support provider is selected automatically for this target. Fresh experiments
derive support from admitted public sources and qualify its actual execution;
the handwritten compiler, calibration and runtime are private final references.

The selected integer RTL has 8-bit signed operands, a 20-bit MAC result and
32-bit accumulator storage. Those are different quantities. A mathematical
int32 golden needs proof that internal partial sums cannot overflow, or an
independent width-aware model. The software requantization rounding parameter
also must not be confused with the RTL's integer rounding shift.
Compute datapath facts also preserve `declared_carrier`: exact FIRRTL port types,
directions, signedness and element widths, including row-vector declarations.
This keeps an explicit `SInt<20>` observation inspectable even when no tensor
quantization format exists for twenty bits. Carrier signedness alone does not
establish a quantizer scale/zero-point or whole-operation overflow policy.

Produce a new source bundle from the exact selected FIRRTL:

```sh
python -m merlin.targetgen.rtl.source_selection \
  --target gemmini --generator gemmini --config GemminiRocketConfig \
  --core-root Gemmini --firrtl /selected/elaboration/design.fir \
  --hierarchy /selected/elaboration/top_module_hierarchy.json \
  --firtool /selected/circt/bin/firtool --output /generated/gemmini/source-1 \
  --drop-annotation-class 'chisel3.experimental.EnumAnnotations$EnumComponentAnnotation' \
  --drop-annotation-class 'chisel3.experimental.EnumAnnotations$EnumDefAnnotation' \
  --drop-annotation-class 'chisel3.experimental.EnumAnnotations$EnumVecAnnotation'
python -m merlin.targetgen.rtl.circt_introspect --target gemmini \
  --source-bundle /generated/gemmini/source-1/source-selection.json \
  --out /generated/gemmini/source-1/facts.json
merlin-target-tools rtl-source-audit \
  --source-bundle /generated/gemmini/source-1/source-selection.json \
  --facts /generated/gemmini/source-1/facts.json \
  --hardware-spec examples/gemmini/platform/hardware.yaml \
  --output /generated/gemmini/source-1/validation.json
```

The producer preserves original FIRRTL, records a separate generation input
when explicitly removing non-functional Chisel enum metadata, runs the selected
`firtool`, extracts an exact HW module closure, and derives hierarchy from actual
FIRRTL instances. It never substitutes watchdog or datapath behavior. Old
hierarchy files may precede specialization/deduplication; discrepancies remain
diagnostics, not guessed aliases. `validation.json` compares direct source slices
with extracted values and records the storage-versus-compute precision gap.

Source consistency verifies provenance, not compiler or numerical conformance.
Operation, quantization, transfer and host support stay unreviewed until their
own execution receipts exist. Use fresh facts for a fresh frozen Phase 0 run;
preserve old artifacts unchanged. See the [Phase 0 walkthrough](../phase0/README.md).

### Independently characterize the arithmetic cell

The example provides deterministic vectors for the exact selected `MacUnit` and
an independent SpecIR signed multiply-add reference. All 65,536 input pairs are
checked at zero addend, then 11,520 accumulator-boundary cases check truncation,
overflow and signed interpretation. This checks the cell's 20-bit wrapping result;
it does not qualify a complete mesh, DMA transaction, readout or compiler.

```sh
python examples/gemmini/verification/characterize_cell.py \
  --hw-source /generated/gemmini/source-1/core.hw.mlir \
  --specir-root /selected/SpecIR \
  --circt-opt /selected/circt/bin/circt-opt \
  --verilator /selected/bin/verilator \
  --output /generated/gemmini/cell-1
```

Inspect `characterization.json`, `cases.json`, `vectors.txt`, `cell.hw.mlir`,
`verilog/`, `harness.cpp`, and `execute.stdout.log`. The harness is generated by
the shared producer, never hand-authored in the output folder. The receipt binds
the original HW source, independent reference sources, vectors, tool binaries,
emitted files and executed native binary. `verify_characterization` from
`merlin_experiments.phase0.cell_probe` checks a saved or moved artifact's hashes
and zero failed outputs without reopening live RTL. Selecting the receipt for a
frozen run still requires separately pinning its bytes; it is not a compiler
certificate or proof that the finite vector set covers every possible addend.

Mathematical integer capsule goldens are valid only in a bounded domain. The
shared `integer_partial_sum_bound` requires actual operand/initial-addend values
and the reduction extent. For unrestricted signed int8 inputs, K=31 passes its
conservative i20 bound and K=32 does not. Wider accumulator storage or a small
final result does not prove that intermediate partial sums avoided wrapping.

The host manifest now lists only typed matmul/batch-matmul candidates found in
its pinned schedule. No standalone normalization or activation is claimed.
Exact `ops` selectors do not widen to a shared semantic family: a declaration
for `linalg.matmul` does not cover a captured `linalg.generic` contraction.
Inspect `coverage/operation-accounting.json` against the selected schedule
before reviewing the host declaration. A transform-interpreter success alone
does not establish that a selector matched. Even a native whole-program result
matching its saved golden is finite host-execution evidence, not per-operation
RVV support, accelerator execution, or a reviewed host numerical contract.
Typed load/readout candidates describe bit-preserving crossings; they remain
unreviewed and never imply that FP32-to-int8 quantization, dispatch, or DMA
execution was already implemented. The selected hardware readout recipe supplies
symmetric, per-tensor W8A8 to the generated quantization contract; the authored
spec limits that format to contractions. Per-channel software epilogues need a
separate route and qualification rather than being inferred from a format name.
The spec explicitly declares an FP32 scale encoding: the selected Gemmini
configuration's `gemmini_params.h` defines `acc_scale_t` as `float`, while CIRCT
alone sees only a 32-bit command carrier. This software-visible fact makes a
scoped diagnostic capture recipe derivable; it does not review the format or
qualify quantization on a model.

For a status-only check of configured tools and prebuilt simulator files, run
`python examples/gemmini/verification/probe_oracles.py`. Set `MERLIN_CHIPYARD` to your
checkout; optional overrides are `MERLIN_GEMMINI_SPIKE`,
`MERLIN_GEMMINI_VERILATOR`, `MERLIN_RISCV_GCC` and `MERLIN_GEMMINI_HARNESS_DIR`.
Status mode runs nothing and is not a provenance or executable-validity check.
`--run spike` or `--run verilator` explicitly executes the selected prebuilt
test on provisioned resources; its exit status does not certify a generated compiler.
The generated harness ABI can separately be checked with
`MERLIN_RTL_FACTS=/generated/gemmini/source-1/facts.json python
build_tools/scripts/check_kernel_abi_arg_order.py --verbose`.

### Observe one selected-core DMA matmul numerically

[probe_native_dma.py](probe_native_dma.py) uses ModeLIR's existing Gemmini
DMA driver with an ARC library built from the selected core. It compares
every byte of one 16×16 INT8 matmul with an independent NumPy golden, then
freezes RTL, intermediate HW/Arc/LLVM files, the native library, tools,
driver source, inputs and outputs. Supply the exact artifact paths shown by
--help; the output directory is produced by the example, not hand-edited.

```sh
python -I /generated/gemmini/native-1/runner.py \
  --replay-bundle /generated/gemmini/native-1
```

That command verifies all frozen bytes and re-executes independently of the
original source checkout. The tested mvout_acc command saturates its INT32
accumulator to INT8 DRAM output; it does not qualify bit-preserving INT32
readout, the full Rocket SoC, or all shapes and values.

### Check the direct residual-sum and ReLU path

`probe_residual_readout.py` builds one target-local bare-metal program from an
iteration workload's selected `group_capsules.json`. It uses two ordinary
scaled loads into the accumulator, then a narrowing ReLU/scale readout; it
does not call the vendor `LOOP_WS` residual helper. A disassembly screen
rejects any other Gemmini instruction class. Spike and the selected
GemminiRocketConfig Verilator run the same ELF, and every output is compared
with an independent single-rounding reference and a separately rounded
device-path model. Gain above one is moved from both saturating loads to the
readout scale. The Phase 0 review status is not changed by this probe.

```sh
export PYTHONPATH=src:packages/merlin-experiments/src
export MERLIN_TARGET_PATH=${MERLIN_INDEPENDENT_TARGET_SUPPORT:?independent runtime support required}
export MERLIN_CHIPYARD=/selected/chipyard
python examples/gemmini/verification/probe_residual_readout.py \
  --phase0 /generated/gemmini/run/phase0 \
  --groups /generated/gemmini/group_capsules.json \
  --chipyard "$MERLIN_CHIPYARD" --campaign full \
  --output out/artifacts/probes/gemmini-direct-residual-1
```

`full` covers all 65,536 ordered signed-i8 operand pairs at the selected
group's two scales. It embeds the independently computed i8 expectation in
the ELF and compares every output on Rocket, emitting only a mismatch count
so simulator UART traffic stays bounded. `smoke` prints a 256-pair
boundary/cancellation grid for direct transcript inspection. The group report
must use the same target contract and CIRCT facts as the frozen Phase 0 run;
the selected OOT provider contract can route the same graph differently.
For a bounded RTL run, `--campaign boundary_rows` or `boundary_cols` checks
4,096 ordered pairs: all 256 values on one operand against the 16 recorded
boundary/cancellation values on the other. These are not exhaustive; the
receipt records the exact operand sets and pair count.
The generated `receipt.json` binds the input facts, readout ABI, group
report, ELF, disassembly, tools, and both transcripts. A passing result
checks this arithmetic path for one pair of scales, not a Phase 1 compiler,
all scales or shapes, or simulator build provenance. Bind the Verilator
binary to its separate selected-source build attestation for that last claim.
`--rebind-receipt` can produce a separate equivalence record for an earlier
passed numerical run when the frozen model, group arithmetic, generated C,
expectations, ELF and transcripts still match. It does not rerun either
simulator or change an unreviewed SW-spec admission decision.

### Check generated contraction kernels on native simulators

[probe_native_kernel.py](probe_native_kernel.py) checks two generated Phase 0
contraction capsules against an independent scalar integer matmul, the capsules'
goldens, native Gemmini Spike, and native Gemmini Verilator. First produce a
corpus with `isa/SY_contraction_i8_aligned` and
`isa/SY_contraction_i8_partial`, and a selected source bundle whose hashes are
bound by that corpus's `_evidence/evidence-manifest.json`. Select the OOT
Gemmini support and Chipyard toolchain explicitly:

```sh
export PYTHONPATH=src:packages/merlin-experiments/src
export MERLIN_TARGET_PATH=${MERLIN_INDEPENDENT_TARGET_SUPPORT:?independent runtime support required}
export MERLIN_CHIPYARD=/selected/chipyard
python examples/gemmini/verification/probe_native_kernel.py \
  --corpus /generated/gemmini/corpus-1 \
  --source-evidence /generated/gemmini/source-1 \
  --output-root out/artifacts/probes/gemmini-kernel-1
```

When refreshed, independently validated facts are stored outside the selected
source bundle, pass `--facts-evidence /generated/gemmini/facts-1`. The probe
checks that the facts validation binds the selected source hash and that the
Phase 0 manifest binds both facts and source/core HW bytes. Without the option,
the existing flat source bundle layout remains supported.

The output root must resolve beneath this checkout's ignored `out/` tree and
must not overlap either input bundle. `receipt.json` binds the selected source,
corpus manifest, probe, support files, tool binaries, and per-case program,
input and output hashes. Each case directory contains `command_buffer.json`,
`inputs.json`, `outputs.json`, `spike/main.c`,
`spike/merlin_gemmini_c0.elf`, `spike_console.log`,
`verilator_console.log`, and `spike_disassembly.txt`. For an existing output
root, add `--audit-existing` to re-execute the saved ELF on Spike; a saved
Verilator console is reused only when its hash and the ELF hash match the prior
receipt. Without that flag, both engines execute the newly compiled program.
For a frozen experiment run, point `--corpus` at its `phase0/capsules` directory;
the probe binds the adjacent `phase0/evidence-manifest.json`. Standalone corpora
with `capsules/_evidence/evidence-manifest.json` remain readable.

The current check covers exactly two fixed int8-input/int32-output shapes:
16×32 by 32×16 (256 outputs) and 16×31 by 31×15 (240 outputs). Their generated
stimuli use operand values 0–3, zero initial accumulator, and no epilogue. The
probe requires zero tile padding, compares all 496 outputs exactly, and checks
that the same ELF contains custom-3 RoCC instructions and runs on both engines.
Each simulator execution has a 180-second wall timeout. The selected Verilator
binary rejects the tested cycle-cap flags, so there is no independent simulated
cycle limit; `receipt.json` records those rejections. This is finite numerical
evidence, not a proof for every input, model, or operator. The generated
capsules' software admission remains `unknown`, and the receipt does not prove
that the selected Verilator binary was built from the exact selected RTL source.

For a finite signed boundary probe, [probe_signed_i8.py](probe_signed_i8.py)
reuses the generated `SY_contraction_i8_partial` interface and injects explicit
`-128`/`127` operands through the selected OOT LLVM emitter and runner harness.
It requires the separately verified MacUnit characterization receipt. The
31-deep reduction has a conservative maximum partial sum of 507,904, below the
signed i20 positive limit of 524,287. It compares all 240 full i32 outputs and
both directions of scaled i8 saturation against independent scalar results.
The emitted ELF must contain custom-3 RoCC instructions, and a bounded Spike
instruction trace must show those exact PCs executing. The probe keeps only the
matching trace lines, not the full trace. Run it with the same source, facts, corpus,
support package and toolchain selections used for the Phase 0 diagnostic:

```sh
export PYTHONPATH=src:packages/merlin-experiments/src
export MERLIN_TARGET_PATH=${MERLIN_INDEPENDENT_TARGET_SUPPORT:?independent runtime support required}
export MERLIN_CHIPYARD=/selected/chipyard
export MERLIN_RTL_FACTS=/selected/facts/facts.json
export MERLIN_M2M_VENV=/selected/model2MLIR/.venv
export MERLIN_CLANG=/selected/clang
python examples/gemmini/verification/probe_signed_i8.py \
  --corpus /selected/phase0/capsules \
  --source /selected/source-bundle \
  --facts /selected/facts \
  --cell-receipt /selected/cell/characterization.json \
  --output out/artifacts/probes/gemmini-signed-i8-1
```

The default receipt is explicitly `spike_passed_rtl_not_run`; Spike is a
functional model. Add `--rtl` to execute each exact ELF on the selected
prebuilt Verilator, bounded by `--timeout` (180 seconds maximum). A Verilator
timeout writes an `rtl_timeout_incomplete` receipt with the exact ELF hash and
bounded stdout/stderr tails, then exits nonzero; it is not an RTL numerical
verdict. This finite diagnostic does not
review the software spec or establish model-level integerization, physical
layout, tails or aliasing.

To check that missing build link, run
[attest_native_simulator.py](attest_native_simulator.py) against a fresh ignored
output directory. It re-lowers the selected FIRRTL with the explicitly selected
Chipyard firtool, checks the Gemmini core RTL files byte-for-byte, regenerates
and compares the Verilator C++ model, then rebuilds and compares the entire
simulator executable to the one hashed in the kernel receipt:

```sh
python examples/gemmini/verification/attest_native_simulator.py \
  --source-evidence /generated/gemmini/source-1 \
  --chipyard /selected/chipyard \
  --firtool /selected/chipyard/.conda-env/riscv-tools/bin/firtool \
  --verilator /selected/chipyard/.conda-env/bin/verilator \
  --kernel-receipt out/artifacts/probes/gemmini-kernel-1/receipt.json \
  --output-root out/artifacts/probes/gemmini-verilator-build-1
```

`receipt.json` in that output records input and tool hashes, the rebuild steps,
core RTL and C++ file counts, and the exact binary equality. The fresh RTL,
Verilated model, and rebuilt simulator remain alongside it. This is
reproducible build provenance for the selected core inside one full SoC binary,
not a proof of RTL behavior. Spike's `libgemmini.so` is a separately hashed
functional model: neither this build check nor agreement with its output makes
Spike RTL-derived. A different Chipyard build or toolchain requires a new
attestation. A later kernel receipt can reuse it only through the byte binding
below.
On a read-only Chipyard tree, the attester copies the selected FIRRTL and projects
only the two hierarchy-output annotation filenames into its ignored artifact;
it records both annotation hashes and the exact path changes, then still requires
byte-for-byte equality of generated core RTL, Verilator C++, and the executable.

If a later kernel receipt uses the **same selected source and simulator bytes**,
bind it to the existing build attestation without rebuilding Chipyard:

```sh
python examples/gemmini/verification/bind_native_simulator.py \
  --attestation out/artifacts/probes/gemmini-verilator-build-1/receipt.json \
  --attested-kernel-receipt out/artifacts/probes/gemmini-kernel-1/receipt.json \
  --current-kernel-receipt out/artifacts/probes/gemmini-kernel-2/receipt.json \
  --source-evidence /generated/gemmini/source-1 \
  --simulator /selected/chipyard/sims/verilator/simulator-chipyard.harness-GemminiRocketConfig \
  --output-root out/artifacts/probes/gemmini-verilator-binding-2
```

This checks the old receipt named by the attestation, both execution receipts
and their saved same-ELF/console files, selected source, current simulator,
retained rebuilt binary, and every retained selected-core RTL file against
their recorded SHA-256 values. A new ELF is
allowed: the binding links the same RTL binary, not the programs' numerical
results. It writes a separate generated receipt without editing existing
evidence. Source-to-FIRRTL generation and the attester's build procedure are
not reverified by this inexpensive step.

### Diagnose one headline-derived kernel window

`probe_headline_kernel.py` takes a captured `frontend-trace.json` and its exact
`model.mlir`, one prepared-graph source node ID, selected RTL facts, and a
capability contract. It refuses a stale trace or ambiguous operation. The
generated capsule records the original operation ordinal, operand types,
model/trace hashes and full matrix geometry, then derives a small integer
window from the RTL mesh edge. It accepts a traced `linalg.matmul`, or a
traced `linalg.generic` only when the exact captured MLIR has the signed
i8×i8→i32 matmul maps, iterators, provenance and reduction body. Other generic
loops are not treated as contractions. For example, TinyLlama's `k_proj` body
`g:prepared:root:n280` in the 8-token prefill capture has source geometry
8×2048×256; the selected 16-wide mesh gives an 8×32×16 diagnostic window.
No dimensions are copied into the generator.

```sh
python examples/gemmini/verification/probe_headline_kernel.py \
  --capture "$HEADLINE_ROOT/tinyllama-prefill" \
  --source-node-id g:prepared:root:n280 \
  --rtl-facts "$RTL_ROOT/facts.json" \
  --support-contract "$MERLIN_TARGET_PATH/contracts/target_contract.yaml" \
  --output-root out/artifacts/probes/headline-tiny-kproj-1
```

The default generates `capsule.yaml`, `capsule.interface.mlir` and
`generation.json` without running a simulator. Add `--native` for independent
scalar-versus-Gemmini Spike output checks; add `--rtl` to execute the same ELF
on Verilator. Native mode requires the explicitly selected OOT support contract,
`MERLIN_TARGET_PATH` pointing to that support package, and `MERLIN_EXT_CHIPYARD`
pointing to the selected Chipyard toolchain/simulator build. The example's
Phase 0 contract supplies the authored corpus issue order; the probe checks
that its shared compute-unit and encoding declarations agree with the OOT
provider contract and records both contract hashes.
Before native execution, the probe checks that the source projection, capsule
and interface still match `generation.json`. The numerical receipt records the
SHA-256 of all three generated files, so a result cannot be silently reassigned
to a different generated diagnostic.
Add `--acc-scale 0.5` to generate an i8 readout capsule with a non-identity,
exactly representable FP32 per-tensor scale. This selects the OOT LLVM-MLIR
emitter (the legacy C conformance driver only accepts raw i32 output); set
`MERLIN_RTL_FACTS` to the same selected facts, `MERLIN_M2M_VENV` to the selected
MLIR toolchain environment, and `MERLIN_CLANG` to the selected compiler. The
probe checks the emitted harness's literal inputs against an independent
scalar FP32-scale, round-to-nearest-even, saturating int8 reference on Spike
and, with `--rtl`, the same ELF on Verilator. The receipt records the emitter,
lowered MLIR, harness and ELF digests. This remains a synthetic window, not
quantized-model equivalence; its numerical receipt alone is not a qualified
simulator source pin.
For a newly generated `merlin.gemmini-headline-window-numerical.v1` receipt,
use the same `bind_native_simulator.py` command above with its
`numerical_receipt.json` as `--current-kernel-receipt` and add
`--facts-evidence /generated/gemmini/verified-facts`. The binder checks that
the generated capsule/interface, selected facts and validation, saved ELF,
Spike/Verilator consoles, selected source and simulator executable all retain
the exact recorded bytes. Older headline receipts without a simulator hash
cannot be retrospectively promoted. A successful binding still says nothing
about numerical equivalence to TinyLlama, SmolVLA or ResNet50.
Each simulator has a 180-second wall timeout. The source capture supplies
geometry only: FP32/BF16 model captures do not establish an int8 model path.
An actual quantized capture and integerization evidence are required before
claiming the model invokes a corresponding int8 contraction, though they are
not required to check this synthetic Gemmini kernel's arithmetic.

For an integerized capture, inspect `frontend-trace.json`'s
`mlir.operations` for an `linalg.generic` with a prepared-graph source ID and
`tensor<...xi8>` operands. Pass that ID to `--source-node-id`; the probe checks
the parsed operation, not just those trace labels. The following are example
source selections, not hard-coded corpus entries:

| Capture | Source node | Observed contraction | 16-wide diagnostic window |
| --- | --- | --- | --- |
| TinyLlama 8-token prefill (`k_proj`, FP32 capture) | `g:prepared:root:n280` | 8×2048×256 | 8×32×16 |
| TinyLlama 8-token prefill (`k_proj`, int8 capture) | `g:prepared:root:n283` | 8×2048×256 | 8×32×16 |
| SmolVLA int8 denoise | `g:prepared:root:n321` | 50×32×720 | 16×32×16 |
| ResNet50 W8A8 | `g:prepared:root:n361` | 12544×147×64 | 16×19×16 |

For the int8 TinyLlama row, the prepared `_int_mm` node traces to the original
layer-0 `k_proj` `aten.linear` call. Each run needs a fresh `--output-root`;
its `generation.json` and
`numerical_receipt.json` identify the capture, selected facts and contracts,
lowered MLIR, ELF and saved console bytes. With `--native --rtl`, the receipt
records a same-ELF Spike/Verilator comparison. To tie that comparison to the
selected RTL source and exact simulator executable, run the separate binary
binder with the verified facts and original build attestation. A passing
synthetic window is a representative kernel check, **not** numerical
equivalence of any of these complete models or proof of model-wide coverage.

### Execute one staged Phase 1 candidate

After `merlin-target-tools stage-int-mm-admission --build-dir ... --out ...`
has produced a `candidate-build-receipt.json`,
[`probe_staged_candidate_spike.py`](probe_staged_candidate_spike.py) links the
receipt's exact kernel and descriptor-shim objects with explicitly selected
Gemmini support, then runs one bounded signed-int8 matmul on Spike. It checks
all output elements against an independent scalar result, rejects a wrong
shape through the descriptor, and records which disassembled package
instructions executed. Select the matching stage report, compiler package,
RTL facts, and OOT support directory explicitly:

```sh
python examples/gemmini/verification/probe_staged_candidate_spike.py \
  --target gemmini --stage-report /generated/gemmini/stage-1/report.json \
  --build-dir /generated/gemmini/stage-1/build \
  --package /selected/gemmini/compiler-package \
  --facts /generated/gemmini/source-1/facts.json \
  --support "$MERLIN_TARGET_PATH" \
  --output out/artifacts/probes/gemmini-staged-1
```

The output directory must be fresh under `out/`. Inspect its `receipt.json`,
`inputs.json`, `expected.json`, `observed.json`, `harness.c`, ELF,
`disassembly.txt`, and `spike_package_instructions.trace`. The receipt remains
`incomplete_support_provenance` while support-source identity or the
compiler-package-to-support binding is unproven. A finite Spike match does not
review the software spec, prove RTL execution, or establish whole-model offload.

### Reproduce the accumulator load-order race

[acc_race/](acc_race/README.md) holds bare-metal reproducers for two loads into
the same accumulator rows (overwrite, then accumulate) that complete out of
issue order. Each wrong element is classified by which operand survived, and a
fenced harness copy serves as the control. Spike and in-order RTL simulators
never show the race; the board and a reordering memory model do. The measured
evidence and its consequences for verification are in the
[design note](../../../docs/design/accumulator_load_order.md).

### Measure open-coded nest placement levers

[probe_load_placement.py](probe_load_placement.py) builds one binary carrying
the same contraction under several scheduling policies: load states, move-in
placement, readout placement and weight reuse. It runs the binary on a selected
RTL engine with the unit's own counters, and the hardware sequencer serves as
the element-for-element oracle. Geometry is read from the harness headers.
`--harness` and `--cc` select a harness tree and compiler when no backend is
installed.

```sh
python examples/gemmini/verification/probe_load_placement.py \
  --target gemmini --shape 64x512x1024 --bank occupancy \
  --emulator /selected/gsim/emulator --out out/artifacts/probes/gemmini-load-placement-1
```

Compare arms only within one binary: an arm's absolute count moves with the
other arms in the same binary. The findings are in the
[design note](../../../docs/design/open_coded_nest_costs.md).

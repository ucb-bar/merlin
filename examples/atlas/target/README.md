# Target inputs and source qualification

`software-spec.yaml` is authored software-visible semantics, not generated tests.
`hardware.yaml` selects deterministic source production and direct audit questions.
`host-capabilities.yaml` is separately bound to immutable host compiler bytes.
Neither an ISA name nor the host package's FP32 strategy certifies FP8/BF16
operation support.

The selected integrated source is `EE290SimConfig` in `bringup-chipyard` at
`426a862f97f98938660772bd8a5d8f41316d15c3`. Its recorded
`generators/atlas-npu` gitlink is
`0079c0541111197741a231c002e3843fa6f545b2`. The target descriptor
checks both revisions when resolving that source; set
`MERLIN_EXT_BRINGUP_CHIPYARD` to the pinned checkout. Earlier standalone
AtlasCore observations retain their own execution tier and are not an
integrated EE290SimConfig qualification. A new elaboration, source audit,
and integrated execution are still required.

The separate [hand-authored Atlas MLIR dialect](https://github.com/ucb-bar/atlas-mlir/tree/handwritten-implementation)
shows typed machine operations and an LLVM/assembly handoff for fixed diagnostic
tiles. Its [dialect reference](https://github.com/ucb-bar/atlas-mlir/blob/5485aa0aaeca222e1c158507460db06a621eeba4/docs/dialect-reference.md)
documents the operation and pass contracts. This example's
[`software-spec.yaml`](software-spec.yaml) remains the authored Merlin software
declaration; the OOT reference does not replace it or qualify all its admitted
semantics.

[`contracts/target_contract.yaml`](contracts/target_contract.yaml) is the loadable
prototype capability declaration. Its matching
[`contracts/residual.yaml`](contracts/residual.yaml) feeds fact-backed derivation.
The current registry reads the first path for reference discovery, while the
manifest deriver reads the second path from the resolved target base. Until
those APIs share one authored input, their parsed declarations are kept equal
apart from `facts_source: rtl` and checked by a targetgen regression test.
The declaration carries the curated MXU FP8/BF16 candidate and runner identity;
it has no copied mesh size, memory capacity, opcode table, operand block-scaling
claim or executable backend. Registry loading and schema validation alone do not
qualify those capabilities. Keep the selected OOT provider authoritative for
execution, and supply one coherent selected source bundle and fresh facts for
derivation and audit.

The minimal software spec describes selected numerical/domain rules, operation
placement/signature constraints, unresolved quantization parameters and explicit
transfer candidates. It intentionally does not repeat backend configuration,
opcode tables, memory sizes, calibration history, RTL-audit hashes or test counts.
The selected OOT provider owns protocol/runtime policy; deterministic tools produce
the source, coverage and numerical qualification records alongside generated
artifacts. Choosing FP8/BF16 is not proof of block scaling or whole-network support.

Operation constraints sit directly under their names. A bit-preserving crossing
uses `copy: {dtype: bf16, layout: row_major_contiguous}` rather than repeating
source/result dtype and layout. Generated artifacts retain the expanded consumer
contract and exact input bytes. Unknown quantization fields are not filled by
this shorthand. See the [authoring guide](../../../docs/guides/phase0_specification.md).

Legacy v1 inline `capability_contract` files still load. New minimal files require
an explicitly selected same-target backend contract; there is no filesystem lookup
or guessed protocol inside the software-spec reader. Typed SW admission is screened
independently from that provider's compatibility declarations.

[`descriptor.yaml`](descriptor.yaml) declares policy, workloads and external
resources; [`tooling.env.example`](tooling.env.example) lists local tool locations
without loading them automatically. Runtime implementation and named-program
oracles belong to the OOT support package. Select that provider and ModelIR
explicitly before extraction:

```sh
export MERLIN_TARGET_PATH=/path/to/atlas-mlir/merlin-support
export MERLIN_MLC_DIR=/path/to/ModelIR
```

The provider's `runner.program_emitter` owns model-specific encoding policy;
Merlin has no bundled assembler patch or same-name fallback. See the selected
companion revision in
[`target_support.json`](../../../build_tools/upstreams/target_support.json).

The selected matrix cell contains E4M3 operands, an exact 13-bit custom product
carrier, and BF16 add-and-round output. The product carrier is not another tensor
quantization format. Operand exponent-zero handling, finite overflow clamping,
special values, addend subnormals and reduction order need independent numerical
qualification; the declaration deliberately keeps operation admission unreviewed.
`compute_datapaths[].declared_carrier` separately records exact FIRRTL port types
and unsigned carrier widths. A `UInt<8>` carrier is not itself proof of E4M3, nor
does it establish an unsigned integer arithmetic operation: the selected arithmetic
structure and independent numerical characterization supply separate evidence.
The selected HW-MLIR also exposes 8-bit `scaleE8M0` command ports, matching the
source's E8M0-named pack/pop controls. That carrier observation does not establish
the block scope, exponent transformation, or a TorchAO scale representation.
The direct source audit now checks all three `ScalarCore` scale command port
widths and the `ScalingFactorRegFile` port geometry: an `i5` write index, `i8`
write data, and exactly 32 contiguous `i8` outputs. A changed/missing port
fails this structural check. These ports do not prove register behavior, wiring
through every command path, or any numerical scale interpretation.
In the selected Atlas source, `ScalarCore` reads that register for
`MXU_POP_FP8`; both MXU sequencers use the byte when packing a BF16
accumulator row to FP8. `VFP8PACK` has its own BF16-to-FP8 pack path. Matrix
operand/weight push and compute do not establish an E8M0 scale applied to
incoming FP8 values. The 32 scale registers are software-selectable entries,
not evidence of a 32-element quantization block. The OOT backend's current
`scaling: block_e8m0` declaration and `must_supply_e8m0_block_scales`
obligation are intent to review, not an RTL-qualified operand format.
Consequently `software-spec.yaml` leaves model-operand scale encoding and
block size unresolved. To resolve them, review the intended tensor-to-FP8
conversion and any scale compensation at each BF16/FP8 boundary, bind an
executable backend route, and compare non-unit-scale cases with the selected
RTL. The generated Phase 0 quantization contract may list a readout-derived
candidate, but until those checks pass it emits no model capture recipe; do
not promote the diagnostic FP8 candidate into a realizable model format.

Produce evidence from one selected elaboration, not a mixture of standalone
spec-generated hardware and Chipyard memory/hierarchy sources:

```sh
python -m merlin.targetgen.rtl.source_selection \
  --target atlas --generator atlas-npu --config EE290SimConfig \
  --core-root AtlasTile --firrtl /selected/elaboration/design.fir \
  --hierarchy /selected/elaboration/top_module_hierarchy.json \
  --firtool /selected/circt/bin/firtool --output /generated/atlas/source-1
python -m merlin.targetgen.rtl.circt_introspect --target atlas \
  --source-bundle /generated/atlas/source-1/source-selection.json \
  --out /generated/atlas/source-1/facts.json
merlin-target-tools rtl-source-audit \
  --source-bundle /generated/atlas/source-1/source-selection.json \
  --facts /generated/atlas/source-1/facts.json \
  --hardware-spec examples/atlas/target/hardware.yaml \
  --output /generated/atlas/source-1/validation.json
```

The producer records exact source/tool digests, emits SoC and core HW dialect
artifacts, and derives the hierarchy from actual FIRRTL instances. It retains
the supplied old hierarchy as a diagnostic comparison without inventing module
aliases. Per-unit accumulation-buffer copies are separate address spaces, not
extra banks of one larger store. `compute_datapaths` and `storage_datapaths`
preserve arithmetic formats and physical storage widths separately.

Inspect `source-selection.json`, `firtool.log`, `hierarchy.json`, `soc.hw.mlir`,
`core.hw.mlir`, `facts.json`, and `validation.json`. A verified source-consistency
record is not numerical, transfer, simulator or compiler certification. Regenerate
a fresh frozen Phase 0 run with these facts; do not rewrite old receipts.
See the [Phase 0 walkthrough](../phase0/README.md).

### Independently characterize the arithmetic cell

The example drives the exact selected `E4M3FMA` through CIRCT-generated
SystemVerilog and Verilator, comparing raw output bits against SpecIR's independent
exact-rational product/add with one BF16 nearest-even rounding. The deterministic
domain comprises all 51,076 pairs of signed-zero or finite-normal FP8 operands
with exponents 1–14, plus 8,192 seeded finite-normal BF16 addends. FP8 exponent
15, BF16 addend subnormals, NaNs and infinities are deliberately outside this
qualification. `--domain multiply` selects only the zero-addend census.

```sh
python examples/atlas/target/characterize_cell.py \
  --hw-source /generated/atlas/source-1/core.hw.mlir \
  --specir-root /selected/SpecIR \
  --circt-opt /selected/circt/bin/circt-opt \
  --verilator /selected/bin/verilator \
  --output /generated/atlas/cell-1 --domain accumulate
```

Inspect the generated `characterization.json`, raw `cases.json`/`vectors.txt`,
`cell.hw.mlir`, `verilog/`, `harness.cpp`, executed binary and stage logs. The
shared producer generates every harness and records byte identities; nothing is
manually written in an output directory. `verify_characterization` from
`merlin_experiments.phase0.cell_probe` detects saved-artifact tampering and failed
native execution. Its finite cell scope does not establish mesh reduction order,
block scaling, VPU semantics, host conversion, dispatch or full-model lowering.

The separate host manifest lists only FP32 matmul/batch-matmul candidates from
the pinned RVV schedule. BF16 readout needs an explicit host conversion before an
FP32 host operation; the bit-preserving transfer candidates do not silently add
that conversion. The FP8 quantization declaration remains unresolved where block
size/scale behavior is not yet independently qualified.

### Observe one selected-core program numerically

[probe_native_program.py](probe_native_program.py) runs an explicit 32×32
finite E4M3 matrix program through an ARC library built from the selected
AtlasCore subtree. It re-extracts that subtree from the selected AtlasTile
HW file, independently derives the BF16 golden, checks all 1,024 output
values and the real TileLink DMA/halt observations, then freezes the source,
build intermediates, tools, OOT ModeLIR driver, program and results. Use
--help to supply exact selected source and ARC artifact paths; outputs go
under a new generated artifact root. The driver needs an explicit,
source-checked scalar/halt_now manifest path when the public io_halted port
is optimized away. It never guesses that the reset-true scalar/halted
register means completion.

```sh
python -I /generated/atlas/native-1/runner.py \
  --replay-bundle /generated/atlas/native-1
```

Replay verifies frozen bytes and re-executes without reopening the original
RTL checkout. This finite AtlasCore result is not a GSIM, whole-SoC,
full-precision-domain, or whole-network certificate.

For a broader numerical check of that same frozen 32×32 instruction stream,
[`probe_native_numerics.py`](probe_native_numerics.py) preloads three new FP8
matrix pairs and compares all 3,072 BF16 output elements bit-for-bit to
SpecIR's exact-rational, per-step RNE reference. The cases cover both signs,
all E4M3 mantissas, finite-normal exponents 1–14, dense nonzero accumulation,
and asserted positive/negative BF16 ties. It records the selected ARC binary,
program, RTL, ModeLIR driver, SpecIR and input/output SHA-256 values in a fresh
ignored `result.json`.

The probe models the observed physical program ABI: raw weight rows are output
columns (`C[i,j] = sum_k A[i,k] * W[j,k]`), and BF16 output is the two 32×16
column halves. The diagonal-only case cannot distinguish this from a logical
untransposed matrix multiply or from a four-quadrant output packing.

```sh
PYTHONPATH=src python examples/atlas/target/probe_native_numerics.py \
  --bundle /generated/atlas/native-1 \
  --specir-root /selected/SpecIR \
  --output out/artifacts/probes/atlas-native-numerics/new-run
```

This reuses a frozen native build; it does not recompile it. The fixed program
cannot qualify tail or batched shapes, a compiler-selected program, or an
application model. A mismatch remains recorded in the output and returns 2.

`python examples/atlas/target/setup.py` only reports machine prerequisites by
default. `--write-env`, `--sync-npu-model` and `--materialize-target-package`
explicitly request local changes; target derivation does not certify a simulator
or install the OOT provider. Keep local paths and credentials in the descriptor's
selected ignored `experiment.env`, never in these inputs. Existing environment
variables take precedence, and a missing selected file has no cross-target fallback.
Tooling changes require a fresh frozen run for verified execution.

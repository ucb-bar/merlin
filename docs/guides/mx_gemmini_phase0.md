---
title: MX Gemmini Phase 0 contract
kind: guide
status: draft
owner: core
last_verified: 2026-09-29
related: [phase0_specification, model2mlir, integrations]
code_refs: [examples/mx_gemmini/target/software-spec.yaml, examples/mx_gemmini/phase0/recipe.yaml, build_tools/scripts/synth_capsule_corpus.py, src/merlin/targetgen/software_spec.py]
---

# MX Gemmini Phase 0 contract

## Authority and scope

The [pinned Gemmini RTL configuration](https://github.com/ucb-bar/gemmini/blob/f0167390b56fb315deea90ac1fc3983772e92d82/src/main/scala/gemmini/ConfigsFP.scala)
and its [MX requantizer](https://github.com/ucb-bar/gemmini/blob/f0167390b56fb315deea90ac1fc3983772e92d82/src/main/scala/gemmini/MxRequantizer.scala)
are the authority for supported modes and hardware behavior. The selected
`standaloneMxFPConfig` / `GemminiMxFPStandaloneConfig` uses a 16 by 16
weight-stationary array, 32-element MX scale groups, and BF16 accumulator
storage/readout. Its per-row internal accumulator precision is narrower than
BF16 until the final row. It disables array nonlinear activations,
normalization, and max pooling. The base `defaultMxFPConfig` has 12-bit
operand lanes and no LUT; its standalone controller path requests `lut.get`.
It is therefore not the selected three-format standalone configuration.
An explicit source selection compiled 99 Gemmini and MxGen Scala files from
the pinned revisions in an otherwise clean Chipyard checkout. It used SBT
source overrides because Chipyard's committed Gemmini gitlink points to a
different revision; no Chipyard source was changed. The resulting standalone
FIRRTL has SHA-256
`15fbf36a96d69f1b7cd379338e760188aa14f6d69378826ec1ca372316e8f352`.
Its bytes match the earlier diagnostic FIRRTL after only the source-location
prefix is normalized. CIRCT converted this source-bound FIRRTL to HW IR and
verified all 14,948 hierarchy instances against its declarations. The
Gemmini role probe recovered the 16 by 16 mesh. The generic fact reader
recovered storage widths and the RoCC interface but did not classify the MX
compute datapath; its unsigned 8-bit scratchpad elements must not be read as
an int8 compute mode.
Direct inspection of that FIRRTL finds 256 `MacUnit` modules from `PE.scala`
that instantiate MxGen `MxFpMul`, with separate 12-bit activation and weight
ports and MX format controls. Their recoded accumulator port widths occur
128/32/80/16 times at 40/44/48/68 bits, matching the 8/2/5/1 row precision
schedule. This is structural evidence of an MX compute path in the saved
elaboration; the generic fact reader does not yet project its modes or
numerical behavior.
Selecting a narrow, diagnostic MX capability declaration against that FIRRTL
also recovered the 16 by 16 mesh and declared the source-observed 32-element
scale group. With these inputs, fresh synthesis rejects all six BF16/int8
accelerator cells in the retained conformance requirement. That declaration
remains a local diagnostic until its source closure and MX datapath are reviewed.

The source audit of the pinned Gemmini and MxGen revisions found these selected
configuration facts:

| RTL path | Observed fact |
| --- | --- |
| `ConfigsFP.scala::standaloneMxFPConfig` | Inherits a 16 by 16 mesh, standalone MMIO, two accumulator banks, four scratchpad banks, 256 KiB scratchpad, 64 KiB accumulator, and 12-bit operand lanes. It enables the requantizer and 6-bit LUT entries with 4-bit indices. |
| `Arithmetic.scala::mac_mx` and `MxParameters.scala::mxGemmini` | The selected 12-bit operands infer the three-format PE with symmetric FP4 E2M1 mode 0, FP6 E3M2 mode 4, and FP8 E4M3 direct mode 8. The wider `allMxFPConfig` also supports E2M3, E5M2, and quad E4M3 mode 9; those modes are outside this contract. |
| `ConfigsFP.scala::mesh*PrecisionList` | All 16 product rows use recoded `MxFloat(4,4,4)`; accumulator rows are 8×`(4,5,4)`, 2×`(4,6,4)`, 5×`(4,7,4)`, then 1×`(8,8,4)`. |
| `MxRequantizer.scala` and `QuantLut.scala` | E4M3 single with `lut_en=0` is direct 8-bit; E3M2 is stored as 4-bit indices into 6-bit LUT entries; FP4 is direct 4-bit. Each LUT line holds 16 entries. The compiler must plan codebook contents, uploads, and index packing. |
| `ScaleFactorMem.scala` | Separate activation and weight E8M0 banks use format-dependent 16-wide or 32-wide addressing. A generic row-major scale copy is insufficient. |
| `GemminiISA.scala` and `ExecuteController.scala` | `CONFIG_EX` selects activation, weight, and output MX formats in rs1 bits 10–15. `CONFIG_SCALE_MEM` is funct 26; rs1 bits 33–59 hold tile bounds, 60–61 select scale banks, 62 requests a requantizer counter reset, and 63 selects scale residency. `MX_LOAD_SCALES` and `MX_LOAD_LUT` are functs 27 and 29. |

The first compiler encoding selects `CONFIG_EX` operand codes 0/1/2 for
MXFP8/MXFP6/MXFP4 and output code 3 to disable MX requantization for BF16
readout. FP6 codebook uploads use funct 29: each 16-entry E3M2 line is 12
bytes packed least-significant-bit first; selectors 1 and 0 address the
activation and weight LUTs. The selected LUT has 64 lines per operand; the
DMA reads through the next 8-byte boundary, so its source buffer needs
padding. `QuantLut.scala` selects a line by activation row or weight column
shifted right by the `CONFIG_SCALE_MEM` rs2 granularity field. Its selected
DIM16 read counters are 6 bits, so one configured spatial window reaches at
most 128 rows and columns; the 64-line LUT may impose a smaller limit for a
given shift. An out-of-tree compiler prototype maps FP6 element codes exactly
to caller-supplied codebooks, fails if a code is absent, and packs activation
and weight indices in the RTL's nibble order. Codebook selection and any
approximate projection remain numerical policy decisions. Funct 30 clears
the runtime LUT state before a subsequent direct FP8
or FP4 contraction. Scale uploads use funct 27, an
8-byte-aligned source address, and a byte count divisible by 8. Direct FP8
uses A[M][K] and B[K][N] bytes. Direct FP4 and LUT-indexed FP6 use
A[M/2][K] and B[K][N/2] bytes: the low nibble holds the even activation
row or weight column, and the high nibble holds the odd one. These are
source-buffer layouts; scratchpad allocation and DMA order remain compiler
decisions. The selected
DIM16 scale memory advances one logical row per two K tiles; each logical
row holds 16 scales in direct FP8 and 32 in FP6/FP4, with separate activation
and weight banks. One active buffer holds 4 KiB per operand; larger contractions
need tiled scale uploads and bank scheduling. The software spec records the
physical bank address formulas. A layout prototype converts logical
activation scales [M][K/32] and weight scales [N][K/32] or [K/32][N] into
K-block-major upload rows for each planned wave. Its payload builder checks
operand and scale shapes together and slices both at the same K boundaries.
The prototype declares an explicitly selectable Merlin support provider with
no executable backend or matrix lowering. Selecting it exposes unreviewed
metadata; it does not enable compilation or qualify hardware behavior.
Three source-bound 32×32×32 diagnostics used this out-of-tree packer's A/B
codes and E8M0 scale bytes with distinct values in row and column halves.
The FP6 case also used the packer's exact code-to-index mapping and packed
16-line activation and weight LUTs. Hand-written C command schedules ran
those bytes on the selected RTL simulator. Each format matched all four BF16
output quadrants with zero mismatches and exited zero. These validate one
packed payload path per format. A bounded out-of-tree emitter then generated
the configuration, LUT and scale uploads, operand transfers, loop, and BF16
readout from coherent payloads at 32×32×32 and 64×64×64. All six emitted
programs matched their four BF16 quadrants exactly on that simulator. The 64³
FP6 program uploaded 32 LUT lines per operand. The emitter refuses other
single-window shapes. A separate 32×32×64 diagnostic split K into two
32-element waves, with different activation values in each wave. Its emitted
programs reloaded both scale banks and operand tiles, retained the FP6 LUT,
and set `ex_accumulate` on the second loop. All three formats matched the
BF16 quadrants with zero mismatches. The emitted programs' load images were
byte-identical to the source-bound RTL programs that produced those results.
Other K-wave shapes and a general schedule remain unqualified. The provider
still has no Merlin executable backend. The out-of-tree package reproduces all
nine emitted C sources with:

```sh
python -m mx_gemmini_support.bringup --format mxfp6 --case split32x64 --output /configured/artifact-root/mxfp6.c
```

The other formats and `square32`/`square64` cases use the same command. It
prints a source hash and refuses to overwrite existing output; the package
tests pin all nine source hashes.
An out-of-tree bridge now accepts model2MLIR's rank-2 Linear operand handoff
without changing TorchAO or introducing target packing into model2MLIR. A
reproducible integration test transforms one 32×32×32 Linear with TorchAO in
each of the three formats, uses its real uint8 element codes and E8M0 scale
bytes, and packs them with caller-selected exact FP6 codebooks. Its emitted C
source hashes exactly match three programs executed on the source-bound RTL
simulator. Those programs each exited zero with no BF16 mismatches across four
distinct output quadrants. A second test set starts from all-zero TorchAO
blocks, checks zero element codes and E8M0 scale byte 104, and reproduces
three more RTL-tested programs with exact BF16 zero output. This demonstrates
bounded capture-to-RTL compatibility; it does not establish functional
attention packing, arbitrary shapes, full-model execution, or an admitted
Phase 0 capsule.
The pinned software header's residency
comment calls rs1 bit 62, but its macro emits bit 63, matching
`ExecuteController.scala`; the compiler must use bit 63.
Scale reads have no programmable base row: each configured loop begins at
logical row zero only if the previous loop completed its full I/J/K counter
cycle. Bit 62 does not directly reset the `ScaleFactorMem` read counters; it
reaches `MxRequantizer` instead. An out-of-tree layout prototype partitions K
at 32-element block boundaries, limits each wave by both 4 KiB operand
windows and the 9-bit K field, and packs each wave from local byte offset
zero. The bounded 32×32×64 two-wave diagnostic above checks one accumulation
sequence. Ordering larger capacity-driven uploads and loops, preserving BF16
state across them, and comparing their results with the selected RTL simulator
remain open compiler qualifications.

The [software spec](../../examples/mx_gemmini/target/software-spec.yaml) is the
authored software-facing proposal. Its `unreviewed` status is intentional.
The [Phase 0 recipe](../../examples/mx_gemmini/phase0/recipe.yaml) derives
capsule membership from that spec, hardware facts, and exact capture inputs.
The retained hand-authored recipe and historical corpus are reproduction
inputs only.

## Operand formats and numerical contract

| Mode | Element | Maximum finite magnitude | Stored operand path | Block scale |
| --- | --- | ---: | --- | --- |
| MXFP8 | OCP E4M3 | 448 | direct 8-bit code | E8M0 per 32 |
| MXFP6 | E3M2 | 28 | LUT-indexed nibble path | E8M0 per 32 |
| MXFP4 | E2M1 | 6 | direct nibble path | E8M0 per 32 |

BF16 blocks use a scale exponent of
`floor(log2(max(abs(block), 2^-23)))`. The selected
[BF16-to-MX rounding logic](https://github.com/ucb-bar/gemmini/blob/f0167390b56fb315deea90ac1fc3983772e92d82/src/main/scala/gemmini/BF16ScalaRoundToTiny.scala)
uses round-to-nearest-even for the declared element modes and preserves
representable MX element subnormals. BF16 source subnormals flush to zero
before element conversion; a zero block gets E8M0 code 104, as also checked
by the three 32×32×32 all-zero diagnostics above. A nonfinite block
gets E8M0 code 255 and follows the RTL poison path. The FP4 path rounds through
E3M1 before E2M1 and canonicalizes underflow zero to positive zero. The array
has a format-specific product path and a 16-lane accumulator schedule;
ordinary PyTorch matmul of dequantized operands does not reproduce that
schedule. Nonfinite blocks, negative zero, and underflow edges require
explicit RTL comparison.

The first shape contract is deliberately bounded: K must be a multiple of 32;
M and N must be multiples of 16 for MXFP8 and 32 for MXFP6/MXFP4. Rank 2 to
4 independent batches are candidates. Unsupported tails and broadcasts stay
outside accelerator admission until measured. These are software restrictions,
not a claim that the RTL is incapable of other shapes.

## Whole-model selection

Quantize both operands of every eligible `nn.Linear` and visible functional
contraction, including attention QK and PV matmuls. Keep softmax, masking,
normalization, residual and other elementwise work on the host. A fused
scaled-dot-product-attention node hides its two contractions; capture must
expose them or refuse the MX coverage claim. Report module count, functional
contraction count, and every skipped site. A module-only TorchAO transform
cannot establish whole-model coverage.
BF16 accelerator readout may be widened to FP32 for host operations. The
software spec declares both BF16 and FP32 host dtypes as an unreviewed policy;
the TorchAO capture currently returns FP32 at that seam for FP32 inputs. A
reviewed host lowering and numerical comparison must establish which path is
used in an admitted whole-model execution.

A local TorchAO extension uses `AOBaseConfig`,
`register_quantize_module_handler`, and `quantize_` without modifying TorchAO.
The module handler stores static MX weight codes and E8M0 scale bytes and
quantizes activations dynamically. A graph pass inserts Q/DQ on visible
functional contractions. Its FP6 codes are not the selected hardware's 4-bit
LUT indices, and it does not select each 16-entry codebook line or upload
codebooks and banked scales. This is a bringup route: its eager outputs use PyTorch
BF16 arithmetic and are **not** hardware goldens. Its generic MLIR import
records `prov.mx_capture_contract` with the selected RTL pin, format, and
contraction census. Generic floating-point matmuls still do not prove MX
lowering; typed lowering and simulator execution remain separate obligations.

For an architecture smoke test, use the one-layer, random-weight TinyLlama
loader with sequence length 32 and eager attention. Full-checkpoint accuracy,
nonlinear-host semantics, and placement require their own evidence. The
[upstream model2MLIR MX branch](https://github.com/ucb-bar/model2MLIR/tree/feat/mx-gemmini-torchao)
at `ce31865` supports a prequantized capture handoff and exposes
`linear_contraction_operands` for one rank-2 Linear site. That helper presents
A codes as [M][K], B codes as [K][N], activation scales as [M][K/32], and
weight scales as [N][K/32] to an out-of-tree packer. It does not provide FP6
codebook selection or an executable command schedule. In an isolated
diagnostic run, Merlin captured this one-layer model in each of MXFP8,
MXFP6, and MXFP4: all eight eligible Linear modules and both visible
attention matmuls were quantized, no sites were skipped, and the MLIR had
zero opaque calls with an explicit MX capture contract. The three captures
used random weights and token inputs. Their frontend trace is unavailable
and their loader provenance is undeclared, so they are not admitted
application captures or hardware arithmetic evidence.

The [Pi0](https://github.com/chloe-wong/pi0-quant),
[SmolVLA](https://github.com/chloe-wong/smolVLA-quant), and
[microscaling](https://github.com/chloe-wong/microscaling-quant) studies are
comparison and workload inputs; they do not supersede the selected RTL. The Pi0
and SmolVLA studies separate matrix sites (`Linear`, attention QK/PV) from
normalization, nonlinear and elementwise sites and report per-layer errors.
Use those site inventories to check whole-model coverage, while keeping their
vector-unit model separate: this selected MX configuration disables array
nonlinear activations, normalization and max pooling. The microscaling study
offers independent MX operand codes and a systolic arithmetic model for L0
cross-checks; neither can replace the selected RTL simulator at L2/L3.

## Numerical policy to review

The RTL fixes the element modes and transfer encoding. The following software
choices still need one reviewed answer before a Phase 0 corpus or full-model
accuracy claim is admitted:

| Decision | Evidence required |
| --- | --- |
| FP6 codebook selection | Specify the 16 E3M2 entries for each activation-row and weight-column LUT line, the update granularity, and the behavior when a value is absent. Compare the selected projection with the RTL LUT path. |
| Exceptional inputs | Decide whether zero blocks, BF16 subnormals, signed zero, overflow and nonfinite blocks are in scope. Check each admitted case against the selected RTL and an independent numerical model. |
| Host seam | Fix BF16 or FP32 transfer/readout policy for each retained host operation and verify the resulting mixed execution numerically. |
| Whole-model acceptance | Declare per-site coverage and error measures, model-level thresholds, and the reference checkpoint/input set before interpreting TinyLlama or VLA results. |

The TorchAO extension supplies operand codes and a contraction census. Its
PyTorch matmul is not an oracle for the selected PE reduction schedule.

## Admission sequence

1. Pin and inventory the RTL commit, configuration, submodules, elaboration
   products, Spike build, and Verilator build. Compare each binary's source
   identity to the selected closure.
2. Review each format's input and output codes, E8M0 scales, LUT projection,
   product truncation, accumulator order, and BF16 readout against RTL and
   independent model vectors. Include ties, subnormals, zero blocks, large
   exponents, and nonfinite blocks.
3. Derive Phase 0 capsules and retain L0 through L3 outcomes per capsule.
   Do not substitute TorchAO fake quant or a local model match for Spike or
   Verilator evidence.
4. Capture a full-model contraction census and host-operation coverage.
   Compare the TinyLlama and VLA numerical results against a reviewed policy
   for every operation. Admit a compiler only after its typed lowering,
   transfers, and simulator outputs satisfy the selected contract.

Current status: the selected source and MxGen mode test have been checked at
the pinned revisions. Explicit source-selected compilation and standalone
elaboration succeeded. CIRCT emitted split Verilog with inline memories, and
Verilator 5.022 lint passed on all 658 filelist modules with zero errors and
843 warnings. A split-memory emission instead required memory wrapper
implementations that were not supplied in that diagnostic invocation. A local
TestDriver simulator was then linked from the source-bound inline-memory
Verilog; its SHA-256 is
`a72c7314defed09d8281fab6bfeffcd2eaab7ae851a6c1034f8b06f4eaea9fd9`.
The pinned FP8 and FP4 64×64×64 bare-metal tests, and the FP6 128×128×128
LUT-indexed E3M2 test, each exited zero and matched every BF16 golden value.
The FP4 source has a copied success label that says “fp8”; its `CONFIG_EX`
operands select FP4 code 2. These diagnostic tests do not replace Merlin Phase 0
capsules. The spec remains `unreviewed` because L0–L3 capsule results,
reviewed host semantics, and whole-model accuracy remain pending. The retained Phase 1
`hwbringup_mx_v0` ABI describes an older
default/GPU-local mapping and cannot certify this standalone configuration.
The example's selected synthesis profile is also `unverified_legacy`, so Phase 0
preflight refuses execution. Fresh synthesis requires a same-target backend
capability contract, selected RTL facts with derived tile geometry, a newly
derived conformance requirement, and digest-bound inputs. The retained
conformance requirement calls BF16 and int8 accelerator contractions admitted;
those cells conflict with the selected three-format software spec. Synthesis
now rejects such conflicting cells before producing a profile. The old
generated residual also lists BF16/int8 compute and FP32 accumulation outside
the selected software contract; it cannot be promoted without RTL review.
The isolated checkout has no declared application-capture roster, so a fresh
conformance diagnostic derives no requirement. Select and verify exact MX
application captures before replacing either historical input. A source-bound
fact bundle and simulator binary now exist for the selected elaboration, but
they have not qualified a Merlin compiler or the full numerical contract.

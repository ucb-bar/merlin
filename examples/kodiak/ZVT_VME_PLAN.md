# Kodiak Zvt/VME configuration path

## Current hardware boundary

The Kodiak ASIC and `firesim_kodiak_opu` bitstream use the legacy BME OPU instruction words.
Saturn's source tree also contains a VME implementation, but that does not make this decoded
OPU configuration execute Zvt. FPGA programming cannot change the ASIC's instruction decoder.
Merlin currently fails closed when `saturn_vme` is requested on the Kodiak OPU board.

Iris's separate VME configuration passes pinned Zvt Spike and a direct SerialTL RTL gate for
K33, biased-edge with bias, and batch-3 cases. That establishes a tested VME component and
encoding path, not a Kodiak VME SoC or bitstream. See Iris `handwritten-implementation` commit
`e13f7cc` and `evidence/zvt_three_case_20261106/`.

## Required configuration

1. In a managed exact-pin Chipyard task, port Kodiak's one-core base integration from
   `generators/saturn/chipyard/OPUConfigs.scala` at the Kodiak OPU pin into the Iris VME
   Saturn pin `536774bd` and Rocket matrix decoder pin `fc745a8`. Define a distinct
   `KodiakVME1CoreConfig` using `WithKodiakBase(nShuttleCores=1)`,
   `WithEICGPassthrough`, and `WithShuttleVectorUnit(512, 256, VectorParams.opuMxParams)`.
   Elaborate it before assuming the source APIs are compatible.
2. Define a distinct `FireSimKodiakVME1CoreConfig`, HWDB key, and bitstream. Begin with
   one core; the previous two-core OPU U250 design did not route. Preserve Kodiak's DRAM
   base `0x80000000`, its existing FireSim memory-size adjustment, and its scratchpad at
   `0x580000000` if the base integration is retained. Do not reuse the OPU bitstream key.
3. Add a separate Merlin `saturn_vme` Kodiak board/unit contract with explicit VME pins,
   geometry, instruction audit, Zephyr image, and fail-closed bitstream matching. The
   candidate int8 tile edge is `VLEN/8 = 64`; confirm it from elaborated hardware.

The host-side Zvt matrix CSRs need Rocket `Matrix.scala` support for `sf.msetmtype` and
`sf.msettn/m/k`. The RTL source encodes those under OP-V `0x57`, `funct3=7`, with
`funct7=0x42/0x44`; Saturn's int8 matrix multiply uses OP-VE `0x77`, `funct6=111100`.
Tile moves, zero, and discard are separate OP-V forms. Check every encoding against the
pinned Zvt Spike implementation and Iris `vme.h`; that header uses `rd[0]` for A sign and
`vtype.altfmt[8]` for B sign. These are source-audit starting points, not a claim of a
decoded Kodiak bitstream.

## Verification gates

1. Prove source pins and elaborate the one-core Kodiak VME SoC and FireSim configurations.
2. Derive each Zvt word from the elaborated decoder and compare with the pinned Zvt Spike
   and Iris compiler/header encodings.
3. Compile one exact Kodiak VME Zephyr ELF and pass the same image on Zvt Spike. Require
   full output hashes, guard checks, and a plain-RVV or scalar control.
4. Run direct RTL cases on the Kodiak VME configuration, including K33, tails, bias,
   batches, and the full 44-case suite. Iris's three-case pass is a component check only.
5. Build a new bitstream and run that exact ELF on matched one-core FireSim. Record SHA-bound
   image, decoder, bitstream, output, and target-cycle receipts. Measure valid MAC throughput;
   physical row duty additionally requires the counter patch and an instrumented bitstream.

The managed Chipyard counter task currently lacks initialized Saturn/FireSim submodules.
A separate managed exact-pin task has Chipyard `6a9b4cd`, FireSim `b084672`, and Rocket
`885dd596`, but its Saturn gitlink is `2dae15d`; the needed OPU/VME pins are not available
through that task's current managed nested-repository interface. Provision the VME pins in
a compliant managed workspace before editing RTL or claiming this gate complete.

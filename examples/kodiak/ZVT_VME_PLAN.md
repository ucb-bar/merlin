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
   The pinned VME RTL has `(DLEN/8)²` physical int8 MAC cells, so `DLEN=256` gives
   **1,024 MAC per fully active cycle** even though the logical tile is 64 × 64. This is
   a capacity bound, not measured array duty; a later utilization report must account
   for clock gating, MAC enable, row/column masks, and valid model work.

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

The initial elaboration checkpoint used Chipyard `b4542fac`, Saturn VME `536774bd`,
and Rocket matrix-decoder `fc745a8`. For reproduction, use the published Chipyard
`0d911e2d` and Saturn `4359c36` pins below. Distinct one-core SoC and FireSim
configurations passed source compile and target FIRRTL elaboration after a separate Rocket
CSR guard fix. The first passing SBT logs report Java `20.0.2-internal` and Espresso;
fresh published-source validation passed with JDK 17 and Espresso. The fresh VLEN512 biased
K-tail Zvt ELF passed pinned Spike. The current OPU bitstream remains BME only. That exact
35 × 63 × 33 biased K-tail ELF has since passed direct Kodiak VME Verilator/SerialTL with
the same pinned Spike digest, zero output and guard errors, and `KT DONE`; the larger RTL suite
and a VME bitstream remain pending. The separate exact-pin OPU counter task has passed Firechip
compile, target FIRRTL elaboration, Golden Gate, and Verilator metasim build using Espresso.
The exact-pin 64³ OPU counter metasim run passed its bracketed active-row and gate-low checks:
2,048 active-row events at 128 MAC slots each account for all 262,144 valid MACs, with zero
gate-low events. The SHA-bound receipt is in the managed Chipyard task
`opu-counter-goldengate-pin/.tmp/metasim_matrix_hashonly/utilization_receipt.json`.
Counter U250 synthesis and a counter-equipped bitstream remain pending. Keep the
configurations and receipts separate.

The tested VME source is published on Chipyard fork branch `feat/kodiak-zvt-vme` at
`0d911e2d4ff722800829da70af7768c2755ca332` and Saturn branch `feat/kodiak-zvt-vme`
at `4359c36b710bb230335fc2866a28bd3f1911ed6e`. Chipyard's public Rocket gitlink stays
at `fc745a8`; its `patches/kodiak-vme/README.md` and SHA-checked application script carry the
focused scalar-Rocket CSR guard fix. A fresh managed checkout applied that patch and passed
JDK 17 Firechip compilation and FireSim FIRRTL; its generated device tree was byte-identical
to the source used for the direct RTL result. No VME FPGA bitstream has been built.
The Chipyard branch head adds only a host TraceRV `libdwarf` API fix over that source state.
Golden Gate generated SystemVerilog with the same hash before and after the host fix, and the
Verilator metasim binary built. The exact fresh Zvt ELF then passed bounded FireSim metasim:
the biased 35 × 63 × 33 case had the pinned Spike/direct RTL digest `0x15d98143c9a17f64`,
zero output and guard errors, `KT DONE`, and exit zero. Its kernel interval was 2,761 target cycles;
the full simulator ran 1,840,130 target cycles. The SHA-bound receipt is in Chipyard managed task
`kodiak-vme-portability-2/.tmp/vme_k33_metasim/receipt.json`.
This does not establish FPGA bitstream timing or physical Kodiak ASIC support.
The metasim receipt is pinned to Chipyard `5311b459`; the newer published branch head adds
only `patches/kodiak-vme/apply-firesim-worktree-build.sh` and its SHA-verified build patch for
the Iris FireSim base `e1bdea8`. Hardware gitlinks and generated RTL are unchanged.

The original VLEN512/DLEN256 `FireSimKodiakVME1CoreConfig` U250 build passed synthesis and
`opt_design`, then failed the placement DRC `UTLZ-1`: it requires 2,208,803 slice LUTs against
1,728,000 available, an excess of 480,803. The bounded build ended with rc 1 and produced no
bitstream. Its log and rc are in managed Chipyard task `kodiak-vme-portability-2/.tmp/` as
`vme-u250-synthesis.log` and `vme-u250-synthesis.rc`. Do not submit its Zvt ELF to the existing
BME OPU bitstream. A distinct VLEN512/DLEN128 compact VME configuration is undergoing exact-pin
source, RTL, and area gates. Its exact 35 × 63 × 33 biased K-tail ELF passed pinned Spike,
direct RTL, and FireSim Verilator metasim. The compact U250 `synth_1` report now records
1,267,125 / 1,728,000 CLB LUTs (**73.33%**) with zero synthesis errors; its
[SHA-bound report and receipt](https://github.com/copparihollmann/chipyard/blob/1cc1690a608cc995b6c4b896f9e19c8ee4cef556/patches/kodiak-vme/evidence/compact-u250-synthesis.json)
are published on the Chipyard feature branch. This is a post-synthesis estimate. Placement,
routing, timing, bitstream generation, and FPGA execution remain unverified, so it is not a
proven U250 image yet.

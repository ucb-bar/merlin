# Target identity for the later Spike timing model

This is a public, frozen target description for a new Phase 1 release. Release
preparation copies `task/` into the release, and the generated bundle grants
that copy to the compiler agent. The selected Phase 1 capability contract and
RTL facts, also granted by the bundle, govern emitted instructions and runtime
behavior. Their configuration names and hashes identify the actual launch.

The planned Phase 2 timing machine is the full Alveo U250 FireSim design named
`FireSimGemminiRocketConfig`. The `full_u250_board` machine selects hardware
config `alveo_u250_firesim_gemmini_rocket_30mhz`. Its pin is
`firesim_gemmini_rocket_30mhz` in `merlin/contract/hardware_pins.yaml`, with
bitstream SHA256
`0af6f26e1cbad1c03f9e5873217505a5100bf2d0d068205cfa486522b24fc603`.
That pin records a FireSim generator revision distinct from the older
`gemmini_rtl` pin used for the selected `GemminiRocketConfig` RTL facts.

The FireSim board catalog declares one non-vector Rocket hart and 16 GiB of
DRAM mapped at `0x80000000`; the selected DTS SHA256 is
`c4a2009a405945ea98f731713e0f4e7dc483031ab8cd5156d871e4eb7594f1df`.
All six Phase 1 whole-model Spike programs select this same board entry for
their host build. The private model specification must pin this FireSim DTS;
the historical Verilator-DTS Spike board is retained only for old runs.
The memory span is an address-map declaration, not a measured usable-capacity
certificate. The full design's ABI header SHA256 is
`3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924`;
it supports full-width accumulator readout. The historical lean U250 design
has a different ABI and is a different target.

Phase 1 capsule timing is measured on its selected gSIM elaboration. The
currently installed emulator uses `GemminiGsimSerialClkConfig` and FIRRTL
SHA256 `089d053b62523b5047e71ccca31dfbbe27d4e5daeab24050227a923e7b4628c8`.
That emulator is based on the older `gemmini_rtl` revision. It is not an
equivalence witness for the FireSim bitstream. The older hand-edited
`gemmini_gsim_model_testharness` pin is also unsuitable for this new run.

The current Phase 0 RTL facts come from a `GemminiRocketConfig` Verilator
elaboration. The planned FireSim bitstream uses a different generator revision.
Their common ancestry does not establish equal accelerator features or equal
cycle counts. A FireSim-targeted Spike model needs a reviewed fact-level
comparison or facts regenerated from the matching FireSim source, plus paired
measurements on the exact FireSim bitstream. Until then, Phase 1 gSIM capsule
labels describe only the selected gSIM design and must carry its identity.

For the next experiment intended to feed this FireSim model, select the
`gemmini_rocket_chipyard` / `gemmini_rocket_rtl` source revisions of the pinned
bitstream before Phase 0 derivation. Add the exact
`build_tools/chipyard/GemminiGsimFireSimAlignedConfig.scala` file to a clean
descendant of the recorded Chipyard revision and keep the Gemmini gitlink at
its recorded revision. This new configuration composes the FireSim chip tweaks
and IO cell models with the gSIM clock and serial adapters. Elaborate
`GemminiGsimFireSimAlignedConfig` from that committed source; regenerate RTL facts from that exact
TestHarness FIRRTL, the gSIM executable and its strict build receipt, and the
selected execution contract together. The source-to-FIRRTL elaboration receipt
must bind the selected configuration, the reviewed Chipyard descendant and the
exact Gemmini, Saturn and FireSim submodule revisions.
The configuration must retain the gSIM clock/serial harness adapters while
matching FireSim's 16 GiB address map, CPU ISA and chip-visible bus/peripheral
settings. `GemminiGsimSerialClkConfig` alone retains the older chip settings.
At the recorded Chipyard revision, `FireSimGemminiRocketConfig` composes
`WithFireSimConfigTweaks` with `GemminiRocketConfig`. The FireSim tweaks set
16 GiB external memory, 1 GHz bus frequencies, SerialTL width 4, UART policy,
debug/clock-gating/monitor policy, trace IO and block-device presence. The new
gSIM configuration must carry each chip-visible setting that changes the
elaboration while selecting the gSIM TestHarness clock instantiator and
serial-clock punchthrough. Review the resulting generated
DTS and RTL differences; a shared base class does not make them equivalent.
Compare the chip portion of that elaboration with the FireSim build and record
the Gemmini parameters and ABI header identity.
Require a passing matched-ELF capsule check on gSIM and FireSim before fitting
any transfer from gSIM cycles to board cycles. A FireSim wrapper is not a drop-in
gSIM executable, even when it encloses the same chip configuration.

The operator can check the Phase 1 Spike executable, extension, source revisions
and full-board ABI, DTS and private program board selection against the Phase 2
machine with:

```sh
python -m merlin_experiments.phase1.spike_alignment \
  --contract CONTRACT --chipyard-root CHIPYARD --spike-runtime SPIKE_RUNTIME \
  --machines MACHINES --hardware-pins PINS \
  --board-catalog CATALOG --private-model-spec SPEC
```
Set `SPIKE_RUNTIME` to the built `fire_aligned_30mhz` functional package's
`runtime/` directory. Its deployment receipt and executable, Gemmini extension
and Boost runtime library must remain together. Launch both Phase 1 and Phase 2
with the explicit 16 GiB map at `0x80000000`; the Phase 1 contract must use
`--isa=rv64imafdcb_zicsr_zifencei_zihpm_zfh_zba_zbb_zbs --priv=m -p1
-m0x80000000:0x400000000` before
`{elf}` and select that runtime's library directory. Use the release-selected
contract, `examples/gemmini/phase2/whole-model-machines.yaml`,
`merlin/contract/hardware_pins.yaml`, the release-selected board catalog and
private full-model specification. A passing report proves those selected Spike,
ABI, board and DTS identities align; its `rtl_equivalence_proven: false` is deliberate.

Before launch, run the independent gSIM alignment check with the selected
release contract and facts:

```sh
python -m merlin_experiments.phase1.gsim_alignment \
  --contract CONTRACT --rtl-facts FACTS \
  --chipyard-root CHIPYARD --hardware-pins PINS \
  --board-catalog CATALOG --gsim-dts GSIM_DTS
```

Require `gsim_source_aligned: true`. This verifies the reproduced elaboration
receipt, source commits recorded for the FireSim bitstream, facts FIRRTL,
strict emulator build receipt, and matching host memory map and ISA in the
generated gSIM and FireSim DTS files. The historical bitstream's build-time source
state was not completely preserved, so matching commits alone cannot prove
chip equivalence. Structural comparison and matched-ELF measurements are still
required before timing transfer.

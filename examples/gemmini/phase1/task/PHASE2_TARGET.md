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
retained `gemmini_gsim_model_testharness` pin names
`GemminiGsimMonitorsConfig`, whose chip configuration derives from
`GemminiRocketConfig`; its source FIRRTL SHA256 is
`477e16d0615508b434d373f9e4dcce1ebff6ec25041e534079f3d068044aedb7`.
An actual run uses the simulator and contract in its sealed release; these
retained pins do not override that selection.

The current Phase 0 RTL facts come from a `GemminiRocketConfig` Verilator
elaboration. The planned FireSim bitstream uses a different generator revision.
Their common ancestry does not establish equal accelerator features or equal
cycle counts. A FireSim-targeted Spike model needs a reviewed fact-level
comparison or facts regenerated from the matching FireSim source, plus paired
measurements on the exact FireSim bitstream. Until then, Phase 1 gSIM capsule
labels describe only the selected gSIM design and must carry its identity.

For the next experiment intended to feed this FireSim model, select the
`gemmini_rocket_chipyard` / `gemmini_rocket_rtl` source revisions of the pinned
bitstream before Phase 0 derivation. Elaborate `GemminiRocketConfig` and the
gSIM harness configuration from those same source bytes; regenerate RTL facts,
the gSIM executable and its strict build receipt, and the selected execution
contract together. Compare the chip portion of those elaborations with the
FireSim build and record the Gemmini parameters and ABI header identity.
Require a passing matched-ELF capsule check on gSIM and FireSim before fitting
any transfer from gSIM cycles to board cycles. A FireSim wrapper is not a drop-in
gSIM executable, even when it encloses the same chip configuration.

The operator can check the Phase 1 Spike executable, extension, source revisions
and full-board ABI, DTS and private program board selection against the Phase 2
machine with:

```sh
python -m merlin_experiments.phase1.spike_alignment \
  --contract CONTRACT --chipyard-root CHIPYARD \
  --machines MACHINES --hardware-pins PINS \
  --board-catalog CATALOG --private-model-spec SPEC
```
Use the release-selected contract, `examples/gemmini/phase2/whole-model-machines.yaml`,
`merlin/contract/hardware_pins.yaml`, the release-selected board catalog and
private full-model specification. A passing report proves those selected Spike,
ABI, board and DTS identities align; its `rtl_equivalence_proven: false` is deliberate.

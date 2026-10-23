# Native FASED SimDRAM integration — 2026-10-23

The production `mm_t` implementation is linked into the selected gSIM emulator.
It shares memory with the existing TSI loader and dump path. The native FASED
RTL, not a fitted approximation, determines request admission and response timing.

Actual acceptance:

- 28 reads and 12 writes completed across 756 cycles with zero Verilator/gSIM
  signal mismatches; up to 10 reads and 3 writes outstanding, repeated IDs,
  partial/narrow writes and response backpressure.
- W was presented before AW for six cycles and correctly backpressured.
- Four malformed geometry/protocol cases were rejected with explicit errors.
- The linked whole-SoC emulator ran a public CPU numerical memory test and
  exited successfully at 77,346 target cycles, including loader/boot work.

`probe.json` gives counts and completion records. Compressed trace/stimulus files
preserve exact bytes. `provenance.json` pins source, runtime settings, adapter,
reused native objects and binaries without machine-specific paths.
`whole-soc-smoke.json` records the actual numerical ELF execution; its cycle
count is not a benchmark or calibration label.

Deployed bitstream source, runtime overrides and AXI edge metadata remain
unavailable from read-only local-wrapper/SSM-document inspection. The selected
native source/configuration is pinned, but deployed-FireSim equivalence remains
**NOT_ESTABLISHED**. No campaign simulator or fitted model was activated.

Reproduction and limitations: `build_tools/gsim/reference/PRODUCTION.md`.
Full local receipts, ELF, binaries and logs are preserved under
`out/artifacts/fased-simdram-20261023/` in the implementing task.

# Native FASED boundary canary — 2026-10-23

Eleven numerical read/write transactions completed. The independently generated
Verilator and gSIM cores agree on all meaningful outputs for 312 post-reset
target cycles (zero mismatches). This is the actual source FASED timing core,
not the source-derived approximation previously used by Spike.

`probe.json` records operations and completion cycles; `comparison.json` records
exact comparison identities. `provenance.json` pins source, tools, selection and
generated artifacts without machine paths. The two `.jsonl.gz` files preserve
exact trace bytes when decompressed; their hashes are in `comparison.json`.
They are analysis evidence, not grading answers or inputs to the agent loop.

The selected source/configuration is a **prototype compile selection**, not a
verified deployed-FireSim contract. Runtime defaults are explicit and effective
port-width truncation is recorded. No FPGA or full-SoC run was performed. The
canary uses one outstanding request, aligned 64-bit INCR transactions and full
write strobes. Production gSIM memory replacement, multiple outstanding/order
coverage, deployment pinning and whole-system calibration remain unqualified.

Implementation/reproduction instructions: `build_tools/gsim/reference/README.md`.

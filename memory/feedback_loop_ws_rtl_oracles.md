---
name: feedback-loop-ws-rtl-oracles
description: Before declaring "RTL bug" on Gemmini features, stand up canonical chipyard tests on the same bitstream as an oracle — RTL bug ≠ first reasonable conclusion when canonical code works
metadata:
  type: feedback
---

When a Gemmini codegen feature (LOOP_WS, LOOP_CONV_WS, etc.) misbehaves on FireSim and you can't immediately pin the cause, **DO NOT default to "RTL bug" as the conclusion**. The user has rejected this conclusion twice in succession — both times the root cause was actually in our codegen contracts vs the hardware spec.

**Why:** chipyard's `gemmini-rocc-tests/bareMetalC/tiled_matmul_ws*.c` plus EXO exercise the same Gemmini RTL successfully on the same FireSim Shuttle bitstream. If they work and ours doesn't, OUR codegen is wrong — not the hardware.

**How to apply:** before writing up an "RTL bug" finding, run the equivalent canonical chipyard test for the failing shape on the SAME FireSim bitstream:
- Add `tiled_matmul_ws_<shape>.c` under `/scratch2/agustin/chipyard-autocomp/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/` + register in `Makefile`.
- Build: `cd .../build && conda activate /scratch2/agustin/chipyard/.conda-env && make BAREMETAL_ONLY=1 bareMetalC -j 8`.
- Stage: `cp .../build/bareMetalC/<name>-baremetal /scratch2/agustin/chipyard/sims/firesim/deploy/workloads/bench-canonical-tiled-matmul-ws/bench-canonical-tiled-matmul-ws.elf`.
- Run: `bash /scratch2/agustin/merlin/tmp/run_canonical_firesim.sh bench-canonical-tiled-matmul-ws`.
- Interpret: if canonical PASSES → bug is in our codegen, diff RoCC streams against `gemmini.h::gemmini_loop_ws` macro emission. If canonical FAILS → only then is "RTL/libgemmini" a credible conclusion.

Two real bugs found via this path (2026-05-25/26):
- LOOP_WS contractually requires `(I*K + K*J)*DIM ≤ BANK_NUM*bankRows/2` per `libgemmini/gemmini.cc:720`; canonical wraps each `gemmini_loop_ws` call in a host `tiled_matmul_outer` triple-loop sized to fit half-SPAD. Our codegen had been passing whole-matmul dims to one LOOP_WS, blowing past the contract for big matmuls.
- LOOP_WS_CONFIG_STRIDES_AB/DC takes strides in **elements**, while CONFIG_LD/CONFIG_ST take **bytes**. Canonical passes them as separate quantities; we were reusing the byte-stride value for both. Silently invisible for any matmul where internal `I = 1` (the bogus factor multiplied by zero), but corrupted memory for the unaligned dronet matmuls.

Related: [[project_dronet_gemmini_resolved_stack_overflow]] — another case where a "stack/runtime" first impression turned out to be a different specific root cause (Zephyr 96 KB worker-stack overflow).

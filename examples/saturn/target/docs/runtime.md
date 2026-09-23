# Saturn runtime adapter

Saturn is a **Merlin runtime adapter**, not its own runtime. The Merlin command buffer
(RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT + tensors table) is executed by:

- **simulator** — the Python engine (`merlin.runtime.simulate`), reference oracle.
- **baremetal (spike)** — `merlin.runtime.backends.spike`: the command buffer is
  compiled into a bare-metal driver around the hand-written RVV kernel
  (`merlin/runtime/baremetal/spike/rvv_matmul_i8.S`), partitioned row-wise across
  harts, and run with `spike --isa=rv64gcv_zfh_zvfh -pN`. Outputs are gated on
  equality with `reference_outputs(cb)`.
- **vcs** — `merlin.runtime.backends.vcs`: the *same ELF* replayed on a pre-built
  Saturn VCS simulator (`MERLIN_SATURN_SIMV`); never builds RTL.
- **zephyr** — later: the `spike_riscv64` SMP board in zephyr-chipyard-sw, exercising
  the RVV context-switch path (`zephyr_ws/zephyr/arch/riscv/core/v.c`).

Command encoding: the abstract opcodes map 1:1 (Merlin owns the command buffer);
"residency" is a packed weight kept live in memory across the region, so RES_PACK and
EVICT are counted layout/budget events with no data movement on this target.

Metrics: `cycles` is the hart-0 `mcycle` delta between the start and final barriers;
counters (pack/hits/evictions/commits) are counted by the generated driver; byte
counters use the simulator's formulas. Everything normalizes onto `metrics.schema.yaml`.

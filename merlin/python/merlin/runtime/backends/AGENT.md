# AGENT.md — merlin/python/merlin/runtime/backends

## Purpose

Merlin runtime **execution backends**: run the same Merlin command buffers the Python simulator executes, on real ISAs/simulators — spike (bare-metal multicore RVV) and pre-built Saturn VCS RTL sims.

## What belongs here

- `rvv_codegen.py` — command buffer → C driver around the hand-written RVV kernel.
- `spike.py` — compile (chipyard riscv gcc) + run (`spike --isa=rv64gcv_zfh_zvfh -pN`) + parse + normalize metrics.
- `vcs.py` — replay the same ELF on `MERLIN_SATURN_SIMV` (never builds RTL).

## What does not belong here

- Target dialects or lowering (that is `xdsl_dialects/`).
- The bare-metal harness C/asm (that is `merlin/runtime/baremetal/spike/`).
- RTL builds, vendored simulators, or anything that mutates the chipyard checkout.

## Interfaces

- Input: command-buffer dicts (command_buffer.schema.yaml), same as `merlin.runtime.simulate`.
- Output: `{outputs, metrics, raw_metrics, correct, console}` with metrics normalized onto `COMMON_METRIC_NAMES` (extras under `target_specific`).
- Env: `MERLIN_CHIPYARD` (default `/path/to/chipyard`), `MERLIN_RISCV_GCC`, `MERLIN_SPIKE`, `MERLIN_SATURN_SIMV`.

## Invariants

- **Correctness gate**: every backend run is compared against `reference_outputs(cb)`; `correct` must be True for a run to count. Residency/parallelization must never change results.
- Generated epilogue C must match `tensor.py` semantics bit-exactly (rounding arithmetic shift, saturating i8).
- Tests must auto-skip (not fail) when the toolchain is absent (`spike.available()`).
- Keep the matmul kernel as hand-written `.S` — see `merlin/runtime/baremetal/spike/AGENT.md` for the GCC 13.2 intrinsics bug that forced this.

## Testing expectations

`merlin/python/tests/test_rvv_spike.py` (skips without toolchain); codegen-only tests run everywhere.

## Notes for future agents

`cycles` from spike is mcycle delta on hart 0 between the start barrier and the final barrier; counters (pack/hits/evictions/commits) are counted by the generated driver itself, byte counters are computed statically by codegen using the simulator's formulas.

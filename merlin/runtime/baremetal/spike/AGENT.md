# AGENT.md — merlin/runtime/baremetal/spike

## Purpose

Merlin-owned bare-metal harness for running command buffers on **spike** as a multicore RVV CPU: startup (`crt.S`), HTIF console/exit (`htif.c/h`), memory layout (`link.ld`), and the hand-written RVV matmul kernel (`rvv_matmul_i8.S`).

## What belongs here

- Machine-mode startup, HTIF protocol, linker script, and RVV kernel library `.S` files consumed by `merlin/python/merlin/runtime/backends/spike.py`.

## What does not belong here

- Generated per-command-buffer drivers (those are emitted into a work dir by `rvv_codegen.py`).
- Target-specific runtime models — this is a Merlin runtime backend; targets adapt to it.

## Interfaces

- Compiled together with the generated `main.c` by `backends/spike.py` using the chipyard `riscv64-unknown-elf-gcc` (`-march=rv64gcv`), run with `spike --isa=rv64gcv_zfh_zvfh -pN`.
- Output protocol on the HTIF console: `OUT <name> <rows> <cols> v...`, `METRIC <name> <value>`, `HART <id>`, `DONE` — parsed by `backends/spike.py`.

## Invariants

- `crt.S` must: install the fail-fast trap handler **before** anything else, enable `mstatus.VS`, give each hart its own stack, and release secondary harts only after `.bss` is cleared (`boot_ready`) — removing any of these reintroduces silent hangs or races.
- Keep the matmul kernel in hand-written assembly: GCC 13.2's vsetvl fusion miscompiles mixed-EEW RVV *intrinsics* (e32m4 config fused to e8m1 → `vsext.vf4` becomes illegal). Do not rewrite with intrinsics without verifying on the actual toolchain.
- Epilogue semantics in generated drivers must match `merlin/python/merlin/runtime/tensor.py` exactly (rounding arithmetic shift, saturating i8 clamp).

## Testing expectations

`merlin/python/tests/test_rvv_spike.py` (auto-skips when the toolchain is absent).

## Notes for future agents

Toolchain resolution lives in `backends/spike.py` (`MERLIN_CHIPYARD`, default `/scratch2/agustin/chipyard`). The trap handler encodes failures as exit code `0x1000 | (mcause << 1) | 1`.

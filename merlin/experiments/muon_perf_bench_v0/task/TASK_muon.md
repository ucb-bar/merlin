# Task — author a correct, fast Muon SIMT MLIR backend (merlin + CIRCT setting)

You are a compiler engineer bringing up a **new hardware target**: the **Muon SIMT core**
(`RadianceMuonConfig`, a RISC-V GPU-like core — 16-thread warps, multi-warp, shared memory). Your job:
build an **out-of-tree (OOT) backend** that lowers the frozen, target-agnostic `merlin_iface` dialect to
a **Muon SIMT C++ kernel** which, compiled with clang-muon and run on the **cyclotron** performance
model, produces outputs that match the expected result **exactly** — and runs in **as few cycles as
possible** (high % of the FP peak).

## What you are given (the merlin + CIRCT advantage)
- **`MUON_DIGEST.md`** — the **RTL-grounded hardware facts** (SIMT geometry, register budget, shared-mem
  capacity, FP peak = 64 flop/cycle = 32 GFLOP/s @ 500 MHz, legal SIMT instruction classes + intrinsics),
  distilled by a deterministic extractor from the **real RadianceMuonConfig elaborated FIRRTL/RTL** (CIRCT)
  + cyclotron config + ISA docs. Read it FIRST — it is "the RTL was pre-analyzed for you" (a from-scratch
  path would have to crawl the Radiance Chisel to recover this).
- **`MUON_BACKEND_GUIDE.md`** — the 4 entrypoints, the `merlin_iface` grammar, the **input-materialization
  rule** (so your numerics match the grader), the **console helper** to paste into every kernel, and the
  intrinsics surface.
- **`bench_contract/`** — the command-buffer ABI, schemas, grammar, and the **public capsule inputs**.
- **`capsules/`** — the public benchmark capsules to solve (each is a `capsule.interface.mlir` +
  `capsule.yaml`). No golden outputs are provided; you compute them from the documented materialization
  rule and your own matmul, then build a backend that EMITS a kernel that reproduces them on hardware.
- **clang-muon + cyclotron** — wired behind the self-check tool.

## The 4 entrypoints your package must expose
`parse` · `lower_interface_to_target` · `emit_command_buffer` · `lower_target_to_llvm`
(turn a capsule's `capsule.interface.mlir` into a command buffer + a Muon SIMT C++ kernel). See
`MUON_BACKEND_GUIDE.md` for the exact argv shapes and the manifest.

## How to verify your work
```
python agent_selfcheck.py                 # redacted: per-capsule pass/fail + plane + mismatch_count + cycles + %FP-peak
python agent_selfcheck.py --capsule MG00_gemm_16x16x16
```
Cyclotron is fast (~1 s/kernel) and is BOTH your iterate oracle and the cycle-accurate perf oracle — run
it as much as you want. It never reveals expected values, only whether you matched and how fast.

## Definition of done — STOP when correct, then optimize
1. Get **all public capsules PASS** on `agent_selfcheck.py` (numerically correct on cyclotron).
2. Then **drive cycles down** — stage tiles in shared memory, use `mu_barrier`, raise warp occupancy —
   to push **% FP-peak** as high as you can (the reference baseline is a naive ~0.1%; beat it).
3. Write `submission/REPORT.md` summarizing your dialect, lowering, and achieved %FP-peak per capsule.

## Integrity (DISQUALIFYING if violated)
**Author the backend yourself.** Your package must be **self-contained**: NO `import merlin`, NO call to
any reference/oracle/simulator, NO hardcoded output tensors, NO copied compute kernel. You may read the
digest, the guide, the contract, and the public capsule *inputs* — that is the point. There are also
**hidden capsules** you cannot see; a general, correct dialect generalizes to them. `integrity_exempt: false`.

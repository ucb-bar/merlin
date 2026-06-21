# Task — author a numerically-correct Gemmini MLIR backend (realistic setting)

You are a compiler engineer. Your job: build an **out-of-tree (OOT) MLIR backend** for the **Gemmini**
systolic-array accelerator that is **numerically and functionally correct** — i.e. for each benchmark
capsule, the command buffer your backend emits, run on the **real Gemmini RTL (verilator/VCS)**, produces
outputs that match the expected result exactly.

## What you are given (realistic HW bring-up — what ships with the RTL)
- `gemmini/rtl/` — the **Gemmini Chisel RTL** (the hardware itself; the ground truth for how it behaves).
- `gemmini/README.md` + `gemmini/README_HWBRINGUP.md` — the architecture/ISA overview that ships with it.
- `gemmini/isa_include/` — the ISA C headers (`gemmini.h`, `gemmini_params.h`): opcodes, `DIM`, dtypes.
- `gemmini/example_kernel/matmul_ws.c` — **ONE** worked example kernel (single-tile weight-stationary
  matmul): the canonical "hello world" showing how to drive the ISA. This is the **only** example you
  get — you must **generalize from it + the RTL + the ISA headers** to every benchmark op (matmul,
  movement, conv, attention, …). This mirrors real bring-up: you have the hardware and one smoke test,
  not a library of finished kernels.
- The bench contract (`bench_contract/`): the command-buffer ABI, schemas, grammar, integrity policy, and
  the **public capsule inputs** (the problems to solve).
- The LLVM/MLIR-23 toolchain to build your package.
- **A self-check tool** (`agent_selfcheck.py`) — see below.

## The 4 entrypoints your package must expose
`parse` · `lower_interface_to_target` · `emit_command_buffer` · `lower_target_to_llvm`
(turning a capsule's `capsule.interface.mlir` into a command buffer in the frozen ABI). Your package is
graded only through these CLI entrypoints and must be **integrity-clean**: no `import merlin`, no calls to
any reference/oracle, no hardcoded outputs, no copied compute kernels.

## How to verify your work — two tools (use them as much as you want)
**1. spike self-check — fast, for ITERATION (seconds):**
```
python agent_selfcheck.py --sim spike --capsules all     # functional: same integer numerics + trace check
```
It builds your package, runs the capsules on spike, and returns **pass/fail + mismatch_count + the failing
plane** (never the expected values). Iterate on this until it is **fully clean**.

**2. verilator — the cycle-accurate RTL cert, but SLOW (~2.5 min/capsule). Run it ASYNC so it never
blocks your turn** — fire per-capsule jobs and poll:
```
python simjob.py submit --sim verilator --capsules A1_mvin_mvout   # returns a job_id immediately
python simjob.py poll   --job-id <id>                              # {state: queued|running|done, result}
python simjob.py list                                              # your jobs
```
Fire several per-capsule jobs and collect results; do **NOT** `--sim verilator --capsules all` synchronously
(that would block your turn for ~an hour). Spike to iterate, verilator (async, per-capsule) to cert.

> **VCS (L4) is UNAVAILABLE** in this environment (no adapter; it segfaults). Do not attempt it — verilator
> is the cycle-accurate cert oracle.

## Definition of done — STOP when correct
1. First get **`agent_selfcheck.py --sim spike --capsules all` FULLY clean** (all pass + trace clean). Do
   **NOT** write `READY_FOR_BARRIER` before that — a premature READY just costs you a round.
2. Then write the empty file `submission/READY_FOR_BARRIER` and end your turn; the harness runs the
   verilator (L3) cert on the real RTL.
3. **A verilator failure or timeout is NOT the end** — you get the redacted cycle-accurate failures back and
   may fix and retry (within the round budget). You are finished the moment **all public capsules pass on
   verilator (L3)**.

## Integrity (DISQUALIFYING if violated)
**Author the dialect yourself.** You may read the vanilla Gemmini repo, its docs, and its example
*kernels* (C code showing ISA usage) — that is the point. You may NOT copy or adapt a pre-existing MLIR
*backend/dialect/package* (e.g. anything under `generated_targets/`, prior `runs/*/submission`, or any
other ready-made `gemmini-opt`/dialect you find on the filesystem), and you may NOT read capsule
`golden.yaml` files or any reference-oracle to obtain expected outputs and hardcode them. Your package is
graded only through its 4 entrypoints and is scanned for these violations; a submission that copies a
prior backend or embeds golden outputs is **disqualified**. Build it from the spec + the example kernels.

## What we measure (be efficient)
We record your **total cost, total tokens, and wall-clock development time** to reach a correct dialect, plus
how many self-checks failed and which simulators/tools you used. **Cycle counts / performance are NOT graded
— only numerical/functional correctness.** Aim to reach a fully-correct backend with the least cost and time.
There is no benefit to over-optimizing or polishing beyond correctness; stop as soon as the barrier passes.

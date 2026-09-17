# Starter plan — C++ OOT + INFRA arm  ·  build your OWN MLIR dialect+passes and integrate it via our scaffold

You author a **C++ out-of-tree MLIR backend** (your own dialect + passes plugin) for Gemmini. Think of it as:
*"I built my own MLIR dialect and want to integrate it into this project's scaffold/runtime."* You get the
**generic C++ OOT scaffolding + the shared runtime/contract** to stand it up — but you build the dialect
work yourself. You do NOT get the Python (xDSL) authoring kit, a pre-built input dialect, the CIRCT RTL
tools, or the oracle. The real work — defining your dialect(s) and writing the lowering — is yours, in C++.

## A. Use the generic C++ OOT scaffold to stand up YOUR dialect (don't hand-roll the package plumbing)
1. **Scaffold the package**: `merlin.targetgen.generate.mlir_scaffold` generates a generic C++ OOT MLIR
   package skeleton from a dialect plan you give it — a target-dialect ODS skeleton (empty ops/types you
   fill in), the `gemmini-opt` driver wiring the 4 entrypoints, and the CMake to build against the provided
   LLVM/MLIR-23. `generate.target_repo` lays out the package; `generate.llvm_plan` helps the LLVM/RoCC
   lowering. Run tools with the repo's `.venv/bin/python` (the interpreter that has `xdsl` + `merlin`).
2. **Define your OWN input + target dialects in C++.** There is **no pre-built `merlin_iface` dialect** here
   — you write the dialect/parsing for the frozen input grammar yourself (it's a small, regular MLIR
   grammar; see the contract docs). Register your dialects in your tool, parse the capsule, lower through
   your target dialect. (This is the "integrate your own dialect" part — the scaffold gives you the package
   structure, you supply the dialect + passes.)
3. **Plug into the shared runtime**: you only emit (a) `command_buffer.json` (the frozen ABI) and (b)
   lowered LLVM/RoCC MLIR. The shared runtime executes + grades them — you do NOT call it directly (the
   oracle is off-limits and not needed). Conform to the 4 entrypoints the scaffold wires:
   `parse` / `lower_interface_to_target` / `emit_command_buffer` / `lower_target_to_llvm`.

## B. The lowering you author (the real work, in C++)
- **matmul-family**: load stationary operand → preload → compute into accumulator → read out w/ scale/relu.
- **movement** (mvin/mvout): load→store, **NO compute**.
- **conv2d**: lower to a **2D im2col matrix first**, then reuse the matmul path (don't feed 4D to the 2D unit).
- **Two RTL invariants**: config-before-use (emit `CONFIG_*` before the first dependent compute/store); tile
  to the array **DIM** and respect scratchpad/accumulator capacities (the public ISA header gives DIM + dtypes).
- Emit BOTH the command buffer AND the LLVM/RoCC `.insn` custom-3 stream; the canonical WS tile sequence is
  `FLUSH → CONFIG_EX → CONFIG_LD → MVIN → CONFIG_ST → PRELOAD → COMPUTE_PRELOADED → MVOUT`.

## C. Build, self-check, iterate
- The package is **C++** (`manifest.yaml`: `language: cpp` + a `build` block that compiles
  `mlir_oot/build/bin/gemmini-opt` against `-DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR`). The runner builds it.
- Iterate on **spike** (fast) via `agent_selfcheck.py --sim spike`; verilator is the cycle-accurate cert —
  run it **async/per-capsule** via `simjob.py submit/poll` (it's minutes/capsule, don't block your turn).
- Don't declare `READY_FOR_BARRIER` until spike + trace are clean. A verilator failure is **not** terminal —
  you get the redacted cycle-accurate failures back and may fix & retry. **VCS (L4) is unavailable.**

## D. What you do NOT have (and shouldn't look for)
- No CIRCT RTL-facts tools (`gen_isa_module`/`gen_rtl_digest`/`gen_numeric_facts`/`facts.json`) — derive the
  ISA from the public header + the scaffold.
- No Python authoring kit (`oot_starterkit`/`synthesize`/`xdsl_dialects`) — you build in C++.
- No oracle / reference / prior backends — emit your own command buffer; never read an answer.

Goal: reach verilator 25/25 by authoring a correct C++ OOT dialect+passes ON the Merlin infra, at lower
cost/rounds than building the same backend from scratch. Stop the moment all public capsules pass.

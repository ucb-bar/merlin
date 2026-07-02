# Starter plan — BASELINE (from-scratch C++ MLIR) arm

You are bringing up a C++ out-of-tree MLIR backend for an accelerator from its RTL + ISA headers. **This
condition ships no example kernels** — derive the lowering from the ISA header + RTL alone. Before writing
code, commit to this plan (it is target-agnostic — it tells you *method*, not the answer):

## 1. Derive the ISA encoding ONCE, up front (do this first — it's the #1 time sink if you flail)
From the ISA headers, extract into a single table you write once: the custom opcode, funct3, and the
**legal funct codes + their names** (config / load / store / preload / compute / flush / loops). Emitting
any funct outside that set produces an UNKNOWN instruction the hardware rejects — enumerate the legal set
before lowering anything. Do not re-grep the headers per op; build the table once and reference it.

## 2. Lowering plan per op-family (write these as distinct pass cases)
- **matmul-family** (matmul, k-accum, linear, attention): load stationary operand → preload → compute into
  accumulator → read out with scale/activation.
- **movement** (mvin/mvout): load→store, **NO compute** instruction. (Forcing a compute into a pure-move
  op is a common bug.)
- **conv2d**: lower to a **2D im2col matrix first**, then reuse the matmul path. Do not feed 4D tensors to
  the 2D compute unit.
- **padding / edge**: handle boundary tiles explicitly.

## 3. Two invariants the RTL enforces — bake them into the emitter
- **Config-before-use:** emit every `CONFIG_*` before the first instruction that depends on it
  (compute needs exec-config; store needs store-config).
- **Tile to the array DIM** and keep scratchpad/accumulator residency within the stated capacities.

## 4. Iterate cheaply
Use the self-check tool on **spike** (fast) to iterate; only run **verilator** once your structure is
clean (it's minutes per run). Read your OWN emitted command-buffer + decoded trace from the self-check
output to debug — that shows exactly what you produced.

Goal: functional + numerical correctness on all public capsules (verilator), with the fewest rounds and
least token spend. Stop the moment you pass.

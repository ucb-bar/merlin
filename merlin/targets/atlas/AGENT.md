# Atlas target (prototype)

Atlas (a self-hosted-ISA FP8 NPU with a 32×32 systolic MXU) as a Merlin **tensor-resident**
target. Status: **prototype**, certification **uncertified** (mesh/dtype facts from the
generator config + mlc arc discovery, not silicon).

Atlas is the **second** target proving the 4-arm ladder generalizes beyond gemmini. Unlike
gemmini (RoCC `.insn`, `endpoint_kind: inline_asm_insn`), Atlas has its **own opcodes and
instruction memory**, so its `endpoint_kind` is **`external_backend`**: the deliverable is a
`kernel.S` the target's own assembler turns into IMEM words. It is NOT RoCC, and NOT an
ISA-less command-buffer target.

## Contract
`contracts/target_contract.yaml` is **generated**, not hand-authored — regenerate with
`merlin.targetgen.capability_manifests.write("atlas", base=<this dir>)` (the derived
`atlas_manifest()` builder; provenance-tagged, `requires_human_review`). It lives here — a
first-class **reference** target dir, exactly like `merlin/targets/gemmini/` — so it is
resolved by `target_registry.resolve("atlas")` as `kind="reference"` and sits **outside** the
`out/artifacts/targets/atlas/` champion/answer-surface tree the harness masks. (Keeping the
contract out of the answer surface is load-bearing: the launcher reads the contract to build
each arm's prompt, and that dir is `chmod 000`-locked as an answer surface before any spend.)

## Oracle
The program-oracle (`merlin.targetgen.program_oracle`) assembles the arm's emitted `kernel.S`
via the model's own ISA (npu_model — never a hardcoded opcode table) and runs it on the mlc
Arc cosim (`cosim_atlas.run_program`), reading DRAM back per the capsule's `cb` tensors.
Validated bit-exact (`max_abs_diff=0.0`).

## Corpus
The atlas capsule corpus is FP8-e4m3 in / BF16 accumulate with FLOAT (tolerance) goldens from
an independent specir refmodel — see `merlin/contract/capsules/atlas/`. It is NOT the gemmini
integer (i8×i8→i32, exact) corpus, which a float MXU physically cannot match.

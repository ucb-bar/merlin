# Gemmini target prototype — C0 RTL certification

The first concrete step of Merlin's RTL-grounded target-generation path: a hand-curated
Gemmini target whose C0 matmul is **certified against real Gemmini RTL** by differential
execution. This is the gate that will later certify *generated* targets.

## What this proves

- A Merlin command buffer (`RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT`) lowered for the
  `gemmini` target, with a generated low-level Gemmini kernel, computes a 16×16×16
  `i8 × i8 → i32` matmul **bit-exactly** on the Gemmini **Verilator RTL** sim:
  `RTL output == reference_outputs(cb) == simulate(cb)`.
- The execution oracle is real and reproducible (see `gemmini_rtl_oracle_status.md`), and a
  measured cycle anchor is recorded (`artifacts/capsule-bench/gemmini/certification_c0.yaml`, cycles = 241).

## What this does NOT prove

- No epilogue (requant/relu), no multi-tile / non-DIM-aligned shapes, no overflow-stressing
  data (the deterministic C0 values fit i8, so i8 and full-i32 readout coincide), no calibrated
  cost model (241 cycles is a single anchor, not predictive), no FireSim, no whole models, no
  agentic generation, no CIRCT fact extraction. Those are later milestones.

## Oracle ladder

| Level | Oracle | `derived_from_rtl` | Role |
|---|---|---|---|
| 0 | merlin `reference_outputs` + `simulate` | n/a | pure Python, no toolchain |
| 1 | spike + gemmini extension | **false** | **bootstrap only** (functional model) |
| 2 | Gemmini Verilator RTL (`GemminiAndOPUShuttleConfig`) | **true** | **certification** |
| 3 | FireSim (same RTL) | true | later — scale + memory-timing fidelity |
| 4 | silicon | true | if available |

**Spike is bootstrap only.** `libgemmini.so` is a hand-written functional model, not derived
from RTL — spike results are never marked `rtl_certified`. **Levels 2 and 3 run the same RTL**
for functional correctness; FireSim upgrades scale and memory-timing fidelity, not basic
correctness.

## Why the command buffer stays target-independent

Gemmini reuses the existing opcodes (`RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT`) — no
Gemmini-specific opcodes. The opcode set is Merlin's hard ABI between compiler and runtime
adapters; keeping it target-independent is what makes the simulator a cross-target oracle and
metrics comparable. The Gemmini ISA (mvin/preload/compute/mvout) lives entirely in the runtime
kernel codegen (`runtime/backends/gemmini_codegen.py`), not in the dialect or the command buffer.

## Why there is no direct RTL → target-dialect lowering

> RTL is not compiled directly into the Merlin target dialect. RTL is elaborated by existing
> hardware tooling into structural evidence and an executable oracle; a target spec constrained
> by that evidence is compiled by Merlin targetgen into the dialect; the generated target is
> accepted only after differential certification against Merlin's reference and the RTL oracle.

## Pieces (this PR)

- `merlin/targets/gemmini/` — hand-curated contract + dialect plan (capacities provenance-tagged,
  pending confirmation against `Configs.scala`).
- `xdsl_dialects/targets/gemmini.py` — compiler-level `pack/matmul/commit/release` dialect
  (clone of saturn; NOT the ISA).
- `xdsl_dialects/lowering/{target_lowering,runtime_lowering}.py` — `gemmini` rows.
- `runtime/backends/gemmini_codegen.py` — command buffer → bare-metal C (explicit low-level
  intrinsics, not `tiled_matmul_auto`).
- `runtime/backends/gemmini.py` — compile + run on spike/verilator + parse + gate; returns
  `oracle.{kind, derived_from_rtl}`.
- `tools/probe_gemmini_oracle.py`, `docs/gemmini_rtl_oracle_status.md` — Step-0 oracle proof.
- `tests/test_gemmini_c0.py` — L0 reference==sim, pipeline descent, codegen smoke, spike
  bootstrap, Verilator C0 certification.

### Diagnostic routing (failure level → responsible plane)

- L0 fails → merlin cb semantics / reference / simulator / lowering.
- L0 passes, spike fails → Gemmini codegen or spike invocation.
- L0 + spike pass, Verilator fails → runtime/kernel plane (config order, fences, alignment, RoCC
  state, mvin/mvout layout, stationary transpose, accumulator addressing) — **not** spec/dialect.
  (RTL is stricter than spike; expect this transition.)

## Status beyond C0 (milestones B/C/D)

**B — conformance battery + recording.** Certifiable (bit-exact, three-way) rungs:
`C0` matmul, `C1` +relu, `C4` multi-tile 32³ (K-accumulation), `C4e` edge 16×24×20 (zero-pad),
`C5` reuse — **all five RTL-certified on Verilator, three-way bit-exact** (cycles
211/211/882/529/841). `requant` (C2/C3) and `transpose` (C6) are
**documented divergences, not oversights** (see `gemmini_requant_reconciliation.md`; transpose
needs an interface-level change). Recording uses the **aet** substrate
(`merlin/eval/gemmini_suite.py`: manifest + metrics + origin-tagged artifacts + FailureRecord);
the resumable cartesian dispatcher is `merlin/eval/gemmini_dispatcher.py` +
`merlin/experiments/gemmini_cert/`.

**C — agentic generation (Claude Code CLI, Opus, no API key).**
`merlin/targetgen/agent/{claude_cli,kernel_slot}.py`. The agent saw the command-buffer ABI + a
Gemmini ISA reference + C0/C1 single-tile examples (never the reference outputs) and synthesized
`generate_driver`; the harness certified it on the **held-out** shapes it never saw (C4 multi-
tile, C4e padded, C5 reuse) — **bit-exact, first attempt**, and the kernel is **RTL-certified on
Verilator** (held-out C4 three-way bit-exact, `derived_from_rtl=True`, 1131 cyc). The certified
kernel + transcript are committed under `merlin/experiments/gemmini_cert/agent_generated/`.
Cheat-scan + held-out discipline prevent peeking. The LLM proposes; the oracle disposes.

**D — FireSim (L3) + CIRCT facts.**
`merlin-rtl-introspect` (`merlin/targetgen/rtl/introspect.py`) extracts **structure-only** facts
from the elaborated Gemmini FIRRTL (mesh 16×16 from the tile hierarchy, scratchpad 262144 B from
the source-tagged `smem`, i8/i32 datapath, RoCC/TLB interfaces) and **reproduces the hand-curated
contract capacities** — the validation that lets a generated spec be trusted. FireSim (L3) runs
the **same RTL** as Verilator (L2), so C0–C5 functional correctness is already earned at L2;
FireSim upgrades memory-timing fidelity (FASED) + scale, not correctness, and executing on the
shared FPGA is gated on the queue/bitstream (the documented next step).

## Next phase — `merlin-rtl-introspect` (CIRCT MLIR-pass version)

```
Chisel/FIRRTL → CIRCT/firtool → hw/seq/comb/firrtl MLIR → Merlin extractor →
  rtl_facts.yaml / decoder_table.yaml / memory_table.yaml / interface_table.yaml
```
Then an agent proposes `family_classification` / `target_spec` / `dialect_plan` /
`runtime_adapter_plan` (constrained by those facts), Merlin's deterministic targetgen emits the
dialect + lowering + runtime scaffold, and **this C0 gate certifies the generated artifact**.
The extractor is structure-only and is validated by reproducing the hand-curated facts above.

---

## Recorded provenance: headline results now live in the AET ledger

The MLIR-faithful Gemmini results (non-requant C-rungs **and** quantized Q-rungs) are recorded
through the **AET substrate + resumable dispatcher** — they are no longer hand-authored YAML or
`/tmp` scrollback. Each `(rung × oracle × codegen_backend)` cell is an **isolated run directory**
with origin-tagged, attributable artifacts:

```
runs/gemmini_cert/runs/gemmini-conformance/<rung>_<oracle>_<backend>_seed000/
  run_manifest.yaml        # status, oracle{kind, derived_from_rtl}, cycle_accurate,
                           # cycle_source/window, memory_model, codegen_backend, toolchain SHAs, cycles
  artifact_manifest.json   # content-addressed, origin-tagged index
  generated/command_buffer.json   (GENERATED            — workload/runtime contract)
  generated/kernel.mlir           (COMPILER_GENERATED   — the emitted MLIR; THE recorded kernel)
  generated/gemmini_kernel.ll/.o  (COMPILER_GENERATED   — lowered LLVM IR + object)
  generated/harness.c             (COMPILER_GENERATED   — thin data/print glue, no compute)
  artifacts/console.log           (ORACLE_OUTPUT        — the oracle's raw stdout)
  logs/{metrics,events,params}.jsonl, logs/failures.jsonl (on mismatch)
```

**Provenance by plane** (the anti-blur ledger): generated input vs compiler output vs oracle
output never blur, because each artifact carries an `ArtifactOrigin` tag.

**Oracle ladder, recorded explicitly:**
- spike → `oracle.kind: spike_gemmini_functional`, `derived_from_rtl: false`, **`cycle_accurate: false`** (bootstrap only — spike cycles must never be read as RTL cycles).
- Verilator → `oracle.kind: rtl_verilator`, `derived_from_rtl: true`, **`cycle_accurate: true`** (the certification gate).

**Codegen selector** is a first-class, separate axis (NOT overloaded onto the oracle):
`codegen_backend ∈ {mlir_inline_asm_rocc, legacy_c}`. The legacy C path is retained (the agent
currently targets it; it is a useful comparison path) and still recorded as `kind=kernel`.

**Failure routing** is recorded in `failures.jsonl`: L0-fail → reference/sim/lowering; L1(spike)-fail
→ codegen/spike-invocation; **L1-pass/L2-fail → runtime/kernel/RTL interaction** (candidate causes
listed: config ordering, fences, block_stride, mvin/mvout layout, accumulator addressing, stationary
transpose, RoCC state, alignment, sim invocation).

**Reproducibility (the acceptance test):** the headline table is regenerated *strictly from
recorded manifests* — `python -m merlin.eval.gemmini_dispatcher --summary-only` — not from any
hand-authored file. The sweep is resumable: cells already passing are skipped, failed/missing
cells re-run. Spike is bootstrap-only; Verilator is the certification gate.

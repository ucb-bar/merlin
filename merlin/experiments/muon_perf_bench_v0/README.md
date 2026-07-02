# Muon perf bench — the Merlin+CIRCT methodology, retargeted to RadianceMuon

This is the **4th-arm methodology** (Merlin xDSL tooling + scaffold + CIRCT-compiled-from-RTL hints)
from the Gemmini experiment, pointed at a **different target**: the **Muon SIMT core**
(`RadianceMuonConfig`, in the separate `/scratch/agustin/projects/chipyard` checkout), with
**cyclotron** as the simulator oracle instead of spike. **Gemmini is not involved** — this is a
parallel path; the frozen Gemmini target-gen is byte-for-byte unchanged (only additive,
target-agnostic widenings to the shared `merlin_iface` grammar + command-buffer/capsule schemas to
admit fp32 + SIMT instruction classes).

Performance is reported **conservatively against the Muon SIMT FP peak**:
`64 flop/cycle = 32 GFLOP/s @ 500 MHz` (2 cores × 16 lanes × 2 flop/FMA).

## What runs today (validated end-to-end)

```
                merlin_iface (frozen, target-agnostic, fp32)
                          │  muon-opt  (4 contract entrypoints)
                          ▼
   command_buffer.json ── reference_v0 ── SIMT C++ kernel
                          │                     │ clang-muon (Vortex rv32, +vortex)
        L0 golden  ◄──────┤                     ▼
        L1 sim     ◄──────┘            kernel.radiance.elf
                                              │
                          ┌───────────────────┴───────────────┐
                          ▼ cyclotron --timing (L2, primary)   ▼ VCS RadianceMuonConfig (L3 cert)
                   cycles + GFLOP/s + %FP-peak           cycle-exact difftest
                                                         (honest-unavailable; kernel-launch WIP)
```

Run the perf bench (reference backend):

```bash
.venv/bin/python experiments/muon_perf_bench_v0/scripts/run_muon_perf.py --run-id ref_v0
# -> runs/ref_v0/perf_table.md + perf_results.json
```

Run one capsule directly:

```bash
.venv/bin/python -m merlin.targetgen.muon_capsule_runner \
  --package generated_targets/muon/reference_v0 \
  --capsule experiments/muon_perf_bench_v0/kernels/MG00_gemm_16x16x16 \
  --runs-root /tmp/muon_runs
```

Reference-backend baseline (naive global-memory GEMM, one threadblock — the *ceiling to beat*):

| kernel | cycles | GFLOP/s | % FP peak |
|---|---:|---:|---:|
| MG00_gemm_16x16x16 | 181177 | 0.023 | 0.07% |
| MG_gemm_32x32x32   | 757127 | 0.043 | 0.14% |
| MG_gemm_64x64x64   | 2877619 | 0.091 | 0.28% |

(The reference is deliberately unoptimized — no SMEM staging, no tiling, both cores recompute. The
agentic Merlin+CIRCT backend's job is to close this gap toward the 32 GFLOP/s peak.)

## Pieces

| component | path | role |
|---|---|---|
| Muon backend | `merlin/python/merlin/runtime/backends/muon.py` | clang-muon compile + cyclotron/VCS run + OUT/DONE parse + GFLOP/s |
| SIMT codegen | `merlin/python/merlin/runtime/backends/muon_codegen.py` | command buffer → fp32 SIMT C++ kernel (reference emitter) |
| oracle adapters | `merlin/python/merlin/targetgen/muon_oracles.py` | cyclotron (L2) + VCS (L3) adapters |
| parallel runner | `merlin/python/merlin/targetgen/muon_capsule_runner.py` | Muon tier ladder; reuses generic helpers, zero Gemmini coupling |
| reference backend | `generated_targets/muon/reference_v0/` | `muon-opt` (4 entrypoints) — integrity-exempt ceiling |
| corpus + report | `experiments/muon_perf_bench_v0/{kernels,scripts}/` | fp32 GEMM capsules + GFLOP/s-vs-peak table |

## Toolchain (env overridable)

`MERLIN_CHIPYARD` (=`/scratch/agustin/projects/chipyard`), `MERLIN_RADIANCE_KERNELS`
(=`/scratch2/agustin/radiance-kernels`), `MERLIN_MUON_CLANG`, `MERLIN_MUON_CYCLOTRON`,
`MERLIN_MUON_CONFIG`, `MERLIN_MUON_VCS`.

## Agentic rounds (merlin + CIRCT, single arm) — READY TO RUN

A fresh sandboxed `claude` agent authors a self-contained Muon SIMT backend each round from the
answer-free workspace + the **RTL-grounded `MUON_DIGEST.md`** (the "+CIRCT" advisory, extracted by
`merlin.targetgen.rtl.muon_introspect` from the real RadianceMuonConfig FIRRTL/hierarchy). The operator
grades on cyclotron and hands back a redacted verdict. Cyclotron is the fast iterate AND cert oracle, so
there is no verilator/simjob/broker.

```bash
# launch the single-arm agentic loop (spawns a real claude agent — costs $ + time)
.venv/bin/python experiments/muon_perf_bench_v0/scripts/run_muon_qa_loop.py \
    --run-id muon_circt_0001 --model claude-opus-4-8 --effort high --max-rounds 6
# -> runs/muon_circt_0001/{final_report.md, submission/, rounds/, qa_history/}
```

The agent's redacted self-check (it runs this itself, as many times as it wants):
```bash
python agent_selfcheck.py            # per-capsule pass/fail + plane + mismatch_count + cycles + %FP-peak
```

Harness pieces (all validated end-to-end except the paid agent spawn):

| component | path | role |
|---|---|---|
| CIRCT facts | `merlin/python/merlin/targetgen/rtl/muon_introspect.py` | firtool/RTL → `muon_facts.json` (evidence-stamped) |
| digest | `merlin/python/merlin/targetgen/rtl/gen_muon_digest.py` | `muon_facts.json` → `MUON_DIGEST.md` (the +CIRCT advisory) |
| task | `task/TASK_muon.md` | the agent's contract |
| guide | `input_bundles/muon_rtlchecks_public_v0/MUON_BACKEND_GUIDE.md` | grammar + console helper + materialization rule + example |
| self-check | `scripts/agent_selfcheck.py` | redacted cyclotron grade (no golden values) |
| driver | `scripts/run_muon_qa_loop.py` | round loop: assemble ws → spawn agent → grade → verdict → finalize |

Regenerate the RTL facts/digest:
```bash
.venv/bin/python -m merlin.targetgen.rtl.muon_introspect    # -> merlin/targets/muon/contracts/rtl_facts/muon_facts.json
.venv/bin/python -m merlin.targetgen.rtl.gen_muon_digest    # -> MUON_DIGEST.md
```

## Not yet built (next)

- **Static advisory checker** (`muon_checks` + `muon_decode`/`muon_trace_check`): a per-round pre-screen
  over the agent's emitted kernel (SMEM ≤128 KiB, live regs ≤128, threadblock ≤ lanes·warps, barrier
  safety) feeding expected-vs-got hints, like Gemmini's `qa_check_rtlchecks`. Today the RTL digest is the
  standing +CIRCT advisory and cyclotron is the gate; this would add a cheaper static pre-screen.
- **Golden-kernel ceiling**: also run the hand-written autocomp golden SIMT kernels as a second
  comparison alongside the FP-peak %.
- **VCS-RTL L3 cert**: currently honest-unavailable (kernel-launch difftest stalls at `nu.invoke`, WIP).

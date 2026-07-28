---
title: Gemmini target-dialect-generation experiment (case study)
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-22
related: [getting_started, reproducibility, targetgen, adding_a_target, target_publishing, experiment_abi]
code_refs:
  - merlin/experiments/capsule_bench/targets/gemmini
  - merlin/experiments/gemmini_cert
  - merlin/experiments/gemmini_perf_bench
  - merlin/python/merlin/targetgen/capsule_runner.py
  - merlin/python/merlin/targetgen/oot_runner.py
  - merlin/python/merlin/targetgen/publish.py
  - merlin/python/merlin/benchharness
---

# Gemmini target-dialect-generation experiment

Gemmini is the **case study** for the target-dialect-generation tool: it asks *how well can an agent
author a correct, RTL-conformant Gemmini MLIR OOT backend*, under increasing amounts of Merlin help,
in a cheat-proof sandbox — then certifies and publishes the result. This guide is the runnable
end-to-end pipeline. It documents **how to run it**, not any particular result.

Every step is fail-closed: correctness gates first, a missing sim/toolchain records `not_run` (never a
false pass), and the answer surfaces (goldens, hidden capsules, the reference oracle) are masked from
the agent and re-verified before any spend.

## The four arms (same task, capsules, hidden set, grader)

| arm | `--arm` | what it gets | driver |
|---|---|---|---|
| C++ scaffold (raw) | `raw_baseline` | public spec + ISA headers, no Merlin tools | `run_baseline_qa_loop.py` |
| C++ + Merlin infra | `cpp_merlininfra` | + Merlin build/runtime infra | `run_baseline_qa_loop.py` |
| xDSL + Merlin tooling | `merlin_assisted` | + xDSL dialect/scaffold generators, starter kit | `run_baseline_qa_loop.py` |
| Merlin + CIRCT | (rtlchecks) | + CIRCT-compiled-from-RTL checks as advisory feedback | `run_rtlchecks_qa_loop.py` |

Convergence bar = **all 20 public capsules pass** (spike L2, then a verilator L3 barrier); the 5 hidden
capsules are graded only in the final audit. (`A0/A2/A4/B0` is a round-0 quick-check subset, NOT the
target.) The same four submissions are re-profiled as perf-bench backends (§4).

## 0. Prerequisites (all resolve via `.env` — never hard-code)

**Shared base:** complete the base install + `.env` setup in [Getting started](getting_started.md)
first, then `check_repro_env.py` to confirm the Gemmini capabilities (`gemmini_spike`,
`gemmini_verilator`, `chia`, `llm_api`) are runnable here.

**Workflow-specific prerequisites:**

- **Required — `bwrap` (bubblewrap) on `PATH`**: the cheat-proof sandbox that filesystem-isolates the
  agent (`scripts/sandbox_toolchain.py`).
- **Required — sim toolchain** via `ext_path('chipyard')` / `.env MERLIN_CHIPYARD` →
  `.conda-env/riscv-tools/bin` (spike, riscv64-unknown-elf-gcc) + `sims/verilator`. Clang-23 via
  `MERLIN_CLANG_INSTALL`. The bwrap toolchain binds and the driver-side sim broker
  (`simjob_broker.py`) both read these from `.env`. spike (L2) + verilator (L3) fully certify;
  **VCS (L4) and FireSim (L5) are not fresh-machine reproducible** (Synopsys license / FPGA — see
  [Getting started §5](getting_started.md)) and fail-closed to `not_run`.
- **Required for a REAL agentic run — `ANTHROPIC_API_KEY`** (+ optional `MERLIN_LLM_MODEL`). Unset ⇒
  the proposer uses its deterministic mock fallback: the sandbox/cert/perf gates still run, but no real
  agentic authoring happens.
- **Optional — chia venv** (only for the Chia fan-out): the isolated `out/build/chia-venv` (never the
  main `.venv`) —
  `uv venv out/build/chia-venv --python 3.13 && uv pip install --python out/build/chia-venv -e /path/to/chia -e .`
- **Required — `.compat_lib/libidn.so.11 → .so.12`** shim at the repo root (the conda cmake needs `.so.11`).

## 1. Prove the sandbox BEFORE any spend (mandatory gate)

```bash
# every legit tool works + every answer is masked, per arm. Exit 0 only if all green. No agent, no $.
.venv/bin/python merlin/experiments/capsule_bench/targets/gemmini/scripts/test_sandbox.py --arm merlin_rtlchecks
.venv/bin/python merlin/experiments/capsule_bench/targets/gemmini/scripts/test_sandbox.py --arm raw_baseline
.venv/bin/python merlin/experiments/capsule_bench/targets/gemmini/scripts/test_sandbox.py --arm merlin
# static cheat-clean gate: no answer content in any shipped tool/prompt (grep over source)
.venv/bin/python merlin/experiments/capsule_bench/targets/gemmini/scripts/verify_no_cheat.py
```
Both must be green (`🟢 sandbox GO`, `✅ VERIFY_NO_CHEAT: PASS`). The sandbox tmpfs-masks all of
`/scratch*`, binds back only the legit toolchain + the workspace (bound LAST so no deny-mask clobbers
the agent's cwd), unsets the nested-session vars so the spawned `claude` connects fresh, and masks the
experimenter memory. Sims the agent requests run OUTSIDE the sandbox via the async `simjob_broker`
(redacted verdicts only).

## 2. Run the sweep (all four arms)

`launch_ab_batch` locks the answer surfaces (chmod 000) + runs `verify_no_cheat` before launching, then
backgrounds a chain. **Do not set a tight `--round-timeout`** — the default is 4 h (14400 s); a short
cap is net-detrimental (it doesn't cut the work, just forces more rounds, each adding a grading pass +
context re-read + rate-limit exposure). The rate-limit watchdog (`--max-rate-limit-waits`, default 8)
sleeps to the window reset and resumes the same round unattended.

```bash
cd merlin/experiments/capsule_bench/targets/gemmini/scripts
.venv/bin/python launch_ab_batch.py --tag <tag> \
  --arms baseline,cpp_merlininfra,merlin,merlin_rtlchecks \
  --mode sequential            # one arm-chain after another; --mode parallel if the 5h bucket has headroom
# monitor via each run's qa_loop_state.yaml (NOT the .log — block-buffered):
#   out/runs/gemmini/capsule-bench/<arm>/<run-id>/qa_loop_state.yaml
```
A single arm (e.g. to iterate/debug): `run_baseline_qa_loop.py --run-id <id> --arm <arm> --sandbox bwrap`
(`--resume` continues from `qa_loop_state.yaml` after a process death; `--no-oracle`/`--skip-hidden` for
fast dev). Runs land under `out/runs/gemmini/capsule-bench/<arm>/<run-id>/` (submission + per-round
verdicts in `qa_history/` + `cost_time_toolcalls.yaml` with the active-vs-rate-limit-wait split).

For unattended survival across session/usage limits, the QA loop's own watchdog handles the five-hour
window; for weekly-limit or process-death resilience use `--resume`, or drive through `aet run --resume`.

## 3. Certify (RTL conformance)

```bash
# C-rungs C0/C1/C4/C4e/C5 (i8 matmul + relu + edge + reuse), three-way bit-exact reference==spike==RTL.
.venv/bin/python merlin/experiments/gemmini_cert/run.py --simulators spike            # bootstrap (fast)
.venv/bin/python merlin/experiments/gemmini_cert/run.py --simulators spike,verilator  # + RTL certification
```
Resumable via `out/runs/gemmini/cert/ledger.jsonl` (skips already-correct cells). The cert backend
(`mlir_inline_asm_rocc`) + the capsule oracle share `runtime/backends/gemmini.py`, which resolves the
toolchain through `.env`. RTL facts are extracted by `targetgen/rtl/circt_introspect.py`
(`firtool --ir-hw` → HW-dialect) and compiled into FileCheck assertions by `rtl_check_compiler.py`.

## 4. Perf-bench (cross-approach profile)

```bash
.venv/bin/python merlin/experiments/gemmini_perf_bench/scripts/run_perf_bench.py \
  --kernels all --approaches golden,baseline,merlin_targetgen,merlin_native   # + agentic_* backends
```
The `agentic_*` backends resolve the latest submission per arm live from `out/runs/gemmini/capsule-bench/`
(a not-yet-run arm is an honest skip). golden(C-lib) and IREE-dialect are additional reference approaches.
Outputs under `out/runs/gemmini/perf-bench/` + `out/artifacts/plots/gemmini/perf-bench/`.

## 5. Publish a certified champion

```bash
# verify against a LOCAL bare remote first (no GitHub):
git init --bare /tmp/gemmini-mlir.git
.venv/bin/python -m merlin.targetgen.publish publish --target gemmini \
  --remote file:///tmp/gemmini-mlir.git --execute
git -C /tmp/gemmini-mlir.git branch --list   # baseline (hand_v0) or stable/<package_id> (a champion)
```
Remote from `merlin/targets/publish.yaml` (`targets.gemmini` = `git@github.com:ucb-bar/gemmini-mlir.git`)
or `MERLIN_PUBLISH_REMOTE_GEMMINI`. Gate (`publish._check_gate`, mlir_oot family): `status ==
rtl_certified` **or** an `oot_runner.certify` pass. Baseline packages → the `baseline` branch; each
certified champion → its own `stable/<package_id>` branch + `v<ver>-<pkg>` tag; idempotent by fingerprint.
The identical flow works for **rvv** (`--target rvv`, vector_schedule family, gate = spike/rtl/k1_verified,
remote `ucb-bar/rvv-mlir`). **A real GitHub push (drop the `file://`) needs an explicit human go-ahead.**

## Honesty invariants
- Correctness gates before any wall/cycle number; a fail sorts last; `not_run` is never a pass.
- The agent never sees goldens / hidden capsules / the reference oracle (masked + `verify_no_cheat` +
  a post-run transcript audit; the per-round audit conservatively flags reads of any `expected_*`
  example, adjudicated at `finalize`).
- Runs are high-variance — compare distributions (n≥3), never a single run; keep round budgets equal
  across arms for a fair comparison.
- Publishing is verified against a `file://` remote; GitHub push is human-gated.

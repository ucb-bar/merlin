# capsule_bench_v0 — A/B coordination contract (raw_baseline ⇄ merlin_assisted)

Purpose: both Claude sessions (this one drives **raw_baseline**; the other drives **merlin_assisted**)
must produce the *same artifact class*, satisfy the *same interfaces*, and be *graded + measured by the
same substrate*, so the comparison is apples-to-apples. This file is the single source of truth for
what MUST be identical, what MAY legitimately differ, and exactly how runs are conducted.

Repo root: `/path/to/merlin` (`$REPO`).
Experiment: `$REPO/experiments/gemmini_capsule_bench_v0` (`$EXP`).

---

## 1. What the experiment compares

One task — *generate an out-of-tree MLIR Gemmini target backend* — produced by two agent arms, graded
by one frozen benchmark (`capsule_bench_v0`). We compare **artifact quality** (which capsules pass, at
which oracle tier, instruction-trace correctness, numeric exactness, integrity) and **process**
(wall, cost, tokens, tool-calls, #iterations to converge). Cycles are **diagnostic only — never gate
pass/fail**.

## 2. The two arms — the ONLY allowed difference

Both arms get: the frozen `merlin/contract/`, the public/dev capsules (`isa/`, `layers/`,
`model_slices/`), public Gemmini headers (`gemmini.h`, `gemmini_params.h`), the task, and the
LLVM/MLIR-23 toolchain. Both are denied: the Merlin runtime **reference/simulator** (the oracle/cheat
surface), `capsules/hidden/`, all prior backends (`merlin_native_v0`, `agent_spec_v0/v1`,
`hand_smoke_oot`), `grader_private_v0`, and `runs/`.

| | raw_baseline | merlin_assisted |
|---|---|---|
| bundle | `input_bundles/raw_baseline_public_v0` | `input_bundles/merlin_assisted_public_v0` |
| extra ALLOWED (authoring only) | — | `merlin/python/merlin/targetgen/{synthesize,generate}/`, `merlin/python/merlin/xdsl_dialects/`, `targetgen/contract/interface_emit.py` |

The Merlin tools are **authoring aids only**. The submitted package must still pass the non-exempt
integrity scan: **no `merlin.runtime.reference`/`merlin.runtime.simulator` imports, no
`reference_outputs`, no reading the answer, no copying/calling a finished backend.** `integrity_exempt:
false` for both.

---

## 3. SHARED FROZEN CONTRACT — both backends MUST satisfy these identically

This is the agreement surface. Source of truth: `$REPO/merlin/contract/`. Do **not** redesign any of it.

### 3.1 Package manifest + the 4 CLI entrypoints (`schemas/manifest.schema.json`, `mlir_oot_backend_contract.yaml`)
`submission/manifest.yaml` required keys: `artifact_type: mlir_oot_target_backend`, `target: gemmini`,
`language: python|cpp`, `authoring{mode,...}`, `integrity_exempt: false`, `entrypoints{tool}`,
`commands{...}` (+ optional `build{command,tool_output[,configure]}` for C++). The package is invoked
**only** through these argv templates (never imported), `{tool}`→`entrypoints.tool` (or
`build.tool_output`), `{input_mlir}`/`{output_json}` substituted by the runner:
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse+verify; nonzero exit on diagnostics.
- `lower_interface_to_target`: `{tool} --convert-iface-to-gemmini {input_mlir}` — emit gemmini-dialect
  MLIR (must parse + `verify()`).
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid
  `command_buffer.json`.
- `lower_target_to_llvm`: `{tool} --convert-iface-to-gemmini --convert-gemmini-to-llvm-rocc
  {input_mlir}` — `llvm.func @gemmini_kernel(...)` of RoCC `.insn r 0x7b` inline-asm.

### 3.2 Input grammar — `merlin_iface` v0.1 (`interface_grammar.md`)
Module attrs `merlin_iface.{version,target,abi_version}` (reject unknown `version`). Ops:
`tensor` (leaf, has `name`,`role`) · `resident_pack` (→`!merlin_iface.resident`, `layout`) · `matmul`
(`(tensor, resident)->!merlin_iface.acc<i32>`) · `commit` (`acc -> tensor`, attrs
`name`,`epilogue`,`output_dtype`,`acc_scale`) · `evict`. **Preserve `name` attrs verbatim** — leaf
data is materialized deterministically by name (this is why hidden capsules generalize and hardcoding
fails).

### 3.3 command_buffer ABI (`command_buffer_abi.yaml`, `schemas/command_buffer.schema.json`)
Top: `{abi_version, target, commands[, tensors, ...]}`. Opcodes: `RES_PACK{src,dst,layout}` ·
`MATMUL_RESIDENT{lhs,rhs,dst}` (i8×i8→i32 accumulate) · `MATMUL` (non-resident) · `COMMIT{src,dst,
bias?, epilogue, output_dtype, acc_scale?, requant_shift?}` · `EVICT{handle}`. `epilogue` = ordered
subset of `[bias_add, requant, acc_scale, relu]` applied in listed order then cast to `output_dtype`.
**`acc_scale` (f32) = `clamp_i8(round_half_to_even(acc * scale))`.** Integer = exact `==`, never a
tolerance.

### 3.4 Kernel ABI (`mlir_oot_backend_contract.yaml`)
`void gemmini_kernel(ptr weight, ptr lhs_0..lhs_{R-1}, ptr out_0..out_{R-1})`; arg order = `[resident
weight] ++ [matmul lhs in command order] ++ [commit outputs in command order]`; pointees row-major,
edge tiles zero-padded to DIM=16. The **runner owns** the harness, link, Spike/Verilator invocation,
and the `OUT/METRIC/DONE` print — the package only emits the kernel.

### 3.5 RoCC encoding the decoder accepts (`targetgen/rocc_decode.py`)
custom-3 `.insn r 0x7b, 0x3, <funct7>, …` — **TWO conformant operand forms are accepted** (both
hold both arms to the same gate; the RTL oracle at L2/L3 is the correctness arbiter):
- 2-operand, no result: `.insn r 0x7b, 0x3, <funct>, x0, $0, $1` with `"r,r"` (rd hardwired to x0).
- 3-operand, with rd:   `.insn r 0x7b, 0x3, <funct>, $0, $1, $2` with `"=r,r,r"` (rd bound to a GPR;
  `$0`=output, `$1`/`$2`=the rs1/rs2 inputs; an LHS `%vN = …` carries the result).
Trailing constraint clobbers (e.g. `~{memory}`) are tolerated. funct: CONFIG=0
(EX/LD/ST refined by `rs1 & 0x3`), MVIN=2, MVOUT=3, COMPUTE_PRELOADED=4, COMPUTE_ACCUMULATE=5,
PRELOAD=6, FLUSH=7; region opened/closed by `"fence"`. **Operands may be ANY MLIR SSA name** (`%c0`,
`%w`, or `%0`) — see §7. Constants via `llvm.mlir.constant`, arg bases via `llvm.ptrtoint %argN` +
`llvm.add`. _(2026-06-18: the 3-operand form was added after the merlin backend emitted valid RoCC
with a bound rd that the 2-operand-only decoder mis-classified as all-UNKNOWN — see §7.)_

### 3.6 Oracle ladder + pass rule (`oracle_runner_contract.yaml`, `scoring.yaml`)
L0 reference==simulate (pure Python, always) · L1 spike functional (`derived_from_rtl:false`) · L2
verilator RTL (`derived_from_rtl:true, cycle_accurate`) · L3 firesim (deferred). Three-way bit-exact
**`golden == reference(cb) == simulate(cb) == oracle`**, integer exact. A capsule **passes** only when
its `required_oracle_tiers` all pass + trace_check passes + numeric exact. A skipped oracle is recorded
skipped, never silently passed.

### 3.7 Instruction-trace class check (`targetgen/trace_check.py`)
Each capsule's `expected_instruction_coverage.yaml` lists required instruction classes (pilot
A0/A2/A4/B0: `[FLUSH,CONFIG_EX,CONFIG_LD,MVIN,CONFIG_ST,PRELOAD,COMPUTE_PRELOADED,MVOUT]`). No
`forbidden_classes` on the pilot, so extra classes (e.g. COMPUTE_ACCUMULATE/LOOP_WS for B0's K=32) are
allowed. Decode must yield zero UNKNOWN, open with FENCE, MVOUT count = tile count, etc. Beyond
classes, trace_check also enforces **mode bits**: i8 readout, relu activation bit, non-identity
acc_scale on CONFIG_ST, and (for `k_accumulate`) an accumulate-onto PRELOAD. Address/stride fields are
not asserted explicitly — they are covered transitively by the **bit-exact** oracle (a wrong
address/stride → wrong output → numeric fail), which is a stronger guarantee than a field check.

**CONTRACT REFINEMENT (both arms must adopt):** `B0_quantized_linear_i8` is K=32 (two K-tiles) but
previously declared modes `{i8, acc_scale}` only — so the accumulate check never fired on the one
K>16 pilot capsule (inconsistent with `H2_k_accum_hidden`, which declares `k_accumulate: true`). Fixed
in `merlin/contract/capsules/generate_corpus.py` (B0 `modes += k_accumulate:True`) and the two generated
B0 files. Effect: B0's trace must now show an accumulate-onto PRELOAD for the 2nd K-tile. This is free
for a correct WS K-tiling backend (the raw_baseline rb_pilot_0002 submission already satisfied it).
**Both arms must grade B0 with this refinement present.**

### 3.8 Integrity (`integrity_policy.md`) — identical gate for both arms
`integrity_exempt:false`; no merlin runtime-reference/simulator imports, no `reference_outputs`, no C
compute kernel, no copied/called bareMetalC, no high-level Gemmini C lib (`tiled_matmul_auto`) as the
answer, no embedded/hardcoded outputs, no reading hidden/goldens/prior backends. Must emit BOTH the
command buffer AND the LLVM/RoCC lowering.

---

## 4. Grading pipeline — identical for both arms

Module: `merlin/python/merlin/targetgen/capsule_grade.py::grade(package_dir, capsules_root, runs_root,
labels, contract)`. It loads+integrity-scans+builds the package once, runs each capsule
(`capsule_runner.run_capsule`: parse→target→cb(schema)→llvm→RV64→rocc_decode→trace_check→L0/L1/L2),
and writes `score_capsule.json` (+ per-capsule `capsule_result.json`, `generated/instruction_trace.json`,
`numeric_report.yaml`). Official harness: `scripts/grade_agent_run.py --run-dir <run> --arm <arm>
--capsules <root>` → **public/dev phase → freeze → hidden phase** (refuses if the submission changed
after freeze). Same `bench_contract` + same `oracle_adapters` (spike+verilator) for both arms.

---

## 5. Experiment protocol — how runs are conducted (must match on both sides)

1. **Stage** a fresh workspace from exactly one bundle (allowed only; answers withheld). Sandbox note
   below.
2. **Launch** the agent non-interactively, identical invocation:
   `claude --print --model claude-opus-4-8 --effort high --permission-mode bypassPermissions
   --output-format stream-json --verbose < TASK.md`, cwd = workspace. Telemetry parsed from the
   stream-json `result`/`usage` events → `cost_time_toolcalls.yaml` (wall, api, turns, tool_calls,
   tokens in/out/cache, cost).
3. **Iterate-to-pass with a redacted QA gate** (this experiment's mechanism — see §6). Sum telemetry
   across rounds = total effort to converge. Same round cap + same per-round wall cap for both arms.
3a. **Finalize turn (both arms).** The multi-round relaunch grades AFTER the agent exits, so the
   converging round's agent never saw a passing verdict and would leave a stale "not yet passing"
   status in REPORT.md. On `all_pass`, the driver runs ONE bounded finalize turn (passing verdict +
   "update REPORT.md/docs only, do not touch code"), re-grades, restores the pre-finalize submission
   if the turn regressed the package, and guarantees the frozen report's final status line matches the
   verdict (programmatic stamp as last resort; `qa_loop_summary.finalize` records which). Identical
   for both arms.
4. **Freeze** the submission (hash), then run the **hidden** phase (operator-only). Hidden = same
   structure, different deterministic data ⇒ proves the backend computes (not memorizes).
5. **Aggregate**: `scripts/gen_reports.py` appends a row to `reports/comparison_table.md`
   (arm, run_id, wall, tokens, cost, tool_calls, public N/4, hidden M/3, tier, numeric, integrity,
   first-failure, iters). Cycles only as diagnostics.

**Sandbox reality (IMPORTANT, must be the same for both arms):** `bwrap` **crashes the Bun-based
`claude` binary** in this environment (SIGILL / FailedToOpenSocket). So both arms run `--sandbox none`
with isolation enforced by: (a) a **golden-masked workspace** (the `bench_contract` copy has every
`golden.yaml` / answer file dropped; `capsules/hidden` absent), and (b) a **post-run transcript audit**
that flags any genuine *read* of an answer path, and (c) hidden-capsule generalization. If one arm were
sandboxed and the other not, the comparison breaks — keep them identical.

## 6. The redacted QA gate / iterate-to-pass loop (apples-to-apples feedback)

Both arms must iterate against the **same QA signal**: a per-round verdict containing ONLY
`{capsule, status, failure_plane, trace_violations, numeric_status, mismatch_count, tiers, all_pass}`
— **no golden/expected/reference/oracle values, no command buffer, no numeric diffs.** Built by
`scripts/qa_check.py` (operator-side; full grade in an operator-only `runs_root`, then scrubbed).
Driver: `scripts/run_baseline_qa_loop.py` (golden-masked copied workspace → relaunch agent each round
with `qa/verdict.json` → grade → repeat until `all_pass` or round cap → freeze → public+hidden →
table). The merlin_assisted arm must use the **identical gate semantics** (same redaction, same
capsules, same tier set) — only its input bundle differs.

---

## 7. ⚠️ CRITICAL coordination items — both sessions must agree

1. **Use the SAME (patched) grader.** I fixed two **blocking** grader bugs in `$REPO` that both arms
   must be graded under (they strengthen, never weaken, the grader):
   - `targetgen/contract/schemas.py::validate` — schemas carry a *relative* `$id`, so jsonschema tried
     to fetch `#/$defs/operand` and **crashed `trace_check` for every capsule with rs1/rs2**. Fix:
     pop `$id` before validate.
   - `targetgen/rocc_decode.py` — regexes hard-coded **numeric** SSA names (`%(\d+)`), so any backend
     emitting **named** SSA values (`%c0`, `%w`) decoded as all-UNKNOWN and failed trace_check
     unfairly. Fix: accept any MLIR SSA id `%([A-Za-z0-9_.$-]+)`, key the ssa table by name.
   → The Merlin-assisted run **must be graded with these fixes present** (same git state), else the two
   arms are scored by different graders. If the Merlin backend emits numeric SSA names it was passing
   by luck; with the fix, naming no longer matters — which is the point.
2. **Same pilot scope + same capsule subset:** A0_config_smoke, A2_single_tile_matmul, A4_acc_scale_i8,
   B0_quantized_linear_i8 (public) + H0/H1/H2 (hidden). Grade both arms with `scripts/pilot_capsules/`.
3. **Same launch + telemetry + freeze + hidden + table.** Identical model/effort/flags; telemetry from
   the same stream-json parser; freeze before hidden; one comparison_table.
4. **Same integrity gate** (§3.8) and **same redacted QA semantics** (§6). The Merlin arm may *author*
   with Merlin tools but must not get richer grading feedback than the redacted verdict.
5. **Cycles are diagnostic only** on both sides — do not optimize for them, do not rank on them.

### Must-be-IDENTICAL vs MAY-differ
- **Identical:** contract (`merlin/contract/`), 4 entrypoints + manifest schema, merlin_iface input,
  command-buffer ABI, kernel ABI, RoCC decode expectations, oracle ladder + pass rule, trace-class
  check, integrity policy, grader (patched) + capsule subset, launch/telemetry/freeze/hidden/table,
  QA-gate redaction, sandbox mode.
- **May differ (the independent variable):** the agent's input bundle (Merlin authoring tools vs none),
  and consequently *how* the backend is authored and its internal design — as long as the submitted
  package satisfies every shared interface above and passes the integrity scan.

## 8. Reproduce (both arms; merlin side swaps `--run-id`/bundle via the arm)
```
cd $EXP/scripts
export PATH=$REPO/third_party/llvm-install/bin:/path/to/chipyard-autocomp/install/bin:$PATH
export MERLIN_MLIR_INSTALL=$REPO/third_party/llvm-install
export MERLIN_CLANG=/path/to/merlin-iree/build/host-merlin-release/install/bin/clang-23
# raw_baseline (this session):
python3 run_baseline_qa_loop.py --run-id rb_pilot_0002 --model claude-opus-4-8 --effort high \
        --max-rounds 8 --round-timeout 2700 --qa-timeout 1200 --sandbox none
# merlin_assisted: same driver/flags, arm=merlin_assisted bundle (the loop driver currently hardcodes
#   ARM="raw_baseline"; parameterize ARM before running the Merlin arm so staging uses the
#   merlin_assisted bundle — everything else stays identical).
python3 -m capsule_grade ...        # or scripts/grade_agent_run.py for the official frozen record
python3 gen_reports.py              # append the row to reports/comparison_table.md
```

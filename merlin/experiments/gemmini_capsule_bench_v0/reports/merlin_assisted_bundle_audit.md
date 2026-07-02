# merlin_assisted bundle audit (pre-launch)

**Scope:** validate that the `merlin_assisted` arm is graded by the *same* substrate as `raw_baseline`
and differs only in its authoring-tool bundle, and that the bundle leaks no answers, no hidden data, no
prior backends, and no callable oracle route. Produced before any real `merlin_assisted` run. The
backend under grading is still hand-authored elsewhere — this audits the **arm setup**, not a result.

Sources: `input_bundles/merlin_assisted_public_v0/input_bundle_manifest.yaml`,
`input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml`, `../../COORDINATION.md`,
`../task/TASK_pilot.md`, `results/gemmini/experiment_preflight_report.md` (GO_FOR_PILOT), and an Explore
audit of the allowed Merlin tool dirs.

## 1. Freeze-awareness / parity check (same contract as raw_baseline)

Both arms compile the same capsules through the same grader; the *only* declared difference is the
extra authoring tools in the merlin bundle.

| Dimension | raw_baseline | merlin_assisted | identical? |
|---|---|---|---|
| Frozen contract `bench_contract/` | allowed (ro) | allowed (ro) | ✅ |
| Pilot capsules | A0/A2/A4/B0 (+ hidden H0/H1/H2) | same | ✅ |
| Numeric policy (exact int; `acc_scale=clamp_i8(round_half_to_even(acc*scale))`) | contract | contract | ✅ |
| QA verdict structure (`qa_check.py`, redacted) | shared | shared | ✅ |
| Hidden protocol (post-freeze; repair disabled) | `grade_agent_run.py` | same | ✅ |
| Capsule grader (`capsule_grade.py` → `capsule_runner`) | shared | shared | ✅ |
| Trace checker (`trace_check.py` + `rocc_decode.py`) | shared | shared | ✅ |
| Integrity rules (`integrity_exempt:false`) | shared | shared | ✅ |
| Runtime ABI (4 entrypoints, kernel ABI, command-buffer ABI) | shared | shared | ✅ |
| Launcher + sandbox | `run_baseline_qa_loop.py`, `--sandbox none` | same (via `--arm merlin_assisted`) | ✅ |
| **Authoring tools (independent variable)** | none | `synthesize/`, `generate/` (−`runtime_adapter.py`), `xdsl_dialects/` (−`lowering/`), `interface_emit.py` | **intended diff** |

No blocking inconsistency in the shared contract. The grader, schemas, capsules, and QA semantics are
untouched by this round (only the merlin bundle config + the arm-aware launcher changed).

## 2. Allowed files (summary)

`bench_contract/` (+ isa/layers/model_slices public-dev capsule inputs), public Gemmini headers
(`gemmini.h`, `gemmini_params.h`), the task dir, the LLVM/MLIR-23 toolchain, and the Merlin **authoring**
tools: `targetgen/synthesize/`, `targetgen/generate/` **(minus `runtime_adapter.py`)**,
`xdsl_dialects/` **(minus `lowering/`)**, `targetgen/contract/interface_emit.py`.

## 3. Denied files (summary)

`merlin/runtime/reference.py`, `merlin/runtime/simulator.py`, **`generate/runtime_adapter.py`**,
**`xdsl_dialects/lowering/`** (the two oracle-callable helpers), `merlin_native_v0/`,
`agent_spec_v0_mlir_oot/`, `agent_spec_v1_mlir_oot/` (and `hand_smoke_oot/` via the grader-side audit),
`capsules/hidden/`, `grader_private_v0/`, and `runs/`.

## 4. Findings

### 4.1 Suspicious allowed paths — RESOLVED (the central finding)
An Explore audit of the allowed tool dirs found **two callable oracle routes** the redacted-QA-only raw
arm lacks:
- `merlin/python/merlin/targetgen/generate/runtime_adapter.py:141` — emits a `semantics.py` template
  containing `from merlin.runtime import simulate as _simulate, reference_outputs as _reference`, i.e.
  a generated `reference(command_buffer)` that returns golden outputs.
- `merlin/python/merlin/xdsl_dialects/lowering/pipeline.py:120,123` — `execute()` does
  `from merlin.runtime import outputs_match, reference_outputs, simulate` then returns a `correct`
  verdict; it is re-exported by `lowering/__init__.py:16`.

The scaffold/dialect generators themselves are **SAFE**: no finished gemmini backend, no `.insn r 0x7b`
emission, no golden/expected outputs embedded (verified). `interface_emit.py` is clean (imports only
`re`/`typing`).

**Mitigation applied this round (merlin-arm config only; no shared grading change):**
1. **Bundle tightened** — `runtime_adapter.py` and `xdsl_dialects/lowering/` added to `denied`
   (deny-wins inside allowed dirs). The launcher stages the allowed tool dirs **minus** these
   sub-paths (`assemble_copy_workspace` copy-minus), so the workspace carries no pointer to them.
2. **Transcript audit strengthened** (`run_baseline_qa_loop.py`, applies to both arms, strictly
   additive/detection-only): now flags dotted oracle **use** in agent-authored code/inline-python
   (`from merlin.runtime`, `reference_outputs(`, `pipeline.execute(`, `outputs_match(`) and reads of
   the oracle sub-paths — not just slash-path reads. Arm-aware so the merlin arm's *allowed* tools are
   not mis-flagged.
3. **Integrity scan** on the submission remains the final gate.

**Honest limitation:** under `--sandbox none` (the mode both arms run — bwrap crashes the `claude`
binary here) the real `merlin` package is importable from disk, so the *load-bearing* boundary is
(2)+(3), not the bundle masking; (1) removes the in-workspace convenience pointer and becomes real
masking under any future bwrap-capable environment. This is the same isolation model already accepted
for the raw arm (the reference oracle has always been reachable on disk under `none`; isolation = golden
mask + transcript audit + integrity scan). The merlin tooling does not lower that bar — it is brought up
to it.

### 4.2 Are prior backends visible? **No.**
`agent_spec_v0/v1_mlir_oot`, `hand_smoke_oot`, `merlin_native_v0` are all denied and carry
`CANARY_FORBIDDEN.txt` guards (pre-flight confirmed they are invisible in the staged workspace).

### 4.3 Is hidden data visible? **No.**
`capsules/hidden/` (with `CANARY_HIDDEN.txt`) and all `golden.yaml`/`expected_command_buffer*` are
denied and golden-masked out of the copied workspace; the hidden capsules are installed only by the
operator post-freeze.

### 4.4 Is final-artifact integrity enforceable? **Yes.**
`integrity_exempt:false`; the non-exempt integrity scan rejects merlin runtime/reference/simulator
imports, `reference_outputs`, copied/called kernels, and embedded outputs; the strengthened transcript
audit catches authoring-time oracle use; the post-freeze hidden phase catches memorization.

## 5. Verdict

The merlin_assisted arm uses the identical benchmark contract, capsules, numeric policy, QA redaction,
grader, trace checker, integrity rules, runtime ABI, launcher, and sandbox mode as raw_baseline. The
only difference is the authoring-tool bundle, and its single material risk (callable oracle helpers) is
closed by bundle tightening + a strengthened, arm-aware transcript audit + the integrity scan.

**GO_FOR_MERLIN_PILOT**

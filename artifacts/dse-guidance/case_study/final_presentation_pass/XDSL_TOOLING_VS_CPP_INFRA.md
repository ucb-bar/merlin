# The non-CIRCT Merlin tooling — what the Python/xDSL arm has that the C++ arm doesn't

*Grounded in the actual `capsule_bench_v0` A/B artifacts, the real tooling source, and the measured run
results. Companion to `CIRCT_RTL_ARTIFACT_JOURNEY.md` (which covers the CIRCT moat separately).*

This answers one precise question: **between the two middle arms — "C++ + Merlin infra" and "Python (xDSL)
+ Merlin infra" — what is the difference beyond the implementation language?** The short answer: a real,
tested **Python authoring kit** (parser + command-buffer builder + generic transforms + typed dialects +
verifier + plan synthesizers). The honest answer also says **what is and isn't measured** — and on the
measurement, the picture is more humbling than the tooling inventory suggests.

---

## 1. The four arms (run families under `experiments/gemmini_capsule_bench_v0/runs/`)

| # | Arm | Run family | What it's given |
|---|---|---|---|
| 1 | raw C++ | `raw_baseline/rb_*` | LLVM/MLIR-23 toolchain + public ISA headers only |
| 2 | **C++ + Merlin infra** | `cpp_merlininfra/rbinfra_*` | + generic C++ OOT scaffolders (`generate/{mlir_scaffold,llvm_plan,target_repo}`) + shared runtime/contract |
| 3 | **Python (xDSL) + Merlin infra** | `merlin_assisted/merlin_*` | + the **Python authoring kit** (§3) |
| 4 | Python + infra + CIRCT | `merlin_assisted/merlincirct_*` | + the CIRCT moat (`rtl/`, `facts.json`) — see the CIRCT journey doc |

Arms 2 and 3 are deliberately constructed so arm 2's Merlin allow-set is a **strict subset** of arm 3's
(verified by `scripts/verify_no_cheat.py`, "kit parity" check). So **arm-2-vs-arm-3 isolates exactly the
Python authoring kit** — both already share the generic C++ scaffold + shared runtime/contract ABI.

---

## 2. The difference, by bundle design (allow/deny sets)

Source of truth: the two bundle manifests.
- `input_bundles/cpp_merlininfra_hwbringup_v0/input_bundle_manifest.yaml`
- `input_bundles/merlin_assisted_hwbringup_v0/input_bundle_manifest.yaml`

The C++ infra arm's manifest states it plainly: *"It does NOT get the Python authoring kit
(`oot_starterkit/synthesize/xdsl_dialects` — what the merlin run had), a pre-built `merlin_iface` input
dialect, the IRDL spec/generator … or the CIRCT RTL moat."* Both arms **do** get
`generate/{mlir_scaffold, llvm_plan, target_repo}` and the shared runtime/contract.

So, beyond language, arm 3 (Python) is granted these five things arm 2 (C++) is denied:

| Granted to Python arm only | Path | Denied to C++ arm because |
|---|---|---|
| OOT starter kit | `targetgen/oot_starterkit/` | "Python authoring kit — C++ arm" |
| Typed xDSL dialects | `xdsl_dialects/` | "xDSL dialect framework — C++ arm" |
| Plan synthesizers | `targetgen/synthesize/` | "Python synthesis tooling" |
| xDSL scaffold generator | `generate/xdsl.py` | "xDSL scaffold generator — C++ arm" |
| Pre-built iface parser | `contract/interface_emit.py` | C++ arm derives the input dialect from generated ODS |

---

## 3. What that tooling actually is (real code, with tests — verified)

All of the following was read in source and is **working code, not placeholder** (tests named per item).

### 3a. `targetgen/oot_starterkit/` — the hw-agnostic authoring kit
Its own header states the rationale: *"the abc4 analysis showed agents rebuilt ~570 LOC of
target-INDEPENDENT plumbing … this kit provides exactly that plumbing + generic compiler transforms — and
NOTHING target-specific."* Six modules:

- **`iface.py → parse_interface()`** — wraps the tested `contract.interface_emit.parse_interface_mlir`
  (the frozen-grammar parser). Real one-liner pass-through; the agent doesn't rebuild a parser.
- **`cmdbuf.py → CommandBufferBuilder`** — accumulates tensors/commands and **validates against
  `merlin/contract/schemas/command_buffer.schema.json` before writing** (`jsonschema.Draft7Validator`).
  Removes the serializer AND the `command_buffer_schema` failure plane. (Test:
  `test_bench_contract.py::test_command_buffer_schema_fails_closed`.)
- **`transforms.py → im2col / tile_to_dim`** — real, target-agnostic transforms. `im2col` computes the
  output dims + (out_h·out_w, k) matrix shape + a recipe; `tile_to_dim` is the standard
  `for mo/no/ko in range(...,DIM)` systolic tiling. The agent still maps the resulting matmul to *its*
  target's instructions.
- **`dialect.py → parse_to_verified_ir()`** — loads the typed `merlin_iface` xDSL dialect, parses the input
  into a `ModuleOp`, and calls `module.verify()` so a broken graph raises at parse time (the C++-verifier
  equivalent).
- **`verify.py → validate / verify_module / structural_checks`** — runs xDSL's native verifier; also a
  `legal_functs()` that parses the **public** `gemmini.h` `#define k_* N` table (NOT `facts.json` — that's
  the CIRCT moat) and structural checks (decode-clean, config-before-use). Explicitly the Python equivalent
  of MLIR's free verifier.
- **`scaffold/`** — structure-only skeleton (empty dialect + 4 entrypoints + verifier hooks; **no op
  lowering** — the agent writes all lowering).

**Anti-cheat (verified):** grep found **zero** Gemmini-specific content in the kit — no funct table, no
opcodes, no goldens, no target op-lowering. Identical for every arm allowed it.

### 3b. `xdsl_dialects/` — five real typed dialects
`interface` (resident tensors, accumulators, commits…), `contract` (facts/obligations/legality), `runtime`
(devices/buffers/command-buffers), `schedule` (chosen decisions), `dse`. Real IRDL ops/types with verifiers,
e.g. `interface.ResidentPackOp` has a `verify_()` enforcing layout match. (Tests: `test_xdsl_interface.py`,
`test_xdsl_contract.py` — build / verify / round-trip, with negative cases.)

### 3c. `targetgen/synthesize/` — real plan synthesizers
`dialect_plan.py`, `llvm_extension_plan.py`, `runtime_adapter_plan.py`, `target_contract.py`,
`zephyr_plan.py`. Each emits schema-valid plan dicts (`toy_npu` fully concrete; real targets get a
`requires_human_review` conservative skeleton). (Test: `test_contract_validation.py`.)

### 3d. `generate/xdsl.py` vs `generate/mlir_scaffold.py` — both real emitters
`xdsl.py` emits a working IRDL Python dialect (the Python arm's scaffold); `mlir_scaffold.py` emits idiomatic
ODS/TableGen + C++ verifier + CMake (the scaffold **both** the C++ and Python arms may use to stand up the
package). (Tests: `test_xdsl_contract.py`; generated-target round-trips.)

**Bottom line on the tooling: it is real, tested, and genuinely target-agnostic.** On paper, that is the
arm-2-vs-arm-3 difference.

---

## 4. What actually happened in the measured runs

> **Provenance note.** The authoritative results are the runs the live trajectory plots consume —
> `scripts/gen_trajectory_v2.py::ARMS`: `raw_baseline/rb_abc11`, `cpp_merlininfra/rbinfra_abc11`,
> `merlin_assisted/merlin_abc9`, `merlin_assisted/merlincirct_abc9`. (An earlier `abc4` batch is **stale**
> and must not be cited.) Numbers below are from each run's `cost_time_toolcalls.yaml` + `qa_loop_summary.yaml`.
> Still **N=1 per arm**; "converged" = the QA-loop iteration target of 20 capsules. The separate full
> 25-capsule (20 public + 5 hidden) L2+L3 `full_suite_audit` is the correctness grading and is not re-tabulated here.

| Arm (authoritative run) | Converged (/20) | Rounds | Cost | Tokens | Tool-calls | Active wall |
|---|---|---|---|---|---|---|
| raw C++ — `rb_abc11` | ✅ 20/20 | 5 | $147.34 | 82.0M | 442 | 3.94 h |
| **C++ + Merlin scaffold** — `rbinfra_abc11` | ❌ **17/20** | 5 | **$159.48** | 84.3M | **677** | **6.45 h** |
| **Merlin Python tooling** — `merlin_abc9` | ❌ **19/20** | 10 | **$86.43** | 45.8M | **352** | 3.46 h |
| **Merlin + CIRCT** — `merlincirct_abc9` | ✅ **20/20** | **1** | **$52.73** | **29.2M** | **137** | **1.03 h** |

### 4a. The arm-2-vs-arm-3 result (the question this doc answers)
Beyond language, the two middle arms now have a measured delta:
- **C++ + Merlin scaffold (`rbinfra_abc11`)** was the **worst** arm — did **not** converge (17/20), the most
  expensive ($159.48), the most tool-calls (677), and the longest (~6.45 h). It also did not actually invoke
  `mlir_scaffold` (it hand-wrote the CMake + dialects despite the prompt).
- **Merlin Python tooling (`merlin_abc9`)** did better on every axis — 19/20, ~half the cost ($86.43), ~half
  the tool-calls (352), ~3.46 h active — though it too did **not** fully converge (one capsule short, over
  10 rounds, with a large rate-limit wait).

So the Python authoring arm beat the C++-scaffold arm on correctness **and** effort — not merely a language
difference. (Honest caveat: N=1; neither middle arm reached 20/20, and `merlin_abc9`'s own submission still
hand-rolled much of the plumbing the kit provides — see §4c.)

### 4b. The headline across all four: CIRCT wins decisively
`merlincirct_abc9` is the standout: **20/20 in a single round**, cheapest ($52.73), fewest tokens (29.2M),
fewest tool-calls (137), ~1 h, **zero** rate-limit waits. Raw C++ also converged (20/20) but the expensive
way (5 rounds, $147, ~4 h). The two middle arms did not converge. So on these runs the ranking by
cost-to-converge is **CIRCT ≪ raw-C++**, with the two infra-only arms not converging — the opposite of the
stale abc4 picture (where CIRCT had appeared *more* expensive).

### 4c. Caveat on what the kit's value is vs isn't (still honest)
The Python kit is real and tested (§3), but `merlin_abc9` did **not** demonstrate a clean kit-amortization
win — it landed at 19/20 and still hand-rolled plumbing rather than leaning fully on `oot_starterkit`. The
decisive lever in this batch was the **CIRCT-grounded feedback** (arm 4), not the authoring kit per se. A
clean "kit fully used vs C++ scaffold" head-to-head, and N>1 for stable magnitudes, remain the pending
experiments.

---

## 5. The honest bottom line

- **What the Python/xDSL arm *has* that the C++ arm doesn't (beyond language):** a real, tested authoring
  kit — `oot_starterkit` (`parse_interface`, `CommandBufferBuilder`, `im2col`/`tile_to_dim`, typed-dialect
  `parse_to_verified_ir`, `verify`), the five typed `xdsl_dialects`, the `synthesize/` planners,
  `generate/xdsl.py`, and the pre-built `interface_emit` parser — all target-agnostic, all denied to the
  C++ arm by bundle design.
- **What is measured (authoritative abc9/abc11 runs, N=1):** the Python-tooling arm (`merlin_abc9`, 19/20,
  $86, 352 calls) **beat** the C++-scaffold arm (`rbinfra_abc11`, 17/20, $159, 677 calls) on both correctness
  and effort; and **Merlin+CIRCT** (`merlincirct_abc9`) won outright — 20/20 in one round at $52.73.
- **What is not yet cleanly isolated:** the *kit's own* amortization win (neither middle arm fully converged;
  `merlin_abc9` still hand-rolled plumbing), and N>1 for stable magnitudes. The decisive measured lever here
  was the CIRCT feedback, not the authoring kit alone.
- **Legitimacy framing:** the classic, honest reason a framework beats from-scratch is amortizing the
  target-independent plumbing (parser + serializer + dialect scaffold + verifier) that C++ rewrites every
  time. That tooling is built and real; on these runs the *measured* differentiator was CIRCT, with the
  Python kit ahead of the C++ scaffold but short of a clean amortization demonstration.

### Re-checkable sources
`scripts/gen_trajectory_v2.py` (the `ARMS` mapping); per-run `runs/{raw_baseline/rb_abc11,
cpp_merlininfra/rbinfra_abc11, merlin_assisted/merlin_abc9, merlin_assisted/merlincirct_abc9}/`
(`cost_time_toolcalls.yaml`, `qa_loop_summary.yaml`); `reports/plots/fig_traj_*` (the live plots);
`input_bundles/{cpp_merlininfra_hwbringup_v0,merlin_assisted_hwbringup_v0}/input_bundle_manifest.yaml`;
`merlin/python/merlin/targetgen/oot_starterkit/*`, `synthesize/*`, `generate/{xdsl,mlir_scaffold}.py`,
`contract/interface_emit.py`, `merlin/python/merlin/xdsl_dialects/*`; tests `test_bench_contract.py`,
`test_contract_validation.py`, `test_xdsl_interface.py`, `test_xdsl_contract.py`; gate `scripts/verify_no_cheat.py`.

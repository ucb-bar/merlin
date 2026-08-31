---
title: "The arm ladder — what tooling each arm carries, where it lives, and what it buys"
kind: reference
status: current
owner: targetgen
last_verified: 2026-08-30
related: [rtl_derived_compiler_tooling, experiment_abi, gemmini_experiment, target_agnostic_core, adding_a_target, reproducibility]
code_refs:
  - merlin/python/merlin/targetgen/tool_registry.py
  - merlin/python/merlin/targetgen/generate_bundles.py
  - merlin/python/merlin/targetgen/target_experiment.py
  - merlin/python/merlin/targetgen/bundle_grants.py
  - merlin/python/merlin/targetgen/rtl/
  - merlin/python/merlin/kernels/cca.py
  - merlin/python/merlin/kernels/cca_contract.py
  - merlin/experiments/capsule_bench/harness/launch_ab_batch.py
  - merlin/experiments/capsule_bench/harness/selfcheck_broker.py
  - merlin/experiments/capsule_bench/harness/simjob_broker.py
  - merlin/experiments/capsule_bench/harness/isa_tools_broker.py
  - merlin/experiments/capsule_bench/harness/cca_broker.py
  - merlin/experiments/capsule_bench/harness/qa_check_rtlchecks.py
---

# The arm ladder — what each arm carries

An **arm** is a rung of a fixed methodology: the same target, the same capsule corpus, the same task,
the same model, the same grader, the same sandbox — and *only* the authoring tools differ. The point of
the ladder is attribution. If arm-4 beats arm-3, the difference must be traceable to exactly one
addition, or the result means nothing.

This document is the catalog: what each rung grants, where that code lives, how it works, what it can
tell you, and what it has measurably done to experiment outcomes. For the deep `input → artifact` chain
of the RTL tools specifically, see [the RTL-derived compiler tooling](../design/rtl_derived_compiler_tooling.md);
this page is the wider, per-arm view and the mechanism behind it.

---

## 1. One catalog, not two

`merlin/python/merlin/targetgen/tool_registry.py` is the single place that says what a rung adds.

It exists because the rungs used to be declared in two unrelated places that could not be varied
together: file-read grants came from per-rung literal lists inside `generate_bundles`, and the
interactive (brokered) tools were gated by *a substring test on the arm name* inside the experiment
driver. "This arm, plus exactly one more tool" was therefore not expressible — it needed a hand-authored
bundle manifest (which then bypasses the generator the readiness gate compares against) **and** a driver
edit. An ablation you cannot express is an ablation nobody runs.

Now a rung is a **set of tool names**, and an ablation cell is that set plus or minus one element:

```python
_ASSISTED = ("merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools")
ARM_TOOLS = {
    "raw_baseline":     (),
    "cpp_merlininfra":  ("cpp_oot_generators",),
    "merlin_assisted":  _ASSISTED,
    "merlin_rtlchecks": _ASSISTED + ("rtl_generators", "rtl_facts"),
    "merlin_eqsat":     _ASSISTED + ("eqsat_seam",),
}
```

Two properties are load-bearing:

- **The ladder is nested.** Every set is a superset of the one before it, which is what makes a delta
  attributable to one addition.
- **Not everything is ablatable.** `merlin_infra` (`merlin/python/merlin/common/`) is marked
  `ablatable=False` because every other granted tool imports it. Dropping it does not remove *a*
  capability, it silently removes several — `targetgen/rtl/facts.py` opens with
  `from merlin.common.paths import artifacts_dir, targets_dir`, so without the grant the RTL generators
  die in the sandbox with `ModuleNotFoundError`. This was measured across three live runs (codex 6 hits,
  codex2 6, nemotron 5): every model tried the granted generators, failed, and either worked around them
  or stopped reaching for them — so the treatment was partly unavailable to all of them and the
  arm-4-vs-arm-3 contrast was *understated*. `arm_tools(..., drop=("merlin_infra",))` now raises rather
  than producing a cell that measures the wrong thing.

`arm_tools(arm, add=..., drop=...)` resolves a cell; `cell_suffix()` names it so a run directory records
which cell produced it, and is empty when nothing was varied — which is what keeps every default bundle
id, and therefore every existing run path, unchanged.

---

## 2. The common substrate — what every arm gets

Deliberately **not** in the registry: the substrate. Ablating it would not measure a Merlin capability,
it would measure whether the benchmark is runnable at all.

Measured for `radiance` (`generate_bundles`, arm-1's complete allow set — 9 entries):

| grant | what it is |
|---|---|
| `merlin/contract/` | the frozen ABI v0.1 |
| `merlin/contract/capsules/<target>/isa/` | the capsule corpus (the task) |
| `merlin/contract/capsules/<target>/model/`, `.../model_slices/` | corpus siblings — whole-model capsules |
| the target's declared ISA header(s) | shared hardware spec |
| `experiments/capsule_bench/targets/<target>/task/` | the task statement |
| `third_party/llvm-install/` | LLVM/MLIR 23 toolchain |
| the hwbringup set, mounted **as** the target name | RTL + ISA + README + one example kernel |
| `agent_selfcheck.py` | the redacted self-check shim |

Plus two brokers every arm reaches (see §5): the **redacted self-check** and the **async sim-job**
runner. Both are substrate, not treatment — a raw baseline that could not run a simulator would not be
a baseline, it would be a strawman.

The corresponding arm-1 deny set is short and total:

```
merlin/                                        Merlin internals (no tools for the raw arm)
out/artifacts/targets/<target>/<prior>/        prior backend / exemplar (answer surface)
merlin/contract/capsules/<target>/hidden/      hidden capsules + goldens
experiments/.../input_bundles/grader_private_v0/   grader-private
experiments/.../runs/                          prior submissions
```

Arm-1 denies `merlin/` wholesale; every higher rung re-grants a named subset of it. That is the whole
mechanism — the ladder is a series of exceptions carved out of one denial.

---

## 3. The rungs

Measured grant deltas, `generate_bundles` on `radiance` (allow-set sizes; each delta is exactly the
tools that rung adds):

| arm | driver | bundle id | allow | deny | adds |
|---|---|---|---|---|---|
| **arm-1** `raw_baseline` | `run_baseline_qa_loop.py --arm raw_baseline` | `raw_baseline_hwbringup_v0` | 9 | 6 | — (substrate only) |
| **arm-2** `cpp_merlininfra` | `run_baseline_qa_loop.py --arm cpp_merlininfra` | `cpp_merlininfra_hwbringup_v0` | 12 | 19 | `cpp_oot_generators` (+3) |
| **arm-3** `merlin_assisted` | `run_baseline_qa_loop.py --arm merlin_assisted` | `merlin_assisted_hwbringup_v0` | 22 | 11 | `merlin_infra`, `xdsl_kit`, `cca_spine`, `isa_tools`, `cca_tools` (+13 paths) |
| **arm-4** `merlin_rtlchecks` | `run_rtlchecks_qa_loop.py` | `merlin_assisted_rtlchecks_hwbringup_v0` | 24 | 9 | `rtl_generators`, `rtl_facts` (+2) |
| **eqsat** `merlin_eqsat` | `run_eqsat_qa_loop.py` | `merlin_assisted_eqsat_hwbringup_v0` | 24 | 11 | `eqsat_seam` (+2) |

Note the deny column: arm-3 has **more** denials than arm-4 (11 vs 9), because arm-3 explicitly denies
the two paths arm-4 grants:

```
merlin/python/merlin/targetgen/rtl/            CIRCT RTL generators (CIRCT arm only)
merlin/<backend_package_dir>/contracts/rtl_facts/   RTL facts (CIRCT arm only)
```

The contrast is symmetric by construction: what one rung allows, the rung below names as denied. A
grant that quietly appeared in both would make the arms incomparable, and this is what makes that
visible.

The **eqsat** arm shares arm-3's denials on purpose. An arm that also gained the RTL facts would differ
in two ways and its result would not attribute to the equivalence seam.

Arms are launched by `launch_ab_batch.py` (`ARMS` table) or fanned out over Ray by `chia_ab_batch.py`.
Both merlin arms write to `out/runs/<target>/capsule-bench/merlin_assisted/<run-id>/` — the rtlchecks
driver drops a `TRACK_RTLCHECKS` marker so they stay distinguishable downstream. **Run directories are
keyed by the bundle track, not the arm name.**

---

## 4. The tools

### `cpp_oot_generators` — arm-2
`merlin/python/merlin/targetgen/generate/{mlir_scaffold,llvm_plan,target_repo}.py`

Generic C++ out-of-tree backend generators: an MLIR scaffold, an LLVM lowering plan, a target repo.
Arm-2 exists to separate "Merlin gave you infrastructure" from "Merlin gave you a compiler abstraction" —
it is the status-quo C++ OOT path with scaffolding, and it is kept a strict subset of the xDSL arm
(hence its 19 denials, the `_CPP_DENY_AGN` block).

### `merlin_infra` — arm-3+, not ablatable
`merlin/python/merlin/common/`

Path/YAML/schema helpers every other granted tool imports. Not an oracle, grader or answer surface —
granting it widens no moat, and withholding it disables the tools above it. See §1.

### `xdsl_kit` — arm-3+
`targetgen/synthesize/`, `targetgen/generate/`, `xdsl_dialects/`, `targetgen/contract/interface_emit.py`,
`targetgen/contract/linalg_iface.py`, `targetgen/oot_starterkit/`

The xDSL authoring kit: dialect synthesis, the generators, the dialect definitions, the interface
emitters, the out-of-tree starter kit. This is the "author a target dialect in Python rather than C++"
treatment.

### `cca_spine` — arm-3+
`merlin/python/merlin/kernels/{cca,cca_compare,cca_contract,action_catalog,microkernel}.py`,
`targetgen/rtl_backend.py`

The **Common Compute Abstraction** — the target-agnostic vocabulary every source (framework kernel, our
compiler output, DSE view) decompiles into, at any level (`asm | source_ast | mlir | dse`). It is the
*where/how* of modifying a compiler: extract a CCA, diff two, check the CCA↔action bijection, walk the
escalation ladder, author a microkernel.

Critically it is **not** RVV-specific and not keyed on target name. A region carries a `backend` tag and
only the relevant **facets** are populated, and which facets those are is a property of the target's
*engines* (`merlin.kernels.engines`), not of its name — a target is a set of compute engines:

- `vector` (a lane engine: RVV, or an accelerator's VPU) — SEW, LMUL, VL strategy, tail
- `spatial` (an array engine, systolic **or** outer-product) — PE-array dims, dataflow, accumulator residency
- `simt` (a threads-of-control engine) — warps, shared-memory residency, barriers, divergence
- engine-agnostic `compute` and `memory` facets

Measured justification for the engine framing: 62% of atlas's engine-driving expert kernels touch its
*vector* engine, so describing it by one facet because it is "an NPU" would leave most of its corpus
unexpressed.

`cca_contract` enforces a machine-checkable invariant, scoped per backend: every CCA field is classified
`IDENTITY`/`LEVER`/`BACKEND_STUB` (capture-completeness), and for a backend the set of `LEVER` axes
equals the set of axes `action_catalog` actually routes to a compiler seam (exposure-completeness). A
lever with no route — the abstraction promising something the compiler cannot change — is a bijection
break, and so is a routed axis with no field.

### `rtl_generators` — **arm-4 only**
`merlin/python/merlin/targetgen/rtl/`

The delta that defines arm-4, alongside `rtl_facts`. Contents:

- **`circt_introspect.py`** — deterministic, target-agnostic RTL fact extraction. Prefers mlc's
  target-agnostic RTL discovery (the version-matched core HW dialect: decoder-derived legal-opcode set,
  discovered mesh DIM, memory capacities) and falls back to a FIRRTL grep + HW-port parse only for a
  target that actually ships those artifacts. **Every path is resolved from the `target` argument.**
  Every fact carries `evidence` — the exact RTL/source token it came from. No LLM: the hardware is the
  source of truth.
- **`facts.py`** — the single place mapping target → its facts artifact and purgeable scratch dir. RTL
  facts are a *generated artifact*, cached at
  `out/artifacts/cache/rtl_introspect/<target>/facts.json` (gitignored) and regenerated on demand by
  `ensure_facts` when cold. There is no committed `facts.json` pin; a target's only tracked definition is
  its reviewed yaml.
- **`gen_isa_module.py`** — promotes the facts from a post-hoc *checker* into a *generator*: emits a
  self-contained encoder module the backend builds on, so the agent writes only op-lowering (tiling,
  im2col), not the ISA encoding it would otherwise re-derive from headers. The two failure classes the
  checker caught — unknown custom-3 instructions and use-before-config — are exactly encoding mistakes,
  and generating the legal funct table plus an ordering-checked emitter makes them structurally
  impossible.
- **`gen_rtl_digest.py`** — renders the facts as one human-readable spec sheet (module map, funct table,
  memory map, datapaths, dataflow, legal-sequencing rules) so the arm reads **one digest instead of
  crawling 55 RTL files**. Exploration was the arm's biggest token sink in abc4 (~67 find/grep/cat
  commands).
- **`gen_numeric_facts.py`** — narrows the structural screen's numeric blind spot: from the RTL datapath
  facts (input dtype, accumulator width, scale/activation semantics) it emits a checker that flags
  numeric-*shape* mistakes — wrong accumulation width, mismatched output dtype, missing scale on a scaled
  op — **without computing or comparing any golden value.** It does not certify numerics; it makes a class
  of width bugs cheap to catch.
- also: `extract_module.py`, `gen_arc_ports.py`, `gen_iface_irdl.py`, `gen_rocc_replay.py`,
  `introspect.py`, `mlc_bridge.py`, `opu_isa.py`, `replay_json_to_h.py`, `spatial_introspect.py`.

**Anti-cheat position**: all of this is RTL-derived *structural* truth — what the hardware **is**. Module
hierarchy, port widths, capacities, the legal funct set. It contains no lowering answer: no command
buffers, no goldens, no per-capsule instruction sequences.

### `rtl_facts` — **arm-4 only**
`merlin/<backend_package_dir>/contracts/rtl_facts/` (descriptor-derived)

The facts already extracted from *this* target's RTL — the generators' output, granted directly, so the
arm need not re-run extraction. Named indirectly in the registry, as the descriptor attribute
`rtl_facts_pin`, so the table stays free of target names.

> Note the indirection is real, not cosmetic: for `radiance` this resolves to
> `merlin/targets/muon/contracts/rtl_facts/` — **muon**, not radiance — because it follows the
> descriptor's `backend_package_dir`.

### `eqsat_seam` — eqsat arm only
`targetgen/contraction_egraph.py`, `targetgen/persistent_equivalence.py`

An e-graph over real IR plus the persistent equivalence store. The treatment under test is the *seam*
itself: the agent registers its own implementation as an alternative in an e-class and the extractor
chooses.

### `isa_tools` and `cca_tools` — arm-3+, brokered
No repo grant at all; see §5.

---

## 5. Brokers — the tools that cannot live in the sandbox

A brokered tool contributes **no path** to the manifest. It is staged into the workspace by the driver,
so it appears in the grant list only by its absence. There are four brokers, all in
`merlin/experiments/capsule_bench/harness/`, all following the same pattern: watch a channel dir inside
the agent's bind-mounted workspace, answer each `req_<id>.json` with a `resp_<id>.json` plus a `done_<id>`
marker, exit on a `STOP` sentinel.

| broker | channel | arms | why it must be outside |
|---|---|---|---|
| `selfcheck_broker.py` | `.qa_channel` | **all** | the redacted self-check internally needs the oracle (`merlin.runtime.reference`) to compute pass/fail, and the oracle is masked inside the sandbox |
| `simjob_broker.py` | `.qa_channel` | **all** | runs the agent's requested sims outside the box, redacted |
| `isa_tools_broker.py` | `.isa_channel` | 3, 4, eqsat | the ISA model derivation needs the target model venv, masked inside |
| `cca_broker.py` | `.cca_channel` | 3, 4, eqsat | `merlin/kernels/types.py` shadows stdlib `types` when the kernels dir is flat on `sys.path` (circular-import crash), and `regions.py` needs the unstaged `merlin.common.paths` |

**`simjob_broker` is a constrained sim-runner, and the constraint is load-bearing.** A request may only
name `{sim ∈ spike/verilator/vcs, capsules that exist under the public set, debug from a whitelist,
workers (clamped)}`. The broker maps those onto a **fixed** `agent_selfcheck.py` argv — it never execs
anything the request names. So the agent gets full sim power on its own submission and the harness
capsules, but cannot read goldens or run arbitrary shell outside the sandbox. It also holds a **global
cross-arm verilator semaphore**, so N concurrent arms do not launch N× the verilator load.

**`isa_tools` and `cca_tools` are oracle-free by construction**, and are gated to the assisted arms not
because they leak anything but because unaided raw-ISA authoring is precisely what the baseline measures.
The assembler encodes the syntax the agent chose; the disassembler and linter inspect the agent's **own**
emitted words; `check_bijection` diffs public schema against public routes; `escalation_ladder` lists the
public FLAG→KNOB→HEURISTIC→PASS→CODEGEN ladder for an axis. No capsule input, no expected output.

---

## 6. How an arm binds to a target

No arm definition mentions a target. The binding happens in three derived layers, which is what lets a
new accelerator drop a descriptor and get the whole ladder — no hand-authored, per-target YAML.

1. **The descriptor** — `merlin/experiments/capsule_bench/targets/<target>/target_experiment.yaml`
   supplies `target`, `backend_package_dir`, `rtl`, `hardware_spec`, `capsule_corpus`, `toolchain`,
   `grading`, `answer_surfaces`. `generate_bundles` reads it and emits the target-specific half of every
   bundle (ISA headers, hwbringup set, corpus, and the derived `rtl_facts` / irdl pins and prior-backend
   deny surfaces).
2. **The registry** — supplies the target-*agnostic* half: literal merlin module paths, byte-identical
   for every target.
3. **The facts** — `circt_introspect` extracts the hardware truth from the target's own elaborated RTL at
   run time, resolved from the `target` argument, cached per target, every fact carrying its evidence
   token.

Six targets currently carry descriptors: `radiance`, `gemmini`, `mx_gemmini`, `atlas`, `saturn_opu`,
`saturn_opu_rvv`.

Where a target legitimately differs, it differs *in the descriptor*, and the effect is visible in the
generated bundle. Compare the substrate for two targets:

```
radiance  ISA header:  .../hwbringup_radiance_v0/isa_include/isa_definition.py   (a self-hosted ISA)
gemmini   ISA headers: .../gemmini-rocc-tests/include/{gemmini.h,gemmini_params.h}  (RoCC C headers)

radiance  corpus: merlin/contract/capsules/radiance/isa/  + model/ + model_slices/
gemmini   corpus: merlin/contract/capsules/isa/           + layers/ + model/ + model_slices/
```

Grants are validated by `merlin/python/merlin/targetgen/bundle_grants.py` and gated by
`build_tools/scripts/check_bundle_grants.py` (pre-commit + CI): every declared grant must resolve to
bytes that actually ship, distinguishing *absent* from *generated-on-demand* from *reachable only via
committed machine-local bytes*. A manifest that promises a tool the stager cannot deliver would
otherwise credit an arm with a tool it never carried, silently.

---

## 7. What you can learn from a run

Per run directory (`out/runs/<target>/capsule-bench/<track>/<run-id>/`):

| artifact | what it tells you |
|---|---|
| `input_bundle_manifest.yaml` | the exact grants this run shipped with — the arm, as delivered |
| `environment.yaml` | resolved toolchain / env |
| `oracle_preflight.yaml` | `oracle_available`, `sim_via`, `verdict: GO/NO-GO` — refuses to spend on a target whose oracle cannot grade |
| `codegen_smoke.yaml` | `codegen_ok` + reason — whether the compile path emits a runnable kernel *with the correct result* before any agent round |
| `selfcheck_log.jsonl` | every brokered self-check the agent requested |
| `rounds/round_NN.prompt.txt`, `.transcript.jsonl`, `.codex_events.*.jsonl` | the full agent trace, per round |
| `qa_loop_state.yaml` | live progress — **monitor this, not the `.log`, which is block-buffered** |

Arm-4 additionally gets an **advisory `rtl_checks` block** appended to each round's verdict by
`qa_check_rtlchecks.py`: deterministic RTL-grounded structural checks (FileCheck over the emitted
dialect MLIR + decoded trace, with bounds from the extracted facts). Every `expected` is derived from RTL
facts + declared shape + ISA rules, never from a golden, and **it does not change pass/fail** — it is
per-round feedback, not a gate.

### Reading a score honestly

Always read `tier_reached`, never a bare score. The certification tiers are not interchangeable: a
capsule can be `status=pass` at L2 (functional) and still fail L3 (RTL-certified). Quote the gate you
mean — `x/N at L2` and `y/N at L3` are different claims, and the gap between them is often the
interesting number (see §8).

---

## 8. Measured effect on experiments

These are point-in-time results, not durable facts. They are recorded here because "what does the
tooling buy" is the question the ladder exists to answer, and the answer has changed as harness defects
were fixed.

**radiance v11 (2026-08-24, gpt-5.6-sol high, frozen champions re-graded).** Identical capability,
2× cheaper to reach:

| | arm-1 | arm-4 |
|---|---|---|
| public | 33/36 | 33/36 |
| compiler-earned | 27 | 27 |
| held-out | 9/12 (9/9 winnable) | 9/12 (9/9 winnable) |
| **rounds to reach it** | **5** | **0** |
| wall / commands | 8.17 h / 511 | 4.00 h / 211 |

**radiance v12 (2026-08-25/26, 3 rounds each).** First run where the arms diverge, after a barrier
defect was fixed. Supersedes v11's "identical" conclusion:

| gate | arm-1 | arm-4 | gap |
|---|---|---|---|
| headline (L2) | 31/36 | 33/36 | +2 |
| compiler-earned | 25 | 27 | +2 |
| **L3-gated** (`status=pass` **and** `tiers.L3=pass`) | **26/36** | **31/36** | **+5** |

Cost: arm-1 171.6M tok / 508 calls / 7.08 h; arm-4 102.0M / 325 / 5.79 h. The two capsules that separate
them are both fp16 and both public (`RP10_gemv_batched_fp16_pt`, `RP17_k_chain_fp16_pt`); the fixture set
is identical across arms, so this is not a grading artifact.

**The most important line in that table is the L3 row.** At L2 the tooling looks worth +2. Under the
RTL-certified gate it is worth +5 — the arms' L2 scores understate the difference, because structurally
wrong code can pass a functional check and not a hardware one. An experiment that only ever quotes the
headline will systematically under-measure the treatment.

**gemmini abc9 (June 2026, Opus-4.8 via Claude Code).** 19/20 grader-verified and L3/verilator-certified
on arm-3, spanning the full corpus. Establishes the realistic ceiling and refutes an earlier "capability
wall" claim: the task is agentically solvable, so a 0/N is evidence about the model or the harness, not
about the task.

**atlas arm-4 (0/11, $52, 12 rounds).** Genuine agent skill, not broken tooling — proven by running the
shipped known-good `MatmulProgram` through the *same* oracle and getting correct output on both L2 and L3
while the agent's kernel produced all-zeros. Worth stating because an earlier atlas 0/11 *was* broken
tooling (no oracle, suite-literal blind loop), and the two are indistinguishable from the score alone.

The recurring lesson across all of these: **a harness limitation reads exactly like an agent failure.**
Check the preflight verdicts (`oracle_preflight.yaml`, `codegen_smoke.yaml`) and read a mid-run trace
before attributing a low score to the model.

---

## 9. Limits worth stating

- **Only `radiance` declares a hidden corpus.** `te.hidden_corpus()` returns `None` for `gemmini`,
  `atlas`, `mx_gemmini`, `saturn_opu` and `saturn_opu_rvv` — so their bundles carry no hidden-set deny
  entry and their runs have no held-out measurement at all. Held-out numbers exist for radiance and
  should not be assumed elsewhere.
- **The advisory block is advisory.** `rtl_checks` never changes pass/fail. Arm-4's advantage has to show
  up through the agent acting on the feedback, not through a scoring change.
- **`gen_numeric_facts` does not certify numerics.** It catches a class of shape/width bugs cheaply so
  fewer sim runs are strictly required. A sim is still what certifies numbers.
- **Brokered tools are invisible in a manifest diff.** Two bundles can have identical allow-sets and
  still differ in brokers. Check `brokers_for(arm_tools(arm))`, not just the paths.
- **Nesting is a property of paths, not of tool names.** By tool name, arm-2 is not a rung at all:
  `cpp_oot_generators` is not in `_ASSISTED`. It is nested only because its three file grants sit inside
  the `targetgen/generate/` **directory** arm-3 grants. `test_every_rung_is_contained_in_the_rung_above_it`
  now pins that directory-aware containment across arm-1 → arm-2 → arm-3 → arm-4; without it the invariant
  rested on a coincidence of spelling.
- **Exclusion uses two mechanisms, and only the effect is the invariant.** `targetgen/rtl/` is *explicitly
  denied* to arm-3 (belt and braces — it is one widened grant away from exposure), while the eqsat seam
  modules sit at a path no arm-3 grant covers and are excluded by *deny-by-default* alone. Both are
  correct. `test_a_rung_cannot_read_any_tool_the_rungs_above_it_add` therefore asserts **reachability**,
  not the presence of a denial, and covers the two grants most likely to drift: `rtl_facts` (path derived
  from the descriptor, not spelled in the registry) and `eqsat_seam`.

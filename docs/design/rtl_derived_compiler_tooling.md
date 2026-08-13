---
title: "Design: the RTL-derived compiler tooling — what it extracts, how it is used, and its honest value"
kind: design
status: draft
owner: targetgen
last_verified: 2026-08-13
related: [target_agnostic_core, compiler_plane, dialect_test_bar, expert_gap_attribution, capsule_bench]
code_refs:
  - merlin/python/merlin/targetgen/rtl/facts.py
  - merlin/python/merlin/targetgen/rtl/circt_introspect.py
  - merlin/python/merlin/targetgen/rtl/gen_isa_module.py
  - merlin/python/merlin/targetgen/rtl/gen_rtl_digest.py
  - merlin/python/merlin/targetgen/rtl/gen_numeric_facts.py
  - merlin/python/merlin/targetgen/rtl_backend.py
  - merlin/python/merlin/targetgen/rtl_check_compiler.py
  - merlin/python/merlin/targetgen/rtl_check_runner.py
  - merlin/experiments/capsule_bench/harness/qa_check_rtlchecks.py
  - merlin/experiments/capsule_bench/harness/isa_tools_broker.py
  - merlin/experiments/capsule_bench/harness/simjob_broker.py
  - merlin/experiments/capsule_bench/harness/agent_selfcheck.py
  - merlin/python/merlin/targetgen/sandbox/bwrap.py
---

# The RTL-derived compiler tooling — an honest, exhaustive accounting

This is the reference companion to the reviewer-facing narrative ("Why an RTL-derived compiler converges
faster"). It catalogs **every** arm-3/arm-4 tool as a concrete `input → what it does → generated artifact`
chain, tags each by its role in the compiler stack, and states — without spin — where the value is real and
where it is ceremony. It exists so the claim can be audited fact-by-fact, not taken on trust.

The central question it answers: *if a good ISA doc suffices for a correct compiler, why extract facts from
the RTL?* The short answer is the keystone frame below; the long answer is the per-tool catalog.

## Keystone: we generate one slice of the compiler, and RTL facts are that slice's preconditions

A full pipeline is `frontend → high-level opt → mid lowering → [target dialect + lowering passes +
encoding] → scalar regalloc/sched → assembler/linker → runtime/ABI`. Everything outside the brackets is
stock and target-independent; we reuse it. We **generate only** the bracketed backend seam:

| We generate | needs | fact source |
|---|---|---|
| target **dialect** (op set) | op *semantics* | ISA / contract — minimal RTL |
| lowering **passes** | tile to **mesh DIM**; fit **scratchpad/accumulator capacity**; emit **legal funct**; **config-before-use / fence** | **RTL-only / RTL-corrected** — docs omit these |
| **encoding** | opcode; funct; operand layout | RISC-V const + contract (not RTL); funct (RTL-corrected); ABI |

The lowering passes are where legality, resource, and sequencing constraints live — and for a *spatial*
accelerator those are **correctness** facts, not performance facts: a tile that overflows the scratchpad
does not run slow, it aliases and returns the wrong answer. That is why RTL matters even in a
correctness-only regime, and it is a compiler contribution (the passes' preconditions), not infrastructure.

**The dividing test used throughout this doc:** does a fact feed the dialect's *semantics* or a pass's
*precondition*? If it feeds neither — if no emitted pass or instruction changes because of it — it is not
adding compiler value, by definition. Tools are tagged `[precondition]`, `[semantics]`, `[feedback]`,
`[navigation]`, or `[ceremony/dormant]` accordingly.

## Tool catalog

### 1. `rtl/facts.load_facts` — the fact bundle `[precondition]`
- **Input:** the target's elaborated RTL (`firtool --ir-hw`) + its ISA definition, via mlc/CIRCT
  (`circt_introspect.build_facts` → `mlc_bridge` `circt-opt` HW graph → `comb.icmp eq` decoder fan-out).
  Cached at `out/artifacts/cache/rtl_introspect/<target>/facts.json`.
- **Does:** extracts mesh DIM, scratchpad/accumulator byte capacity, datapath dtypes, and the
  decoder-derived `funct_decode_table` (legal_funct, custom_opcode, funct3, names). Fails closed: a
  non-RoCC / SIMT target yields `"facts": {}`.
- **Artifact (gemmini, verbatim key fields):**
  ```json
  "arrays":[{"name":"mesh","rows":16,"cols":16}],
  "memories":[{"name":"scratchpad","bytes":262144},{"name":"accumulator","bytes":65536}],
  "funct_decode_table":{"custom_opcode":123,"funct3":3,"legal_funct":[0,...,24,126],
    "evidence":"...29 comparisons -> 26 legal opcodes; vs header: phantom=[25] missing=[126]",
    "header_only_functs":[25],"decoder_only_functs":[126]}
  ```
- **Value over the doc:** exactly three facts are genuinely RTL-only AND consumed — **mesh DIM**,
  **capacity**, **decoder-corrected legal set**. `custom_opcode`/`funct3` are RISC-V constants + the
  reviewed contract (not RTL); names are header-derived. The undeniable RTL-corrects-doc example: header
  phantom funct 25 / missing funct 126.

### 2. `rtl/gen_isa_module` — the generated RoCC encoder `[precondition]`
- **Input:** `facts.json` + the capability manifest's ABI block.
- **Does:** emits a Python encoder (`--header` for C++) where `Instr.__post_init__` raises on any funct ∉
  `LEGAL_FUNCT`, and `Program.finalize()` enforces config-before-use. Illegal programs are unrepresentable
  by construction — the mechanism behind faster convergence.
- **Artifact:** `LEGAL_FUNCT` includes 126, excludes 25 (the decoder correction, baked into the compiler);
  `DIM=16`. Fail-closed cross-target: atlas → `n/a: no RoCC custom_opcode … refusing to emit with a guessed
  opcode`; radiance → `n/a: not a RoCC command ISA`.
- **Served-track note:** on the public bundle this generator's *output file* is not pre-staged to the agent;
  its guarantees reach the agent when the agent runs `load_facts`/builds its encoder. `[ceremony]` as a
  pre-staged artifact; `[precondition]` as a capability.

### 3. `rtl/gen_rtl_digest` — the human-readable RTL brief `[precondition]`
- **Input:** `facts.json`. **Does:** renders `RTL_DIGEST.md` (mesh, capacities, funct table, sequencing
  rules). **Artifact:** "Tile every matmul operand to DIM=16"; capacity table; config-before-use rule.
  Build-time/CI on the served track (`[ceremony]` there), a genuine pass-precondition brief otherwise.

### 4. `rtl/gen_numeric_facts` — numeric-shape checker `[precondition, fail-closed]`
- **Input:** `facts.json`. **Does:** emits `numeric_facts.py` + `check_numeric_shapes(cb)`. **Artifact:**
  `INPUT_DTYPE='i8'`, `ACC_DTYPE='i32'`, `ACC_WIDTH_BITS=None`. The width being `None` (lane_bits absent in
  the cached facts) makes the accumulation-width check *disable itself* rather than assume — fail-closed,
  and an honest coverage gap.

### 5. `rtl_backend.derived_levers` — DSE lever surface `[navigation]`
- **Input:** live mlc discovery (`TargetProfile`: legal_opcodes, memory_map, dim). **Does:** returns
  `["spatial.dataflow"]` (+`"spatial.accumulator_resident"` if an accumulator exists). **Caveat:** needs
  *live* mlc (`MERLIN_MLC_DIR`); offline it returns `[]`. Granted to both assisted arms; only arm-4's
  addendum instructs the agent to call it.

### 6. `rtl_check_compiler` + `rtl_check_runner` — the compiled FileCheck test `[feedback]`
- **Input:** `facts.json` + a capsule's declared op/shape. **Does:** `compile_trace_checks` /
  `compile_kernel_checks` bake RTL literals into `CHECK-DAG` directives; the runner renders the agent's
  emitted RoCC trace / `kernel.S` to canonical text and runs the LLVM `FileCheck` binary over it.
- **Artifact (gemmini A2 matmul, generated):**
  ```
  // TRACE-DAG: ABI custom=0x7b funct3=0x3
  // TRACE-DAG: ILLEGAL_FUNCT_COUNT 0{{$}}
  // TRACE-DAG: MVOUT_COUNT 1{{$}}          # ceil(M/DIM)*ceil(N/DIM), DIM=16 from RTL mesh
  // TRACE-DAG: COMPUTE_PRESENT yes
  ```
  Provenance record names each source (legality ← `decoder_icmp_fanout(mlc)`, tiling ← mesh DIM). Result
  persisted to `rtl_checks.json`.
- **Honest scope:** catches ISA-legality + structural defects (illegal opcode/funct, tile-count, capacity,
  static ordering) — a cheap pre-screen a functional sim would also surface, **not** a verilator substitute.
  Bank conflicts, DMA backpressure, pipeline interlocks, X-propagation, real overflow are **verilator-only**.
  Known bug: the kernel-path tiling count is exact even for `resident_reuse` → false rejects.

### 7. `qa_check_rtlchecks` — the arm-4 advisory `[feedback, non-gating]`
- **Input:** the redacted round verdict + the compiled FileCheck (tool 6). **Does:** appends a non-gating
  `rtl_checks` block with per-finding `expected`/`got`/`fix_hint`. **This is the one load-bearing,
  default-config, agent-consumed arm-4 signal.**

### 8. `isa_tools` (broker) — asm/disasm/lint + arc `debug` `[semantics-check / arc-estimate]`
- **Real:** the `debug` subcommand runs the command buffer on the CIRCT/arcilator model compiled from RTL
  (the same L3-oracle cosim), returning per-op cycles + scratchpad/accumulator/DRAM counts + accumulator
  commits, with outputs and pass/fail verdict redacted. asm/disasm/lint are oracle-free over the derived
  `IsaModel`. **Requires a running broker** (see §sandbox).

### 9. `simjob` (broker) — async spike/verilator/vcs oracle `[cycle-accurate]`
- **Real** async dispatch; non-blocking (agent keeps working while verilator runs); returns the redacted
  verdict. Only the `--debug` symbolic flag is a no-op stub.

### 10. `agent_selfcheck` (broker) — on-demand redacted grader `[functional→cycle-accurate]`
- **Real:** builds + grades an isolated copy exactly like the between-rounds grade, redacting only the
  golden value. Granted directly in hwbringup/realistic bundles; broker-served under bwrap.

### 11. CCA spine (`cca`, `cca_compare`, `cca_contract`, `action_catalog`, `microkernel`) `[navigation]`
- **Real, oracle-free:** maps an expert-vs-ours CCA divergence to the concrete compiler seam +
  FLAG→KNOB→HEURISTIC→PASS→CODEGEN escalation ladder + forkable status. Import-granted; the only heavyweight
  tool reachable without a broker. Zero correctness signal.

## Enforcement: the sandbox is what makes the tools real

The broker tools (8–10) start only under an enforced sandbox; the safe pattern is a thin in-box shim that
forwards over a channel to a driver-side broker running the real implementation **outside** the box, so the
oracle never enters the agent's view. `sandbox/bwrap.py` builds the deny-by-default argv (system RO, answer
surfaces masked, only granted bundle files bound). **Enforced bwrap is the default** — it both seals the
answer surface and brings the broker tools live. (History: bwrap was briefly disabled after a DNS
misconfiguration made the agent's `claude` session hang; the fix binds the systemd-resolved dir, adds an
XDG runtime dir, and drops inherited `CLAUDE_CODE_*` nesting markers.) Do **not** grant `program_oracle`
for direct import — it returns unredacted outputs; the broker is the safe boundary.

## Overengineering ledger (said plainly)
- Generators 2–4: build-time/CI only on the served track — ceremony as pre-staged artifacts.
- Tools 8–10: dormant unless a sandbox/broker runs (now default-on).
- Tool 6's "verilator-only defect" framing is inaccurate — it is an ISA-legality pre-screen.
- Discarded extraction: accumulator banks / addr-width / pipeline-depth computed then thrown away;
  `ACC_WIDTH_BITS=None`.
- Load-bearing set is small: **mesh DIM, capacity, funct-legality correction, sequencing** + the advisory.

## Soft spots (reviewer kill-shots)
Confound (info vs feedback vs instruction); RTL vs encoder design (24/26 functs are header-derivable);
structure-not-semantics overclaim; generalizability (radiance derives empty facts); oracle circularity
(facts + verilator share the RTL/CIRCT flow); effect-size/variance; integrity (`sandbox=none` is
detection-not-prevention).

## Experiments that would settle it
- **X1** factorial info×feedback×instruction · **X2** doc- vs RTL-seeded legality encoder · **X3 (crux)**
  a capsule bit-wrong unless the RTL-only capacity/sequencing fact is known · **X4** doc-vs-RTL across
  gemmini/atlas/radiance · **X5** FireSim-vs-verilator + injected-wrong-DIM · **X6** component ablation ·
  **X7** N≥10 variance · **X8** fact-attribution (% of correctness-critical edits using an RTL-only fact).
X3 is the single highest-leverage artifact.

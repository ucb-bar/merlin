# The CIRCT / MLIR RTL-artifact journey

*How OSCAR Merlin turns the **real Gemmini RTL** into machine-checkable compiler artifacts — every stage,
the file it produces, the exact code that produces it, and why.*

This is the long-form, snippet-by-snippet companion to the one-slide figure
`experiments/gemmini_perf_bench/reports/fig_circt_tooling.png`. Every path, number, and snippet below is
quoted from the working tree (mostly under `merlin/python/merlin/targetgen/rtl/` and
`merlin/targets/gemmini/contracts/rtl_facts/`). Where a claim is **measured**, the measurement source is
named; where it is **designed but not yet re-measured**, it says so in those words.

---

## 0. One-paragraph thesis

The Gemmini accelerator is described in Chisel/Scala RTL. We run that RTL through **CIRCT/`firtool`** to get
MLIR (the `hw` and `arc` dialects), then run small **deterministic, no-LLM extractors** over that MLIR to
produce a single provenance-stamped artifact, **`facts.json`** — the mesh size, the memory capacities, and
the legal RoCC instruction table, each carrying the exact RTL token it came from. That one artifact then
fans out into **five consumers**: an advisory *checker*, a one-page *digest*, a generated *ISA encoder*, a
*numeric-shape* checker, and an *IRDL* dialect bridge — plus a second CIRCT track (**arcilator**) that
compiles the same RTL into a *native executable simulator*. The point: the hardware is analyzed **once, by a
compiler**, and everything downstream (the agent that writes the backend, the checks that grade it, the sim
that certifies it) reads that analysis instead of re-deriving it from headers.

---

## 1. The whole journey on one page

```
 ┌─ Chisel/Scala RTL ───────────────────────────────────────────────────────────────────────┐
 │  GemminiISA.scala (funct table)   Scratchpad.scala (smem)   AccumulatorMem   Mesh/Tile      │
 └───────────────┬──────────────────────────────────────────────────────────────────────────┘
                 │  firtool  (CIRCT)
        ┌────────┴─────────┬───────────────────────────────┐
        │ --ir-hw          │ (FIRRTL .fir + hierarchy.json) │  --ir-... → arc
        ▼                  ▼                                ▼
  gemmini_soc.hw.mlir   *.fir + top_module_hierarchy.json   cell.arc.mlir ─► gemmini.ll ─► gemmini_arc
   (CIRCT HW dialect)    (elaborated FIRRTL)                 (CIRCT Arc dialect → LLVM → native sim)
        │                  │                                            │
        │ circt_introspect │ introspect (v1 grep)                       │ gen_arc_ports / gen_rocc_replay
        │ (v2: acc + funct) │ (mesh + scratchpad)                       ▼
        └────────┬─────────┘                              arc_results.json (bit-exact + cycles)
                 ▼
        ┌──────────────────┐
        │   facts.json     │  ← the single provenance-stamped artifact (mesh, mems, funct table)
        └───────┬──────────┘
   ┌────────────┼───────────────┬───────────────────┬────────────────────┬───────────────────┐
   ▼            ▼               ▼                   ▼                    ▼                   ▼
 rtl_checks   gen_rtl_digest  gen_isa_module     gen_numeric_facts    gen_iface_irdl     (re-used by all)
 (A) checker  (B) one-page    (C) generated      (D) numeric-shape    (E) IRDL dialect
 advisory     digest          ISA encoder        checker              spec (no C++ ODS)
   │
   ▼  runs against the agent's decoded kernel
 rocc_decode → instruction_trace.json ──► CheckReport (expected/got/ratio, advisory)
```

The decode side (`rocc_decode → instruction_trace.json`) is what the checker grades; spike/Verilator/arc
remain the correctness gates. Each box is a real file; the rest of this doc walks them in order.

---

## 2. The artifacts on disk (re-checkable)

All under `merlin/targets/gemmini/contracts/rtl_facts/` unless noted.

| Artifact | Size | Produced by | Role |
|---|---|---|---|
| `gemmini_soc.hw.mlir` | 21.5 MB | `firtool --ir-hw` | CIRCT HW dialect, full SoC (731 modules) |
| `gemmini.hw.mlir` | 5.3 MB | `extract_module` | `@Gemmini` subtree only (for arcilator) |
| `cell.arc.mlir` | 14 KB | `firtool` (Arc) | CIRCT Arc dialect |
| `gemmini.ll` | 3.3 MB | arcilator → LLVM | Arc lowered to LLVM IR |
| `gemmini_arc`, `gemmini_arc_replay`, `gemmini_arc_stress` | ~650 KB each | clang on `gemmini.ll` + harness | native RTL replay simulators |
| `gemmini.state.json` | 14 KB | arcilator `--state-file` | flat port/state byte map |
| `gemmini_arc_ports.h` | 10 KB | `gen_arc_ports` | C `#define`s + get/set helpers for ports |
| `a2_replay.json` / `a2_replay.h` | 6 KB / 3 KB | `gen_rocc_replay` / `replay_json_to_h` | replay spec + C header |
| `arc_results.json` | — | `gemmini_arc` runs | per-capsule cycles + bit-exact verdict |
| `facts.json` | 3.6 KB | `circt_introspect` | **the central artifact** |
| `firtool_irhw.log` | 105 KB | `firtool` | the elaboration/lowering log |

The whole directory is `.gitignore`-aware (large MLIR/binaries are local build products; `facts.json` is the
small committed distillate).

---

## 3. Stage 0 — the RTL source (what we read FROM)

The ground truth is the elaborated hardware, not documentation. Two representative sources:

**The ISA funct table** — `GemminiISA.scala` (chipyard `generators/gemmini/.../GemminiISA.scala`):
```scala
  // funct values
  val CONFIG_CMD = 0.U
  val LOAD2_CMD  = 1.U
  val LOAD_CMD   = 2.U
  ...
  val STORE_SPAD_CMD          = 23.U
  val LOOP_WS_CONFIG_SPAD_AB  = 24.U
  val LOOP_WS_CONFIG_SPAD_C   = 25.U

  val CONFIG_EX = 0.U      // <- rs1-subfield block begins here; NOT a funct code
```
These 26 `val NAME = N.U` lines (codes 0..25, stopping at `CONFIG_EX`) ARE the legal RoCC funct set the
silicon decodes. We extract exactly this list — never a hand-typed copy.

**The scratchpad memory** — `Scratchpad.scala` declares the banked `smem`, which `firtool` lowers to the
FIRRTL line `smem mem : UInt<8>[16] [4096] @[... Scratchpad.scala ...]` (×4 banks). That single line carries
element width (8), row elements (16), and depth (4096).

---

## 4. Stage 1 — `firtool` (CIRCT) lowers RTL → MLIR

**What we use:** `firtool`, the CIRCT FIRRTL compiler. **Why:** it elaborates the parameterized Chisel into
concrete, fully-typed hardware IR (the `hw`/`seq`/`comb` dialects) where every port width is explicit — and,
on the Arc path, all the way to a runnable model. **How:** `firtool --ir-hw` over the elaborated SoC.

The `firtool_irhw.log` records the real run (note the CIRCT-version warnings, proving this is a genuine
firtool invocation, not a mock):
```
generators/.../tilelink/Monitor.scala:35:7: warning: module contains 114 printf-encoded verification
  operation(s) ... see https://github.com/llvm/circt/issues/6970
```

The output `gemmini_soc.hw.mlir` (21.5 MB, 731 `hw.module`s) contains the explicit port signatures we mine,
e.g. the accumulator memory (the widths that grep over Scala could never give honestly):
```mlir
hw.module private @AccumulatorMem(... in %io_write_bits_addr : i9,
   in %io_write_bits_data_0_0 : i32, ... in %io_write_bits_mask_0 : i1, ...)
```

---

## 5. Stage 2 — extraction: MLIR/FIRRTL → facts (deterministic, NO LLM)

Two tiers. Both are pure parsers; the comment in `circt_introspect.py` states the principle:
> *"Deterministic, no LLM: the hardware is the source of truth."*

### 5a. v1 — `rtl/introspect.py` (grep over the elaborated FIRRTL)

**What/why:** the cheapest facts (mesh count, scratchpad geometry) are already unambiguous in the FIRRTL +
the module-hierarchy JSON; v1 reads them with scoped regexes. **How** (`introspect.py:74-84`):
```python
# Line form: `smem mem : UInt<W>[E] [D] @[... Scratchpad.scala ...]`
sp = _grep(fir, r"smem mem : UInt<[0-9]+>\[[0-9]+\] \[[0-9]+\] @\[[^]]*Scratchpad\.scala[^]]*\]")
if sp:
    m = re.search(r"UInt<(\d+)>\[(\d+)\] \[(\d+)\]", sp[0])
    ebits, row_elems, depth = int(m[1]), int(m[2]), int(m[3])
    banks = len(sp)
    facts["memories"].append({
        "name": "scratchpad", "banks": banks, "row_elems": row_elems,
        "depth": depth, "elem_bits": ebits,
        "bytes": banks * row_elems * (ebits // 8) * depth,   # 4*16*1*4096 = 262144
        "evidence": f"{banks}x `smem mem : UInt<{ebits}>[{row_elems}] [{depth}]` @ Scratchpad.scala"})
```
The mesh is scoped via the instance tree (`top_module_hierarchy.json`), counting `Tile`s under `Mesh`
(`_count_tiles_under_mesh`) → 256 tiles → 16×16. v1 honestly leaves the accumulator size `None` ("size left
unextracted rather than guessed", `introspect.py:86-91`).

**Transformation (v1):**
```
INPUT  (FIRRTL):  smem mem : UInt<8>[16] [4096] @[... Scratchpad.scala ...]      (×4)
                  top_module_hierarchy.json: 256 Tile instances under Mesh
OUTPUT (facts):   scratchpad{banks:4,row_elems:16,depth:4096,elem_bits:8,bytes:262144}
                  mesh{rows:16,cols:16,tiles:256,square:true}
```

### 5b. v2 — `rtl/circt_introspect.py` (parse the CIRCT HW dialect + Chisel)

**What/why:** v2 picks up exactly where grep stops — the accumulator capacity (from HW-dialect port widths)
and the full legal funct table (from the Chisel funct block). **How** — accumulator
(`circt_introspect.py:58-95`):
```python
_ACC_MOD_RE   = re.compile(r"hw\.module\s+(?:private\s+)?@AccumulatorMem\b([^\n]*)\)")
_WRITE_ADDR_RE = re.compile(r"io_write_bits_addr\s*:\s*i(\d+)")
_WRITE_LANE_RE = re.compile(r"io_write_bits_data_(\d+)_0\s*:\s*i(\d+)")
...
addr_w = int(am.group(1)); depth = 1 << addr_w          # i9  -> depth 512
lane_bits = lanes[0][1] ...                              # i32 lanes
banks = len(_ACC_INST_RE.findall(hw_text)) or 1          # 2 @AccumulatorMem instances
return {"name":"accumulator","banks":banks,"addr_width":addr_w,"depth":depth,
        "lanes":n_lanes,"lane_bits":lane_bits,"row_bytes":row_bytes,
        "bytes":banks*per_bank, ...}                      # 65536 B total
```
and the funct table (`circt_introspect.py:99-125`) parses `val NAME = N.U` from the `// funct values` block,
**stopping at `CONFIG_EX`** (so the rs1-subfield names that reuse small numbers are not mistaken for funct
codes):
```python
val_re = re.compile(r"\bval\s+([A-Z0-9_]+)\s*=\s*(\d+)\.U\s*(?:$|//)")
for ln in lines[start+1:]:
    if "CONFIG_EX" in ln: break        # rs1-subfield block -> stop
    mm = val_re.search(ln)
    if mm: table.setdefault(int(mm.group(2)), mm.group(1))
```

**Transformation (v2):**
```
INPUT  (HW MLIR): @AccumulatorMem(... io_write_bits_addr:i9, 16x io_write_bits_data_*_0:i32, 64 mask bits)
INPUT  (Chisel):  // funct values  →  val CONFIG_CMD=0.U ... val LOOP_WS_CONFIG_SPAD_C=25.U  (stop CONFIG_EX)
OUTPUT (facts):   accumulator{banks:2,depth:512,lanes:16,lane_bits:32,bytes:65536}
                  funct_decode_table{custom_opcode:0x7B,funct3:3,legal_funct:[0..25],names:{...}}
```

---

## 6. Stage 3 — the central artifact: `facts.json`

`build_facts()` merges v1 (mesh+scratchpad) with v2 (accumulator+funct) and stamps provenance + input SHAs;
`dump_facts()` is a **deterministic cache** (re-extraction is a no-op when the input SHAs are unchanged, and
a change to the extractor code itself invalidates it via `extractor_sha`). The committed result:

```jsonc
{
  "schema_version": "2.0",
  "generator": { "name": "merlin.targetgen.rtl.circt_introspect",
                 "version": "rtl-introspect-v2-circt-hw",
                 "method": "firtool --ir-hw CIRCT HW-dialect port widths (accumulator) +
                            GemminiISA.scala funct block (decode table) + v1 grep (mesh/scratchpad)" },
  "inputs": { "hw_mlir":"gemmini_soc.hw.mlir","hw_sha":"b07472562dbbb873",
              "fir_sha":"fbc95eabdc7fc740","isa_sha":"812cba090cf85b2b" },
  "facts": {
    "arrays":   [ {"name":"mesh","tiles":256,"rows":16,"cols":16,
                   "evidence":"top_module_hierarchy.json: Tile instances under Mesh"} ],
    "memories": [ {"name":"scratchpad","banks":4,"depth":4096,"elem_bits":8,"bytes":262144,
                   "evidence":"4x `smem mem : UInt<8>[16] [4096]` @ Scratchpad.scala"},
                  {"name":"accumulator","banks":2,"addr_width":9,"depth":512,"lanes":16,
                   "lane_bits":32,"bytes":65536,
                   "evidence":"2x @AccumulatorMem instance; per-bank io_write_bits_addr:i9 ..."} ],
    "datapaths":[ {"name":"input","dtype":"i8","evidence":"scratchpad smem UInt<8>"},
                  {"name":"accumulator","dtype":"i32","evidence":"AccumulatorMem SInt<32>"} ],
    "interfaces":[ {"name":"funct_decode_table","custom_opcode":123,"funct3":3,
                    "legal_funct":[0,1,...,25],"names":{"0":"CONFIG_CMD",...,"25":"LOOP_WS_CONFIG_SPAD_C"},
                    "evidence":"GemminiISA.scala // funct values block: 26 codes [0..25] up to CONFIG_EX"} ]
  }
}
```

Every fact has an `evidence` string. A `validate()` method (`circt_introspect.py:186`) cross-checks the
facts against the hand-curated `target_contract.yaml` and `rocc_decode._FUNCT_CLASS`, returning
`{agree, diverge}` — *divergence is surfaced as information* ("RTL is authoritative; contract unconfirmed"),
never silently reconciled. That equality is what lets the curated numbers eventually be retired.

---

## 7. Stage 4 — the five consumers of `facts.json` (why CIRCT pays off)

The same artifact is read five ways. This is the leverage: analyze the hardware once, reuse everywhere.

### (A) The advisory **checker** — `targetgen/rtl_checks.py`

**Why:** the oracle ladder (spike L2 → Verilator L3 → FireSim L5) is expensive; most broken codegen is
broken *before* simulating (malformed ISA, tile count that can't cover the shape, address past capacity).
This catches those statically and gives the agent *graded, expected-vs-got* feedback instead of pass/fail.

**How:** `load_default_facts()` prefers `facts.json`, falling back to curated `DEFAULT_RTL_FACTS`
(`rtl_checks.py:43-76`); `screen()` runs eight T0 (pure-static) checks over the decoded trace
(`rtl_checks.py:411-418`):
```python
rep.checks.append(_check_decode_clean(trace))            # no UNKNOWN custom-3 forms
rep.checks.append(_check_decode_funct_legal(trace, facts))  # funct ∈ legal_funct (from RTL)
rep.checks.append(_check_movement_compute_balance(trace, capsule))  # matmul w/ 0 COMPUTE → fail
rep.checks.append(_check_tile_coverage(trace, capsule, facts))      # MVOUT == ⌈M/16⌉·⌈N/16⌉
rep.checks.append(_check_spad_capacity(trace, facts))    # spad_addr < scratchpad rows
rep.checks.append(_check_preload_before_compute(trace))  # WS micro-sequence
rep.checks.append(_check_config_before_use(trace))       # CONFIG_EX/ST before first use
rep.checks.append(_check_fence_bracket(trace))           # FENCE … FENCE bracketing
```
Each expected value is derived ONLY from RTL facts + the capsule's *declared* shape — never a golden (the
non-overfit invariant, `rtl_checks.py:15-20`). A real finding (the `CheckReport.to_dict()` shape):
```jsonc
{ "id":"T0.tile_coverage", "tier":"T0", "severity":"warn", "status":"fail",
  "message":"MVOUT=240 but 16x256 over 16x16 mesh needs Mt*Nt=16 tiles",
  "expected":16, "got":240, "ratio":15.0,
  "fix_hint":"over-committing outputs ~15x; check the M/N tiling loop bounds" }
```
**Status (honest):** the module header self-labels **"Phase 0, ADVISORY / UN-WIRED … imported by NOTHING in
the frozen runner/grader/schema path … changes no pass/fail verdict."** It informs a verdict; it never is
one. The phased plan to wire it into the agent loop is `rtl_checks_layer_design.md` (Phase 0 done; 1–5 open).

### (B) The one-page **digest** — `rtl/gen_rtl_digest.py`

**Why (the token argument, in the tool's own words):** *"The merlin+CIRCT arm's biggest token sink in abc4
was exploration: ~67 find/grep/cat commands crawling the RTL + headers. … This renders those facts as a
human-readable digest … Reading one digest instead of crawling 55 Scala files is the CIRCT arm's 'the RTL
was pre-analyzed for you' advantage (a from-scratch/no-CIRCT path cannot get this)."*

**How:** `facts.json` → `RTL_DIGEST.md` (module map + full funct table + memory map + datapaths + legal
sequencing rules). A generated example exists at
`experiments/gemmini_capsule_bench_v0/_qa_ws/merlincirct_abc9/workspace/RTL_DIGEST.md`.

### (C) The generated **ISA encoder** — `rtl/gen_isa_module.py` (the "moat")

**Why:** promotes facts from a post-hoc *checker* into a *generator*. From the header: *"the two failure
classes the CIRCT checker caught — UNKNOWN custom-3 instructions and use-before-config — are exactly encoding
mistakes. Generating the legal funct table + an ordering-checked emitter from the RTL makes those bugs
structurally impossible, and removes the single largest hand-written chunk (~300 LOC of rocc.py). … Encoding
is RTL-grounded; only the algorithm is left to the agent."*

**How:** `facts.json` → a self-contained Python ISA-encoder module whose tables are stamped
`"GENERATED from RTL facts by merlin.targetgen.rtl.gen_isa_module — DO NOT hand-edit the tables."` The agent
then writes only op-lowering (tiling, im2col), not the error-prone RoCC encoding.

### (D) The **numeric-shape** checker — `rtl/gen_numeric_facts.py`

**Why:** CIRCT's structural screen can't see "structurally-legal yet numerically-wrong" code — that residual
is why a sim is still needed. This narrows it: from the RTL datapath facts (input dtype, accumulator width,
scale/activation semantics) it emits a checker that flags numeric-**shape** mistakes (wrong accumulation
width, mismatched output dtype, missing scale) **without computing any golden**. It does *not* certify
numerics — it shrinks how many sim runs are strictly required.

### (E) The **IRDL** dialect bridge — `rtl/gen_iface_irdl.py`

**Why:** the C++ out-of-tree backend and the xDSL infra must agree on one dialect spec. Rather than hand-sync
a C++ ODS dialect, this emits the canonical **IRDL** description of `merlin_iface`, which
`mlir-opt --irdl-file=…` registers *dynamically* — a C++ OOT tool parses the frozen `*.interface.mlir`
grammar with zero hand-written dialect code (stock LLVM-23 tools, no custom C++).

---

## 8. The decode side — what the checker grades against

The checks need the agent's emitted kernel as a structured trace. `targetgen/rocc_decode.py` parses the
package's `lowered.llvm.mlir` into `instruction_trace.json` (fail-closed: any unrecognised `.insn` →
`UNKNOWN`). A real one (`selfcheck_out/A2_single_tile_matmul/instruction_trace.json`):
```jsonc
{ "source": ".../A2_single_tile_matmul/generated/lowered.llvm.mlir",
  "abi": { "custom_opcode":"0x7b", "funct3":"0x3" },
  "instructions": [
    { "index":0, "class":"FLUSH",     "funct":7, "rs1":{"raw":0,"kind":"const"}, ... "decoded":{} },
    { "index":1, "class":"CONFIG_EX", "funct":0, "rs1":{"raw":4575657221408489476,"kind":"const"}, ... }
  ] }
```
`screen(trace, capsule, facts)` consumes this. So the loop is: **RTL → facts.json** (left) meets **agent
kernel → instruction_trace.json** (right) inside `rtl_checks`, producing the advisory `CheckReport`.

---

## 9. The second CIRCT track — arcilator native simulation

Besides facts, CIRCT gives a *fast runnable model* of the RTL, used as a quick replay oracle alongside
Verilator.

1. **`rtl/extract_module.py`** — the SoC HW MLIR has 731 modules; to JIT only the accelerator we need
   `@Gemmini` + its transitive instances. A pure text/graph BFS over `hw.module`/`hw.instance` emits a
   standalone `module { … }` (→ `gemmini.hw.mlir`), avoiding SoC blackboxes like the EICG clock gate.
2. **arcilator** lowers that to the **Arc dialect** (`cell.arc.mlir`) → LLVM (`gemmini.ll`) → native
   `gemmini_arc` executable; `--state-file` emits `gemmini.state.json` (`numStateBytes: 363289`, every port
   at a byte offset).
3. **`rtl/gen_arc_ports.py`** turns that state map into `gemmini_arc_ports.h` (`#define`s + get/set helpers
   so the C harness can poke RoCC/TileLink ports by name).
4. **`rtl/gen_rocc_replay.py`** turns a decoded trace + golden into a replay spec (`a2_replay.json`: arg
   addresses + materialized bytes + the exact funct/rs1/rs2 stream); **`rtl/replay_json_to_h.py`** renders it
   to a C header the harness `#include`s.
5. The harness replays into `@Gemmini` and checks bit-exactness. `arc_results.json` shows real runs:
   ```json
   { "capsule":"A0_config_smoke", "cycles":238, "bitexact":true, "wall_s":0.00281 }
   { "capsule":"A3_k_accumulation","cycles":322, "bitexact":true, "wall_s":0.00352 }
   ```

---

## 10. Advisory vs gate — the boundary (what decides correctness)

| Layer | Source | Decides pass/fail? |
|---|---|---|
| `rtl_checks` (T0 static) | `facts.json` + decoded trace | **No** — advisory pre-screen + feedback |
| spike (L2) | functional ISS | bootstrap functional check |
| Verilator (L3) | elaborated RTL, cycle-accurate | **Yes** — correctness oracle |
| arcilator `gemmini_arc` | CIRCT Arc native model | fast replay oracle (bit-exact + cycles) |
| VCS/FireSim (L4/L5) | config-gated | optional, when available |

The figure's "ADVISORY → gates pass/fail" box is this row: CIRCT checks *grade and steer*; the sim *certifies*.

---

## 11. Measured evidence — the authoritative abc9/abc11 A/B

> **Provenance.** Use the runs the live trajectory plots consume (`scripts/gen_trajectory_v2.py::ARMS`):
> `raw_baseline/rb_abc11`, `cpp_merlininfra/rbinfra_abc11`, `merlin_assisted/merlin_abc9`,
> `merlin_assisted/merlincirct_abc9`. An earlier `abc4` batch is **stale and must not be cited.** Numbers
> from each run's `cost_time_toolcalls.yaml` + `qa_loop_summary.yaml`. **N=1 per arm**; "converged" = the
> QA-loop iteration target of 20 capsules (the separate full 25-capsule L2+L3 audit is the correctness grade).

| Arm (run) | Converged (/20) | Rounds | Cost | Tokens | Tool-calls | Active wall |
|---|---|---|---|---|---|---|
| raw C++ — `rb_abc11` | ✅ 20/20 | 5 | $147.34 | 82.0M | 442 | 3.94 h |
| C++ + Merlin scaffold — `rbinfra_abc11` | ❌ 17/20 | 5 | $159.48 | 84.3M | 677 | 6.45 h |
| Merlin Python tooling — `merlin_abc9` | ❌ 19/20 | 10 | $86.43 | 45.8M | 352 | 3.46 h |
| **Merlin + CIRCT — `merlincirct_abc9`** | ✅ **20/20** | **1** | **$52.73** | **29.2M** | **137** | **1.03 h** |

**The measured CIRCT result (decisive):** `merlincirct_abc9` converged **20/20 in a single round**, at the
**lowest** cost ($52.73), **fewest** tokens (29.2M), **fewest** tool-calls (137), ~1 h, zero rate-limit
waits. Raw C++ also converged but the expensive way (5 rounds, $147, ~4 h); the two infra-only middle arms
did **not** converge (Python 19/20, C++-scaffold 17/20). So on the authoritative runs the CIRCT-grounded
advisory feedback is what turned bring-up into a one-shot — the opposite of the stale abc4 picture (which had
wrongly shown CIRCT as *more* expensive).

**What this lets us say, precisely (N=1):**
- *Extraction + facts.json + the five consumers + arcilator* — **built, with artifacts on disk** (this doc).
- *CIRCT-grounded feedback yields the cheapest, fastest, single-round convergence* — **measured** on
  `merlincirct_abc9` (lowest cost/tokens/tool-calls/rounds of all four arms).
- *The "fewer tokens / faster iteration" claim* — now **supported in this batch** (29.2M tokens / $52.73 is
  the lowest of the four), but it is **N=1**; magnitudes need N>1 before stating them as stable.

**Mechanism corroboration (older, label as such):** an earlier `abc4` correlation replayed the CIRCT
pre-screen over all arm×round×capsule traces and found CIRCT-reject ⟹ sim-fail with no false-clean — i.e.
CIRCT is a sound *structural* pre-screen (it cannot judge numerics). Treat that as mechanism evidence from
the stale batch; the authoritative effort numbers are the abc9/abc11 table above.

---

## 12. Reproduce it

```bash
# 1. extract facts from RTL (deterministic; cache hit if SHAs unchanged)
python -m merlin.targetgen.rtl.circt_introspect --out facts.json --validate
# 2. derived consumers
python -m merlin.targetgen.rtl.gen_rtl_digest          # -> RTL_DIGEST.md
python -m merlin.targetgen.rtl.gen_isa_module          # -> gemmini_isa.py (generated tables)
# 3. grade an agent's emitted kernel (advisory)
python -m merlin.targetgen.rtl_checks <instruction_trace.json> --capsule <capsule.yaml>
# 4. arcilator native replay
python -m merlin.targetgen.rtl.gen_arc_ports gemmini.state.json --model Gemmini --out gemmini_arc_ports.h
python -m merlin.targetgen.rtl.gen_rocc_replay <capsule.yaml> <instruction_trace.json> --out replay.json
```

Files to read, in journey order: `rtl/introspect.py` → `rtl/circt_introspect.py` → `facts.json` →
`rtl_checks.py` → `rtl/gen_rtl_digest.py` / `gen_isa_module.py` / `gen_numeric_facts.py` /
`gen_iface_irdl.py` → `rtl/extract_module.py` / `gen_arc_ports.py` / `gen_rocc_replay.py` →
`rtl_checks_layer_design.md` (the wiring plan) → `scripts/gen_trajectory_v2.py` + the four runs'
`cost_time_toolcalls.yaml`/`qa_loop_summary.yaml` (the authoritative abc9/abc11 measurement).

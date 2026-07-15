# agent_spec_v0_mlir_oot — Merlin-side decision memo

Decision-quality summary of the Merlin-assisted artifact, for the conference comparison. All claims
below are backed by recorded artifacts (run dirs + manifests), not scrollback.

## 1. Is the Merlin artifact correct for the *scoped* claim?

**Yes.** From a **clean rebuild** (build tree deleted, rebuilt via the manifest through
`oot_runner.build_package`) into a **fresh runs root** (`runs/agent_spec_v0_oot_rerun`), the package
re-certified **18/18** runs: G0–G5 + hidden h0/h1/h2 on both Spike (bootstrap) and Verilator (RTL),
all `status: pass`, all entrypoints pass, L0 `reference==simulate` pass, three-way bit-exact, and
Verilator `cycle_accurate=true`. Cycles reproduced identically to the first sweep
(308/308/250/250/1006/428 + 308/308/250).

## 2. What exactly is certified?

- **G0** matmul→i32, **G1** +relu→i32, **G2** +acc_scale→i8, **G3** +acc_scale+relu→i8 (single 16×16 tile).
- **G4** tiled 32³ with K-accumulation→i32; **G5** one resident weight reused across two matmuls→i32.
- **Hidden** h0/h1/h2 (renamed-tensor variants → different deterministic data): pass with cycles equal
  to their public counterparts ⇒ the kernel is **data-independent** (does not hardcode outputs).
- Certification = three-way bit-exact `oracle == reference == simulate`; Verilator runs are RTL-backed
  (`derived_from_rtl=true`, `cycle_accurate=true`).

## 3. What exactly is NOT certified (explicit non-claims)

- **Not** general RTL→target generation (RTL is only the certification oracle; provenance class
  `verilator-oracle-only`).
- **Not** a full/complete Gemmini backend; **not** arbitrary tiling or arbitrary residency (G4/G5 are
  specific certified *instances*).
- **No** conv / attention / network / FireSim / cost-model / bias / accumulator-preload support.
- **No** performance optimization, and **no** "Merlin is faster/cheaper/easier" claim — the baseline
  has not been run and there is no process telemetry for either side yet.

## 4. Is the package non-exempt? (and is the integrity policy real?)

**Yes — `integrity_exempt: false`, and the scan demonstrably works.** During the reproducibility
rerun, an edit to the package's provenance wrapper (`runtime/oracle_runner.py`) introduced the
forbidden substrings `merlin.runtime.reference` / `merlin.runtime.simulator` / `reference_outputs`
in *documentation strings*. The integrity scan **caught it and failed those runs closed**
(`plane: integrity, category: forbidden_pattern`). The wrapper was fixed (forbidden literals removed;
canonical stage names recorded in non-scanned YAML/MD instead), the scan now passes, and the matrix
re-ran clean. This is positive evidence that the fairness boundary is enforced, not decorative.

## 5. What role does Merlin tooling play?

- **Recorded structured recipe** (`inputs/{target_spec,dialect_plan,runtime_plan}.yaml`): documented &
  stabilized the op/type set, the interface→target→cb mapping, and the RoCC encoding plan. The
  dialect/passes were **hand-authored to match** these and validated against them — there is **no
  deterministic code generator**. Authoring mode: `agent_generated_from_recipe`
  (= `agent_authored_with_structured_spec_provenance`); `deterministic_generator: false`.
- **xDSL prototype** (`xdsl/dialect.py`): self-contained semantic prototype that pinned the
  gemmini→cb mapping (validated: g0 cb == golden; L0 g0–g3) before the C++ was authored.
- **RoCC encoding ported** from the certified native path (`gemmini_codegen_mlir.py`) — a Merlin
  **tooling/authoring advantage**, not independent rediscovery. Re-expressed in self-contained C++.
- **oot_runner / AET** provided certification + provenance recording (runner-side).

## 6. Information parity vs the baseline

Per `agent_spec_v0_mlir_oot_public_facts_audit.md`:
- **Public (baseline also has it, in-sandbox):** custom-3 opcode, all funct codes, `DIM=16`,
  weight-stationary/relu concepts, `acc_scale` is IEEE-f32, `F1=1.0f`.
- **Merlin tooling advantage (NOT in the sandbox's spike-model `gemmini.h`):** the exact config-word
  bit-packing, the `pack()` tile descriptor, accumulator address bits (`C_ACC/ACC_I8/ACC_ACCUM`), the
  `GARBAGE` sentinel, scratchpad slot convention, and the exact unrolled WS sequence — ported
  pre-assembled. Derivable from broader public Gemmini sources, but the in-sandbox baseline would
  have to derive it. This must be reported as an advantage, not parity.

## 7. What remains before a publishable comparison?

1. **Run the baseline** (raw Claude Code, same contract + sandbox) under a **measured harness**;
   capture wall time / tokens / tool calls / edit count / first-failure plane for **both** sides.
2. Fill the **placeholder baseline row** in `agent_spec_v0_mlir_oot_report.md`.
3. (Optional, stronger) hard **two-repo / two-machine isolation** instead of honor-system + scan.
4. Note the **repo-HEAD drift** (authoring SHA `2a850044` → snapshot SHA `f684ab53`, concurrent
   session) and that the correctness oracles (`runtime/{reference,simulator,tensor}.py`) are
   modified-in-tree; pin/commit before publication.

## 8. Should this be committed or snapshotted?

**Recommend committing/snapshotting** the package + evidence: it is reproducible from a clean rebuild,
integrity-clean, truthfully labeled, and all results regenerate from recorded artifacts. Everything is
currently uncommitted working-tree state alongside concurrent-session changes — snapshot it on a branch
so the conference evidence is pinned and the SHA drift is frozen.

## Verdict

`agent_spec_v0_mlir_oot` is **correct and clean for the scoped claim** and **comparison-ready on the
Merlin side**. The only thing standing between here and a publishable head-to-head is running the
baseline under equal, measured conditions.

> agent_spec_v0_mlir_oot demonstrates that Merlin's structured targetgen flow can produce a non-exempt
> out-of-tree MLIR-level Gemmini target package satisfying Experiment ABI v0.1 and certifying the
> supported rungs through the shared command-buffer/reference/oracle ladder. It does not yet prove
> general RTL-derived target generation.

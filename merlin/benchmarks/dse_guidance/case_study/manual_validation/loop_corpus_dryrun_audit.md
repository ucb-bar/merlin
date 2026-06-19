# P23 Phase 1 — loop-corpus dry-run audit (go/no-go gate: **GO**)

Before regenerating the committed study on the loop-preserving corpus, we dry-ran the full pipeline
into a temp dir (no committed artifact touched) and forensically checked the captures. Result: **GO**.

## 1. Full pipeline runs cleanly on the loop captures
`run_case_study` completed end-to-end on the 10 loop models (all P5–P12 + P20 artifacts produced).
The analysis tools descend into the `scf.for` region transparently (xDSL `module.walk()` is recursive),
so the repeated-region op graph is fully analyzed — it was not before (the flat corpus has no loop).

## 2. What changes vs the committed flat corpus (the diff)
| artifact | flat | loop | why |
|---|---|---|---|
| operator_shape_table (rows) | 1051 | 1385 | loop captures carry the prefix AND the repeated body |
| operator_full_inventory | 8194 | 10651 | same |
| work_coverage (workloads) | 11 | 10 | small_llama excluded (see §4) |
| dataflow_candidate / **timeloop problem shapes** | 161 | **161** | same distinct problem shapes — loop adds *instances*, not new geometry (good: per-op shapes are consistent) |
| sharding_table | 3153 | 4155 | more ops to shard |
| traffic_table (regions) | 17 | 13 | structural role split + small_llama out |

Role attribution is now **structural** (the `scf.for` boundary), corpus-wide: in-loop matmuls →
`repeated_head`, out-of-loop → `backbone_once`/prefix. All 10 attribute FULLY (vs the flat fqn
heuristic which left openvla PARTIAL and rdt/rdt2 UNKNOWN once the loop body lost its fqn).

## 3. Forensic fidelity — each loop body == the real per-step computation
Every wrapper was numeric-checked by its author-agent against the eager unrolled K-step loop, which
PROVES the loop body's op-set is the model's true per-step work (not a wrapper that drops/duplicates):

| model | K (IR) | numeric vs eager unrolled |
|---|---|---|
| smolvla / pi05 | 10 | cos 0.9999994 / 1.0 |
| openvla / molmoact / bitvla / tiny_llama | 7 / 8 / 7 / 7 | bit-exact (token match) |
| xr0 / groot / rdt / rdt2 | 5 / 4 / 5 / 5 | cos 1.0 / 1.0 / 1.0 (rdt +DPM-solver state) / 1.0 |

## 4. Honest exclusion
- **small_llama** (synthetic toy, random init, 2L/128h): its loop wrapper used module-free functional
  weights, which m2m lowered to `linalg.generic` (0 `linalg.matmul`) — the matmul-based pipeline can't
  see its GEMMs. Excluded from the loop corpus (its flat capture remains). The 10 real models stand.

## 5. K reconciliation
The llama loop captures used **K=7** (the actual captured decode length, recovered from the `scf.for`
trip count); `MODEL_ARCH` previously carried an *assumed* K=32 for the llamas. The IR K is the
authority for a loop-preserving capture → Phase 2 sources K from the IR and updates the reference.

## Gate decision
GO. Proceed to Phase 2 (switch the primary corpus to `recaptures_loop`, regenerate, rebaseline the
~26–33 K/count-pinned checks + tests + docs). The flat `recaptures/` are retained; all prior committed
results remain recoverable in git.

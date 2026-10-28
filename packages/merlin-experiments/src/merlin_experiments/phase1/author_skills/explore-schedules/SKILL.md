---
name: explore-schedules
description: "Search the schedule or pass options of one operator family with merlin.transforms.pass_exploration instead of hand-picking them: wrap your lowering as PassFamily options, generate candidates, rank them on training shapes and check the winner on held-out shapes with scripts/holdout_rank.py, then time the winner as a variant. Use when a choice your compiler fixes (order, blocking, whether to apply a pass) could differ per shape, or a case has low array utilization."
---

`SKILL_DIR` = the directory holding this file. Run from the workspace root.

1. **Choose bounded alternatives.** Prioritize large/connected roles and expose a finite set of
   reusable lowering/pass choices. No complete timing: rank on `fast_estimate`, revert if measured
   worse. Inspect the emitted host/device code and keep a compact before/after
   record per trial; `reference/evidence.md` lists what to look for and what to retain. Helpers can
   explore families concurrently against snapshots of the same compiler.
2. **Freeze validation shapes.** Create `ExplorationGeometry(family, operand_shapes)` values, with
   larger and tail shapes in `ValidationPartition(training, held_out)`. It refuses aliases.
3. **Explore real passes.** The source/lowering/emission/traffic callbacks below are your code;
   supply `partition` from step 2. Plain Python IR needs clone/fingerprint/verification adapters;
   xDSL modules use the defaults.
   ```python
   import copy, hashlib, json, sys
   sys.path[:0] = ["submission/compiler_library", "submission", "SKILL_DIR/scripts"]
   from merlin.transforms.pass_exploration import (EmittedProgram, ExplorationGeometry, PassFamily,
       ValidationPartition, generate_candidates)
   from merlin.transforms.pipeline_policy import PolicyFacts, RegisteredPass, ShapePredicate
   from merlin.transforms.pipeline_state import PassSpec
   from holdout_rank import dumps, rank_on_holdout

   sha = lambda ir: hashlib.sha256(json.dumps(ir, sort_keys=True).encode()).hexdigest()
   shape = PassFamily("shape", lambda k: RegisteredPass(
       PassSpec("shape", 0, produces=("shape",), mutates_ir=False, ir_level="mine"),
       lambda s: {"shape": shapes_of(s.module)}))
   def lower(k):  # your lowering with this candidate's option values
       def action(s):
           s.module = my_lower(s.module, **k)
           return {"program": my_emit(s.module)}
       return RegisteredPass(PassSpec("lower", 1, requires=("shape",), produces=("program",),
                                      ir_level="mine"), action)
   families = (shape, PassFamily("lower", lower, (("order", ("mnk", "nmk")), ("block", (1, 2, 4))),
               predicates=lambda k: (ShapePredicate(0, 0, multiple_of=k["block"]),)))
   space = generate_candidates(families, output="program", max_candidates=16)
   geometries = (*partition.training, *partition.held_out)
   report = rank_on_holdout(partition, {g: source_ir(g) for g in geometries}, space,
       lambda s: PolicyFacts(s.stamp(), shapes=shapes_of(s.module)), output="program",
       emission=lambda r: EmittedProgram(json.dumps(r.artifact).encode(), "json", sha(r.module)),
       metrics=lambda program, receipt: {"operand_bytes": count_operand_bytes(program.data)},
       objectives=("operand_bytes",), domain="my-emitted-operand-traffic-v1",
       clone=copy.deepcopy, fingerprint=sha, verify_ir=lambda ir: None, ir_level="mine")
   print(dumps(report))
   ```
   This screens operand bytes of the emitted product, not cycles; missing metrics stay unknown and
   shape predicates establish applicability (`reference/evidence.md`).
4. **Read structural evidence.** `winner_bindings` describes the training choice;
   `holds_on_held_out` checks the same objective on independent shapes, not hardware performance.
   If it fails, retain alternatives or derive applicability from shape/resource facts, never IDs.
5. **Check and time.** Run affected Spike numerics and `ab-library-pass` for comparable cycles.
   A timing KEEP supports a provisional decision; an inconclusive result does not stop structural
   exploration. Retain numerically correct hypotheses as knobs/variants, without claiming speedups.
   Idle timing uses declared alternatives (`qa/idle_tuning.json`). Preserve the correct baseline.

API: `submission/compiler_library/merlin/transforms/pass_exploration.py` (`generate_candidates`,
`explore_candidates`, `rank_observed_metrics`). No static score replaces missing cycles.
Per-class incumbents switched on cycles covering cutoffs (`merlin.transforms.shape_policy`): the
`shape-schedule-selection` skill; library PipelineVariants: `heavy-pass-recipes`.

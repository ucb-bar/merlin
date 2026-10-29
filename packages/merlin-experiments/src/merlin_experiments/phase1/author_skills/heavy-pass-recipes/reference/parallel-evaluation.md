# One schedule/effect adapter, several real alternatives

`parallel_schedule_recipe.py` connects the existing heavy passes to
`pass_exploration.explore_candidates` and `evaluated_exploration.evaluate_emitted`.
It constructs ONE checked counted source per shape, branches through registered
actions, emits with ONE unchanged callback, then streams unique bytes to the
external evaluator. It does not copy compiler trees or invent cycle prices.

```sh
python .agents/skills/heavy-pass-recipes/scripts/parallel_schedule_recipe.py \
  --library submission/compiler_library --facts rtl_facts/facts.json \
  --output ab/parallel-example-NEW --compile
```

Omit `--facts` for the explicitly synthetic neutral example. The two default
geometries are derived from array/store extents, with aligned and ragged outputs.
Repeat `--shape M,K,N` to use your source geometries. The neutral source-check
budget is intentionally small; larger jobs need your ordinary numerical owner.
The example needs a K-segmented block and enough output blocks to expose a
readout change; unsupported geometry refuses rather than bypassing the premises.
`--compile` uses the selected `mlir-translate` and native compiler from the normal
toolchain resolver. A missing tool or a rejected product fails with its retained
stderr; it does not silently substitute a compiler.

The registered `original`, `batched`, and `ready` actions apply the actual
`batch_dense_segmented_readouts` and `schedule_dense_ready_readout` consumers.
An `identical` alias deliberately emits original bytes and skips repeated
evaluation. Every branch retains its pass receipt and its actual changed or
unchanged schedule and emitted SHA. Loops remain counted. Full completions map
only through the original checked drain callback.

The neutral callback checks projected integer effects against the source and
optionally compiles actual LLVM to native objects. External effect declarations
are unresolved: these are NOT executable device kernels, accelerator numerical
evidence, cycle-model feedback or a speedup result. All such timings remain
UNKNOWN/unranked. Counts never substitute for cycles.

## Bind your existing backend once

Use `--adapter tools/schedule_owner.py`. Its small public interface is:

```python
def binding(shape, caps):
    # Build your source schedule once using existing shared scheduling APIs.
    # Return the same callbacks for every alternative, not a per-pass emitter.
    return dict(source=checked_source,
                transform=apply_registered_alternative,  # (checked_result, mode)
                emit=emit_same_target_product,           # -> (LLVM text, evidence)
                shapes=original_operand_shapes,
                context=selected_pipeline_context)

def identity(shape, binding):
    return exact_exploration_identity  # actual source/ABI/input/reference/facts/
                                      # library/provider/scope/ROI/context pins

def evaluate(program, receipt, identity):
    # Use ordinary tool-owner leases. Compile/link this EXACT emitted program,
    # run the original numerical and completion checks, then selected observation.
    # Return CompletedObservation(identity, program SHA, original feedback SHA,
    #   numerical_status, timing_status, complete_roi, cycles), or None if pending.
    return externally_completed_observation
```

`binding.transform` receives `original`, `batched`, `ready` or `identical`; it
calls the shared checked consumers with your selected admissions. The example's
neutral adapter supplies executable code for this wiring. Replace its neutral
`encode`/`project`, admission identities and terminal effect declarations with
your existing selected bindings. It supplies no target opcode or capability.
`binding.emit` uses the SAME checked-result `.emit_index_program(control,
lower_effect, complete_epoch=...)` on all sides. Finish the original package ABI
with the same emitter; retain original initialization, completion, tail and
numerical configuration requirements. No extra per-effect fence is added.

The evaluator receives only emitted bytes, a plain pipeline receipt and exact
domain; it does not need mutable IR or a compiler directory copy. Reference
immutable inputs and small overlays from the owner. Optional
`hold_inputs(request)` returns an existing lifecycle lease context, acquired
before submit and held through compact completion publication. Optional
`on_completion(request, completion)` must only append/enqueue small records;
enqueue slow verification/cleanup on a caller-owned maintenance executor.
Cleanup must respect the lease and run after release. A slow cleanup must not
block admission or consume the evaluator's ordinary resource allowance.

The optional CLI owns a bounded thread executor; library callers instead pass
their EXISTING executor to `run(..., executor=..., max_in_flight=...)`. This is
not a separate CPU reservation: evaluators still use the ordinary shared broker
or resource leases. Requests interleave geometries, so the first shape's entire
family cannot monopolize admission. `completions.jsonl` is appended after each
callback; `report.json` retains candidates, dedup, exact domains and per-domain
cycle rankings. UNKNOWN never ranks and no result automatically changes a policy.

## Remaining ordinary launch integration

The ordinary capsule `agent_selfcheck`/joint-performance owner consumes a complete
authored compiler, original capsule, passing native ELF producer and selected
runtime. A neutral effect-only stream lacks that contract. To use that owner,
connect your existing source/effect adapter to its SAME declared manifest output,
retain its native producer/input/scope custody, and delegate to its normal
numerical/selected-cycle path. Reopen the exact feedback and map only complete
numerically passing ROI cycles to `CompletedObservation`. A bare LLVM file or
command count cannot be relabelled as an observed capsule. The example proves
registration, emission and concurrent external-callback connectivity; supplying
that target/ordinary-owner binding remains an explicit construction task.

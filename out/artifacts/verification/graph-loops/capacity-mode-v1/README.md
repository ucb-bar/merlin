# Additive pending-capacity mode — sealed Phase0

This mode extends the existing actual4499 counted-loop/storage boundary with a kernel-checked
bound on pending work at EVERY state of the complete dynamic trace. It imports the exact
existing memory/completion semantics. It never clears pending jobs at period boundaries.
Physical capacity, effect footprints, completion visibility, input initialization and external
nonaliasing remain target/declaration premises. Concurrency pairs are bound in full and checked
by actual Python production admission; no new concurrency-pair or hardware theorem is claimed.

| Paper row | Evidence |
| --- | --- |
| Actual boundary | Read-only4499 `periodic_effects.check_drained_period` / `_closed`,3period/30call and independently7period/70call actual captures. This is counted-loop representation/materialization, not an optimizing pass or universal recurrence proof. |
| Existing roots | The unchanged delivered v1 GraphLoops/Epochs/Bound/Storage roots, ResourceGeometry and StateEffects. |
| New roots/domain | `Merlin.GraphLoops.Capacity.execute_sound/check_sound`, `Trace.erase/state_bounds/event`: under an explicitly selected positive pending limit, every event before/after state is bounded, the entire actual trace remains admitted, and final pending work is empty. |
| Actual correspondence | The unchanged v1 actual IR/effect check is composed with a separately re-opened actual adapter and full admission declaration, including concurrency pairs and qualification/facts/declaration identities. Both input copies have separately checked exact paths and identical original bytes. |
| Preserved v1 | Patch724510de637949fe7b1613d2ee5d5225420734a1473fd6d93fa50518a1267570, all11source files and referenced tool/object/IR/runtime/report identities reopen unchanged. No old module/object was rebuilt or replaced. Only8required objects were copied byte-identically into a NEW private capacity profile. |
| Fixed checks | The CASE_ROSTER was fixed before qualification.48tests passed in qualification;48passed again in ONE fresh-process replay batch. The roster contains the original36 and12additive capacity cases. |
| Important refusal | Actual source with three simultaneous independent publishes remains accepted by the v1 Lean memory/initialization model but is refused by actual capacity2 and the new Lean capacity gate. Missing drain, stale admission/limit/concurrency/qualification, stale tool/model/v1 seals and an intervening query writer also refuse. |
| Audit |36public graph roots, zero unapproved transitive axioms; only propext/Classical.choice/Quot.sound. Actual capacity queries print28root reports. No reflection/native_decide/bv_decide/sorry/admit/new axioms/downloads. |
| Source-only delivery | `source-only.patch` adds GraphLoopsCapacity.lean, lean_graph_loops_capacity.py, lean_graph_loops_capacity_capture.py and one test file. SOURCE_FREEZE.json identifies exact source, selected proof closure, full old seal, tool and fixed roster. SEAL.json records qualification/replay/postchecks; EVIDENCE_SEAL.json additionally indexes retained actual query/projection/check evidence. |
| Composition limits | No SCF/countedKernel bridge, adjacent pipeline receipt, allocation-changing pass, numerical/source-definedness, LLVM/native-renderer/hardware/model correctness follows. Python projection/parser/runtime, selected Lean runtime/standard library and declaration truth remain explicit. Original native70event/14output evidence is preserved separately. |

First development failures are retained in actual-capacity-development-v1/v2. v1 compared JSON
list pairs against asdict tuple pairs; v2 incorrectly equated paths of two separately bound copies.
Both fixes preserve complete declaration/event equality and independently bind both copies to the
same original buffers. Development-v3 then checked the actual case; development tests-v1 passed12.
These are named development versions, not retries of a frozen qualification context.

To reproduce, inspect freeze_capacity_v1.py, SOURCE_FREEZE.json, selection.json, proof-closure.json,
CASE_ROSTER.txt and each RUN command. Qualification and replay used the explicitly pinned existing
Lean4.35 and Python, no Lake/downloads. The capsule is not an accepted pipeline receipt or an
independently reviewed target qualification. Frozen files are now READ-ONLY; subsequent work uses
new segmentation leaves and a separate profile. Do not edit the original v1 or capacity modes.

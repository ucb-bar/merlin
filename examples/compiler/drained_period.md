# Counted blocks with bounded storage/completion checks

`merlin.compile.scheduling.periodic_effects.check_drained_period` checks every
iteration of one existing `Kernel` loop while retaining that loop for ordinary
C emission. It does not expand emitted code or invent a target instruction.
The finite verification cost is proportional to the dynamic call count.
Construction checks once; emission rechecks before and after target callbacks.
`report()` returns a detached historical receipt with the checked kernel digest,
without repeating verification or claiming current-input freshness. A caller
must supply `max_dynamic_calls`; exceeding it reports the exact count and refuses,
so the compiler can retain another route. This is not an affine induction proof
or a scalable solution for arbitrarily large asynchronous programs.

The API consumes existing `StorageAdapter` projections, private objects, byte
banks, fixed external windows and reservations. Every iteration reuses
`verify_temporary_storage` for actual access bounds, initialization, aliasing,
placement and completion. Each private object must be completely initialized in
that iteration; every actual asynchronous token must finish before its end.
Pending work and private live-in values cannot cross the iteration boundary.

`resolve_drained_period_admission` requires explicit concurrent instruction pairs
and a fact-referenced pending-work bound. These declarations and qualification
identifiers are assumptions needing target evidence, not authenticated
certificates. The generic tool does not discover queues, infer effects from names
or establish that different named engines can overlap.

```python
admission = resolve_drained_period_admission(
    declaration, facts, adapter=selected_effect_adapter,
    qualification_identity=explicit_assumption_identity,
)
checked = check_drained_period(
    kernel, objects, banks, selected_effect_adapter, admission,
    max_dynamic_calls=budget,
    external_windows=windows,
    private_ownership=True,
)
c_source = checked.emit_c(instruction_set, symbol="candidate")
```

Launch and wait operations must already be actual calls decoded by the adapter.
This first route refuses `unit`, `produces`, `awaits`, `stages`, `sets` and `assumes`
annotations through the existing storage checker; it does not erase them or use
an annotation as a replacement for a completion instruction. Source, adapter
identity, assumptions and effect observations are checked again around emission.
The renderer still owns machine semantics and requires independent execution.

The runnable `drained_period.py` example is a neutral two-slot byte-memory system.
Its generated counted loop issues asynchronous input copies and output writes;
the independent C runtime performs them only when their actual wait commands
execute. It checks all outputs against a separate scalar calculation and reports
a peak of two outstanding operations. This is mechanism evidence, not a target
kernel, preferred schedule or hardware performance measurement.

```sh
python examples/compiler/drained_period.py --output YOUR_TASK_TMP/program.c
cc -std=c11 -O2 YOUR_TASK_TMP/program.c -o YOUR_TASK_TMP/program
YOUR_TASK_TMP/program
```

The focused tests compile and execute one-, three- and seven-iteration versions.
Negative cases reject an early overwrite, missing input completion, missing final
drain, absent concurrency assumption, excessive pending work, mutation during
emission and an external-address error beyond a valid prefix. A deliberately
incorrect renderer still passes neither the independent output comparison nor
any claimed numerical qualification: the checker reports numerical equivalence
as unproven.

Boundaries remain explicit: no numerical recurrence, configuration invariant,
cross-iteration output nonaliasing or full-model equivalence proof is provided.
External writes may legally revisit the same window under this storage-only
checker; a source/numerical checker must decide whether that is correct. Tails
are checked exactly when their calls satisfy the same full private-initialization
and drain requirements; partial initialization and carried work refuse. The
existing synchronous dense adapter is unchanged. No gSIM, Spike, FireSim or
physical timing qualification follows from this result.

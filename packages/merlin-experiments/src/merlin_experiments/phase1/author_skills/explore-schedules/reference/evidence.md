# Exploration evidence: what to inspect and what to retain

Read with steps 1, 3 and 5 of the `explore-schedules` skill.

## When cycle timing is UNKNOWN

UNKNOWN limits cycle ranking, not compiler development. Use cycle headroom when it is available;
otherwise inspect traffic, preparation, storage and completion boundaries in the emitted host/device
code. Look for:

- repeated packing of immutable operands;
- redundant configuration;
- avoidable reloads;
- tile waste;
- drains that exceed the actual producer/consumer dependency.

Use the selected hardware facts and the software semantics to establish that a change is legal.
Consult the relevant library adapter when the current lowering cannot express a change; implement or
improve a reusable pass where needed.

## What to retain per trial

For each useful trial keep a compact record of:

- the before/after effect on the emitted artifact;
- the legality assumptions it relies on;
- the numerical result it affects;
- unresolved performance tradeoffs.

Lower traffic or fewer drains support a hypothesis, not a guaranteed speedup: buffering can add
contention or storage pressure.

## Reading the emitted products

Inspect the actual emitted products with their dynamic loop multiplicities, not static op counts.
The documented snippet screens operand bytes, not cycles; retain several alternatives when resources
trade off. Missing metrics stay unknown, never zero. Commands/instructions are secondary diagnostics.
Shape predicates, not workload identities, establish where a choice applies.

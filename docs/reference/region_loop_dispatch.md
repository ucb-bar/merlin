# Optional host dispatch loops

`merlin.compile.dispatch_c.prepare_dispatch_loop` captures an explicit counted
window over an existing serial `MerlinProgram` and its `DispatchCall` table.
Pass the result as `loop=` to `emit_program_c`. Emission rechecks the complete
dispatch graph, selected-plan receipt, arena plan and ordered wrapper/context
bindings. A stale schedule is refused before sources are returned. Bounds must
fit the flat call table and contain at least two iterations.

The generated `_run_loop` entry accepts a target admission hook. The hook sees
the complete ordered table of existing functions, resolved contexts and exact
input/output pointers, with the prefix and iteration bounds. It returns 1 to
admit counted execution or 0 to choose flat order. Admission happens before any
wrapper runs; hooks must not dispatch, mutate the table, retain its invocation-owned
pointers or recursively invoke the endpoint. Other hook return values fail
before the dispatch state is changed. The ordinary `_run` entry chooses flat
order, and emission without `loop=` retains the existing generated ABI/source.

Both paths use every original compiled variant and preserve the original arena,
pointer bindings and synchronous ordering. A wrapper failure preserves active
state, its actual status and the flat completed-call count. There is no fallback
or replay after execution begins; the existing endpoint drain/reset rule applies.
The caller owns endpoint synchronization during admission and execution. The
complete call table is allocated on the heap for each invocation and freed when
that invocation returns; no static mutable table is shared between invocations.
The call table and operand-pointer slots share one allocation. Loop-enabled
drivers require standard C `malloc` and `free`; allocation failure returns
`INVALID` before admission or dispatch state changes. The hook's schedule and
all table pointers are valid only during that invocation.

The executable C fixture in `test_dispatch_loop_c.py` compares an unrolled driver
and a counted two-iteration driver with prefix/suffix, distinct compiled variants
and contexts, carried pointers and immutable constants. It also checks declined
and absent hooks, invalid admission, and second-iteration failure accounting.
Allocation failure and cleanup are checked on each return path. A 2,048-call
driver also compiles unoptimized with a 4 KiB stack-frame warning limit.
This is a generic runtime qualification; it does not qualify a model's numerical
execution or hardware.

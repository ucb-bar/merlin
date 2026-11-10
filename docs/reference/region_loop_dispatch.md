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

`native-compile --application --application-region-loop` enables the counted
entry in the standard application package. It is off by default; default
compilation skips loop analysis and emits no `application-loop-plan.json`, even
when the target declares analysis parameter attributes. Opt-in compilation
analyzes the current selected regions, checks exact expansion of
`application-loop-plan.json` against every current request, and authenticates
the complete C call table. It uses all recorded trips; no trip count is supplied
by the caller. An application without a repeated window is rejected when this
option is selected. Target-declared analysis parameter attributes do not change
any executable variant.

The normal generated application invocation calls `_run_loop` with compiler
admission of that unchanged table. The low-level `_run` still chooses flat order.
The package manifest records this explicit option, and `native-verify` regenerates
the same loop plan, sources and handoff from the current source and selected plan.
The C driver is the single generic executor; there is no parallel Python callback
executor.

This path regroups the original call table and does not yet share programs.
It compresses host scheduling only and infers no new dependence edges,
buffer lifetimes, allocations or program equivalence. Target code parameterized
inside a region remains the target compiler's responsibility. Distinct embedded
quantizer exponents remain distinct compiled variants. Operand writes, program
loading and optional residency retain their existing target/session behavior.

The executable C fixture in `test_dispatch_loop_c.py` compares an unrolled driver
and a counted two-iteration driver with prefix/suffix, distinct compiled variants
and contexts, carried pointers and immutable constants. It also checks declined
and absent hooks, invalid admission, and second-iteration failure accounting.
Allocation failure and cleanup are checked on each return path. A 2,048-call
driver also compiles unoptimized with a 4 KiB stack-frame warning limit.
This is a generic runtime qualification; it does not qualify a model's numerical
execution or hardware.

`test_application_loop.py` publishes a complete application through the normal
compile API, independently replays it, and compiles and executes its normal
invocation for all 22 recorded trips. It also checks the handoff sources and
refuses a changed publication option. This synthetic gate is separate from any
whole-model numerical qualification.

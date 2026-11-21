# Compiler construction and performance from expert references

Build one reusable compiler from the empty target backend for the complete declared workload inventory.
The main performance objective is to learn from the supplied expert kernels, approach their performance,
and improve beyond it through compiler rules that generalize across operation families and unseen shapes.
Performance work starts as soon as a case passes; continue correctness work on other families concurrently.
Use large and connected workloads as performance targets and small cases as correctness/calibration coverage.

Both study arms receive identical expert kernel files and performance targets. Trust the supplied expert
measurements as targets with their recorded engine, input preparation and timed region. They are available
in `reference-kernels/performance_baselines/baselines.json` and ordinary per-case feedback. Model estimates,
gSIM measurements and FireSim measurements retain their evidence kinds. A ratio requires compatible
execution conditions; a missing matching measurement does not invalidate the supplied expert result.
Some historical targets use hardware loops: learn their mechanisms while respecting this run's ordinary
command policy. The reference instruction policy does not grant permission to emit prohibited commands.

Analyze the expert functions and compare each representative with the actual emitted candidate. Explain
differences in useful compute attainment, operand traffic and retention, transfer shapes, tiling, loop order,
preparation, fusion, overlap and completion. Close the largest observed gaps first. Exact instruction copying
is unnecessary; explain an alternative with equivalent useful effects or better comparable cycles. Preserve
the complete source/fusion boundary, numerical contract, layouts and observable outputs in every comparison.
No case/model/reference-name dispatch or copied per-case schedule can substitute for a compiler rule.

Use the shared transformation library, scheduling dialect, selected hardware facts and existing tools.
Read `OPTIMIZATION_BY_OPERATION.json` and the selected API/recipe catalogs. Explore cooperating transformations
and legal scheduling alternatives, consume the actual scheduling plan in emission, then verify the emitted
mechanism. An import, a changed knob or a scheduling annotation does not establish that an optimization ran.
When the library cannot express a supported mechanism, implement a reusable pass, lowering, adapter or
schedule consumer on the permitted compiler surfaces, with legality guards and a fallback. Compare baseline,
alternatives and ablations through the ordinary selfcheck/sweep/variant tooling.

Keep `OPTIMIZATION_RULES.json` as the running technique index: operation families, shape/resource guards,
supporting hardware facts, implementation and pass order/settings, successful and failed trials, exact
emitted/numerical/timing receipts and unresolved gaps. Reuse existing optimization-ledger/history tools.
Validate on neighboring shapes, tails, capacity boundaries and related operations. Freeze candidates before
independent held-out assessment; use the existing shape-policy machinery and preserve the incumbent outside
validated guards. A technique that only works on one reference shape still needs generalization work.

Parallelize helpers by operation family after the common frontend and encoding are available. Integrate
their checked work into one compiler promptly, preserve good alternatives, and keep authoring concurrent
with immutable-snapshot evaluation. The library, API documentation and ordinary feedback are shared by
both arms. Additional derived reference maps, CCA/lifting tools and their guidance are enabled only in the
guided arm; the raw arm derives its reference analysis from the supplied original files itself.

Use selected tooling for supported, separately labeled reference builds and observations. Reference files
do not replace the selected current hardware facts or numerical checks for generated code. Unknown timing
or mechanism evidence requires a concrete tool/compiler repair task, not an invented zero or speedup.
Do not weaken checks, change evaluation inputs or invent target capabilities. Do not launch another campaign.

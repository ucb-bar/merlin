# Proposed incremental tooling ablation

Status: deferred proposal only. This document defines later comparisons; it does
not prepare configurations, select a launch, or change the current EL1/EL2/EL3
wave. EL4 is being studied elsewhere. Map that run's actual configuration to the
rungs below before adding another full-system run.

## Nested additions from a scratch EL3 baseline

Start each run with an empty target backend and the same fixed EL3 environment:
C++ integration plus lightweight Python/xDSL authoring plumbing. Retain the
ordinary evaluation ABI. Supply no modern scheduling library, optimized pass
implementations, completed target backend or CCA service in the baseline.
Pin the entire EL3 support closure, including its generic logical-ABI,
command-buffer and tensor/numerical helpers; keep those identical at every rung.
Do not subtract capabilities from Fresh6: its existing scheduling/pass use and
inherited implementations would confound these additions.

| Rung | Addition to the preceding rung | Intended adjacent contrast |
|---|---|---|
| B: EL3 scratch | Fixed baseline above | Reference environment |
| S: scheduling abstractions | Scheduling dialect, its semantics and legality checks; no optimized pass implementations or catalog of optimized schedules | B→S: supplied scheduling abstraction and legality assistance |
| P: fixed transformation library | One pinned, reviewed catalog of pass/transform implementations and their applicability contracts | S→P: supplied reusable implementations |
| C: fixed-catalog CCA | CCA raw-reference analysis and suggestions restricted to that same catalog; automatic system kernel learning OFF | P→C: analysis and route-selection assistance |
| L: automatic kernel learning | Same CCA with automatic system kernel learning ON; new learned artifacts require the ordinary validation/admission gates | C→L: automatic extraction/generalization assistance |

These are nested marginal effects conditional on earlier capabilities. They do
not identify a standalone causal effect of every pass. If a rung requires an
additional implementation, expose and pin that dependency before defining the
contrast; never silently include it in the lower rung.

“System kernel learning” means the tool automatically derives reusable kernel
rules/abstractions or transform proposals from permitted references or
source-bound emitted artifacts and maintains the resulting learned artifact
bank. It does not mean LLM learning, extra model training, ordinary conversation
memory, or the agent writing code. C fixes the tool's catalog throughout the
run; L may extend its learned bank through the declared native mechanism.
Record proposals, acceptance/refusal, applicability and actual emitted use.
The native owners and exact switches must be mapped and audited before any
configuration is prepared; an unavailable learning mechanism is not an ON arm.

Every rung permits the author to implement equivalent scheduling, transforms
or analysis manually using its granted language/runtime and the same raw
references. This measures supplied assistance, not absolute expressiveness.

## Hold the experimental conditions fixed

Use identical 152-case public xs inputs (109 core, 40 scalability, three
connected programs), the 14 historical protected cases and their evaluation
policy, numerical contracts/fixtures, raw reference bank, latest judge, target
facts, provisional performance model, author LLM/version/effort, common prompt,
objective, helper limits and feedback budgets. Keep common prompt bytes fixed;
any necessary capability disclosure is minimal, versioned and reported.
Reference examples supply no hardware-cycle target figures. Generated candidate
programs retain the same primitive/no-FSM instruction policy in every rung.

Freeze the existing 34-cell GEMM/QK transfer panel before authoring. Preserve its
one-axis design, GQA sharing, all outputs, arithmetic bounds and independent
fixtures. Keep its sources, fixtures, results and generator outside every
author/helper grant. Preserve the accepted exposure labels: public anchors,
the historically seen Fresh6 M32 GEMM contract, and N2560 geometry exposure with
a different readout. Partial prior exposure is accepted; do not claim every
cell is unseen or remove cells after observing results.

Give every run four hours of authoring, fifteen CPUs and at most eight helpers,
with identical checkpoint/drain rules. Evaluate one frozen compiler, rules and
settings per run on the same contracts, without retuning or per-case winner
mosaics. Contract/fixture qualification alone earns no compiler execution,
ordering, hardware or timing credit.

## Priorities and interpretation

Prioritize B↔S because the abstraction contrast is central, then S↔P. Choose the
next bounded pair from P↔C or C↔L after mapping the external EL4 run and the
remaining scientific uncertainty. An external run is a matched control only
when source, references, budgets, model and tool/catalog/learning settings
actually agree; a shared “EL4” label is insufficient.

If time supports only one further pair, run that pair with a contemporaneous
baseline rather than an incomplete ladder. A leave-one-out comparison can take
priority when it isolates a working, independent feature of the full stack;
verify that its removal does not disable dependent passes or change the task.
Otherwise retain the adjacent-rung design and report the dependency limitation.
Repeat the strongest contrast with contemporaneous baseline runs when possible.
The experimental unit is an author run. Cases and repeated hardware calls are
not author replicas; an unreplicated ladder is descriptive, without p-values or
claims of robust stochastic effects.

Report public diagnostics and coherent-snapshot coverage AUC; lead plus helper
usage and its completeness; time/tokens to source/ELF-joined correctness and
same-domain improvement; static/numeric/ordering failures, repairs and repeated
failures; and explicitly evidenced emitted reuse/fallback. Report final frozen
transfer correctness, coverage, per-cell performance/regressions and exposure
strata separately. Distinguish proposed, admitted and emitted mechanisms, and
model estimates from qualified physical measurements. Missing evidence stays
unknown/null, never zero or an inferred success.

## Capacity for any later launch

At most three of our experiments may run concurrently, and at most five across
the machine including two Claude-owned experiment seats. Recheck live ownership
and retain the existing peer-safe reservations. The current fifteen-CPU pools
are 38–47 plus 56–60; 61–75; and 76–90. Never allocate CPUs 4–37 or 48–55.
This proposal allocates no CPUs and authorizes no preparation or launch.

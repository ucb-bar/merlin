---
title: Compiler construction and representation study
kind: design
status: draft
owner: experiments
last_verified: 2026-10-23
related: [joint_campaign_execution, ../guides/public_capability_planning, ../guides/conditional_compiler_pipelines, ../guides/artifact_evaluation]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_inventory.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/capability_brief.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/telemetry/token_budget.py
  - src/merlin/transforms/pipeline_policy.py
  - src/merlin/targetgen/isa_codegen_cli.py
---

# Compiler construction and representation study

The research question is which abstractions, representations and transformations
help agents develop a correct, generalizable, performant compiler with less work.
A faster known-case kernel, an available API or a declared optimization is not
enough to answer it. Comparisons below are proposed studies, not launched runs.

## Construct capabilities, evaluate continuously

Derive the initial capability inventory from every admitted public case. Keep
numerical policy, operations, layouts, shape regimes and connected graph structure
visible before authoring. Grouping organizes implementation; it cannot replace
individual case obligations. Capsule and model names are traceability only.

```mermaid
flowchart TD
  C[All public source requirements] --> G[Typed graph and region planning]
  H[Public ISA and RTL facts] --> I[Neutral instruction scaffold]
  G --> P[Shared passes and conditional policies]
  P --> S[Counted schedule and physical effects]
  S --> E[Target encoding and emission]
  I --> E
  P --> R[Merlin host preparation and execution]
  E --> X[Immutable compiler and executable]
  R --> X
  X --> V[Rolling numerical and required RTL checks]
  V --> F[Compact feedback and eligible model evidence]
  F --> P
  F --> S
```

Build a complete lowering path for a small capability early, then expand semantic
coverage while its immutable executions proceed. Do not wait for the entire
compiler before the first executable. Assign independent helpers graph, schedule
and host-lowering work behind shared interfaces; integrate into one compiler.
Keep a known working checkpoint and test the boundaries changed by each merge.

For the selected joint corpus, the public denominator is 250: 153 original
correctness, 20 calibration and 77 scalability/performance cases. Original
physical obligations remain mandatory, including their declared expected-refusal
semantics. Calibration witnesses are selected for unresolved mechanisms; large
development cases use Spike. All three cohorts may advance concurrently.
Hidden and complete-model validation remain separate obligations.

Primary performance effort goes to scalability cases. Original small cases have
low performance priority and remain useful regressions and eligible model
evidence. Passing implementations may be optimized; correctness is preserved
through exact-source checks, not by forbidding changes to passing code.

## Representation boundaries and evidence

Reuse the existing graph, counted scheduling and instruction representations.
Introduce a new dialect only when a decision must survive, be verified and be
transformed across lowering boundaries.

| Boundary | Mechanism | Required evidence |
| --- | --- | --- |
| Public ISA to instruction operations | Explicit ISA declaration and generated types/encoders | Independent encoding check; unknown semantics/effects remain explicit |
| Graph to legal regions | Liveness, numerical policies, grouping and fusion discovery | Every externally visible output and other consumer preserved |
| Alternatives to transformed IR | `PipelineSession`, `PassSpec`, semantic predicates, `evaluate_variants` | Consumed transformed artifact, actual pass records and independent correctness |
| Counted schedule to execution | Target-owned storage, configuration and completion effects | Legal dependencies, capacities, overlap and executable behavior |
| Host preparation to inference | Merlin-owned preparation and ownership lifecycle | One-time versus repeated work, invalidation and safe storage reuse |
| Performance selection | Matched scope, model epoch and structural roofline | Qualified estimates within domain; physical validation separately reported |

Generate mechanical instruction scaffolding equally for study arms. It supplies
no tensor compiler, operand packing, schedule or optimized kernel. A run using
it is a fresh optimized backend atop neutral scaffolding, not an empty tooling
environment. The target-owned source extractor must pin public source and refuse
unsupported syntax instead of inferring instruction semantics from names.

Pass variant evaluation enumerates explicit legal alternatives on private IR and
publishes each completed artifact. Machine search should handle parameter trials;
agents propose or improve algorithms, applicability and search spaces. Neither
an `applied` receipt nor successful compilation is a numerical or timing verdict.

`evaluate_variants(..., prefix=(...))` applies common registered transformations
once ahead of alternative exploration. Each alternative clones the prepared IR
and starts a fresh pass session; analyses containing references into the previous
IR are discarded and rederived. This is reuse of actual compiler transformations,
not merely task grouping. Agents can improve the common preparation, individual
passes, branch predicates or target realization independently while keeping the
same representation contracts. The [conditional pipeline guide](../guides/conditional_compiler_pipelines.md)
contains an executable-style example using the shared normalization pipeline.

Before fitting timing parameters, match external-memory behavior with the selected
FireSim configuration. Simplified functional memory and FASED timing are different
reference domains. Compare target-cycle request admission, response ordering and
backpressure, including initial state and runtime overrides. Preserve the deployed
hardware; change the software reference adapter. The strict qualification target
is at most 4% absolute relative error on every predeclared held-out case, with
complete case coverage. Final FireSim results remain independent of calibration.

## Token and feedback policy

New joint sessions record an aggregate soft target of 40,000,000 tokens: 32,000,000
working budget and 8,000,000 integration/repair reserve. Input includes cached
input; output includes reasoning. Count the lead and every descendant helper,
excluding copied parent history from helper usage. This is token traffic, not
billing. Missing usage is unknown and incomplete totals are lower bounds.

The installed controller accepts `--author-token-target` and
`--author-token-reserve`; launch configuration uses `author.token_target` and
`author.token_reserve`. The recorded selection survives resume. Explicit drift
requires a new run; historical runs without a selection gain no invented budget.
Crossing the target reports an overrun and narrows redundant exploration; it
does not stop necessary correctness/performance work or change qualification.

Use the canonical operator live accounting command outside the read-only run:

```sh
python -m merlin_experiments.phase1.telemetry.author_agents \
  --run-dir "$RUN_DIR" --out "$OPERATOR_DIR/author-usage.json"
```

Each refresh records all-author usage and the soft-budget state. Final timing
reports also include it. Preparation/review and separate study runs require their
own visible accounting; they are not hidden inside the author-run denominator.

Fresh inventory-bearing runs use a compact round brief: class denominators,
separate reported/L2/L3 passes, snapshot identity, three shared failure groups,
scoped performance priorities and bounded approach notes. Full redacted records
remain on disk. Give helpers capability-specific briefs and references; avoid
copying the whole history, API catalog or result collection into every turn.

## Readiness and subsequent comparisons

Infrastructure acceptance requires a real consumed pass transformation, independently
checked instruction encoding, all public inventory entries, immutable source
attribution, correct resume behavior and representative end-to-end execution.
Synthetic infrastructure tests establish their tested behavior, not complete
target readiness. Missing native memory equivalence, candidate execution or
timing qualification must remain explicit outstanding gates.

Once ready, freeze definitions for these comparisons before running them:

| Comparison | Isolated question |
| --- | --- |
| Existing workflow versus capability-organized construction | Does seeing all public requirements and implementing semantic families reduce repeated repair? |
| Same algorithms with source-oriented versus typed/composable access | Does the representation improve agent effectiveness independently of supplied algorithms? |
| Fixed library versus editable shared passes | Does compiler co-evolution improve performance and generalization? |

Keep hardware facts, neutral scaffolding, public examples, independent checking,
model versions, budgets and agent limits matched as appropriate. Declare existing
measured scheduling defaults as supplied priors. Private coworker kernels and
private comparison scores are excluded from author inputs. Generic suggestions
need independent public ISA/RTL/compiler justification.

Measure time/tokens to first executable and complete coverage, invalid candidate
rate, reusable pass changes, large-case performance, integration repairs and
generalization to withheld shapes/graphs. Rescore comparable final compilers with
the same observer version. An interface that adds overhead, helps only correctness,
or merely exposes better supplied algorithms is a valid negative or narrower finding.

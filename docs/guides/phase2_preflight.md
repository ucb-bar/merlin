---
title: Prepare and check a Phase 2 experiment
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-20
related: [phase2_corpus_search, phase2_public_diagnostic_launch, phase2_archive_screen, shared_pass_phase2]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_diagnostic_launch.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/timing_domain_readiness.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/compiler_pass_preparation.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/diagnostic_pair_check.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py
---

# Prepare and check a Phase 2 experiment

Setup checks and diagnostic execution can run before formal Phase 1 completion.
The ordinary performance campaign still needs its original functional, corpus,
runtime and timing admission. Successful setup or process completion cannot issue
those qualifications. Keep each refused member and every missing dependency in
the complete experiment inventory.

## Preserve the starting compiler

Select the exact Phase 1 package, full file inventory, numerical policy, hardware
facts, software contract, compiler profile, toolchain and runtime. Keep that
package immutable. The public diagnostic launcher verifies it and creates
separate baseline and candidate directories before retaining either revision.
The candidate edit contract fixes the permitted compiler functions and pass
directories before authoring starts.

An explicitly selected additional shared library applies to both treatments.
The original preparation owner may add its reviewed connection to the copies
before edit authority is frozen. This never changes the original Phase 1 package
or retrospectively certifies its use of that library. Current compiled products
and complete outputs still need independent checks in the selected environment.
See [shared-pass integration](shared_pass_phase2.md) for pass ordering, ordinary
policy choices, additional mechanisms and qualification boundaries.

## Check the complete development inventory

Retain original public source declarations, full inputs, independent references,
numerical contracts, source identities, objective/guard roles and measurement
boundaries. A shape inventory or an independent reference does not prove that
the compiler realizes the accelerator work.

Phase 0's original materialization and writer owners admit executable public
programs. Target experiment entrypoints may compose these owners with the
selected compiler and runtime. They must preserve the complete source, final
outputs, private exclusion checks and every refusal. A metadata-only intake
cannot replace actual compilation, accelerator-effect checks or final writer
publication. Physical capacity and overlap need emitted storage and completion
evidence in addition to logical shape pressure.

Keep the performance cohort separate from cheap functional guards. Freeze its
membership, roles and contracts before scored search; successful cases cannot
silently replace slow, unsupported or failed objectives.

## Inspect archived timing without launching a target

The installed timing preflight reopens the explicitly selected public archives,
per-record bindings, original observations and complete declared cohort:

```sh
merlin-phase2-timing-readiness \
  --inputs TIMING_INPUTS --inputs-sha256 TIMING_INPUTS_SHA \
  --output FRESH_TIMING_REPORT
```

The input schema is documented in
[archive calibration and readiness](phase2_archive_screen.md). The command
launches no compiler, simulator, author or fit. It preserves source refusals,
unsupported semantics, exact TRAIN source coverage, unavailable observations,
original held gates and optional capacity-validation obligations.

Exit zero means that the report was written. Read each member and archive's
status. The report does not create a live Count, native-semantics, source,
timing or runtime owner. Raw activity bounds, complete source coverage and
statistical validation remain separate gates. Missing compatible public labels
must return to the calibration producer; changing ranges or thresholds cannot
qualify an unsupported prediction. Phase 2 iteration can remain Spike-only once
the declared domain has actually passed those gates.

## Check a launch plan, then execute an admitted diagnostic

Use the existing launcher for a nonexecuting plan check:

```sh
merlin-phase2-public-diagnostic \
  --plan PLAN --plan-sha256 PLAN_SHA \
  --output FRESH_PLAN_REPORT --check-plan
```

This checks plan data and inventory; it does not construct the live factory or
queue. Execution requires the original source-pinned factory and compiler
profile, actual allocated CPUs, common CPU slot namespace, finite immutable
clock and enforcing process supervisor. The launcher itself is not an aggregate
CPU, memory or storage supervisor. Tool interpreters and import roots must match
the manifest's actual command, including an explicit `python3` launcher.

Then run an unchanged control and a distinct retained candidate through the
[public diagnostic queue](phase2_public_diagnostic_launch.md). Select the same
library and preparation owner for both. The normal path must perform actual
compilation, complete-output checks, current source/effect checks, requested
fresh counts, feedback validation, decision recording and descendant cleanup.
No saved PASS or copied counter report establishes fresh execution.

An explicit prepared-pair protocol checks both emitted programs after the arms
finish and repeats the original check in the queue's parent before publication.
The target owns instruction effects, storage ownership and completion semantics.
Optional checked schedule metrics expose preserved products and outputs, commands,
requested bytes and allocations. They supply no physical bandwidth, measured
cycles, utilization or complete roofline attainment.

## Start ordinary search only through its original gates

The diagnostic route permanently reports `DIAGNOSTIC_ONLY`, pending deeper tiers
and null measured cycles. Even a passing edited-program pair is insufficient for
an incumbent or a convergence claim.

Once the frozen required cohort, original functional/runtime admission and
qualified timing domain are available, use the existing
[ordinary search configuration and coordinator](phase2_corpus_search.md).
Reserve resources before launch, keep the evaluator separate from compiler
edits, preserve the declared objective and stop rules, and retain a final
verification reserve. The coordinator owns retained candidates, bounded parallel
evaluation, feedback and decisions; setup scripts do not replace that loop.

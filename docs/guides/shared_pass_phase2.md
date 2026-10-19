---
title: Use shared passes in compiler generation
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-18
related: [../reference/shared_compiler_passes, phase2_corpus_search]
code_refs: [src/merlin/targetgen/tool_profile.py, src/merlin/targetgen/tool_profile_shared.py, src/merlin/targetgen/common_transforms.py, packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_stage.py]
---

# Use shared passes in compiler generation

Merlin's shared compiler library provides checked scheduling, storage and
composition algorithms for generated compilers. Select a library release before
Phase 1; Phase 2 inherits that same release and can tune the compiler's decisions
without changing the checks used to admit its output.

This guide covers the `merlin.common_compiler_substrate.v6` release profile.
Versions 1–5 retain their existing APIs and verification policies. An already
certified compiler keeps its frozen profile; changing Phase 2 options cannot
add library access to it.

## Inputs and backend requirements

The compiler needs a selected hardware configuration, a numerical policy, and
resource and code-growth budgets. Optional controls can restrict the supported
optimization choices.

| Input | What it establishes |
| --- | --- |
| Versioned RTL facts | Extracted dimensions, element widths, capacities, banks and address limits |
| Software and ISA specification | Arithmetic, layouts, conversions, instruction effects, ownership and completion rules |
| Numerical policy | Permitted arithmetic and the original admission thresholds |
| Budgets and controls | Bounds on compilation work, code growth and legal optimization choices |

A backend must also supply the applicable instruction implementations, physical
mapping, encoding and decoding, and independent execution qualification.
Structural RTL facts alone cannot establish arithmetic equivalence, completion
semantics or performance costs. Missing facts remain unknown.

Shared entries accept typed source or schedule IR. A compiler must call the
appropriate entry with its selected contracts and qualified backend bindings.
Importing the library does not enable passes on arbitrary backend output or
automatically replace a saved-model compiler's routing. See the
[pass reference](../reference/shared_compiler_passes.md) for each entry's accepted
IR, required capabilities and fallback behavior.

## Prepare a library release

Choose the Merlin source tree, its license directory, and a fresh destination
for the release. `MERLIN_SOURCE_ROOT` must be the absolute canonical Python
source directory (`<checkout>/src`) used by the running Merlin installation;
`MERLIN_LICENSE_ROOT` is the directory containing `LICENSE` and `NOTICE`.

```sh
python -m merlin.targetgen.tool_profile_shared \
  --python-source-root "$MERLIN_SOURCE_ROOT" \
  --license-directory "$MERLIN_LICENSE_ROOT" \
  --output-dir "$TMPDIR/common-v6"
```

The command stages the shared library, verification support, licenses and a
manifest of source identities. Retain the release while a compiler uses it.
Staging files does not itself grant a candidate access to them; bundle creation
selects the permitted APIs and resources.

For experiments that expose additional authoring tools, repeat `--tool NAME`
for the selected registered tools. Each treatment's bundle still determines
which tools and resources that treatment may access.

Pass the prepared release to the normal bundle generator:

```text
--common-substrate merlin.common_compiler_substrate.v6
--python-source-root <the prepared common-v6 root>
```

Keep the hardware and software inputs, public corpus, numerical policy,
toolchain and qualification selections explicit. Phase 1 freezes the release
and its access profile. Run Phase 2 through the ordinary source-revision search
with that inherited profile.

## Choose optimization policies in Phase 2

The compiler can choose supported blocking, loop order, operand retention,
composition, copy elimination and configuration cleanup from the current
operations, shapes, layouts and resources. Changes are ordinary compiler source
revisions; each queued evaluation uses the frozen bytes of that revision.

Pass order is constrained by prerequisites, IR levels and invalidations. Current
entries validate explicit orders. A rewrite that changes instructions requires
new emission and program-size checks; spill insertion requires fresh allocation
and completion checks. Automated search over complete-model pass sequences is
not provided by the release.

Keep physical capacities, numerical policy, qualification and verification gates
fixed during tuning. Most `ScheduleBudgets.max_*` fields bound verification
work; increasing them is not a performance optimization. When costs are unknown,
use the entry's documented conservative policy or supply calibration evidence.

## Guarantees and limitations

The shared checks bind analyses to the current IR, hardware and software
selection, numerical policy and library version. Stale analyses and invalid
explicit ordering are rejected. Repairs use bounded budgets and recompute the
affected analyses.

Emission checks operate on the representations each entry explicitly supports.
Where required, actual emitted instructions and effects must be reconstructed
independently. A selected plan or a successful link alone does not establish
correct execution. Unknown operations, effects and unsupported handoffs remain
barriers. Retain the original legal route when an optimization is unavailable;
otherwise propagate its diagnostic.

Capability requirements differ by pass. For example, device spills require
declared register classes and exact save/reload semantics; complete-row
reductions require the original reduction domain and numerical order. The
library does not infer addressable accumulators, asynchronous completion,
supported masks or new approximation policies from a target name or shape.

Neutral execution tests establish shared-mechanism behavior. They do not qualify
an accelerator backend or establish hardware latency. Consult the
[supported domains and limitations](../reference/shared_compiler_passes.md)
before selecting an entry.

## Compare compilers fairly

Every V6 treatment must receive the same shared algorithms and release identity.
V6 includes generic ISA parsing and generation, so comparisons must account for
that common access. Preserve the separate grants for target facts, software
specifications, runtime tools and independent qualification. Protected reference
implementations and evaluation answers remain outside authoring inputs.

Freeze tuned policies before evaluating unseen workloads. Use identical inputs
and unchanged numerical thresholds, checking complete outputs and carried state.
Measure complete models, including host arithmetic, packing, copies, layout
conversions, dispatch and accelerator work. Report compilation time, peak memory,
code size, commands and requested bytes separately from paired execution timing.
Functional Spike instruction counts are not accelerator latency cycles.

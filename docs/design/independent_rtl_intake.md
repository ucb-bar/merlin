---
title: Independent RTL intake for fresh compiler authoring
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-08
related: [docs/design/component_compiler_convergence.md]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/rtl_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/command_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/accessor_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/source_predicate_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/software_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_generation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_coverage.py
  - src/merlin/targetgen/rtl/source_selection.py
  - src/merlin/targetgen/rtl/introspect.py
---

# Independent RTL intake for fresh compiler authoring

Phase 1 authors a fresh target compiler from the admitted public RTL information,
minimal software semantics, generic compiler library and independent development
inputs. Phase 2 optimizes that compiler. Neither phase may import the handwritten
compiler, its adapters, schedules, transcribed ISA tables, workload captures or
performance history. Independently derived measurement tools have a separate
owner from compiler candidates.

## Structural derivation authority

`issue_independent_hardware_intake` runs the fixed existing FIRRTL-to-HW producer
into a fresh output directory. It checks exact SoC and core HW bytes against the
selected source bundle, derives and checks the instance hierarchy, and runs the
generic structural census on those exact FIRRTL and hierarchy bytes. It accepts
no existing facts table, target-provider import or ISA header. A command stored
in an old receipt is never executed.

The protected coordinator selects the public source bundle and native tool
before authoring and supplies the campaign's forbidden roots. The intake refuses
those paths before reading their source bytes, including symlink indirection.
It binds the exact descriptor, RTL input/output files, producer executable and
generic extraction sources. Saved JSON is an audit record. Only a live issued
`IndependentHardwareIntake` object passes `verify()`; reconstructing the object
from a saved receipt grants no authority.

`fact(path)` returns only a concrete structural value that the executed census
derived. Missing values and `None` refuse. `public_facts()` returns a decoded copy
of the complete structural projection and its explicit unknowns. The complete
`contract/hardware_facts.json` member must match that projection exactly before
a public component view is admitted. This check covers hardware facts; the fresh
authoring input owner separately checks the software spec, complete view, corpus,
generic library and tool runtime.

## Corpus binding

Ordinary `generate_target(..., component_only=True, hardware_intake=intake)`
binds the issued identity through `bind_entries` into the generation identity
and private coverage report. The exact selected raw facts must equal the issued
facts. A support-provider source, instruction-semantics transcription, ISA header
or extra RTL source outside the intake is refused. A legacy component report has
no independent intake binding and cannot authorize a fresh Phase 1 experiment.
Adding a hash to an old report does not rerun or qualify its derivation.

Fresh evidence selection with the issued hardware intake skips provider code,
backend readout hooks, ISA-header taxonomy and backend-declared execution
capabilities. Headers and legacy instruction-semantics resources are refused.
Execution support stays unknown until its independent live runtime owner issues
qualification; hardware observation cannot certify a compiler or support tool.

The minimal software spec owns semantic and numerical choices. Campaign
objectives remain in the recipe. Independent example graphs supply reviewed
operation and effect semantics; shape frequencies, held workload data and
historical cycle labels do not select the corpus or its weights.

## Command-source observations

`issue_independent_command_intake` consumes the live structural intake, an exact
clean tracked public hardware-source checkout/file/Git commit, explicit source
declaration boundaries and the selected native CIRCT serialization tool. It
derives declaration names/values and packed source bundle layouts; parametric
widths remain unknown. It executes a fixed fresh generic serialization command
over the replayed core HW and preserves typed HW parameter attributes with the
generic xDSL parser.

The generic observer traces local module inputs through exact extracts and
contiguous concatenations to equality comparators. Registers, memories,
instances, noncontiguous concatenations and transformations stop that trace.
These observations are local evidence; they do not establish complete ISA
legality, cross-instance routing, numerical effects or prohibited-command roles.
Direct instance-result comparisons are recorded separately with the exact
instance/module/output names and width. Their input correspondence and effects
remain unknown; this observation never aliases a queued output to a module input.
The source commit and configured Git remote record selected local provenance;
they do not authenticate a remote server or reproduce historical elaboration.

The complete `contract/command_facts.json` member can be checked by the live
`IndependentCommandIntake`. Saved receipts cannot recreate the authority.
Physical instruction encoding, entry/completion/memory/timer ABI and simulator
equivalence require independent semantic and execution qualification. Native
fixture tests exercise actual calls, tracking, drift and exclusions; only the
selected real tool/RTL replay supplies target observations.

## Public native accessor observations

`issue_independent_accessor_intake` generates and compiles a fixed native C++
observer from a protected minimal header/type/member specification. The
specification supplies no encoding values, shifts, masks, arbitrary expressions
or instruction effects. The selected public header must match its clean tracked
source before compilation, and every observed non-system compiler dependency
must match the selected public checkout and exact Git commit. A mismatching
installed SDK is refused. Native compiler, specification, public headers, actual
program, executable, compiler output and complete observation records are pinned.

The observer calls the actual public accessors over zero, every single bit,
all ones and independent mixed words. Contiguous field projections are admitted
only when every observed word agrees. Instruction lengths come from actual
source accessor calls; there is no copied length table. `decode_words` executes
the pinned observer again over the requested words, emits the complete input,
output and invocation receipt, and rechecks the live authority. Saved facts and
copied objects cannot authorize decoding. Public facts are model/header ABI
observations. Physical CPU byte order, instruction legality and effects, selected
RTL/bitstream equivalence, ownership, synchronization and timers stay unknown.

An independently reviewed software policy may explicitly prohibit selected
public source symbols. That defines the experiment restriction; it does not
establish hardware numerical behavior. Symbol resolution must use the actual
admitted declaration scope and reject outside-scope aliases and ambiguity.

`issue_independent_source_predicate_intake` follows protected selections of exact
public source bindings and operand names in the same clean tracked checkout as
the command authority. It reads only pure comparisons against admitted symbol
declarations, parentheses, conjunction and disjunction. Unsupported expressions,
numeric transcriptions, other operands, ambiguous definitions or outside-scope
symbols refuse. The exact source expression, line span, referenced symbols and
membership over the admitted finite declaration domain are retained and pinned.
The policy owner can corroborate its explicit forbidden symbols against these
actual source predicates; the predicate reader assigns no roles from names.
Source-to-elaborated-RTL correspondence and physical effects remain unknown.

## Protected minimal software selection

`issue_independent_software_intake` requires the live hardware selection and an
explicit protected review. The review binds the exact independently composed
software source, independent example roster, numerical choices and source-call
correspondences for every operation owner. The issuer validates the closed raw
semantic schema before following graph sources, then replays each original
frontend graph through the ordinary semantic-basis owner. Review status or a
matching source hash alone cannot pass this intake.

The raw software source contains only schema, target, review status, numerical
semantics and operations. Numerical models are generic independent engines
without source/provider hooks. Shape bounds, layouts, performance objectives,
schedules, model/capture/history/profiling fields, evidence narratives and backend
metadata refuse at every admitted level. Extended numeric models need a separate
reviewed minimal schema; they are not silently admitted through arbitrary fields.
The protected selector checks campaign exclusions before source bytes are read.
Every exact review/source/graph and reader byte is pinned. Historical source
origin, hardware numerical support, whole-domain functionality, physical effects
and compiler/runtime correctness remain separate unknowns.

The complete normalized minimal projection is published at
`contract/software_spec.json`. Only the live `IndependentSoftwareIntake` can
verify it; saved receipts or reconstructed objects cannot authorize authoring.
Ordinary component generation binds `software_intake_sha256` into its generation
identity and top-level private coverage report. Evidence selection checks the
same independently selected software source before parsing it; corpus binding
preserves exact reviewed numerical choices and semantic operation selectors.
Fact-derived placement remains separate from this semantic selection.

Fresh authoring must require both the independently issued hardware and minimal
software authorities, matching report identities and exact complete public
projections. A legacy reviewed software specification has no such authority and
cannot enter the author view merely because its bytes match an older report.

## Scope and incomplete evidence

Actual reproduction of selected FIRRTL-to-HW bytes establishes this structural
derivation. It does not establish the historical source-to-FIRRTL build origin,
instruction encodings or numerical semantics, simulator equivalence, bitstream
correspondence, compiler functionality or performance. These are explicit
unknowns. Matching filenames, configuration symbols, receipt status and byte
hashes cannot substitute for independently observed production and qualification.

Fresh experiment acceptance still needs actual ordinary compilation and target
execution, complete output and ownership checks, zero prohibited instructions,
independent feedback qualification, actual author isolation, and matched final
FireSim evaluation. Local native producer fixtures test replay and exclusions;
they do not qualify a hardware target or claim model cycles.

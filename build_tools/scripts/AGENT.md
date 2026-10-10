# AGENT.md — build_tools/scripts

## Purpose

Maintenance/validation scripts (e.g. `check_structure.py`).

## What belongs here

- Tracked source for maintenance, build automation, checks and documentation generators.

## What does not belong here

- Application/library implementations (use `src/merlin/` or the owning optional distribution).
- Generated artifacts, logs, caches or schema definitions.

## Invariants

- These scripts are tracked source, not disposable build output.
- Never commit generated artifacts here.
- Write generated products beneath the configured `out/` root using the shared path helpers.
- Keep checks source-layout-aware across core and optional distributions.
- Installed test origin exceptions name only exact archived test modules,
  paths and SHA-256 bytes in a closed roster retained before collection and
  reopened at completion. Production imports must still come from the own-site
  installation; no namespace-wide test exception or checkout fallback exists.
  Load and check installed parent packages before pytest can create archive-only
  parents with their names during importlib collection.
  This is qualification bookkeeping and an execution tripwire, not isolation.
- Selected native suites may retain their explicitly supplied test-child
  environment before launch. Keep actual values in private operator output,
  bind the complete mapping including its selected record path, and recheck
  record bytes before execution. This is replay input, not runtime authority.
- Native test input mappings are explicit bounded regular JSON files with a
  complete suite-declared string-key roster. Reopen their bytes before and after
  each command; they cannot replace selected tool/source environment keys. Their
  values remain private operator inputs, not source or runtime qualification.
- `host-ranked-descriptors` is core-only and requires the exact 46 pure and two
  native original test identities with zero skips. Compiler Python preserves its
  selected entry/prefix; llc, clang and clean model2MLIR source are explicit
  selections. Packaging proves finite descriptor transport only; logical body,
  physical storage, dependency closure and runtime/effect authority stay separate.
- `original-scalar-binary-sources` archives the exact five source/conversion test
  modules and their scalar fixture support. Its complete 110 pure plus six native
  identities require zero skips and explicit public source/tool selections.
  Preserve original literal/typed argument/body checks and all numerical policy,
  reference, owner, effect, resource and mandatory admission unknowns; packaging
  and finite registered construction never establish those premises.
- Numeric falsifiability uses the optional evaluator's constant-candidate audit.
  Its default public scope is tracked capsule declarations outside hidden paths.
  Missing oracle outputs are unmeasured; an explicitly partial CI invocation must
  still fail on measured accepted constants or malformed public declarations.

- `formal-instruction-selection` retains the four complete declaration/caller
  control modules and their exact sibling fixture support. Every selected case
  must execute without skips. The native issuers and model builds in these
  controls are diagnostic substitutes; installed wiring grants no instruction,
  source-role, numerical, runtime, physical or experiment qualification.
- `rtl-source-bindings` is core-only and retains all 266 original reader,
  operator-command and file-boundary controls with zero skips. Archive the exact
  three counter fixture siblings; production reader/command imports must come
  from the own-site package. Installed source observations grant no instruction,
  endpoint role, counter sample, numerical, runtime or performance authority.

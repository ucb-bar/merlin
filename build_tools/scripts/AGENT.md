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
- Selected native suites may retain their explicitly supplied test-child
  environment before launch. Keep actual values in private operator output,
  bind the complete mapping including its selected record path, and recheck
  record bytes before execution. This is replay input, not runtime authority.
- Native test input mappings are explicit bounded regular JSON files with a
  complete suite-declared string-key roster. Reopen their bytes before and after
  each command; they cannot replace selected tool/source environment keys. Their
  values remain private operator inputs, not source or runtime qualification.
- Numeric falsifiability uses the optional evaluator's constant-candidate audit.
  Its default public scope is tracked capsule declarations outside hidden paths.
  Missing oracle outputs are unmeasured; an explicitly partial CI invocation must
  still fail on measured accepted constants or malformed public declarations.

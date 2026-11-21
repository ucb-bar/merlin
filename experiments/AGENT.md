# Experiment definitions

The catalog selects authored definitions in `definitions/` and target workflows
under `examples/compiler-generation/`. Shared templates belong in `templates/`;
authored target and workload sources belong under `examples/`. Installed test-generation
and backend-generation engines live in `src/merlin/development/`.

Keep generated corpora, bundles, captures, compiler snapshots and reports under
the configured `out/` root. Historical payloads are explicit run inputs; never
silently select a retained artifact as a fresh reviewed release. Preserve original
source and acceptance identities during replay. Definitions resolve paths relative
to themselves; descriptors and frozen records retain their declared conventions.

Do not commit hidden capsule identities, golden answers, credentials or
machine-specific paths.

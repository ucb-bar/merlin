# Public definition resources

Ship the versioned schema in the package wheel. Changing its accepted meaning
requires compatibility tests; unknown command and configuration fields fail closed.

`legacy_entrypoints.json` is the closed adapter-to-source binding table during
migration. It declares execution locations, not hardware facts. Entries must be
relative Python files inside the configured checkout; no shell expressions or
operator-controlled commands are accepted.

`legacy_target_access.json` records historical target-named oracle/grader import
identities that may remain in an installed Python toolchain after their target
support has moved out of tree. This is mandatory host-private sandbox policy,
not an agent grant. Missing or malformed bytes refuse sandbox startup; frozen
run inventories bind these exact bytes.

`phase1_launch_config.schema.json` and `phase0_release_config.schema.json` validate
operator launch and release configs (`merlin_experiments.phase1.launch_config`).
They select operator inputs; they grant nothing to an author and qualify nothing.

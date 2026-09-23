# Public definition resources

Ship the versioned schema in the package wheel. Changing its accepted meaning
requires compatibility tests; unknown command and configuration fields fail closed.

`legacy_entrypoints.json` is the closed adapter-to-source binding table during
migration. It declares execution locations, not hardware facts. Entries must be
relative Python files inside the configured checkout; no shell expressions or
operator-controlled commands are accepted.

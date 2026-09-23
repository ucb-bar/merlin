# Experiment definitions

This is the catalog and its generic templates. Target-specific definitions live
under examples/<target>/experiment.yaml, referenced by this catalog without copies.
Shared authored inputs live in `templates/`; definitions select them explicitly.
Neither location is another implementation
of the phase engines. Keep generated execution artifacts beneath the configured
run root. `reference-data/` holds byte-preserved historical analysis evidence and
explicitly selected public derivation references;
it is read-only input, not an execution destination or a grading answer surface.
Do not commit grading goldens, hidden capsule identities, credentials, or
machine-specific paths. Paths inside definitions are relative to the definition.

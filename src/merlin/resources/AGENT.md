# src/merlin/resources

This package owns reviewed shared contracts, schemas, prompt templates and the
selected public benchmark definitions shipped with Merlin. Resolve them through
`merlin.common.paths`; editable and installed packages must use the same owner.
Update `build_tools/package_resources.json` when shipped membership changes.

Keep executable implementations, generated output, study captures, corpora,
private holdouts and grading answers elsewhere. A prompt or schema is not an
independent correctness oracle. Resource packaging does not grant candidates
access to protected evaluation inputs or permission to change acceptance policy.

# Optional distributions

Each directory owns one independently built distribution. Core primitives live under
`src/merlin/`, never as a copied implementation here. Shared `merlin.*` namespace contributions
must have unique file and command ownership; core owns shared namespace initializers.

Keep phase execution, research studies and optional framework dependencies in their owners.
Register trusted evaluation access identities before moving code. Build products belong under
the repository's `out/build/python/`, using the shared extension setup hook.

Validate real wheels with `build_tools/scripts/check_distribution_layout.py`, not just editable
imports. Existing command and import identities must remain compatible where documented.

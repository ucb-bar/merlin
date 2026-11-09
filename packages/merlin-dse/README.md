# Optional DSE research tools

DSE implementation now ships in the main `merlin` package. From the repository root,
install `python -m pip install '.[dse]'`; select `xdsl` for compiler-backed analysis and
`plots` for optional figures. The main package supplies `merlin-dse`,
`merlin-design-pressure` and `merlin-dse-guidance`. This directory retains tests and
documentation, not another installable code distribution. Missing measurements remain
unavailable, never invented.

The implementation retains the public import names `merlin.dse`,
`merlin.design_pressure` and `merlin.dse_guidance` under `src/merlin`, with
`src/merlin_dse/cli.py` as the typed CLI facade. Existing class identities,
imports, pickles and downstream integrations remain stable without duplicated
modules or import hooks.

The research owners share workload representations and analysis machinery. Shared capture identity, shape/role
classification, evidence ordering/confidence, and deterministic Cartesian sweeps
belong to core; compatibility imports point to those same implementations.

The historical evidence fixture is read-only at
`experiments/reference-data/dse/case_study`, not a grading golden. Tools using it
still require an explicit checkout/reference-data root; it is not wheel data.
Existing research tests remain in `merlin/tests/dse/`; package-seam tests live
here. This packaging move does not change scoring, evidence levels, or numerical
semantics.

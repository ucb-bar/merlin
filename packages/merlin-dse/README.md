# Optional DSE research tools

Install alongside core with `pip install ./packages/merlin-dse`. Existing commands
`merlin-dse`, `merlin-design-pressure`, and `merlin-dse-guidance` are supplied by
this distribution. Install `[xdsl]` for compiler-backed analysis and `[plots]`
for optional figures. Missing measurements remain unavailable, never invented.

The implementation retains the public import names `merlin.dse`,
`merlin.design_pressure`, and `merlin.dse_guidance` through Merlin's extended
package path. This is an intentional migration seam: existing class identities,
imports, pickles, and downstream integrations remain stable without duplicated
modules or import hooks. `merlin_dse.cli` is the distribution-owned CLI facade.

The three research modules are one distribution because they share workload
representations and analysis machinery. Shared capture identity, shape/role
classification, evidence ordering/confidence, and deterministic Cartesian sweeps
belong to core; compatibility imports point to those same implementations.

The historical evidence fixture is read-only at
`experiments/reference-data/dse/case_study`, not a grading golden. Tools using it
still require an explicit checkout/reference-data root; it is not wheel data.
Existing research tests remain in `merlin/tests/dse/`; package-seam tests live
here. This packaging move does not change scoring, evidence levels, or numerical
semantics.

# src/merlin/analysis/mining

This package consumes kernel observations to prepare schedule packages, preserve
fork lineage and reconstruct optimization proposals. `references/` collects
reference-package evidence; `reconstruction/` turns observations into proposals.
These are research consumers, not observer SDK exports.

Host package loading lives in `merlin.compiler.backend.host`. The historical
`RvvPackage` name is that same class. Target schedule rendering, blocking policy
and microkernel realization belong to the selected OOT provider; importing this
package must not select or register one.

`from_strategy.mint_fork` reexports the implementation in `fork.py`. Keep one
implementation, preserve the public call signature, and bind rendering to the
selected provider's source identity. Retain parent/evidence lineage and refuse
changed provider sources during rendering.

One installed Merlin distribution owns this code. Do not restore namespace
package extension, a separate mining extra or a second implementation tree.

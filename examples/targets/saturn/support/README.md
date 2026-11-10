# Saturn support for Merlin

This is a host-owned support provider, not an evaluated compiler submission.
It preserves the Saturn vector backend and dialect formerly stored in Merlin.
Select it explicitly:

```sh
export MERLIN_TARGET_PATH=/absolute/path/to/rvv-mlir/saturn-support
```

The provider target is `saturn`; its backend registers as `saturn_vec`.
The sibling `merlin-support/` provider instead describes `rvv`. Do not rename
either identity or silently use this provider when only RVV was selected.

Merlin supplies shared RVV emitters, the dialect factory, command-buffer
semantics and Spike harness integration. This package owns the Saturn binding,
dialect plan and target contract. Its reference geometry is not a newly
extracted RTL fact. Native execution requires separately provisioned compilers,
Spike and harness resources; importing support does not qualify those tools.

`provenance.json` records the exact source revision and hashes of the seven
migrated files. Historical comments in those byte-preserved files describe
earlier implicit discovery; current Merlin requires explicit support selection.
No backend, simulator or hardware qualification is implied by this migration.
The existing RVV publication's certificate does not cover this new support.

Keep this directory host-private during compiler evaluation. Exported candidate
compiler payloads must not include support implementations or oracle answers.

`plugin.matrix_lowering` requires explicit Saturn support selection. The OPU shim
and matrix-unit declarations now live in this provider; shared int8 contraction
rewriting remains in Merlin. Core `opu_kernel`, `opu_cert`, and `opu_isa`
dependencies still await extraction. Unit and hardware configuration remain explicit.
This interface adds no compiler, simulator, hardware or source-closure qualification.
`matrix_adapter_migration.json` records the contract transformation and new adapter;
the original seven-file provenance remains unchanged.
`opu_shim_migration.json` records the subsequent physical shim/declaration move
and adapter/contract transformations, without rewriting that historical record.

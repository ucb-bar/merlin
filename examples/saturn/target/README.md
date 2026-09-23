# Saturn reference metadata

This directory retains reference contracts and documentation only. Executable
backend and dialect implementations live in the local RVV companion repository's
`saturn-support/` provider. The exact revision is recorded in
[`target_support.json`](../../../build_tools/upstreams/target_support.json).
Those companion changes are local and have not been pushed upstream.

Select that provider explicitly before loading the Saturn dialect or the
`saturn_vec` backend:

```sh
export MERLIN_TARGET_PATH=/absolute/path/to/rvv-mlir/saturn-support
```

The companion's separate `merlin-support/` directory describes target `rvv`;
it does not select Saturn. Neither identity is the `saturn_opu_mxv256d128`
experiment target. These names must not be treated as interchangeable aliases.
The shared generic RVV emitters remain in Merlin; this provider reuses them.

Reference discovery alone cannot load executable support. The selected provider
owns the contract and dialect plan used for execution, without falling back to
this reference tree. Hardware parameters here describe the historical reference
model and are not new RTL-derived facts or proof of an OPU-enabled configuration.

Backend import, pure emission and dialect construction tests do not qualify
compilers, simulators or hardware. The support code remains host-owned and must
not be bundled with an evaluated compiler candidate. Broader OPU support
migration remains unfinished.

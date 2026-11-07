# Saturn reference metadata

This directory retains reference contracts and documentation only. Executable
backend and dialect implementations live in the provider vendored at
[`../support`](../support), a byte-identical copy of the RVV companion's
`saturn-support/` directory at the revision recorded in [`../SOURCE.yaml`](../SOURCE.yaml)
and [`target_support.json`](../../../build_tools/upstreams/target_support.json).

With `MERLIN_TARGET_PATH` unset it is the selected Saturn support, so the Saturn
dialect and the `saturn_vec` backend load without configuration. To try another
revision, select it explicitly:

```sh
export MERLIN_TARGET_PATH=/absolute/path/to/rvv-mlir/saturn-support
```

The companion's separate `merlin-support/` directory, vendored at
[`../../rvv/support`](../../rvv/support), describes target `rvv`; it does not select Saturn. Neither identity is the `saturn_opu_mxv256d128`
experiment target. These names must not be treated as interchangeable aliases.
This historical Saturn command-buffer route reuses Merlin's legacy RVV vector
emitters; the current MLIR compile route uses the pinned `rvv-mlir` provider.

Reference discovery alone cannot load executable support. The selected provider
owns the contract and dialect plan used for execution, without falling back to
this reference tree. Hardware parameters here describe the historical reference
model and are not new RTL-derived facts or proof of an OPU-enabled configuration.

Backend import, pure emission and dialect construction tests do not qualify
compilers, simulators or hardware. The support code remains host-owned and must
not be bundled with an evaluated compiler candidate. Broader OPU support
migration remains unfinished.

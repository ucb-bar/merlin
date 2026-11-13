# Recorded application on bare metal

`merlin.compile.baremetal_harness.emit_harness` consumes `application-c.json`,
a complete mapping of input ids to raw recorded byte files, and the SHA256 of
concatenated reference outputs in interface order. It emits `main.c`, its hash
header, and a build-requirements receipt. No tensor conversion occurs.

The caller selects `platform.h`; this example binds the generated `capsule`
namespace to coherent EE290 IMEM/CSR and direct DRAM callbacks. Other namespaces
or targets supply an adapter with the same `harness_platform_bind` contract.
Generate `ee290_host_map.h` from the selected Atlas map before compiling:

```sh
python examples/atlas/baremetal/generate_host_map.py \
  /path/to/atlas-mlir/runtime/selected-host-map.json /build/ee290_host_map.h
```

Add `/build` to the include path. Select other generated application headers and
namespaces with `HARNESS_APPLICATION_HEADER` and `HARNESS_APPLICATION_NAMESPACE`.
The adapter includes `atlas_host.h` before the application header.

When `model_parameters` is present, pass an explicit `parameter_address` aligned
to `workspace_alignment` (64 bytes when unspecified); there is no default address.
Choose a range inside the board's physical memory, separate from program sections
and device scratch. For SophiaLake's 1 GiB DRAM map a 44,120,064-byte payload at
`0xa0000000` ends at `0xa2a13800`. The generated binder authenticates the payload
at startup. Keep it immutable and retain DRAM between invocations.
The emitter refuses missing or incorrect
pinned math archives when the application declares host linkage. Applications
without math symbols do not require a math supplier.

Compile using the receipt's `-ffp-contract=off -frounding-math`. This adapter uses
200 million host polls by default. Select board startup, linker script, console,
and clock through Baremetal-IDE: `CHIP=penguin26`, `PLATFORM=SIMS` or `CHIP`.
At the bench define `SYS_CLK_FREQ=30000000`, `MTIME_FREQ=30000`, and use
`PENGUIN26_EXIT_TOHOST=ON` for CHIP. Simulator execution is required before
claiming RTL correctness. A successful cross-link supplies no such claim.

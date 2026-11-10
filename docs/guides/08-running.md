---
title: Run a compiled program
kind: guide
status: current
owner: runtime
last_verified: 2026-11-09
related: [07-building, 10-compiler-generation, ../reference/repo_structure]
code_refs:
  - src/merlin/api.py
  - src/merlin/service_cli.py
  - src/merlin/compiler/artifact.py
  - src/merlin/runtime/service.py
  - src/merlin/runtime/linux.py
  - src/merlin/runtime/spike.py
  - src/merlin/runtime/external.py
  - src/merlin/runtime/boards.py
  - src/merlin/runtime/backends/zephyr_model.py
  - src/merlin/compiler/build/zephyr/config.py
---

# 8. Run a compiled program

[Previous](07-building.md) | [Guide index](../README.md) | [Next](09-verification.md)

After [building a program](07-building.md), choose where its executable will run.
`merlin run` and `merlin.api.run` use the same runtime service. They run an
existing product; they do not compile it again.

## Run an existing build

For a native Linux build with an executable and a saved deployment description:

```sh
merlin inspect out/model-build
merlin run out/model-build --runner merlin.linux --timeout 60
```

Replace `out/model-build` with your actual build directory. `inspect` checks
recorded file hashes. The run service selects the artifact whose role is
`executable` and checks its bytes and deployment before invoking the runner.
A build containing only object files or device commands still needs its
appropriate linked entry image or a runner that understands those products.

The same call in Python is:

```python
from pathlib import Path
from merlin.api import run
from merlin.runtime.service import RunOptions

result = run(
    Path("out/model-build"),
    runner="merlin.linux",
    options=RunOptions(timeout=60),
)
print(result.status)
print(result.stdout)
```

A completed process is an execution observation. It does not mean that output
comparison was requested or passed; inspect the result metadata and the
separate verification report.

## Choose the runner and deployment

| Runner | Input and behavior |
| --- | --- |
| `merlin.linux` | Runs a compatible native Linux ELF on this Linux machine, with explicit initial CPU affinity. |
| `spike` | Runs a compatible RISC-V ELF using the existing Spike launcher; supports baremetal and Zephyr deployments. Requires the selected external tools. |
| `firesim` | Prepares files for externally managed execution and later collection. It does not launch FireSim. |

Additional runtime packages can register `merlin.runners` providers. Their
`validate` and `run` methods own the supported product/platform combination.
An ELF suffix does not identify its operating environment or calling convention.

A saved `Compilation` carries a `Deployment`: system, platform, ISA, ABI, hart
count, and optional DRAM origin/size. Here ISA means the instruction set and ABI
means the binary calling convention. These describe the built image, not the
name of the compiler that produced it.

For an ELF supplied outside a Merlin build directory, provide its actual
contract. The CLI requires `--system`, `--platform`, `--isa`, `--abi` and
`--harts`; `--dram-base` and `--memory-bytes` must be supplied together when used.
The Python equivalent is `RunOptions(deployment=Deployment(...))`.
Use the values from that image's build and selected hardware. Declaring them
does not rebuild an incompatible ELF or establish missing build provenance.

## Set harts when building the image

A hart is a hardware thread. Select the supported hart count during compilation.
At run time, `--harts` must match the image's deployment; omitting it uses that
recorded count. Changing a one-hart image to `--harts 4` is refused rather than
silently changing its startup or work partitioning.

For `merlin.linux`, this count selects initial CPU affinity from the process's
available CPUs. It does not create that many threads or reserve exclusive CPUs.
For a RISC-V runner, it selects target harts under the image/runtime contract.
The hardware description must distinguish vector-capable harts where needed.

## Baremetal and Zephyr are different image builds

A baremetal image needs the selected startup code, linker layout, memory map
and console/exit mechanism. A Zephyr image additionally depends on its board
port, device tree and kernel configuration. These choices belong to the target
system and image build; a run-time platform flag cannot convert one into the other.

`runtime/boards.py` holds board-description fields. The existing
`runtime/backends/zephyr_model.py` build path uses
`compiler/build/zephyr/config.py` for configuration derived from the selected
board and execution choices, then supplies its memory/CPU overlays. CPU availability,
vector state, DRAM placement and console support can differ between systems.
Keep those inputs with the selected system rather than applying one global
Zephyr configuration to every backend.

The public Spike adapter checks the ELF architecture and ABI. A Zephyr Spike
run also needs declared DRAM origin and size. These checks precede execution;
they do not establish that every startup or device behavior matches the hardware.

## Prepare a FireSim handoff

Given an already built image with its deployment:

```sh
merlin run out/model-build --runner firesim --mode prepare --output out/firesim-workload
```

The destination must be fresh. Merlin copies the selected artifacts, records
hashes in `request.json`, and identifies the entry image. For Linux it writes
`invoke.sh`, which runs the executable once your guest is ready. Baremetal and
Zephyr are described as boot images, not Linux shell commands. Use `--entry`
when the build has multiple ELF images, and repeat `--input NAME=PATH` to include
explicit input files. `--argument` is supported for Linux guest invocation.

Your existing FireSim workflow loads or stages that handoff and runs it. Farm
setup, bitstreams, instance launch, guest provisioning and job scheduling remain
outside Merlin. The external environment must match the recorded deployment.

After execution, your wrapper supplies `result.json` using the declared
`merlin.external_result.v1` protocol, the request hash, observed exit code and
hashed output artifacts. Collect those existing observations with:

```sh
merlin collect out/firesim-workload --results out/firesim-results
```

Collection checks correspondence and retains the files. Its `collected` status
is not a numerical or timing verdict. Continue to the independent checks for
the selected workload before reporting correctness or performance.

For runtime source locations, use the [repository map](../reference/repo_structure.md).
The [compiler-generation guide](10-compiler-generation.md) explains how generated
backends join the same compile and evaluation flow.

# Chipyard Concepts for Merlin Users

A short glossary of the Chipyard ecosystem, focused on what matters when
running Merlin workloads. You should not need to know these concepts in detail
— the `merlin chipyard` tool handles the interactions for you — but this
reference is useful for understanding what is happening under the hood.

## Chipyard

An SoC design framework from UC Berkeley. It generates complete RISC-V SoC
designs by composing cores, accelerators, and peripherals. Merlin uses
Chipyard-generated designs as execution targets for compiled ML models.

## Config class

A Scala class (e.g., `RadianceGemminiOnlyConfig`) that selects which hardware
components are included in the SoC. The config class determines the core type,
accelerator parameters, memory layout, and peripheral set. Each Merlin recipe
pins one config class.

## FireSim

FPGA-accelerated full-system simulation. Runs on Xilinx Alveo U250 FPGAs.
Supports both bare-metal and Linux workloads. FireSim synthesizes a Chipyard
config into an FPGA bitstream and manages deployment, execution, and result
collection.

FireSim is configured via four YAML files in `sims/firesim/deploy/`:

| File | Purpose | Managed by |
|---|---|---|
| `config_build_recipes.yaml` | Defines available hardware builds | Pre-populated in Chipyard |
| `config_build.yaml` | Selects which recipe to build | `merlin chipyard configure-firesim` |
| `config_hwdb.yaml` | Registers built bitstreams | `merlin chipyard register-hwdb` |
| `config_runtime.yaml` | Selects hardware + workload for a run | `merlin chipyard configure-firesim` |

## FireMarshal

Workload build tool for FireSim. Produces Linux images with your binaries baked
in. Used to package the IREE runtime and compiled model artifacts into a
bootable image for FireSim runs.

```bash
# Build the base image (done once via Merlin):
merlin chipyard build-firemarshal

# Under the hood this runs:
cd $CHIPYARD_ROOT/software/firemarshal
./marshal build br-base.json
./marshal install br-base.json
```

## Workloads

A FireSim workload is defined by a JSON file + an overlay directory:

```text
sims/firesim/deploy/workloads/
├── merlin-iree.json            # Workload definition
└── merlin-iree/
    └── overlay/
        └── opt/merlin/         # Merlin binaries + model artifacts
            ├── install/bin/    # IREE runtime
            ├── *.vmfb          # Compiled models
            └── run.sh          # Entry point script
```

The `merlin chipyard stage-workload` command creates both the JSON and the
overlay directory automatically.

## VCS / Verilator

RTL simulators. Run bare-metal ELFs directly against the generated hardware
design. The `merlin chipyard` tool wraps these:

```bash
# Via Merlin (recommended):
merlin chipyard build-sim gemmini_mx
merlin chipyard run gemmini_mx path/to/elf

# What happens under the hood:
cd $CHIPYARD_ROOT/sims/vcs
make CONFIG=RadianceGemminiOnlyConfig BINARY=path/to/elf LOADMEM=1 run-binary
```

VCS is a commercial simulator from Synopsys (faster, requires a license).
Verilator is open-source (slower, freely available).

## HTIF

Host-Target Interface. Memory-mapped communication (`tohost`/`fromhost`) for
bare-metal programs. The IREE bare-metal runtime uses HTIF for standard output
and program termination signaling. The linker script at
`build_tools/firesim/htif.ld` defines the `.htif` section.

## Where Chipyard lives

Chipyard is a separate repository. Tell Merlin where it is:

```bash
# Persistent (saved in .chipyard_config.json):
merlin chipyard set-path /path/to/chipyard

# Or via environment variable:
export CHIPYARD_ROOT=/path/to/chipyard
```

Merlin's build system and the `merlin chipyard` tool both read this path.
Physical board targets (SpacemiT X60) do not use Chipyard.

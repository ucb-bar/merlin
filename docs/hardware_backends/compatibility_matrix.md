# Compatibility Matrix

Pinned versions for each hardware backend. Keeping these in sync avoids
hard-to-debug mismatches between the compiler, runtime, and hardware.

| Component | Saturn OPU / U250 | Gemmini MX | SpacemiT X60 |
|---|---|---|---|
| Chipyard branch | `main` | `graphics` | N/A |
| Chipyard SHA | `bcb612918b` | `bcb612918b` | N/A |
| Merlin build profile | `firesim` | `firesim` | `spacemit` |
| Target YAML | `saturn_opu.yaml` | `gemmini_mx.yaml` | `spacemit_x60.yaml` |
| Execution method | FireSim U250 | VCS bare-metal | SSH to board |
| Chipyard config class | `FireSimSaturnOPUConfig` | `RadianceGemminiOnlyConfig` | N/A |
| Key submodule: saturn | `1d3515e02b` (opu-fp8) | -- | -- |
| Key submodule: gemmini | -- | `6ad65b90b1` (gemmini-mx) | -- |
| Key submodule: firesim | `b084672c2f` | -- | -- |

> **Note:** Pinned SHAs are maintained in `build_tools/hardware/*.yaml`.

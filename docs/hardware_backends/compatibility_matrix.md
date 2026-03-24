# Compatibility Matrix

Pinned versions for each hardware backend. Keeping these in sync avoids
hard-to-debug mismatches between the compiler, runtime, and hardware.

| Component | Saturn OPU / U250 | Gemmini / U250 | Gemmini MX | SpacemiT X60 |
|---|---|---|---|---|
| Chipyard branch | `main` | `main` | `graphics` | N/A |
| Chipyard SHA | `bcb612918b` | `bcb612918b` | `6a9b4cd950` | N/A |
| Merlin build profile | `firesim` | `firesim` | `firesim` | `spacemit` |
| Target YAML | `saturn_opu.yaml` | `gemmini.yaml` | `gemmini_mx.yaml` | `spacemit_x60.yaml` |
| Execution method | FireSim U250 | FireSim U250 | VCS bare-metal | SSH to board |
| Chipyard config class | `FireSimOPUV128D64ShuttleConfig` | `FireSimLeanGemminiRocketConfig` | `MxGemminiRocketConfig` | N/A |
| Key submodule: saturn | `ea37380016` (opu-int8) | -- | -- | -- |
| Key submodule: gemmini | -- | `6ad65b90b1` | `9c94a394c8` (gemmini-mx) | -- |
| Key submodule: radiance | -- | -- | `bdd970075a` (main) | -- |
| Key submodule: firesim | `b084672c2f` | `b084672c2f` | -- | -- |
| Bitstream status | Building | Built (2025-10-02) | Not yet buildable | N/A |

> **Note:** Pinned SHAs are maintained in `build_tools/hardware/*.yaml`.

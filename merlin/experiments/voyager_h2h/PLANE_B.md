# Plane B — Voyager's own hardware, Voyager's own flow

Reproduction of the Voyager paper's cycle numbers (arXiv 2509.15205v1, Table 4) with the accelerator
release's own flow, on this host. Before merlin is measured against the co-designed compiler on this
RTL, the co-designed pair has to reproduce its own published numbers. Driver:
`scripts/plane_b_regression.sh` (subcommands `env`, `codegen`, `fast-sim`, `block`, `rtl`, `rtl-sim`).

## Pair under test

| Piece | Revision | Where |
|---|---|---|
| Accelerator generator | `e3a725db` (`open-source-release`, code.stanford.edu/voyager/accelerator) | work tree `out/build/external/voyager-accelerator-work` (pin `voyager_accelerator`) |
| Compiler (the release's submodule) | `cac504ef` | `out/build/external/voyager-compiler-cac504ef` (pin `voyager_compiler_accel_pair`) |
| Interstellar | public copy vendored in the later compiler (`src/interstellar`) | `out/build/external/voyager-interstellar-shim` |
| Python env | Python 3.10, torch 2.6.0, protoc / libprotobuf 29.3 | `out/build/voyager-accel-env` |
| HLS | Catapult Ultra Synthesis 2023.1_1/1045460 (upstream tested 2024.2_1) | `/ecad/tools/mentor/catapult/2023.1_1` |
| RTL simulator | VCS V-2023.12-SP1-1, 64-bit (upstream tested T-2022.06-SP2) | `/ecad/tools/synopsys/vcs/current` |
| Technology | `generic`: Catapult's shipped `nangate-45nm_beh` + `ccs_sample_mem` (paper: TSMC 16 nm) | `scripts/tech/generic.tcl` |
| Configuration | INT8, 16x16 PE array (256 MACs), default 1024-element buffers, `CLOCK_PERIOD=1` ns | the paper's 256-MAC point |

## Deviations from the release's documented environment (all outside Voyager's source)

1. **Host multiarch headers.** Catapult's bundled gcc 10.3 does not search
   `/usr/include/x86_64-linux-gnu`, so `<asm/errno.h>` is missing. SystemC build: `-isystem` via the
   Makefile's `BASE_FLAGS`. Catapult's C++ front end (which never sees `BASE_FLAGS`): a symlink
   `lib/asm -> /usr/include/x86_64-linux-gnu/asm` in the work tree's `lib/` (its HLS search path).
2. **One missing HLS math header.** `src/datatypes/StdFloatTypes.h` includes `ac_math/ac_gelu_pwl.h`,
   absent from Catapult 2023.1. Only that file is supplied, from `hlslibs/ac_math` `1fde1dd`
   (Apache-2.0): after Catapult's includes for the SystemC build, and as `lib/ac_math/ac_gelu_pwl.h`
   for Catapult's front end. Catapult's own `ac_types` 4.6 stays: the public `ac_types` HEAD rejects
   Voyager's exponent-only scale type `UFloat<8,8>` (`ac_int<0>` via `ac_std_float::to_float`).
3. **Host multiarch C runtime.** `-B/usr/lib/x86_64-linux-gnu` in `LDFLAGS` (`crt1.o`, `crti.o`).
4. **Loader path, as upstream `env.sh` intends.** The env's newer libstdc++ before Catapult's
   (`GLIBCXX_3.4.29/30` missing from Catapult 2023.1's), Catapult's gcc-10.3 SystemC library after it;
   `PROJECT_ROOT` and `CODEGEN_DIR` exported as `env.sh` does.
5. **Tool versions.** Catapult 2023.1 and VCS V-2023.12 instead of 2024.2 / T-2022.06; generic 45 nm
   libraries instead of the paper's 16 nm, at the paper's 1 GHz clock.
6. **Interstellar.** The release installs it from an SSH-only submodule; the public vendored copy is
   used (it exposes every name `run_tiler.py` imports).
7. **Batch license environment.** Catapult checkout from a long-lived tmux server needs
   `LM_PROJECT` (with `MGLS_LICENSE_FILE`) propagated; without it every feature checkout fails.

## How the release turns simulations into a model number

- `run_regression.py --uniquify_layers` keeps one instance of each identical layer and multiplies its
  measured runtime by the number of identical layers. A "model" number is therefore a **per-layer sum**,
  not one timed invocation of the model.
- `MAX_TILES`: the C++ harness simulates `min(MAX_TILES, L2 tiles)` tiles of a layer, **all** of them
  when the variable is unset; `run_regression.py` assumes 1 simulated tile when it is unset and scales
  by `full_tiles / actual_tiles`. The two defaults disagree (unset = over-count on multi-tile layers).
  The driver exports `MAX_TILES=1000000` so both sides mean "every tile, no extrapolation".
- The layer tile count `run_regression.py` reads comes from `l2_tiling` in `model.txt`; the MobileBERT
  encoder IR at this compiler revision carries none (every layer counts as 1 tile), so for it the
  scaling is the identity.

## Results

### SystemC (functional; upstream: "not cycle-accurate")

`mobilebert_encoder`, INT8 16x16: **14/14 non-skipped layers pass against Voyager's gold model, error
count 0.** `slice_tensor` fails ("Slice indices for the last dimension must be multiples of
OC_DIMENSION!") and is a layer Voyager's own `ci_skip_rules.json` skips for this model.

### HLS (Catapult -> Verilog)

- `ProcessingElement`: synthesized in 59 s at 1 ns, generic 45 nm.
- Full `make rtl`: see below (in progress when this file was first written).

### RTL (VCS, cycle-accurate)

Pending.

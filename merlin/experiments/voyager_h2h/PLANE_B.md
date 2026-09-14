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
| Configuration | INT8, 16x16 PE array (256 MACs), default 1024-element buffers, `CLOCK_PERIOD=5` ns (deviation 8) | the paper's 256-MAC point |
| SystemC compiler for VCS | gcc 9.2.0 + binutils 2.33.1, the VCS S-2021.09 GNU package (upstream documents `VCS_GNU_PACKAGE` S-2021.09) | `/ecad/tools/synopsys/vcs/S-2021.09-SP1-1/gnu/linux` |

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
   libraries instead of the paper's 16 nm. VCS V-2023.12's SystemC 2.3.3 front end rejects the host's
   g++ 13.2 (`SC-SYSCAN-SYSC-COMP`; it accepts 7.3 / 9.2 / 9.5), so its compiles use the gcc 9.2 GNU
   package from the host's S-2021.09 VCS install -- the package generation upstream documents --
   selected through Catapult's own `VCS_EXEC_{VLOGAN,SYSCAN,VCS}` hooks (`-cpp`/`-cc`). That package's
   `g++` wrapper unsets `CPATH`/`C(PLUS)_INCLUDE_PATH` and vlogan spawns syscan with an empty
   `-cflags`, so deviation 1 reaches those compiles through a two-line wrapper
   (`out/build/external/voyager-gcc9-multiarch/{gcc,g++}`) that appends
   `-idirafter /usr/include/x86_64-linux-gnu` and execs the package's wrapper unchanged.
6. **Interstellar.** The release installs it from an SSH-only submodule; the public vendored copy is
   used (it exposes every name `run_tiler.py` imports).
8. **Clock period 5 ns, not 1 ns.** At `CLOCK_PERIOD=1` on the generic nangate-45nm library, HLS of
   `VectorAccumulator` fails to schedule (SCHD-30, "Unsatisfied Circle"): the loop-carried
   `acc_old -> vadd<bfloat16> -> acc_old` recurrence needs a 3-cycle + 0.25 ns bf16 adder core plus
   ~0.97 ns port input delay and muxes. (`InputController`, `ProcessingElement` and `SystolicArrayRow`
   did synthesize at 1 ns.) The release's own README example runs a 40 nm library at
   `CLOCK_PERIOD=5.0`, so the build uses 5 ns and reports cycles as runtime(ns) / 5. HLS inserts
   pipeline stages per clock, so cycle counts at 5 ns on 45 nm are not guaranteed identical to the
   paper's 1 GHz / 16 nm build -- this is the largest single source of expected reproduction gap.
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

- 1 ns attempt (`out/runs/voyager_accel/plane_b/rtl_INT8_16x16_generic_clk1_20260914T203542Z`):
  `ProcessingElement` (59 s), `InputController`, `SystolicArrayRow` synthesized; `VectorAccumulator`
  failed SCHD-30 (deviation 8). Abandoned, rc=2 after 385 s.
- 5 ns build (`out/runs/voyager_accel/plane_b/rtl_INT8_16x16_generic_clk5_20260914T204600Z`, log
  `make_rtl.log`). Catapult's own post-HLS timing estimate (`<block>.v1/rtl.rpt`, "Slack", ns):

  | Block | Slack | Block | Slack |
  |---|---|---|---|
  | ProcessingElement | +1.587 | VectorAccumulator | -1.463 |
  | SystolicArrayRow | +1.587 | VectorFetchUnit | -1.269 |
  | SystolicArray | +1.587 | VectorPipeline | -0.151 |
  | MatrixProcessor | +0.009 | VectorReducer | -2.511 |
  | WeightController | +0.001 | VectorUnit | -2.511 |
  | MatrixParamsDeserializer | +3.921 | InputController | -0.038 |
  | VectorParamsDeserializer | +4.140 | OutputController | -0.192 |

  Seven blocks miss 5 ns by Catapult's estimate on this library. That does not change RTL-simulation
  cycle counts (the generated RTL is cycle-exact at any real clock); it means this build is not a
  timing-closed 5 ns design, and it is one more reason cycle counts are the comparable quantity here,
  not wall time.
- Block RTL digests (`sha256` of `clock_5/<block>/<block>.v1/concat_rtl.v`, first 16 hex):
  InputController `0fd259ad4a6604b4`, MatrixParamsDeserializer `6e7a64263cee51f8`, MatrixProcessor
  `5032e3f4cc97327c`, OutputController `36263aec3a56d2b7`, ProcessingElement `aaca3de2f42fb7bb`,
  SystolicArrayRow `148623d24cd64b4c`, SystolicArray `75f5b163b3fb6598`, VectorAccumulator
  `c205c96c08d7103e`, VectorFetchUnit `bb97477e1797629a`, VectorParamsDeserializer `4bdeb5b7e79bebdf`,
  VectorPipeline `64897a9fb27d37a0`, VectorReducer `36e63f5a43b07e3a`, VectorUnit `c46def47bb0999c9`,
  WeightController `6de71d1bebc1b34e`.

### RTL (VCS, cycle-accurate)

- VCS toolchain smoke test on the 1 ns `ProcessingElement` (block-level SCVerify, `SIM_ProcessingElement`):
  with deviation 5's compiler and wrapper, every system and SystemC header now compiles
  (`ProcessingElement_rtl.o` builds). The build then stops in Voyager's own source:
  `src/Tieoff.h:102: no match for 'operator=' (sc_out<sc_lv<24>> and Int<24,true>)` -- the
  block-level mixed SystemC/RTL harness assigns a datatype to a bit-vector port. The release's
  regression never builds block-level RTL sims (it builds `Accelerator.v1` only), so this is a
  finding about an unused path, not a blocker, unless the top-level build hits it too.

### ResNet-50 (blocked)

Voyager's ResNet-50 codegen calibrates on 10 images streamed from the gated Hugging Face dataset
`timm/imagenet-1k-wds` (`voyager-compiler/test/utils/dataset/imagenet.py`). The token in the repo's
`.env` authenticates (`whoami` succeeds) but its account is not on that dataset's authorized list
(HTTP 403 `GatedRepoError` on `auth-check`), so `make network-proto NETWORK=resnet50` fails in 8 s.
Unblocking needs the account to request access on the dataset page. The alternative, calibrating on
different images, would change quantization scales but not layer shapes or tilings (which set the
cycle count). It would have to be disclosed as a deviation and was not done without approval.

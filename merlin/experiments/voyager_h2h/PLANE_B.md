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
| Connections (RTL testbench compile only) | public `hlslibs/matchlib_connections` 2.2.0, `6a3003b8` (Catapult 2023.1 ships 1.3.0; deviation 9) | `out/build/external/hlslibs-matchlib_connections` |
| Technology | `generic`: Catapult's shipped `nangate-45nm_beh` + `ccs_sample_mem` (paper: TSMC 16 nm) | `scripts/tech/generic.tcl` |
| Configuration | INT8, 16x16 PE array (256 MACs), default 1024-element buffers, `CLOCK_PERIOD=5` ns (deviation 8) | the paper's 256-MAC point |
| SystemC compiler for VCS | gcc 9.2.0 from the VCS S-2021.09 GNU package (upstream documents `VCS_GNU_PACKAGE` S-2021.09), with the host's binutils 2.42 `as`/`ld` (deviation 5) | `/ecad/tools/synopsys/vcs/S-2021.09-SP1-1/gnu/linux` |

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
3. **Host multiarch C runtime.** `-B/usr/lib/x86_64-linux-gnu` in `LDFLAGS` (`crt1.o`, `crti.o`),
   with the env's `lib/` placed before `-L/usr/lib/x86_64-linux-gnu`. VCS's `simv` link puts
   `LDFLAGS` ahead of its own `-L`, and the host's Ubuntu protobuf 3.21 (`libprotobuf.so.32.0.12`)
   would otherwise satisfy `-lprotobuf` instead of the env's 29.3 the harness is compiled against
   (undefined `google::protobuf::Arena::Allocate`, `protobuf_assumption_failed`, ...). Voyager's own
   Makefile already searches the env first (`$(LDLIBS) $(LDFLAGS)`), so its links are unchanged.
4. **Loader path, as upstream `env.sh` intends.** The env's newer libstdc++ before Catapult's
   (`GLIBCXX_3.4.29/30` missing from Catapult 2023.1's), Catapult's gcc-10.3 SystemC library after it;
   `PROJECT_ROOT` and `CODEGEN_DIR` exported as `env.sh` does.
5. **Tool versions.** Catapult 2023.1 and VCS V-2023.12 instead of 2024.2 / T-2022.06; generic 45 nm
   libraries instead of the paper's 16 nm. VCS V-2023.12's SystemC 2.3.3 front end rejects the host's
   g++ 13.2 (`SC-SYSCAN-SYSC-COMP`; it accepts 7.3 / 9.2 / 9.5), so its compiles use the gcc 9.2 GNU
   package from the host's S-2021.09 VCS install -- the package generation upstream documents --
   selected through Catapult's own `VCS_EXEC_{VLOGAN,SYSCAN,VCS}` hooks (`-cpp`/`-cc`). The package's
   own `xbin/` wrapper cannot be used unchanged: it puts the package's binutils 2.33.1 ahead of any
   user flag, and that `ld` rejects this host's glibc 2.39 shared objects (`libm.so.6`, `libdl.so`,
   `libmvec.so.1`: "unknown type [0x13] section `.relr.dyn'", then "skipping incompatible"), so
   `simv` does not link. VCS V-2023.12 also puts its own bundled binutils 2.33.1 on the link's
   `PATH`: with the package's wrapper bypassed, the failing `ld` was
   `vcs/current/gnu/linux64/binutils-2.33.1_64/bin/ld`. Where VCS injects it was not traced; neither
   `bin/vcs`/`bin/syscan` nor the generated makefiles name that path. The driver generates
   `out/build/external/voyager-gcc9-multiarch/{gcc,g++}`, which run the same gcc 9.2 driver with the
   package's Ubuntu `-B/usr/lib/x86_64-linux-gnu` (crt files), clear the gcc include environment as
   the package does, force the host's binutils 2.42 `as`/`ld` first on `PATH` (gcc 9.2 resolves its
   own `cc1plus`/`collect2` through its install prefix and `as`/`ld` through `PATH`), and append
   `-idirafter /usr/include/x86_64-linux-gnu` (deviation 1). VCS's own binutils check requires only
   ld >= 2.33.1 for gcc 9.2. vlogan spawns its own syscan with an empty `-cflags`, so the compiler is
   the one place that include reaches every compile.
6. **Interstellar.** The release installs it from an SSH-only submodule; the public vendored copy is
   used (it exposes every name `run_tiler.py` imports).
7. **Batch license environment.** Catapult checkout from a long-lived tmux server needs
   `LM_PROJECT` (with `MGLS_LICENSE_FILE`) propagated; without it every feature checkout fails.
8. **Clock period 5 ns, not 1 ns.** At `CLOCK_PERIOD=1` on the generic nangate-45nm library, HLS of
   `VectorAccumulator` fails to schedule (SCHD-30, "Unsatisfied Circle"): the loop-carried
   `acc_old -> vadd<bfloat16> -> acc_old` recurrence needs a 3-cycle + 0.25 ns bf16 adder core plus
   ~0.97 ns port input delay and muxes. (`InputController`, `ProcessingElement` and `SystolicArrayRow`
   did synthesize at 1 ns.) The release's own RTL CI (`.gitlab/ci/rtl.yml`, job "RTL Simulation
   (INT8, 16x16)") runs exactly this point at exactly this setting: `DATATYPE=INT8 IC_DIMENSION=16
   OC_DIMENSION=16 TECHNOLOGY=generic CLOCK_PERIOD=5 run_regression.py --models
   vit,bert,resnet18,resnet50,mobilebert_encoder --sims rtl --uniquify_layers --skip_layers`. The
   README example also uses `CLOCK_PERIOD=5.0`. So the build uses 5 ns and reports cycles as
   runtime(ns) / 5. That CI job asserts no runtime or utilization, and it does not run the full
   `mobilebert` model. HLS places pipeline stages per clock, so latencies at 5 ns on 45 nm can differ
   from the paper's 1 GHz / 16 nm build. Throughput did not: every pipelined loop of the 5 ns build
   is II=1 (see "Every layer runs at ~2x its ideal").
9. **Connections library generation in the RTL testbench.** The SCVerify testbench compiles
   Voyager's SystemC design too (Catapult's generated `scverify/sysc_sim.h` includes
   `src/Accelerator.h`), and that source does not compile against the Connections library Catapult
   2023.1 ships (`connections.h` 1.3.0, `connections_fifo.h` 1.4):
   - under that library's pre-HLS default `MARSHALL_PORT` (`Out<T>::dat` is `sc_out<sc_lv<W>>`),
     `src/Tieoff.h:102` (`out.dat = zero`) fails: `no match for 'operator='`;
   - under `DIRECT_PORT` (tried via `-DAUTO_PORT=Connections::DIRECT_PORT`), Catapult's own
     `connections_fifo.h:199` writes a literal `0` into a struct message (`MatrixParams`,
     `PEInput<Int<8>>`, `Pack1D<Int<24>,16>`): `cannot convert 'int' to 'const data_type&'`.
     The public library fixed that in 1.5.0 (2023-08), and made `DIRECT_PORT` the pre-HLS default in
     2.2.0 ("CAT-34924 - use DIRECT_PORT by default for pre-HLS simulation").

   So the testbench's syscan compiles put public `hlslibs/matchlib_connections` tag 2.2.0
   (`6a3003b8`, 2024-09-05, Apache-2.0; `out/build/external/hlslibs-matchlib_connections`) first on
   the include path, with its own defaults. Catapult 2024.2, the release's tested version, is
   contemporary with 2.2.0, but which Connections release 2024.2 bundles was not verified on this
   host. Catapult's `mc_connections.h` includes Connections with angle brackets behind one include
   guard, so the whole library is swapped, never mixed. HLS still used Catapult's own copy. The
   RTL-facing wrapper ports are `SYN_PORT` explicitly in `sysc_sim.h`, and Connections documents every
   port type as bindable to `SYN_PORT`.

## How the release turns simulations into a model number

- `run_regression.py --uniquify_layers` keeps one instance of each identical layer and multiplies its
  measured runtime by the number of identical layers. A "model" number is therefore a **per-layer sum**,
  not one timed invocation of the model.
- `MAX_TILES`: the C++ harness simulates `min(MAX_TILES, L2 tiles)` tiles of a layer, **all** of them
  when the variable is unset; `run_regression.py` assumes 1 simulated tile when it is unset and scales
  by `full_tiles / actual_tiles`. The two defaults disagree (unset = over-count on multi-tile layers).
  The driver exports `MAX_TILES=1000000` so both sides mean "every tile, no extrapolation".
- The layer tile count `run_regression.py` reads comes from `l2_tiling` in `model.txt`; neither
  MobileBERT IR at this compiler revision carries any (every layer counts as 1 tile), so for them the
  scaling is the identity.
- "Identical" means the op's protobuf is equal after deleting every `name`, `memory`, `scratchpad` and
  `node` field (`add_layers`, DeepDiff). Computed with the release's own `get_skip_layers` +
  `add_layers` (RTL rules, INT8, block 16), without simulating:
  - `mobilebert_encoder`: 14 unique layers stand for 29 layer instances (`slice_tensor` skipped by
    `ci_skip_rules.json`).
  - `mobilebert`: 16 unique layers stand for 570 layer instances. `ci_skip_rules.json` lists no rule
    for this model, so its `slice_tensor` layer is simulated, the op the encoder's SystemC run
    rejected ("Slice indices for the last dimension must be multiples of OC_DIMENSION!").

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
  `make_rtl.log`): all 15 solutions (14 blocks + `Accelerator` top) synthesized, `make rtl` rc=0,
  wall 1057 s, zero Catapult errors. Top-level `Accelerator.v1/concat_rtl.v`: 14,977,605 bytes,
  sha256 `d0708330373ef1ea9f6bb985784813093204e9336c8ab171ed9a9a2d4db8efbf`, estimated slack
  -2.511 ns (VectorUnit's path). Catapult's own post-HLS timing estimate per block
  (`<block>.v1/rtl.rpt`, "Slack", ns):

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
- It does: the top-level `Accelerator.v1` build (the one `run_regression.py` makes) failed the same
  way. The run launched then (`rtlsim_mobilebert_encoder_..._20260914T210352Z`) was stopped and
  marked `ABORTED.txt`. The build went through four more failures before it linked: the `Tieoff.h`
  port type; with `DIRECT_PORT`, Catapult's own `Fifo` (deviation 9); binutils 2.33.1 against glibc
  2.39 (deviation 5); host protobuf 3.21 shadowing the env's 29.3 (deviation 3). With all four, a
  clean `make -f scverify/Verify_concat_sim_rtl_v_vcs.mk build` of the 5 ns `Accelerator.v1`
  finishes rc=0 in 354 s, with 0 undefined references, Verdi KDB elaboration "0 error(s)", and the
  `sc_main` simulation binary produced.

#### `mobilebert_encoder` (INT8 16x16, generic 45 nm, 5 ns)

Run `out/runs/voyager_accel/plane_b/rtlsim_mobilebert_encoder_INT8_16x16_generic_clk5_20260914T214254Z`
(`run_regression.log`; the release's own `regression_results/2026_09_14_14_43_05` is linked there with
per-layer logs and `test_results.pkl`): `run_regression.py --sims rtl --uniquify_layers --skip_layers
--keep_build`, `NPROC=6`, `MAX_TILES=1000000`, rc=0, wall 217 s. Its `make -j rtl` and VCS `build`
steps were no-ops (the build above).

- **Pass: 14/14 unique layers** (standing for 29 layer instances), "Error count: 0" in every layer
  log. The one layer log read in full (`ffn_0_intermediate_dense_fused`) matches the gold model on
  65,536/65,536 outputs within 0.001.
- Per layer (ns at 5 ns; cycles = ns / 5):

  | Layer | Count | Runtime ns | Ideal ns | Type |
  |---|---|---|---|---|
  | quantize_default | 1 | 41,130 | 20,480 | vector |
  | bottleneck_input_dense_fused | 2 | 328,305 | 163,840 | matrix |
  | attention_self_query_fused | 2 | 82,620 | 40,960 | matrix |
  | attention_self_value_fused | 1 | 328,380 | 163,840 | matrix |
  | matmul_2_fused | 4 | 21,195 | 10,240 | matrix |
  | softmax_1_fused | 4 | 31,790 | 15,360 | vector |
  | matmul_6_fused | 4 | 21,180 | 10,240 | matrix |
  | attention_output_dense_fused | 1 | 82,550 | 40,960 | matrix |
  | add_4_fused | 3 | 10,410 | 5,120 | vector |
  | ffn_0_intermediate_dense_fused | 2 | 328,380 | 163,840 | matrix |
  | ffn_0_output_dense_fused | 2 | 328,310 | 163,840 | matrix |
  | output_bottleneck_dense_fused | 1 | 328,310 | 163,840 | matrix |
  | add_10 | 1 | 41,135 | 20,480 | vector |
  | classifier | 1 | 5,425 | 2,560 | matrix |

- **Whole-encoder per-layer sum** (runtime x Count x L2 Tiles / Actual Tiles, the release's own
  weighting; all tile counts are 1): 3,290,050 ns = **658,010 cycles**; ideal 327,168 cycles. The
  release prints **Utilization 0.497, Matrix Utilization 0.498**. This is a per-layer sum of
  uniquified layers times their counts, not one timed invocation of the encoder. The paper has no
  row for this encoder; it only anchors the flow.

#### Every layer runs at ~2x its ideal

Ratios range from 2.00 to 2.12 on both matrix and vector layers, at every size from 2,560 to
163,840 ideal ns. That is a halved throughput, not a fixed overhead. What was checked:

- Not the clock: the testbench `sc_clock` and the harness's ideal both come from `CLOCK_PERIOD`
  (log: "Connections Clock: harness.clk Period: 5 ns"), so ideal cycles = ideal ns / 5.
- Not HLS: every pipelined loop on the matrix path (MatrixProcessor, Skewer, InputController,
  WeightController) and the vector path (VectorUnit, VectorPipeline, VectorFetchUnit,
  OutputController) reports "initiation interval 1" (SCHD-43) in the 5 ns logs.
- Not single-port RAM: the buffers map to Catapult's `ram_sync_1R1W`.
- Where it shows: `ffn_0_intermediate_dense_fused` reads 524,288 input bytes = 32,768 beats of 128
  bits, exactly its 32,768 ideal compute cycles, in 65,676 cycles, i.e. one beat per two cycles on
  average. The handshake trace below shows that the memory side is not what limits it.

#### Where the 2x comes from

Traced, and **not a deviation of ours**. It is in Voyager's released source. There are two
mechanisms: one on the matrix path, one on the vector path. Evidence is in
`out/runs/voyager_accel/plane_b/stall_trace_20260914T230518Z/`: per-layer ucli event logs of every
vld/rdy change (`changes.txt.gz`), the reduced tables, and the helper scripts. The driver reproduces
them with `rtl-trace` / `rtl-layer` / `systemc-layer`.

Method. One layer of the existing SCVerify build, run under a ucli script. ucli here rejects
`dump -type VCD`, so the script logs every value change of the listed nets with the simulator's own
time. Reduced per channel: a transfer is vld && rdy at a rising edge.

- **`classifier` is not a matrix layer in RTL.** The log calls it "matrix unit ideal runtime", but
  `matrix_unit_params_in` never fires and every `matrix_unit_*` port is idle. It runs on the vector
  unit. The smallest real matrix layer is `matmul_2_fused` (ideal 2,048 cycles, 4,239 measured).
- **The memory side keeps up.** In `matmul_2_fused` (4,239 cycles):
  - `matrix_unit_input_resp` has vld high on 3,975 cycles; the harness offers data 94% of the time.
  - Its rdy is high on only 2,048 cycles; the RTL accepts half the time.
  - `input_req` has rdy high on all 4,239 cycles; the harness is always ready, and the RTL raises
    vld on 2,048.

  The testbench memory model (`test/common/Harness.cc` `process_read_request` /
  `send_data_response`, a 16-deep `sc_fifo` behind Connections sim ports) is not the stall.
- **The backpressure comes from `MatrixProcessor`.** Cycles 1000-1040, per edge (X = transfer,
  V = vld only, R = rdy only):

  ```
  window_buffer_out (MatrixProcessor input)   XVXVXVXVXVXVXVXVXVXV...
  weight_buffer_read_resp (its weight input)  ...XVXVXVXVXVXVXVXV...
  output_channel (its output)                 XRXRXRXRXRXRXR...
  accumulation_buffer_mu_write_request        ...RXRXRXRXRXRXRX
  ```

  - Every producer upstream holds vld: InputController, the input and weight double buffers, and
    the memory stream.
  - The consumer downstream holds rdy: the vector unit.
  - `MatrixProcessor` alone takes one item per two cycles, and emits one per two cycles.
  - Averaged over 128-cycle bins the whole matrix path moves in lockstep at half rate: 64 input
    beats, 32+32 bank writes, 32+32 bank reads, 32 weight beats and 32 outputs per bin. There are
    no load-then-compute phases.

**Cause 1, matrix path: a depth-1 FIFO inside MatrixProcessor for non-MX datatypes.**

- **Where.** `src/MatrixProcessor.h:24` is `static constexpr int FIFO_DEPTH = SUPPORT_MX ? 8 : 1;`.
  It sizes `accum_to_wb_fifo` (`:47`), the `Connections::Fifo` between the `process_accumulation`
  and `write_back` threads. Every output row of the systolic array crosses it.
- **Why depth 1 halves throughput.** `Connections::Fifo` drives `enq.rdy = !full` with `full`
  registered, and clears `full` one cycle after a dequeue (`connections_fifo.h` `EnqRdy` /
  `FullNext`). A depth-1 FIFO can therefore never enqueue in the cycle it dequeues: one message per
  two cycles.
- **Same in both generations.** This logic is identical in Catapult 2023.1's bundled 1.4.0 (what
  HLS synthesized) and in public 2.2.0 (what the testbench compiles). Their diff is initial values
  and index types only.
- **Which configs get it.** `SUPPORT_MX` is true only for `MXINT8`/`MXNF4`
  (`src/ArchitectureParams.h`), so every INT8, E4M3 and posit build gets depth 1.
- **Not ours.** No tool or library version of ours enters it.

**Cause 2, vector path: 128-bit ports against a 256-bit bf16 vector.**

- **The bound.** `OC_PORT_WIDTH = OC_DIMENSION * WEIGHT_DTYPE_WIDTH` = 128 bits for INT8
  (`ArchitectureParams.h:297`), while `VECTOR_DATATYPE` is bf16. One 16-lane vector is 256 bits,
  i.e. two beats (`VectorFetch.h:369-370`, `vector_fetch_*_num_beats`).
- **What the trace shows.** In `add_4_fused` (2,082 cycles):
  - both `vector_fetch_0_resp` and `vector_fetch_1_resp` transfer on every cycle, 2,048 beats each;
  - `vector_output_data` emits 1,024 words, one per two cycles.

  The port runs at full rate.
- **Why that reads as 2x.** The harness's ideal (`test/common/Simulation.cc:174-175`,
  `num_ops / VECTOR_UNIT_WIDTH`) assumes one vector per cycle, which this port width cannot supply.
  The vector layers' 2x is a bandwidth bound of the configuration, not a stall.

**What rules out our toolchain, library and testbench deviations as the cause.**

- **The pre-HLS SystemC model reproduces it.** The Makefile's `sim` TestRunner is the same harness
  in `CONNECTIONS_ACCURATE_SIM`, with no Catapult, no VCS, no RTL and no SCVerify wrapper
  (`plane_b_regression.sh systemc-layer`). It gives the same numbers:

  | Layer | RTL (VCS) ns | SystemC accurate ns | Ideal ns |
  |---|---|---|---|
  | classifier | 5,425 | 5,220 | 2,560 |
  | matmul_2_fused | 21,195 | 21,075 | 10,240 |
  | add_4_fused | 10,410 | 10,305 | 5,120 |
  | ffn_0_intermediate_dense_fused | 328,380 | 328,275 | 163,840 |

  This rules out Catapult 2023.1 vs 2024.2, VCS V-2023.12, the gcc 9.2 wrapper and the SCVerify
  testbench. The SystemC build needs deviation 9 too: `src/Tieoff.h` does not compile against
  Catapult 2023.1's Connections in accurate mode either. So the release was written against
  Connections 2.x.
- **The Connections swap** changes neither the sim-accurate `Pre()`/`Post()` timing nor
  `Connections::Fifo`. The whole 1.3.0/1.4 vs 2.2.0 `connections.h` diff (419 lines) is
  destructors, logging and the `AUTO_PORT` default (MARSHALL_PORT vs DIRECT_PORT); the harness's
  channels run in bypass (latency 0) in both.
- **No memory-model knob** (burst length, bus width, outstanding requests) can help, because the
  harness is never the side that stalls.
- **`DOUBLE_BUFFERED_ACCUM_BUFFER=true`** (one variable, SystemC) does not help:
  - `matmul_2_fused` becomes slower, 33,930 ns;
  - `add_4_fused` (10,305 ns) and `classifier` (5,220 ns) are unchanged.

**A/B, one variable: `FIFO_DEPTH` 1 -> 2 (non-MX).**

- **Setup.** Patched copy `out/build/external/voyager-accelerator-ab-fifo2`: the work tree minus
  `.git`/`build`/`regression_results`, with `models` and `test/compiler/networks` symlinked. Its
  only source difference is `src/MatrixProcessor.h:24`. Neither the pinned checkout nor the flow's
  work tree was touched.
- **SystemC accurate**
  (`out/runs/voyager_accel/plane_b/systemc_accurate_fifo_ab_20260914T225955Z`):

  | Layer | C2 | depth 1 ns | depth 2 ns | x ideal |
  |---|---|---|---|---|
  | matmul_2_fused | 2 | 21,075 | 15,965 | 2.06 -> 1.56 |
  | ffn_0_intermediate_dense_fused | 8 | 328,275 | 307,805 | 2.00 -> 1.88 |
  | add_4_fused (vector, control) | - | 10,305 | 10,305 | 2.01 -> 2.01 |

- **The SystemC residual is the accumulate round trip.** It is exactly
  1 + (C2 - 1)/C2, where C2 is the layer's reduction-tile count `loops[0][reduction_loop_idx[0]]`.
  Accumulating steps (C2 - 1 of every C2) do `accumulation_buffer_read_address.Push()` then
  `read_data.Pop()` in one iteration (`MatrixProcessor.h:579-582`).
  - In SystemC a thread cannot overlap iterations, so that costs 2 cycles.
  - In HLS the loop is pipelined at II=1 with the Pop scheduled 2 cycles after the Push
    (`MatrixProcessor.tcl` `cycle set ... -equal 2`).

  So once the FIFO is fixed the SystemC model is a pessimistic proxy, and only the RTL shows what
  the fix recovers.
- **RTL** (re-HLS of `MatrixProcessor` and the `Accelerator` top in the patched copy, every other
  block solution reused unchanged; `make rtl` rc=0 in 510 s, MatrixProcessor `concat_rtl.v` sha256
  `8c73d719...`, `Accelerator` `d1737382...`; VCS build rc=0 in 273 s;
  `out/runs/voyager_accel/plane_b/rtl_fifo2_ab_20260914T230402Z`). **This is where the matrix path
  recovers**, because HLS pipelines the accumulate round trip at II=1 while SystemC cannot:

  | Layer | C2 | depth 1 ns | depth 2 ns | x ideal | Error count |
  |---|---|---|---|---|---|
  | matmul_2_fused | 2 | 21,195 | 12,835 | 2.07 -> 1.25 | 0 |
  | attention_self_query_fused | 8 | 82,620 | 41,685 | 2.02 -> 1.018 | 0 |
  | ffn_0_intermediate_dense_fused | 8 | 328,380 | 164,565 | 2.00 -> 1.004 | 0 |
  | add_4_fused (vector, control) | - | 10,410 | 10,410 | 2.03 -> 2.03 | 0 |

  Every layer still matches Voyager's gold model. The big dense layers, which dominate the model,
  land within 0.4-1.8% of ideal; the small `matmul_2_fused` keeps a fixed ~800 ns of per-layer
  pipeline fill, which matters only at 2,048 ideal cycles. The vector layer is untouched, as
  expected for a matrix-path FIFO.

#### `mobilebert` (full model) vs the paper

Run `out/runs/voyager_accel/plane_b/rtlsim_mobilebert_INT8_16x16_generic_clk5_20260914T214631Z`
(release's `regression_results/2026_09_14_14_46_40`), same settings as the encoder, rc=1, wall 209 s.

- **Pass: 15/16 unique layers, covering 569 of 570 layer instances.** The failure is `slice_tensor`
  (1 instance): the harness throws "Slice indices for the last dimension must be multiples of
  OC_DIMENSION!" (E549 uncaught exception) before any accelerator time, the same rejection as the
  SystemC run. `ci_skip_rules.json` skips exactly this op for `mobilebert_encoder` but has no rule
  for `mobilebert`, and the release's CI never runs `mobilebert`. The release records a failed layer
  as runtime 0, so the total below covers the 569 passing instances.
- The 15 passing unique layers are the encoder's shapes, with identical runtimes, at higher counts
  (matmul/softmax x84, dense layers x21-42, `add_4_fused` x63, `add_10_fused` x20), plus `add_230`
  (41,135 ns).
- **Per-layer sum** (uniquified layers x counts; not one timed invocation of the model):
  68,159,850 ns = **13,631,970 cycles** (matrix 12,785,759; vector 846,211); ideal 6,778,368 cycles;
  **utilization 0.497**. The release prints 0.497, matrix 0.498.

| | Paper Table 4 (16x16, 1 GHz, TSMC 16 nm) | This run (16x16, 5 ns, generic 45 nm) | Ratio |
|---|---|---|---|
| MobileBERT cycles | 7.71M | 13.63M (per-layer sum, 569/570 instances) | 1.77x |
| Utilization | 95.1% | 49.7% | 0.52x |
| Ideal cycles | 7.33M (implied: 7.71M x 0.951) | 6.78M | 0.92x |

Reading the gap:

- Almost all of it is the ~2x per-layer throughput halving above, and that is the depth-1
  `accum_to_wb_fifo`. With `FIFO_DEPTH` 1 -> 2 as the only change, the same flow on the same host
  reproduces the paper's order of magnitude (next table).
- The ideal work is itself 7.5% below the paper's implied ideal, so the model this release pair
  compiles (`make network-proto NETWORK=mobilebert`, compiler `cac504ef`) is not exactly the paper's
  MobileBERT-tiny workload (different op set or shapes; not investigated). The paper's model name is
  "MobileBERT-tiny"; the release's network is `mobilebert`, and that they are the same model is
  assumed, not verified.
- The one excluded `slice_tensor` instance cannot matter: even at an `add` layer's cost (~8.2k
  cycles) it would move the total by 0.06%.

#### Whole model, one variable changed

Run `out/runs/voyager_accel/plane_b/rtlsim_mobilebert_fifo2_20260914T233000Z`: the release's own
`run_regression.py --models mobilebert --sims rtl --uniquify_layers --skip_layers`, same settings,
same host, on the patched RTL. Same 15/16 unique layers pass, same `slice_tensor` failure, rc=1.

| | Paper Table 4 (16x16, 1 GHz, TSMC 16 nm) | Release as shipped | `FIFO_DEPTH` 1 -> 2 |
|---|---|---|---|
| MobileBERT cycles | 7.71M | 13.63M | **7.38M** |
| Utilization | 95.1% | 49.7% | **91.8%** |
| Matrix utilization | - | 49.8% | **97.4%** |
| Vector utilization | - | 48.8% | 48.8% |

The release prints 0.918 / 0.974. One line of Voyager's own source moves this flow from 1.77x the
paper's cycles to 0.96x, i.e. 4% fewer cycles than Table 4 reports, on a workload whose ideal work
is 7.5% smaller than the paper's implied ideal.

**What this means for Table 4.**

- The paper's numbers are **not reproducible from the public release as configured**: its INT8 point
  is capped at ~50% of its own ideal by `SUPPORT_MX ? 8 : 1`. Table 4's 95.1% therefore cannot have
  been measured on this configuration of this code. It is consistent with a build where that FIFO is
  deeper -- every `SUPPORT_MX` datatype (`MXINT8`, `MXNF4`) gets depth 8 -- or with an internal
  revision. Which one was not determined here.
- With that one line changed, the release's own flow does reproduce the paper's claim on this host:
  91.8% utilization and 7.38M cycles against 95.1% and 7.71M. So the paper's *architectural* claim
  stands; what the public release ships for non-MX datatypes does not match it.
- The residual 8.2% is **not** the matrix path (97.4% utilized). 433,539 of the 601,839 remaining
  slack cycles (72%) are the vector path's 128-bit ports against 256-bit bf16 vectors, which is a
  property of the configuration's port width, not a stall.
- For the head-to-head this plane exists to support: comparing a merlin backend against Voyager's
  compiler on RTL built from the release as shipped would credit merlin with a 2x that belongs to a
  FIFO depth. Either build both sides with the patched depth, or report the cap alongside the
  numbers.

**Reproduced, with one line changed.** The release's own flow, at its own CI configuration for this
point, gives 49.7% and 13.63M on this host; with `MatrixProcessor.h`'s non-MX `FIFO_DEPTH` raised
from 1 to 2 it gives 91.8% and 7.38M, against the paper's 95.1% and 7.71M.

### ResNet-50 (blocked)

Voyager's ResNet-50 codegen calibrates on 10 images streamed from the gated Hugging Face dataset
`timm/imagenet-1k-wds` (`voyager-compiler/test/utils/dataset/imagenet.py`). The token in the repo's
`.env` authenticates (`whoami` succeeds) but its account is not on that dataset's authorized list
(HTTP 403 `GatedRepoError` on `auth-check`), so `make network-proto NETWORK=resnet50` fails in 8 s.
Unblocking needs the account to request access on the dataset page. The alternative, calibrating on
different images, would change quantization scales but not layer shapes or tilings (which set the
cycle count). It would have to be disclosed as a deviation and was not done without approval.

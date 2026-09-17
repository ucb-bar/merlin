# voyager_accel — onboarding Voyager's own accelerator as a merlin target

Scoping and derivation only. Nothing here is a performance claim, and nothing here was guessed: every
value cites the file and line of the pinned release it was read from, or is recorded as `UNKNOWN`.

**Subject.** The accelerator release pinned as `voyager_accelerator` = `e3a725db` in
`merlin/contract/hardware_pins.yaml`, read at `out/build/external/voyager-accelerator` (pinned
checkout, untouched) and `out/build/external/voyager-accelerator-work` (the flow's work tree, which
holds the built instance). Its paired compiler is `voyager_compiler_accel_pair` = `cac504ef`.

**Target name.** `voyager_accel` — deliberately distinct from the compiler pins
(`voyager_compiler`, `voyager_compiler_accel_pair`) and from the pin name `voyager_accelerator`, so a
merlin target and a hardware revision are never confused. It is already the run-dir axis
(`out/runs/voyager_accel/plane_b/...`, `PLANE_B.md`).

**The built instance every "derived from a built instance" line below refers to** is
`out/build/external/voyager-accelerator-work/build/INT8_16x16_1024x1024x1024_false_false_false/Catapult/generic/clock_5/`
(`DATATYPE=INT8 IC_DIMENSION=16 OC_DIMENSION=16`, default 1024-element buffers, generic 45 nm, 5 ns) —
the release's own RTL-CI point (`PLANE_B.md`, deviation 8). Citations of the form
`concat_rtl.v:N` are into that instance's `Accelerator/Accelerator.v1/concat_rtl.v`.

**The verified cap that rides on every number.** `src/MatrixProcessor.h:24` is
`static constexpr int FIFO_DEPTH = SUPPORT_MX ? 8 : 1;`. `SUPPORT_MX` is true only for `MXINT8`/`MXNF4`
(`src/ArchitectureParams.h:106-147`), so every INT8 / E4M3 / posit build gets a depth-1
`Connections::Fifo` on the accumulate-to-writeback path and is capped at ~50 % of its own matrix
throughput. A head-to-head on this RTL must build BOTH sides patched or report the cap in the cell
(`PLANE_B.md`, "What this means for Table 4").

---

## 1. The fact map

### 1.1 What is a compile-time `-D` knob and what is a property of a built instance

**Compile-time `-D` (`Makefile:12-95`).** Changing any of these is a different device:

| Knob | Where | Default | Notes |
|---|---|---|---|
| `DATATYPE` | `Makefile:12-13,47` | none (`$(error ...)`) | one of `P8_1 E4M3 E4M3_NS E4M3_DW E4M3_DW_NS E5M2 HYBRID_FP8 BF16 FP32 INT8 INT8_32 MXINT8 MXNF4 CFLOAT` (`ArchitectureParams.h:12-158`); selects input/weight/accum/vector dtypes together |
| `IC_DIMENSION` | `Makefile:15-16,48`; `ArchitectureParams.h:168-170` | none (`#error`) | systolic array rows (reduction edge) |
| `OC_DIMENSION` | `Makefile:18-19,49`; `ArchitectureParams.h:172-174` | none (`#error`) | systolic array cols (output edge) |
| `INPUT_BUFFER_SIZE` | `Makefile:51-55`; `ArchitectureParams.h:309-311` | 1024 | entries per bank |
| `WEIGHT_BUFFER_SIZE` | `Makefile:57-61`; `ArchitectureParams.h:317-319` | 1024 | entries per bank |
| `ACCUM_BUFFER_SIZE` | `Makefile:63-67`; `ArchitectureParams.h:325-327` | 1024 | entries |
| `DOUBLE_BUFFERED_ACCUM_BUFFER` | `Makefile:69-72`; `ArchitectureParams.h:164-166` | false | 1 vs 2 accumulator banks (`MatrixUnit.h:27-31`) |
| `SUPPORT_MVM` / `SUPPORT_SPMM` / `SUPPORT_DWC` | `Makefile:75-90`; `ArchitectureParams.h:182-188`, `333-335` | false | add whole units **and their own parameter queues** (`Accelerator.h:57-133,168-200`) |
| `CLOCK_PERIOD` | `Makefile:93-94` | (unset) | HLS target period; also the harness's ns→cycle divisor |
| `NUM_CODEBOOK_ENTRIES` | `ArchitectureParams.h:263-265` | 16 | **changes `MatrixParams::width`** |
| `SUPPORT_CODEBOOK_QUANT`, `DWC_*`, `UNROLLFACTOR` | `ArchitectureParams.h:246-265,333-356` | see file | |

`SUPPORT_MX` is **not** a `-D` knob: it is implied by `DATATYPE` (`ArchitectureParams.h:115,146,178-180`).

**Derivable from a built instance, no recompile.** Everything structural is legible in the generated
Verilog: the complete top-level port list with widths (`concat_rtl.v:262497-262589`), the RAM
geometries (`DoubleBuffer_1024_128` at `concat_rtl.v:262147`, instantiated twice as `input_buffer`
`:262373` and `weight_buffer` `:262426`; `DualPortBuffer_Pack1D_DataTypes_int24_16UL_1024` at
`:262058`, instantiated as `accumulation_buffer` `:262477`), and which optional units exist (their
ports are simply absent).

**NOT derivable from RTL.** The **field names, widths and Marshall order inside the 64-bit parameter
stream**. The RTL sees one 64-bit `..._params_in_dat` port and a `passthrough` blackbox
(`ParamsDeserializer.h:45-49`, `src/passthrough.v`); the meaning of the bits exists only in
`src/Params.h`. **The voyager_accel ABI is source-derived, not RTL-derived** — which is why a merlin
fact extractor for this target cannot be an RTL reader.

### 1.2 Geometry (built instance)

| Fact | Value | Derivation |
|---|---|---|
| Systolic array | 16 × 16 = 256 MACs | `IC_DIMENSION` × `OC_DIMENSION` (`Makefile:48-49`); array at `src/SystolicArray.h`, cell `src/ProcessingElement.h` |
| Input dtype | `int8` (8 b) | `INPUT_DATATYPE DataTypes::int8` (`ArchitectureParams.h:94`); `typedef Int<8,true> int8` (`datatypes/DataTypes.h:30`) |
| Weight dtype | `int8` (8 b) | `ArchitectureParams.h:95` |
| Accumulator dtype | `int24` (24 b) | `ACCUM_DATATYPE DataTypes::int24` (`ArchitectureParams.h:96`); `DataTypes.h:34` |
| Accumulator-buffer dtype | `int24` | defaults to `ACCUM_DATATYPE` (`ArchitectureParams.h:214-216`); confirmed by the built RAM name `DualPortBuffer_Pack1D_DataTypes_int24_16UL_1024` (`concat_rtl.v:262058`) |
| Vector dtype | `bfloat16` | `ArchitectureParams.h:97`; `StdFloat<7,8,false,true,AC_RND_CONV>` (`DataTypes.h:40`) |
| Input port | 128 b | `IC_PORT_WIDTH = IC_DIMENSION * INPUT_DTYPE_WIDTH` (`ArchitectureParams.h:290-294`); built: `input [127:0] matrix_unit_input_resp_dat` (`concat_rtl.v:262536`) |
| Weight/bias/vector-fetch port | 128 b | `OC_PORT_WIDTH = OC_DIMENSION * WEIGHT_DTYPE_WIDTH` (`ArchitectureParams.h:296-300`); built `[127:0]` on `matrix_unit_weight_resp_dat`, `..._bias_resp_dat`, `vector_fetch_{0,1,2}_resp_dat` |
| Matrix→vector result bus | 384 b (16 lanes × int24) | `Pack1D<ACCUM_BUFFER_DATATYPE, OC_DIMENSION>` (`Accelerator.h:46-47`); built: `wire [383:0] matrix_unit_output_dat` (`concat_rtl.v:262595`) |
| Vector lanes | 16 | `VECTOR_UNIT_WIDTH = OC_DIMENSION` (`ArchitectureParams.h:198-200`) |
| Address width (params) | 32 b | `ADDRESS_WIDTH 32` (`ArchitectureParams.h:176`) |
| Address width (bus) | 64 b | `MemoryRequest.address` (`AccelTypes.h:21`); built: `output [95:0] matrix_unit_input_req_dat` = 64 + 32 |

**Derived hazard.** The parameter ABI caps every tensor base at 32 bits while the bus carries 64. A
merlin backend must refuse a base ≥ 2^32 rather than truncate.

**Derived bandwidth bound (already measured).** For INT8 the ports are 128 b while `VECTOR_DATATYPE`
is bf16 over 16 lanes = 256 b, so one vector takes two beats (`ArchitectureParams.h:296-300` vs
`vector_unit/VectorFetch.h:369-370`). This is the vector path's own 2×, separate from the FIFO
(`PLANE_B.md`, "Cause 2").

### 1.3 Memory (built instance)

| Store | Shape | Bytes | Derivation |
|---|---|---|---|
| Input buffer | 2 banks × 1024 × 128 b | 32 KiB | `DoubleBuffer<INPUT_BUFFER_SIZE, INPUT_BUFFER_WIDTH>` (`MatrixUnit.h:42`), two independent arrays `mem0`/`mem1` (`DoubleBuffer.h:12-16`); width `IC_DIMENSION*INPUT_DTYPE_WIDTH` (`ArchitectureParams.h:313-315`); built `DoubleBuffer_1024_128` (`concat_rtl.v:262147,262373`) |
| Weight buffer | 2 banks × 1024 × 128 b | 32 KiB | `MatrixUnit.h:74`; width `ArchitectureParams.h:321-323`; built `concat_rtl.v:262426` |
| Accumulation buffer | 1 bank × 1024 × 384 b | 48 KiB | `DualPortBuffer<Pack1D<int24,16>, ACCUM_BUFFER_SIZE>` (`MatrixUnit.h:110`); `NUM_BANKS = DOUBLE_BUFFERED_ACCUM_BUFFER ? 2 : 1` (`DualPortBuffer.h:9-16`); built `concat_rtl.v:262058,262477` |
| **Total on-chip** | | **112 KiB** | sum of the three; three distinct `ccs_ram_sync_1R1W` instantiations in the built top |

**Banking semantics.** Input and weight are **ping-pong double buffers**, not banked address spaces:
each has two full-depth memories with their own write/read request port pair and their own thread
(`DoubleBuffer.h:22-25,32-48`), so a load into bank *b* overlaps a compute out of bank *1−b*. The
accumulator is a single 1R1W memory unless `DOUBLE_BUFFERED_ACCUM_BUFFER=true`, in which case it
becomes 2 banks × 2 ports and the vector unit gets its own read port
(`MatrixUnit.h:120-131`, `Accelerator.h:49-55`).

**Discrepancy to resolve, not to adopt.** `merlin/contract/perf_reference_targets.yaml:247` names the
design `voyager_accelerator_ic16_oc16_sram96KB_tsmc16_1GHz`. The derived on-chip total for this build
is **112 KiB** (48 KiB of it accumulator, because `INT8` accumulates in int24); `INT8_32`
(`ArchitectureParams.h:99-104`) gives 128 KiB. The `96KB` figure matches neither. Flagged — it is a
label in a reference table, and this map does not change it.

**No scratchpad.** There is no software-addressable on-chip memory: the three buffers are filled only
by the units' own address generators, from parameters. A merlin backend has no `mvin`/`mvout`
equivalent and cannot place a tile by address.

### 1.4 The command / parameter ABI

**Transport.** One `Connections::In<ac_int<64,false>>` queue **per unit**, no shared front end:

| Queue | Declared | Built port | Present when |
|---|---|---|---|
| `matrix_unit_params_in` | `Accelerator.h:22` | `input [63:0] matrix_unit_params_in_dat` + `_vld`/`_rdy` (`concat_rtl.v:262522-262524`) | always |
| `vector_unit_params_in` | `Accelerator.h:139` | `concat_rtl.v:262547-262549` | always |
| `matrix_vector_unit_params_in` | `Accelerator.h:64` | — | `SUPPORT_MVM` |
| `spmm_unit_params_in` | `Accelerator.h:110` | — | `SUPPORT_SPMM` |
| `dwc_unit_params_in` | `Accelerator.h:173` | — | `SUPPORT_DWC` |

**Framing: there is none.** No opcode, no length, no tag. Each deserializer pops exactly
`ceil(T::width / 64)` words and reassembles the struct (`ParamsDeserializer.h:15-26`,
`get_serialized_params<T, 64>`). The word count is therefore a **compile-time constant of the build**,
and a producer that disagrees with it desynchronises the queue permanently with no error.

**Bit order (derived, and self-checking).** `Marshaller::AddField` writes field *n* at
`glob.range(cur_idx+FieldSize-1, cur_idx)` and advances `cur_idx` — so the **first Marshalled field
occupies the LOW bits** (`marshaller.h:156-169`). `GetResult` asserts `cur_idx == Size`
(`marshaller.h:172-175`), i.e. the declared `T::width` must equal the exact sum of field widths.
Word *i* of the stream is bits `[64i+63 : 64i]` (host side `Harness.cc:344-349`; device side
`ParamsDeserializer.h:18-22`); the final word is zero-padded.

#### `MatrixParams` — every field, in Marshall order

Declarations `Params.h:110-182`; Marshall order `Params.h:205-305`. `A = ADDRESS_WIDTH = 32`,
`L = MatrixParams::LOOP_WIDTH = 16` (`Params.h:108`), `D = DTYPE_INDEX_WIDTH = 1`
(`ArchitectureParams.h:274-276`; 2 only for `MXNF4`, `:138`), `E = NUM_CODEBOOK_ENTRIES = 16`,
`WI = DECODED_INPUT_DTYPE_WIDTH = WW = DECODED_WEIGHT_DTYPE_WIDTH = 8` (`ArchitectureParams.h:250-256`).

| # | Field | Bits | Count | Σ bits | Bit offset (this build) |
|---|---|---|---|---|---|
| 1 | `input_offset` | A=32 | 1 | 32 | 0 |
| 2 | `input_scale_offset` | 32 | 1 | 32 | 32 |
| 3 | `weight_offset` | 32 | 1 | 32 | 64 |
| 4 | `weight_scale_offset` | 32 | 1 | 32 | 96 |
| 5 | `bias_offset` | 32 | 1 | 32 | 128 |
| 6 | `loops[i][j]` i=0..1, j=0..5 (row-major) | L=16 | 12 | 192 | 160 |
| 7 | `x_loop_idx[0..1]` | 3 | 2 | 6 | 352 |
| 8 | `y_loop_idx[0..1]` | 3 | 2 | 6 | 358 |
| 9 | `reduction_loop_idx[0..1]` | 3 | 2 | 6 | 364 |
| 10 | `weight_loop_idx[0..1]` | 3 | 2 | 6 | 370 |
| 11 | `fy_loop_idx[0..1]` | 3 | 2 | 6 | 376 |
| 12 | `fx_loop_idx` | 3 | 1 | 3 | 382 |
| 13 | `weight_reuse_idx[0..1]` | 3 | 2 | 6 | 385 |
| 14 | `stride` | 8 | 1 | 8 | 391 |
| 15 | `padding` | 8 | 1 | 8 | 399 |
| 16 | `weight_addr_loops[i][j]` i=0..1, j=0..4 | 16 | 10 | 160 | 407 |
| 17 | `weight_addr_reduction_loop_idx[0..2]` | 3 | 3 | 9 | 567 |
| 18 | `weight_addr_weight_loop_idx[0..1]` | 3 | 2 | 6 | 576 |
| 19 | `weight_addr_fy_idx[0..1]` | 3 | 2 | 6 | 582 |
| 20 | `weight_addr_fx_idx` | 3 | 1 | 3 | 588 |
| 21 | `input_dtype` | D=1 | 1 | 1 | 591 |
| 22 | `use_input_codebook` | 1 (bool) | 1 | 1 | 592 |
| 23 | `input_burst_size` | 10 | 1 | 10 | 593 |
| 24 | `input_num_beats` | 4 | 1 | 4 | 603 |
| 25 | `input_pack_factor_lg2` | 4 | 1 | 4 | 607 |
| 26 | `weight_dtype` | 1 | 1 | 1 | 611 |
| 27 | `use_weight_codebook` | 1 | 1 | 1 | 612 |
| 28 | `weight_burst_size` | 10 | 1 | 10 | 613 |
| 29 | `weight_num_beats` | 4 | 1 | 4 | 623 |
| 30 | `weight_pack_factor_lg2` | 4 | 1 | 4 | 627 |
| 31 | `input_code[0..15]` | WI=8 | 16 | 128 | 631 |
| 32 | `weight_code[0..15]` | WW=8 | 16 | 128 | 759 |
| 33 | `input_code_zero_idx` | 4 | 1 | 4 | 887 |
| 34 | `head_size_lg2` | 8 | 1 | 8 | 891 |
| 35 | `is_resnet_replication` | 1 | 1 | 1 | 899 |
| 36 | `is_generic_replication` | 1 | 1 | 1 | 900 |
| 37 | `num_channels` | 2 | 1 | 2 | 901 |
| 38 | `fx_unrolling_lg2` | 3 | 1 | 3 | 903 |
| 39 | `input_y` | 16 | 1 | 16 | 906 |
| 40 | `input_x` | 16 | 1 | 16 | 922 |
| 41 | `has_bias` | 1 | 1 | 1 | 938 |
| 42 | `is_mx_op` | 1 | 1 | 1 | 939 |
| 43 | `is_fc` | 1 | 1 | 1 | 940 |
| 44 | `merge_heads` | 1 | 1 | 1 | 941 |
| 45 | `input_transpose` | 1 | 1 | 1 | 942 |
| 46 | `weight_transpose` | 1 | 1 | 1 | 943 |
| 47 | `write_output_to_accum_buffer` | 1 | 1 | 1 | 944 |
| 48 | `weight_dequant` | 1 | 1 | 1 | 945 |
| 49 | `dq_scale_offset` | 32 | 1 | 32 | 946 |
| 50 | `dq_zero_point_offset` | 32 | 1 | 32 | 978 |
| — | *(`SUPPORT_SPMM` only)* `is_spmm`, `spmm_data_offset`, `spmm_indices_offset`, `spmm_indptr_offset` | 1+32×3 | 4 | 97 | 1010 |

**Total for this build: 1010 bits = 16 words of 64 (1024 b, 14 zero pad bits).** This reproduces the
release's own declared width exactly — `base_width` 712 + `extra_width` 298 + `spmm_extra_width` 0
(`Params.h:184-201`) — which is the Marshaller's `cur_idx == Size` invariant and the cheapest
correctness check a packer can run.

#### The other parameter structs

| Struct | Width | Words | Derivation |
|---|---|---|---|
| `VectorParams` | 1084 b | 17 | `4*address_gen_width(186) + 12 + 36 + 32 + 18 + 4 + 8 + A − 16 − 18 + codebook(135) + sparse(97)` (`Params.h:938-952`) |
| `VectorInstructionConfig` | 1582 b | 25 | `VectorInstructions::width(135, :639) * 8 + 4 + 16 + ApproxUnitConfig::width(434, :1444-1445, from NUM_MAXES=6/NUM_RANGES=7/NUM_COEFFS=3 at ArchitectureParams.h:8-10) + OutlierFilterConfig::width(48, :1513)` (`Params.h:1560-1562`) |
| `DwCParams` | 349 b | 6 | `Params.h:1740-1744` |
| `MemoryRequest` | 96 b | — | `AccelTypes.h:24`; built `[95:0]` |

**A vector operation is always a PAIR, in this order: `VectorParams` then `VectorInstructionConfig`**
(`ParamsDeserializer.h:169-198`) — 17 + 25 = **42 words** on `vector_unit_params_in`.

**Routing is the host's job.** Nothing in the stream says which unit an op is for. The release's own
driver dispatches on struct type and on `is_fc` / `is_spmm` **before** serialising
(`Harness.cc:352-397`). A merlin backend must own the same decision.

#### Loop-nest semantics (what `loops` means)

- The 12 counters are **one odometer**: `loops[0][0]` outermost … `loops[1][5]` innermost, incremented
  innermost-first at `MatrixProcessor.h:401-417` and `:617-630`. Level 0 is the L2 (off-chip tile)
  level, level 1 the L1 (on-chip) level.
- `x_loop_idx[l]`, `y_loop_idx[l]`, `reduction_loop_idx[l]`, `weight_loop_idx[l]`, `fy_loop_idx[l]`,
  `weight_reuse_idx[l]` each name **which of the six slots at level `l`** carries that dimension;
  `fx_loop_idx` exists at level 1 only (`Params.h:118-124`, read at `MatrixProcessor.h:211-216`).
- `weight_addr_loops[2][5]` + its own idx fields are a **second, independent** address generator for
  the weight stream (`Params.h:129-136`, consumed in `WeightController.h`).
- **Width hazard, derived:** `MatrixProcessor`'s counters are `LOOP_WIDTH = 10` (`MatrixProcessor.h:23`)
  while `MatrixParams::LOOP_WIDTH` is 16 (`Params.h:108`). A bound ≥ 1024 is silently truncated inside
  the unit. The release's own tiler guards it (`while (x0 >= 1024 || ...)`, `Tiling.cc:524`).

**Closed-form GEMM mapping** (`Tiling.cc:486-574`), the reference a merlin backend must match or beat.
For `M × K @ K × N` with `x0·x1 = M`, `k0·k1 = N / OC_DIMENSION`, `c0·c1 = K / IC_DIMENSION`:

```
loops             = {{x1, 1, k1, c1, 1, 1}, {c0, k0, 1, 1, 1, x0}}
x_loop_idx        = {0, 5}      y_loop_idx   = {1, 4}
reduction_loop_idx= {3, 0}      weight_loop_idx = {2, 1}
fx_loop_idx       = 3           fy_loop_idx  = {4, 2}
weight_reuse_idx  = {4, 5}      stride = 1      padding = 0
```

**Capacity constraints the release itself enforces** (`Tiling.cc:524-560`) — these are the real
capability bounds, and they are derived, not guessed:

```
x0 * c0                  <= INPUT_BUFFER_SIZE      (1024)
k0 * c0 * IC_DIMENSION   <= WEIGHT_BUFFER_SIZE     (1024)
x0 * k0                  <= ACCUM_BUFFER_SIZE      (1024)
x0                       <  1024                   (MatrixProcessor LOOP_WIDTH)
```

#### Memory addressing (units master their own memory)

Each unit issues `MemoryRequest{address, burst_size}` and consumes an **untagged, in-order** stream of
port-width beats:

- `address = <stream>_offset + element_index * dtype_width / 8` — `send_packed_request`
  (`Utils.h:60-68`); `dtype_width` is looked up from the dtype index (`Utils.h:65`,
  `DataTypes.h:200-228`). So offsets are **byte** addresses and the index is in **elements**.
- `burst_size` is in **bytes**; the responder returns `ceil(burst_size / (port_width/8))` beats
  (`Harness.cc:194-206`). `<stream>_burst_size` in `MatrixParams` is set to `fetch_width / 8`
  (`MatrixOps.h:414,455`).
- The input element index is a full address-generator expression over the loop counters
  (`InputController.h:215-241`), with alternate layouts selected by `is_resnet_replication`,
  `is_generic_replication`, `merge_heads`, `input_transpose`.
- Streams, all independent 96 b request + 128 b response port pairs: matrix `input`/`weight`/`bias`
  (`Accelerator.h:23-31`), vector `vector_fetch_0/1/2` (`:141-150`), plus MX scale streams under
  `SUPPORT_MX` (`:34-40`).

#### Result egress — the fact that shapes the first milestone

**The matrix unit never writes memory.** Its only output is `matrix_unit_output`, a combinational
`Pack1D<int24,16>` channel wired directly into the vector unit (`Accelerator.h:46-47`, bound at
`:220` and `:304`). The **vector** unit owns every store: `vector_output_data` (128 b) +
`vector_output_addr` (32 b), and the MX-scale and sparse variants (`Accelerator.h:152-163`).

So **one matmul whose result must reach DRAM is two parameter submissions**: 16 words on
`matrix_unit_params_in` and 42 words on `vector_unit_params_in`. There is no matrix-only path to
memory short of `write_output_to_accum_buffer` (`Params.h:171`), which keeps the tile on chip.

### 1.5 The completion protocol

- Per unit, two **`Connections::SyncOut` channels carrying no data**: `<unit>_start` and `<unit>_done`
  (`Accelerator.h:43-44,100-101,131-132,165-166,198-199`). Built: `output matrix_unit_start_vld;
  input matrix_unit_start_rdy; output matrix_unit_done_vld; input matrix_unit_done_rdy`
  (`concat_rtl.v:262543-262546`), i.e. a bare ready/valid pulse pair.
- **There is no interrupt, no status register, no MMIO, no completion queue and no operation id.** The
  host *blocks* on the pulse: `matrix_unit_start.SyncPop()` (`Harness.cc:441`),
  `matrix_unit_done.SyncPop()` (`Harness.cc:538`).
- A fused matrix+vector op pops matrix-start then vector-start (`Harness.cc:441,456`) and matrix-done
  then vector-done (`Harness.cc:538,548`).
- Between operations the release's own driver **serialises**: it waits `operation_done` before
  enqueueing the next op's params (`Harness.cc:392-396`). Whether a unit will accept parameters for
  op *N+1* while executing op *N* is therefore **not established by the release** — see UNKNOWNs.
- Ordering across units is implicit: the queues are independent, so a matrix op and a vector op that
  must be ordered are ordered only by the host's start/done discipline.

---

## 2. The drafts

| Artifact | Path | Status |
|---|---|---|
| Target descriptor | `merlin/experiments/voyager_h2h/targets/voyager_accel/target_experiment.yaml` | draft, hand-authored (the only hand-authored descriptor form merlin has) |
| Capability-manifest side-input | `merlin/targets/voyager_accel/contracts/residual.yaml` | draft, hand-authored |
| Capability manifest | `merlin/targets/voyager_accel/contracts/target_contract.yaml` | **not written** — it is GENERATED (`merlin.targetgen.capability_manifests.write_oot_target`) and must never be hand-edited. It cannot be generated yet; see gap G1. |

Both drafts carry a `# cite:` comment on every derived value. Values that are not derivable are
recorded as `UNKNOWN` — in the field itself where merlin's validator allows a free value, and in the
residual's `unknowns:` block where it does not (see gap G11).

**The promotion boundary, and why the descriptor is not under `capsule_bench/targets/`.**
`descriptor_for('voyager_accel')` resolves the path above exactly as it would resolve one under
`capsule_bench/targets/` (`target_experiment.py:665-666`, glob `*/targets/{t}/{f}`). What
`capsule_bench/targets/` additionally means is membership of the **live roster**:
`merlin/tests/infra/test_sandbox_isolation.py:44-47` globs that one directory and generates and
audits a bundle for every descriptor in it, and `build_tools/scripts/_target_roster.py:22` reads it to
extend the no-target-name gate. Measured: placing this descriptor there fails 6 sandbox tests —
correctly, because voyager_accel has no capsule corpus, backend, oracle adapter or sim toolchain.
**Moving that one file into `merlin/experiments/capsule_bench/targets/voyager_accel/` is the act that
makes this target live**, and it belongs with the first milestone, not before it.

**Derivation check that passes today:**
`.venv/bin/python -c "from merlin.targetgen import capability_manifests as cm;
cm.validate(cm.manifest_for('voyager_accel'))"` — the residual is discovered
(`cm.discovered_targets()` now includes `voyager_accel`), derives, and validates against
`merlin/schemas/target_contract.schema.yaml`.

---

## 3. UNKNOWNs (recorded, not guessed)

| # | Fact | Why it is not derivable |
|---|---|---|
| U1 | Clock frequency | Not a property of the release. The pinned build targets 5 ns on generic 45 nm and seven blocks miss it by Catapult's own estimate (`PLANE_B.md`, HLS table). The paper's 1 GHz / TSMC 16 nm point is not reproducible here. |
| U2 | DRAM latency, bandwidth, burst policy | The release ships no SoC and no memory model: the testbench answers every request from an array with no stalls (`Harness.cc:189-221`) behind an `sc_fifo` (`Harness.h:32-41`). Every published cycle count is against a perfect memory. |
| U3 | Bus arbitration / number of real memory ports | The top exposes 6 independent request/response pairs (3 matrix, 3 vector) plus MX/SPMM/DWC variants; nothing in the release says how they share a real interconnect. |
| U4 | Parameter-queue depth | `Connections::In<ac_int<64>>` declares none; the harness binds a `Connections::Combinational` (`Harness.h:28-29`). The backlog a real SoC would allow is unspecified. |
| U5 | Whether a unit accepts op *N+1*'s parameters while op *N* runs | The release's driver never tries (`Harness.cc:392-396`). Until measured, a merlin backend must assume it does not. |
| U6 | Vector instruction semantics (`op_type`, `vector_op0..3`, `vdest`, `reduce_op`, the approximation-unit coefficients) | Defined in `src/vector_unit/VectorOps.h` and `test/toolchain/VectorOps.h`; deliberately out of scope for this pass. The *ABI* (widths, order, 42 words) is derived above; the *meaning* of the fields is not. |
| U7 | Area, power, technology numbers | Build- and PDK-dependent; the generic 45 nm build is not the paper's 16 nm point. |
| U8 | On-chip SRAM figure in `perf_reference_targets.yaml` (`sram96KB`) | Contradicts the derived 112 KiB (INT8) / 128 KiB (INT8_32). Provenance of the 96 KB label not established. |
| U9 | Whether `MXINT8`/`MXNF4` (the only `SUPPORT_MX` datatypes) change the completion or queue protocol | Not read in this pass; the MX builds add two more request/response pairs per unit (`Accelerator.h:33-41`) and change `MatrixParams::width` via `DTYPE_INDEX_WIDTH`. |
| U10 | An RTL-derivable form of the parameter ABI | None exists: the struct layout lives only in `src/Params.h` (see §1.1). Any merlin fact extractor for this target must read C++ headers, which merlin has no extractor family for (gap G1). |

---

## 4. First executable milestone — one matmul layer, merlin-produced parameters

**Goal.** Drive one GEMM layer's parameters, produced by merlin, through the release's own already-built
testbench, and compare the bytes it writes against an independent reference.

**What already exists** (`PLANE_B.md`): the plane-B environment, the 5 ns INT8 16×16 Catapult build, a
linked SCVerify VCS binary, the pre-HLS SystemC `TestRunner`, and a driver with single-layer
subcommands (`scripts/plane_b_regression.sh {systemc-layer,rtl-layer,rtl-trace}`).

**What is missing.** Three things, all on merlin's side of the boundary:

- **M1 — a parameter injection path.** Nothing in the harness reads parameters from a file.
  `Harness::send_params` (`Harness.cc:352-397`) consumes a `std::deque<BaseParams*>` built by
  `map_operation` (`test/toolchain/MapOperation.cc:17-46`) from the legacy `param.proto` `model.txt`.
  A merlin-produced buffer needs a small additive TestRunner that reads 64-bit words and pushes them
  straight into `matrix_unit_params_in` / `vector_unit_params_in`, with a matching start/done reaper.
- **M2 — data placement.** `DataLoader` places tensors at addresses taken from the protobuf
  (`MatrixOps.h:365` `get_address(input)`) out of `$CODEGEN_DIR/networks/<model>/<DATATYPE>/tensor_files`
  (`Simulation.cc:120-131`). A merlin buffer must declare its own `{tensor -> base, bytes}` and have
  them written into `ArrayMemory` before reset.
- **M3 — a gold model.** `run_gold_model` is driven by the protobuf op list, not by `MatrixParams`
  (`Simulation.cc:204-243`), so injecting parameters bypasses Voyager's gold model entirely. For a
  standalone matmul the honest reference is merlin's own exact int8→int24 GEMM plus the requantise the
  `VectorInstructionConfig` asks for — stated as merlin's reference, not Voyager's.

### Step list

Run everything from the repo root with `TMPDIR` pointing at a large scratch filesystem. The pinned checkout
(`out/build/external/voyager-accelerator`) is never written; all work happens in a patched copy, the
pattern `PLANE_B.md` already established for the FIFO A/B.

1. **Pin check.** `.venv/bin/python -c "from merlin.common import provenance as P;
   print(P.verify('voyager_accelerator'), P.verify('voyager_compiler_accel_pair'))"`
   — both clean before anything else.

2. **Make a work copy for the injector** (the pinned tree and the flow's work tree stay untouched):
   `cp -a --reflink=auto out/build/external/voyager-accelerator-work out/build/external/voyager-accelerator-inject`
   minus `.git`, `build/`, `regression_results/`, with `models` and `test/compiler/networks` symlinked
   — the same recipe as `voyager-accelerator-ab-fifo2` (`PLANE_B.md`, "A/B, one variable").

3. **Add the injector** (~120 lines, additive, no existing file edited) to the copy:
   - `test/common/ParamInjector.cc` — reads `$PARAM_FILE` (a flat little-endian `uint64` array) and
     `$MEM_FILE` (a JSON manifest of `{path, base, bytes}` blobs), loads the blobs into `ArrayMemory`,
     pushes `$MATRIX_WORDS` words into `matrix_unit_params_in` and `$VECTOR_WORDS` into
     `vector_unit_params_in`, then pops matrix-start, vector-start, matrix-done, vector-done in the
     order `Harness::record_start`/`record_done` establishes (`Harness.cc:429-458,526-550`).
   - **The width assertion that makes the ABI self-checking**: the injector prints
     `MatrixParams::width`, `VectorParams::width`, `VectorInstructionConfig::width` and aborts unless
     the word counts it was handed equal `ceil(width/64)`. Because those are compile-time constants of
     the same `-D` set, this is a real cross-check, not a comment.
   - a `Makefile` fragment building `$(BUILD_DIR)/cc/ParamInjector` beside the existing `TestRunner`.

4. **Emit the buffer from merlin.** A packer (`merlin/experiments/voyager_h2h/scripts/voyager_params.py`,
   to be written) that takes `(M, N, K, base_in, base_w, base_bias, base_out)` and emits the 16 + 42
   words, packing fields LSB-first in the Marshall order of §1.4 and the loop nest of §1.4's
   closed form. It must *fail closed* on every derived bound (§1.4 capacity constraints, the 10-bit
   counter, the 32-bit offsets) rather than truncate.

5. **Gold.** `.venv/bin/python merlin/experiments/voyager_h2h/scripts/voyager_params.py --gold` writes
   the exact int8 GEMM result at the output base, as a byte image to diff.

6. **Run it on the pre-HLS SystemC model first** (seconds, no Catapult, no VCS):
   ```
   VOYAGER_WORK=$PWD/out/build/external/voyager-accelerator-inject \
   PARAM_FILE=<...>/params.u64 MEM_FILE=<...>/mem.json \
   merlin/experiments/voyager_h2h/scripts/plane_b_regression.sh systemc-layer inject matmul_1 \
     $PWD/out/runs/voyager_accel/plane_b/inject_systemc_<TS>
   ```
   (`systemc-layer` already builds and runs a `cc/` binary in the work tree; it needs one added case
   for `ParamInjector`.)

7. **Then the same buffer on the built RTL**, reusing the existing 5 ns `Accelerator.v1` SCVerify build
   — no re-HLS, because the injector touches only `test/`:
   ```
   VOYAGER_WORK=$PWD/out/build/external/voyager-accelerator-inject \
   PARAM_FILE=... MEM_FILE=... \
   merlin/experiments/voyager_h2h/scripts/plane_b_regression.sh rtl-layer inject matmul_1 \
     $PWD/out/runs/voyager_accel/plane_b/inject_rtl_<TS>
   ```

8. **Control, and it is the one that makes the milestone mean anything.** Run the *same shape* through
   the release's own path (`plane_b_regression.sh rtl-layer mobilebert_encoder matmul_2_fused ...`) and
   compare both the output bytes and the cycle count. If merlin's buffer and Voyager's own mapping
   produce different cycles for the same shape, that difference is the first plane-B result — and it
   must be reported with the depth-1 FIFO cap (`MatrixProcessor.h:24`) stated, or with both sides
   patched.

9. **Record it** under `out/runs/voyager_accel/plane_b/` with `merlin.common.provenance.record()` for
   `voyager_accelerator` and `voyager_compiler_accel_pair`, plus the merlin commit and the three
   toolchain deviations `PLANE_B.md` lists.

**Estimated blockers, in likelihood order:** (a) the injector's start/done pop order must match what
the units actually pulse when there is no `map_operation` to shape the op — cheapest to settle on the
SystemC model in step 6; (b) `Connections::Combinational` depth-1 semantics may require the injector
to push params from its own thread rather than inline; (c) a matmul with `has_bias=false` still reads
the bias stream in some paths (`WeightController.h:761-764`) — check before assuming a bias base can
be null.

---

## 5. Gap list — what merlin's target contract cannot express for this hardware

| # | Gap | Evidence | What would have to change |
|---|---|---|---|
| **G1** | **No fact-extractor family can read this target.** `FamilyProfile.fact_extractor` is one of `circt_static`, `simt_config`, `opu` (`merlin/python/merlin/targetgen/families.py:55-95`); all three read elaborated RTL. voyager_accel's geometry is in C++ headers and its ABI is *only* in `src/Params.h` (§1.1). | `families.py:55-95`; `ArchitectureParams.h`; `Params.h` | A fourth extractor family (`headers` / `hls_source`) that parses the release's own `-D` set + the built Verilog port list, or an explicitly declared all-residual target with a committed, hand-derived fact bundle. Without one, `merlin-onboard` refuses rather than fabricates (`targetgen/onboard.py:169-173`) — correctly. |
| **G2** | **`kind` and `endpoint_kind` are coupled the wrong way for this device.** Voyager's matrix unit *is* a systolic array, but it has no command ISA. `kind: systolic` forces `encoding_required=True` and the `rocc_insn` trace gate from the family profile; only `kind: spatial` reaches `command_buffer` by default. | `families.py:75-95`; `target_experiment.py:1182-1188` (`encoding_required=prof.encoding_required` — **no override path**). MEASURED on the draft residual: `load_capability_manifest` returns `endpoint_kind='command_buffer'` (override honoured), `trace_gate=None` (correctly cleared), `encoding_required=True` (stuck). | Make `encoding_required` overridable by the manifest the way `trace_gate` already is (`target_experiment.py:1182-1183`), or split the datapath axis (`kind`) from the control axis (`endpoint_kind`) in `FamilyProfile`. Declaring `kind: spatial` to get the right endpoint would misdescribe the datapath and misroute fact extraction — the wrong fix. |
| **G3** | **Per-unit parameter queues have no representation.** merlin's command buffer is one flat `commands` array with one `target` (`merlin/contract/schemas/command_buffer.schema.json`). voyager_accel has 2–5 independent queues, and *routing is the producer's obligation* (`Harness.cc:352-397`). | schema `required: [abi_version, target, commands]`; `Accelerator.h:22,64,110,139,173` | Either a per-command `unit`/`queue` field in the ABI, or a documented rule that the target's runtime backend owns routing (workable today, but then the L0 reference simulator cannot model cross-queue ordering). |
| **G4** | **The schedule — the thing plane B measures — is not expressible.** A `MatrixParams` *is* a 12-deep loop nest plus two address generators. merlin's `MATMUL` command carries operands and a small closed attribute set (`command_buffer_abi.yaml` `opcodes:`), no tiling, no loop order, no buffer residency. | `Params.h:117-136`; `command_buffer.schema.json` `commands[].attributes` | A schedule-carrying attribute the reference simulator actually models. `attributes` is `additionalProperties: true`, so a schedule can ride along today — but opaquely, which means merlin's own L0 gate cannot check it and a scheduling bug reaches RTL undetected. |
| **G5** | **Bus-mastering units / no scratchpad.** merlin's memory model is `{resident, accumulators}` and the command vocabulary assumes explicit movement (`RES_PACK`, `MOVEMENT`, `COMMIT`, `EVICT`). voyager_accel has no software-addressable on-chip memory at all: the units fetch from DRAM themselves under parameter control (`Utils.h:60-68`, `InputController.h:215-241`). | `merlin/targets/gemmini/contracts/residual.yaml` `memory_model`; `command_buffer_abi.yaml` opcodes | A `memory_model` axis for *unit-mastered, no-scratchpad* devices, so `capacity_fit`-style obligations bind on the **loop-tile** bounds of §1.4 instead of on a scratchpad byte budget. |
| **G6** | **No interrupts / no completion queue.** merlin's runtime promises include `command_buffer` and `metrics`; there is no vocabulary for "blocking ready/valid pulse per unit, no op id" (`Accelerator.h:43-44`, `concat_rtl.v:262543-262546`). | as cited | A `completion:` block in the manifest (`{kind: sync_pulse|interrupt|poll, per: unit|device, tagged: bool}`). Today the only honest place for it is a free-text note. |
| **G7** | **Result egress is cross-unit.** A matmul's result reaches memory only through the vector unit (`Accelerator.h:46-47,304`; `Accelerator.h:152-155`), so a single `MATMUL` command is two submissions on two queues. No contract field says a compute unit cannot store. | as cited | `compute_units[*]` needs an egress/`stores_to` axis; `contains:` (the schema's existing unit-composition field) is about containment, not dataflow. |
| **G8** | **The ABI's word count is a build constant with no declared home.** 16 / 17 / 25 words here, different under `MXNF4` (`DTYPE_INDEX_WIDTH=2`, `ArchitectureParams.h:138`) or a different `NUM_CODEBOOK_ENTRIES`. A merlin backend that hardcodes 16 breaks silently on a re-parameterised build. | `Params.h:184-201`; `ParamsDeserializer.h:15-26` | The manifest needs a derived `abi.words_per_op` per queue, cross-checked at run time against the build — the same discipline `encoding.readout_bits` gets from `addr_len` today. |
| **G9** | **No registry entry shape fits.** `merlin/contract/compute_endpoints.yaml` requires `encoding.source` ∈ {`rtl_facts`, `isa_encoding`}; voyager_accel has neither. | `compute_endpoints.yaml:19-90` | A third `source` (e.g. `params_struct`) naming the header the struct layout comes from. Deliberately not added here: an unrecognised value could break the mining consumers. |
| **G10** | **Provenance has no slot for "which `-D` set".** The pin registry records a commit; two builds of the same commit at different `IC_DIMENSION` are different devices with different ABIs. | `hardware_pins.yaml:886-912`; `Makefile:118` (the build dir *is* the config) | A per-result `build_config` field (the release's own `BUILD_DIR` string is a ready-made canonical form: `INT8_16x16_1024x1024x1024_false_false_false`). Recommended before any voyager_accel number is cited. |
| **G11** | **A closed-vocabulary field cannot say "unknown".** `compute_units[*].scaling` and `.requant` validate against closed sets with no unknown member, so a target that has the mechanism but whose arithmetic has not been read must either omit the key (reads as "none") or guess. Measured: `scaling: UNKNOWN` raises `ValueError: scaling 'UNKNOWN' not in ['block_affine', ...]`. | `merlin/python/merlin/targetgen/compute_units.py` (the vocabulary check) | Admit an explicit `unknown` member in each closed vocabulary, or a per-field `unknown: [scaling, requant]` list the deriver honours. Worked around here with a free-form `unknowns:` block in the residual, which no consumer reads. This is the same fail-closed principle the repo already applies to RTL facts — an unknown must be recordable, not silently absent. |

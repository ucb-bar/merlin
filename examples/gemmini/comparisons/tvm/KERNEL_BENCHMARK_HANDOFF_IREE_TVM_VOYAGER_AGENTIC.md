# Kernel benchmark handoff: IREE, TVM, Voyager and agentic compilation

## Next-campaign scope and budget

The next handoff covers **ResNet50 plus the 16 kernel cases below**. Full-model
TinyLlama and SmolVLA are backlogged; T01–T07 remain active isolated kernels and
must not be confused with those backlogged model runs. The earlier "few trials"
policy for those models is superseded for this kernel campaign.

Requested TVM tuning: **50 warmup trials + 5 model-guided trials per workload**,
using stock MetaSchedule EvolutionarySearch/XGB, or stop earlier when the legal
space is exhausted. Count candidate schedules, not batches, executions, or timing
repetitions. Three baseline wall-watchdog calibrations are separate and charged
to the same occupied-slot budget. An empty stochastic batch is not proof that
all legal schedules have been enumerated. Record budget stop, search failure,
and verified exhaustion separately. Record fewer than five guided trials when
warmup/training did not complete; do not call random predictions guided.

ResNet's last confirmed time caps are **4 hours DSE** and **30 minutes per final
baseline/tuned E2E run**. Kernel aggregate wall budget and precedence of the
ResNet time cap versus the requested candidate count are awaiting clarification;
this document does not authorize an unbounded paid campaign. Candidate median
watchdogs and confirmed remote termination remain mandatory.

The meaning of "per workload" for multi-output cases versus internal TensorIR
tasks is awaiting confirmation. Regardless of tuning granularity, published
cycles for T01/T02 include all 32 heads; T04/T06 include both/all three outputs.
Never substitute a single contraction's cycles for the complete case.

This is a **preparation specification**, not a claim that 16 TVM kernel ELFs,
other compilers' adapters or FireSim results already exist. The current TVM
chunk driver measures extracted contractions; it cannot simply relabel those
measurements as complete grouped-kernel costs. First supply the exact 16 inputs,
independent references, complete-call builders and measurements below. Existing
ResNet chunk fixtures do not replace these nibble-pattern kernel inputs.

For IREE, Voyager and agentic compilation, retain their native optimization
interfaces and disclose their search method/budget. MetaSchedule is the TVM
method, not an assumed interface for the other systems. Use the same numerical
and measurement contract; unsupported target paths must be reported explicitly.

Authoritative supplied definitions follow, including input generator and historical
context. They override similarly named historical model/capsule recipes. Historical
cycles are context only. Current TVM support/proof limitations and runner requirements
remain in [FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) and
[TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md).

Requested work: compile and execute the 16 kernel workloads below, optimize them through the selected compiler, and return correctness, completed-call cycles and reproducible artifacts. Run the same logical workloads for IREE, TVM, Voyager and the agentic implementation. Record each compiler/fork and target adapter explicitly.

This is a kernel study, not full ResNet50/TinyLlama inference. The integer tensors and scales are supplied inputs. No model download, calibration dataset, TorchAO calibration, perplexity experiment or end-to-end quality threshold is needed for these tests.

## 1. Read these definitions first
All three-number GEMM labels in the supplied table mean M × K × N, not M × N × K. The operation is A[M,K] @ W[K,N] -> Y[M,N].
784×256×1024 therefore means A[784,256] @ W[256,1024]; do not swap 256 and 1024.
The historical case ID ZA_gemm_12544x64x64_native_i8 is misleading: its actual shapes are M=12544, K=64, N=256. Use tensor shapes, not that ID’s last number.
Operands are signed INT8, zero point 0. Accumulation is INT32. The output is INT8 or INT32 as specified per case. Do not make every output INT8 or every output INT32.
The INT8 arithmetic below is a fixed numerical contract. Do not change scales, introduce unsigned reinterpretations, approximate the contraction, or change its mathematical scope to improve time.
Shapes are compile-time constants. Shape specialization and autotuning are allowed; embedding answers or substituting constant output tensors is not.
A returned cycle count is usable only for an implementation that produces every required output correctly.
## 2. Six standalone GEMMs
All arrays are C-contiguous row-major logical tensors. None of these six has bias, activation, transpose, residual addition or non-unit quantization scale.


| ID | Table label: M×K×N | A: INT8 | W: INT8 | Y: INT8 | Exact computation |
| --- | --- | --- | --- | --- | --- |
| G01 | 512×512×512 | [512,512] | [512,512] | [512,512] | sat_i8(A @ W) |
| G02 | 12544×256×64 | [12544,256] | [256,64] | [12544,64] | same |
| G03 | 12544×64×256 | [12544,64] | [64,256] | [12544,256] | same |
| G04 | 3136×512×128 | [3136,512] | [512,128] | [3136,128] | same |
| G05 | 3136×128×512 | [3136,128] | [128,512] | [3136,512] | same |
| G06 | 784×256×1024 | [784,256] | [256,1024] | [784,1024] | same |

Here sat_i8 clamps the completed INT32 sum to [-128,127], then stores INT8. Casting with wraparound is wrong. These are native INT8 exits, not FP32 outputs and not INT32 outputs later compared after a different epilogue.

Historical logical input roster, in allocation order: W, then A0. Output: Y0.

## 3. Three spatial convolutions
All three have batch 4, kernel 3×3, stride 1, dilation 1, groups 1, channel bias, no activation, and INT8 output with scale 1. These are cross-correlations, as in ordinary deep-learning convolution; do not reverse the spatial kernel.

Important historical boundary: the actual interface already supplies the larger input tensor below. The operator applies VALID convolution, padding=0 to it. The historical timing harness filled the entire input—including its outer rows/columns—with deterministic data. It did not replace that border with zeros. Do not silently substitute SAME convolution on an H×W input.


| ID | Table output geometry | Actual IFM: INT8, NHWC | Flattened W: INT8 | B: INT32 | Y: INT8 |
| --- | --- | --- | --- | --- | --- |
| C01 | B4, 56×56, 64→64 | [4,58,58,64] | [576,64] | [64] | [4,56,56,64] |
| C02 | B4, 28×28, 128→128 | [4,30,30,128] | [1152,128] | [128] | [4,28,28,128] |
| C03 | B4, 14×14, 256→256 | [4,16,16,256] | [2304,256] | [256] | [4,14,14,256] |

Weight indexing and reference:


```text
k = ((kernel_y * 3 + kernel_x) * CI + input_channel)
W4[kernel_y,kernel_x,input_channel,output_channel] = W[k,output_channel]

acc[b,y,x,o] = B[o] + sum_{r=0..2,s=0..2,c=0..CI-1}
                        IFM[b,y+r,x+s,c] * W4[r,s,c,o]
Y[b,y,x,o] = clamp(acc[b,y,x,o], -128, 127)
```

The historical output ABI flattens batch/spatial positions: [B*H*W,CO]. This has the same contiguous byte order as NHWC. Bias is added once per output, in INT32, before saturation. No ReLU or pooling is included.

Historical input roster: IFM, then W in the INT8 arena; B is separate INT32 storage. Bias seeds for the generator in §6 are C01=0x6000, C02=0x5000, C03=0x4000.

Direct, implicit-GEMM and materialized-im2col algorithms are all allowed. If im2col is constructed during the call, its CPU/accelerator time and scratch storage count. An extra experiment may use zero-halo SAME inputs, but give it a different case ID and do not merge its numbers into these historical rows.

## 4. Seven TinyLlama kernel workloads
These are isolated contractions/projection groups. They do not include RMSNorm, RoPE, softmax, causal masking, SiLU, gating multiplication, residual addition or KV-cache updates. Do not add those operations under these case names; do not claim these kernels constitute complete attention or complete model inference.

Define:

raw(A,W) = signed INT32 matrix product A @ W
q8(acc,s) = clamp(round_to_nearest_ties_to_even(float32(acc) * float32(s)), -128,127)
All scaled exits below use the exact FP32 value 0.0078125 = 1/128. There is no bias or activation. INT32 raw exits have no requantization.


| ID | Name | Inputs / mathematical work | Required outputs |
| --- | --- | --- | --- |
| T01 | att_qk | 32 products, each M16 K64 N16; 4 shared K groups, 8 query heads/group | 32 × [16,16] INT32 |
| T02 | att_pv | 32 products, each M16 K16 N64; 4 shared V groups, 8 probability heads/group | 32 × [16,64] INT8, q8(...,1/128) |
| T03 | mm_down | A[16,5632] @ W[5632,2048] | [16,2048] INT32 |
| T04 | mm_gu | one A[16,2048], two independent Wg/Wu[2048,5632] | Yg and Yu, each [16,5632] INT8, independent q8(...,1/128) |
| T05 | mm_o | A[16,2048] @ W[2048,2048] | [16,2048] INT32 |
| T06 | mm_qkv | one A[16,2048]; Wq[2048,2048], Wk[2048,256], Wv[2048,256] | Yq[16,2048], Yk[16,256], Yv[16,256], all INT8, independent q8(...,1/128) |
| T07 | mm_lm | A[1,2048] @ W[2048,32000], actual M=1 | [1,32000] INT32 |

mm_gu means two GEMMs, not two calls of an already-combined gate/up kernel. mm_qkv means all three projections. Time the complete group and return/check every output. The implementation may share inputs and fuse dispatch; do not report only the fastest member.

Exact grouped-query attention semantics
Let h=0..31, g=h//8, query m=0..15, key position n=0..15, feature d=0..63.


```text
# T01: canonical logical arrays
Q[h,m,d]             INT8  [32,16,64]
K[g,d,n]             INT8  [4,64,16]   # already in contraction orientation
S[h,m,n] = sum_d Q[h,m,d] * K[h//8,d,n]   INT32 [32,16,16]

# T02: canonical logical arrays
P[h,m,n]             INT8  [32,16,16]  # supplied INT8, not generated by softmax here
V[g,n,d]             INT8  [4,16,64]
O[h,m,d] = q8(sum_n P[h,m,n] * V[h//8,n,d], 1/128)
```

There are 32 products, not 32 independent K/V tensors. Neither case includes a 1/sqrt(64) multiplier. T02 is the original single INT8-product recipe, not the multi-plane numerical recipe used by some whole-model captures.

Historical compiler-side tensor roster: K0,K1,K2,K3, then A0..A31; outputs Y0..Y31. For T01, Kg=[64,16], Ah=[16,64], Yh=[16,16] INT32. For T02, Kg=[16,64], Ah=[16,16], Yh=[16,64] INT8. The matrices are individually contiguous; the historical allocator aligns each to 4096 bytes. Canonical stacked tensors may be constructed from them without changing values.

Physical layouts expected by the original expert functions
If running the original Jack/Exo functions, use these mappings from the canonical logical tensors. The function’s const output pointer declarations do not mean outputs are read-only.


```text
att_qk:
  qr[m, h*64+d] = Q[h,m,d]                  # qr[16,2048]
  krT[g*64+d,n] = K[g,d,n]                  # krT[256,16]
  sc[h,m,n] = S[h,m,n]                     # sc[32,16,16], INT32

att_pv:
  P_expert[h,m,n] = P[h,m,n]                # P[32,16,16]
  v[n,g*64+d] = V[g,n,d]                    # v[16,256]
  a[m,h*64+d] = O[h,m,d]                    # a[16,2048], INT8
Projection weights in those expert functions use 64-column panels:

wp[n//64,k,n%64] = W[k,n]                  # [N/64,K,64]
mm_gu: concatenate packed gate then up panels: [176,2048,64] = 88+88 panels. mm_qkv: concatenate Q then K then V: [40,2048,64] = 32+4+4 panels. Pack canonical row-major values; do not fill an unrelated packed tensor with another random stream and call it the same input.

```

Packing immutable weights once is allowed in the prepared-weight measurement, but report its time, bytes and lifetime separately. Input/output layout conversion required by the public timed call belongs in that call. If you additionally report a prepared-input microkernel time, label it separately.

## 5. Common numerical acceptance
Read operands as signed two’s-complement INT8, zero point 0. No dynamic scale selection during a kernel.
Form products and bias accumulation in INT32. The reference may use INT64 and explicitly wrap to signed INT32. All supplied signed-nibble inputs are safely below INT32 overflow.
For raw exits, compare every INT32 element exactly. No cosine, RMSE or PPL gate replaces this.
For INT8 exits, compare every INT8 byte exactly, after the specified scale, ties-to-even rounding and saturation. In particular, do not replace saturation with an INT8 cast.
Preserve dimensions, head/group ordering, bias, output precision and independently observable outputs. Validate guards around buffers; poison outputs before execution.
A floating implementation is not an equivalent INT8-target result merely because its final values happen to match small inputs. Record operand, multiplier, accumulator and readout types plus actual accelerator placement.
Use at least one complete independent reference comparison per new binary/case. Subsequent timing repeats can use retained complete-reference digests. Add small tie/saturation/tail checks only when the lowering or numerical conversion changes. Avoid rerunning a large regression suite for every schedule knob.

## 6. Portable deterministic inputs and reference helpers
This reproduces the historical logical compiler-side input arena pattern without requiring Gemmini to initialize it. Every INT8 byte is in [-8,7]. It matches the original SplitMix seed block, scratchpad placement and rotated 256-byte output blocks; it is not numpy.random.

For each case, give make_inputs the INT8 input roster in the order described above. It starts from byte offset 0 and rounds each tensor’s allocation to 4096 bytes. Different compilers must use the same resulting logical arrays. Store little-endian arrays and SHA256 their raw contiguous bytes.


```python
import hashlib
import math
import struct
import numpy as np

MASK64 = (1 << 64) - 1

def splitmix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64

def historical_i8_arena(nbytes, seed=0x5EED):
    seed_block = bytearray()
    for i in range(2048):
        w = splitmix64(seed + i) & 0x0F0F0F0F0F0F0F0F
        sign_bits = (w >> 3) & 0x0101010101010101
        seed_block += struct.pack('<Q', w | (sign_bits * 0xF0))
    spad = bytearray(16384 * 16)
    for row in range(0, 16384, 16):
        src = (row * 16 + (row // 1024) * 272) % (16384 - 256)
        spad[row*16:row*16+256] = seed_block[src:src+256]
    arena = bytearray(((nbytes + 255) // 256) * 256)
    for b in range(len(arena) // 256):
        row = (b * 16 + (b // 1024) * 48) % 16384
        arena[b*256:(b+1)*256] = spad[row*16:row*16+256]
    return arena[:nbytes]

def make_inputs(roster):
    # roster = [(name, shape), ...], INT8 inputs only, in historical order.
    placements, end = [], 0
    for name, shape in roster:
        count = math.prod(shape)
        placements.append((name, tuple(shape), end, count))
        end = ((end + count + 4095) // 4096) * 4096
    arena = historical_i8_arena(end)
    arrays = {}
    for name, shape, offset, count in placements:
        arrays[name] = np.frombuffer(arena, dtype=np.int8,
                                     count=count, offset=offset).reshape(shape).copy()
    return arrays

def make_bias(channels, seed):
    return np.array([(splitmix64(seed+i) & 2047) - 1024
                     for i in range(channels)], dtype='<i4')

def as_i32(values):
    # Defined modular INT32 semantics, without signed-overflow dependence.
    u = np.remainder(np.asarray(values, dtype=np.int64), 1 << 32).astype('<u4')
    return u.view('<i4')

def raw(a, w):
    return as_i32(a.astype(np.int64) @ w.astype(np.int64))

def q8(acc, scale):
    x = np.asarray(acc, dtype=np.float32) * np.float32(scale)
    return np.clip(np.rint(x), -128, 127).astype(np.int8)

def conv_reference(ifm, w, bias):
    batch, hi, wi, ci = ifm.shape
    ho, wo, co = hi - 2, wi - 2, w.shape[1]
    weights = w.reshape(3, 3, ci, co).astype(np.int64)
    acc = np.broadcast_to(bias.astype(np.int64), (batch, ho, wo, co)).copy()
    for r in range(3):
        for s in range(3):
            acc += ifm[:, r:r+ho, s:s+wo, :].astype(np.int64) @ weights[r, s]
    return q8(as_i32(acc), 1.0).reshape(batch*ho*wo, co)

def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()
```

Worked G02 fixture:


```python
x = make_inputs([('W', (256,64)), ('A0', (12544,256))])
y = q8(raw(x['A0'], x['W']), 1.0)
assert y.shape == (12544,64) and y.dtype == np.int8
np.savez('G02.inputs.npz', **x)
np.save('G02.Y0.npy', y)
print({name: array_sha256(a) for name,a in {**x, 'Y0': y}.items()})
```

Remaining rosters:


| Cases | INT8 roster, in order | Separate bias / outputs |
| --- | --- | --- |
| G01–G06 | W[K,N], A0[M,K] | Y0[M,N] INT8 |
| C01–C03 | IFM, W with shapes in §3 | B=make_bias(CO,seed); Y0 INT8 |
| T03/T05/T07 | W[K,N], A0[M,K] | Y0=raw(A0,W) INT32 |
| T04 | Wg[2048,5632], Wu[2048,5632], A0[16,2048] | Yg/Yu = independent q8(raw(...),1/128) |
| T06 | Wq[2048,2048], Wk[2048,256], Wv[2048,256], A0[16,2048] | Yq/Yk/Yv = independent q8(raw(...),1/128) |
| T01 | K0..K3, each [64,16], then A0..A31, each [16,64] | Yh=raw(Ah,K[h//8]), INT32 |
| T02 | K0..K3, each [16,64], then A0..A31, each [16,16] | Yh=q8(raw(Ah,K[h//8]),1/128), INT8 |

Historical Exo caveat: old measurements filled Exo’s packed/interleaved physical tensors directly; their logical input checksums were not generally comparable to the compiler-side tensors. For this new cross-system study, derive every packed/interleaved input from the canonical arrays above. This intentionally fixes that comparability problem. Do not demand the old Exo checksum or claim identical historical bytes when using the corrected packing protocol.

## 7. Hardware and instruction-policy tracks
The historical selected execution domain was scalar in-order Rocket with a 16×16 Gemmini array, INT8 operands, INT32 accumulation. The software contract used 256 KiB scratchpad (4×4096 rows×16 bytes) and 64 KiB accumulator (1024 rows×64 bytes).

The retained deployed F2 image identifies:

RuntimeHWConfig: firesim_rocket_singlecore_gemmini_no_nic_l2_llc4mb_ddr3
AGFI: agfi-09268e63eac5f1c08
Memory build: WithAutoILA_FRFCFS16GBQuadRankLLC4MB_BaseF2Config
This is a DDR3/4 MiB LLC deployment, not bare BaseF2Config/LatencyPipe. Archived build summaries name a newer source lineage than the old local pinned gSIM checkout. Do not infer equality of hardware from a similar config name or a local hardware-pins YAML. Record actual bitstream, memory model, full source revisions where available, parameters, software header revision and toolchain.

If your machine uses another Gemmini configuration, still run the numerical suite, but give its cycles a separate hardware column. A GPU/CPU execution of IREE/TVM is a different target comparison; do not label it Gemmini execution.

Report two implementation tracks separately:

Compiler-generated primitive schedule: primitive configuration/load/preload/compute/store/completion operations; no LOOP_WS* or LOOP_CONV_WS* accelerator hardware-loop commands. This is the main comparison with our compiler-generated schedules.
Library/FSM offload: compiler dispatches Gemmini’s SW library hardware loops. This is a valid separately labelled system result and the FSM baseline, not evidence that the compiler synthesized the primitive schedule.
Ordinary CPU counted loops are allowed in both tracks. Calling a supplied expert kernel is a reference-kernel result, not a compiler-generated implementation. Record the distinction.

## 8. Measurement contract
Primary completed-call measurement
initialize inputs, immutable weights and reference outputs
prepare immutable weights once, if the tested profile permits it
poison outputs; finish all preparation and cache conditioning
t0 = target cycle counter
invoke the complete compiled workload/group
wait for its real accelerator completion and output publication
t1 = target cycle counter
compare outputs / guards and record results outside the timed window
Include dynamic configuration, operand movement, computation, result stores, dispatch and synchronization required by this public invocation. Return outputs in the declared layout. Do not stop at command issue, stop before DMA drains, or exclude a layout conversion that the callable requires. For T01/T02, time all 32 heads together. For T04/T06, time all two/three outputs together.

Keep one-time immutable preparation separate from per-call work. Report preparation cycles, packed bytes, scratch/retained bytes and whether it is reused. Do not move mutable activations or state preparation into an untimed one-time step. An additional prepared-input/kernel-only measurement is useful, but must be labelled and must not replace the complete-call result.

Cache state, repetition and instrumentation
The historical harness used a 2 MiB Gemmini load sweep before each timed call. It was called “cold L2” against a 512 KiB source-model L2. That does not prove a fully cold deployed 4 MiB LLC/DDR hierarchy.
Report historical_2MiB_sweep, verified_cold, and warm_repeat as different profiles. Document the eviction mechanism and capacity actually covered; do not silently claim one is another.
Prefer linking compared kernels into one image with fixed per-case A/W/Y addresses. Pin addresses even when linked separately: 4 KiB placement changes have affected previous results.
Retain the first invocation separately. Use balanced/rotated implementation order and at least five timed repetitions when practical. Preserve all samples; report median and range. Do not claim a scheduling win by comparing one cold reference invocation with a warm candidate.
Keep cycle-dependent formatting and UART outside all timed calls, buffering cycle values until the end. This reduces instrumentation disturbance and avoids counter-derived diagnostic control contaminating later model windows.
Perform a small correct baseline first; then compare a bounded number of schedule alternatives concurrently. Reuse unchanged binaries/results. Large kernels do not need repeated gSIM runs.
Engines and metrics
FireSim: measured target cycles; primary for comparisons to this table.
gSIM: cycle-accurate for its selected RTL/memory domain. Report that domain separately from F2 DDR3; do not multiply its cycles by one universal “FPGA correction factor.”
Spike functional: correctness and instruction/command diagnostics. Its functional mcycle/instruction counter is not accelerator elapsed time.
Spike timing model: estimates, labelled with exact model/runtime/scope identities and complete/UNKNOWN status. UNKNOWN stays unranked. Do not fit a model on these private comparison results and call the same results independent validation.
Primary optimization objective: completed-call cycles, with traffic, storage, preparation, overlap and utilization explaining the result. Instruction/fence counts are diagnostics.
Useful compute peak attainment for this 16×16 contract:

useful_MACs / (256 * completed_call_cycles)
GEMM useful MACs=M*K*N; conv=4*H*W*CO*3*3*CI; each attention workload=32*16*64*16; grouped projections sum their products. This is not a physical PE-busy counter. M=1 cannot fill all array rows under the conventional row mapping. Report useful and padded work separately.

A memory roofline needs bandwidth and cache/preparation scope from the same hardware domain. The historical 8 bytes/cycle bound came from gSIM’s memory interface; it is not a measured F2 DDR3 roofline. Unavailable resource-utilization counters should be null, not invented percentages.

## 9. Work requested from each system’s agent
Use normal compiler entry points and disclose any target integration work. Do not assume a stock checkout already has an executable Gemmini backend.


| Agent | Required work |
| --- | --- |
| IREE | Represent the exact integer contractions, grouped outputs and explicit epilogues; compile through the chosen IREE pipeline and actual Gemmini target adapter. Record compiler IR before/after fusion, scheduling and target lowering. If only CPU fallback executes, label that result and the missing Gemmini support. |
| TVM | Build equivalent tensor computations, including exact INT8 exits and shared K/V grouping. Use the selected target integration/tensorization/codegen path and record schedule/autotuning settings and search budget. Distinguish generated primitive code from external-library/FSM calls. |
| Voyager | Use the selected Voyager source revision and its actual Gemmini integration. Verify its default tensor layout, output types, biases and scales against this handoff rather than substituting a model-level default. Record any frontend, layout or backend adaptations. |
| Agentic compiler | Build reusable lowering and schedule choices for all families, using shape/layout/capacity features. Parallelize independent capability/schedule experiments and test immutable candidates. Keep changes to shared transformations and shape rules understandable; report any reference code supplied to the author. |

For all four: target all 16 cases, start with one dense, one conv and one grouped-attention execution, then run independent cases concurrently. Use focused correctness checks for changed paths and short cycle-ranked sweeps; do not repeatedly rebuild/verify the entire stack for a knob change. If unsupported, return the exact lowering refusal and last successful IR stage instead of timing a different workload.

This handoff supplies semantics and baseline context, not a prescribed winning tile size, pass order or expert instruction stream. Any score-aware agent treatment must be labelled when comparing it with an independently blinded construction run.

## 10. Deliverables
Return:

results.csv with one row per case/system/hardware/measurement profile, plus raw samples.
Input and reference-array raw-byte SHA256s; every output’s dtype/shape/element count, mismatch count, first mismatch and guard status. Do not use cosine/PPL to waive an integer mismatch.
Compiler/fork, source and toolchain commits; build/run commands; actual backend and library dependencies; cache/preparation/instruction-policy track.
Reusable source/IR, pass/schedule configuration, emitted LLVM/assembly, ELF and hardware/simulator logs sufficient to rebuild and execute. Record kernel and complete-image hashes.
Optimization log: hypothesis, shared transformation/knob changed, affected shape class, correctness, cycles before/after, traffic/storage/overlap evidence, and keep/reject decision. Preserve unsuccessful trials compactly; delete redundant build products rather than copies of the full compiler.
Suggested CSV fields:

case_id,system,compiler_commit,backend_commit,hardware_id,engine,track,
input_profile,cache_profile,output_dtype,output_elements,input_sha256s,
preparation_scope,preparation_cycles,packed_bytes,first_call_cycles,
cycle_samples,median_cycles,min_cycles,max_cycles,mismatches,guard_failures,
useful_macs,padded_macs,useful_peak_attainment,requested_read_bytes,
requested_write_bytes,scratch_bytes,model_identity,timing_complete,
kernel_sha256,elf_sha256,artifact_path,notes
## 11. Historical comparison table supplied for this study
These are historical context, not correctness goldens or acceptance thresholds. Remeasure a local matching FSM/reference on the same hardware and fixtures. Do not manufacture missing values.

≈ … E: estimated/near-match expert result, not a direct measurement of this exact kernel invocation.
‡: the historical “Exo” column refers to the coworker’s expert Exo-generated TinyLlama kernels, not necessarily upstream optimized Exo. The old packed/interleaved inputs were not generally logical-byte matched to the other implementations; new comparisons must use §6’s canonical mapping.
—: unavailable.
FAIL: incorrect execution; cycles are not rankable. The fresh4 convolution cohort also has documented intermittent target failures; do not treat an unmarked historical number as a new correctness certificate.
Columns come from different retained compiler snapshots and jobs. Small differences may reflect cache/order/placement and run-to-run variation. Do not mix them with newer offline-oracle measurements as if all came from one controlled experiment.

| Shape (GEMM: M×K×N) | FSM | Exo | Merlin | v8 live | v8 final | fresh1 | fresh3 | fresh4 | fresh5 | fresh6 | fresh7 ref | fresh7 cca |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemm 512×512×512 | 558,283 | — | — | 789,435 | 785,177 | 702,833 | 821,259 | 1,120,775 | 862,077 | 818,893 | 787,727 | 793,756 |
| gemm 12544×256×64 | 981,848 | ≈845,309 E | — | 2,009,601 | 1,955,856 | 1,863,534 | 2,108,477 | 2,255,510 | 2,274,181 | 2,416,461 | 2,004,594 | 2,090,400 |
| gemm 12544×64×256 | 1,651,540 | ≈854,521 E | — | 2,753,922 | 2,717,887 | 2,327,521 | 2,652,375 | 2,974,456 | 2,705,008 | 2,697,559 | 2,600,833 | 2,607,368 |
| gemm 3136×512×128 | 879,991 | ≈912,274 E | — | 1,406,475 | 1,351,605 | 1,297,714 | 1,511,837 | 1,739,522 | 1,565,682 | 1,581,616 | 1,396,185 | 1,399,415 |
| gemm 3136×128×512 | 991,086 | ≈882,771 E | — | 1,776,951 | 1,889,567 | 1,411,909 | 1,677,531 | 1,965,382 | 1,717,885 | 1,643,987 | 1,558,704 | 1,558,505 |
| gemm 784×256×1024 | 821,320 | ≈915,238 E | — | 1,699,918 | 1,704,263 | 1,120,434 | 1,347,732 | 1,850,021 | 1,427,823 | 1,317,272 | 1,284,662 | 1,283,908 |
| conv 3×3 56×56 64→64 b4 | 1,944,930 | ≈1,987,935 E | — | 3,264,831 | 2,644,651 | 4,546,491 | 19,165,197 | 47,062,025 FAIL | 11,802,617 | 13,399,139 | 7,070,499 | 7,084,775 |
| conv 3×3 28×28 128→128 b4 | 1,971,151 | ≈2,114,988 E | — | 3,856,600 | 2,127,456 | 3,641,445 | 18,636,669 | 46,511,936 | 10,937,621 | 11,580,402 | 4,903,583 | 4,869,921 |
| conv 3×3 14×14 256→256 b4 | 1,996,584 | ≈1,962,829 E | — | 2,957,556 | 2,168,899 | 4,560,628 | 23,691,821 | 43,748,608 | 11,075,415 | 18,226,725 | 5,178,693 | 15,395,314 |
| att_pv (32 heads) | 37,327 | 5,828 ‡ | 40,777 | 13,812 | 14,950 | 15,037 | 32,564 | 19,094 | 23,045 | 21,903 | 41,382 | 59,437 |
| att_qk (32 heads) | 45,477 | 12,662 ‡ | 49,353 | 18,508 | 25,531 | 24,687 | 34,976 | 27,979 | 34,486 | 27,513 | 20,643 | 55,383 |
| mm_down 16×5632×2048 | 2,544,149 | 4,005,693 ‡ | 3,230,948 | 2,575,681 | 2,495,056 | 14,176,538 | 3,294,367 | 9,623,119 | 3,933,028 | 4,041,161 | 3,809,203 | 3,424,901 |
| mm_gu 16×2048×5632 ×2 | 5,537,130 | 8,150,185 ‡ | 6,583,789 | 5,295,882 | 5,205,464 | 6,683,276 | 6,969,085 | 13,787,313 | 8,207,526 | 8,186,286 | 7,682,964 | 7,061,625 |
| mm_o 16×2048×2048 | 900,917 | 1,535,796 ‡ | 1,219,408 | 857,714 | 867,428 | 1,238,595 | 1,165,852 | 2,793,669 | 1,391,670 | 1,486,531 | 1,311,401 | 1,181,068 |
| mm_qkv 16×2048×(2048+256+256) | 1,211,435 | 1,865,240 ‡ | 1,524,354 | 1,087,474 | 1,205,039 | 1,643,668 | 1,562,300 | 3,484,869 | 1,819,539 | 1,735,078 | 1,586,237 | 1,451,375 |
| mm_lm 1×2048×32000 | 15,906,948 | 22,307,349 ‡ | 19,272,038 | 15,024,379 | 15,100,519 | 17,696,017 | 19,239,249 | 33,996,857 | 23,462,167 | 22,853,847 | 21,680,872 | 16,006,792 |

Historical FSM reference: Gemmini software library gemmini-rocc-tests@6b477a8b43ed, tiled_matmul_auto / tiled_conv_auto, weight-stationary hardware loops, row-major weights. Raw INT32 matmuls select full-C output. INT8 matmuls/convs use the specified identity or 1/128 scale and signed saturation. Attention FSM values include 32 separate head calls, while an optimized compiler may schedule all heads together.

## 12. Source/provenance notes
The definitions were checked against actual retained interface IR, tensor rosters, and the generated historical harness, rather than inferred from the table labels. Original operator source locations on the originating machine:

/work/merlin/tasks/el4-grader-throughput-20261024/
  .tmp/v8-fresh3-firesim/scripts/build_batch.py
  .tmp/v8-fresh3-firesim/build/v8_all/harness.c
  .tmp/v8-fresh3-firesim/build/v8_all/manifest.json
  .tmp/v8-fresh3-firesim/ref/inputs/
  .tmp/v8-fresh3-firesim/ref/inputs3/
  merlin/out/artifacts/perf-studies/all-compilers-firesim/results.json
  merlin/out/artifacts/perf-studies/fresh1-firesim/README.md
These paths are provenance, not dependencies on your machine. The shape/numerical definitions and input generator above are sufficient to construct the workloads without those files. If exact old ELFs, reference sources or hardware manifests are needed, obtain the authorized artifact bundle separately and pin its hashes. Do not assume a local /work/merlin/... path exists elsewhere.

Voyager repository identified for this comparison: https://github.com/jeffreyyu0602/voyager-compiler. Record the actual revision used; this handoff does not assume a particular current build interface. Apply the same revision discipline to IREE, TVM and every Gemmini integration fork.

Completion criterion: 16 explicitly identified workloads attempted by each system, all supported results numerically checked, all unsupported paths explained, and matched-scope measured cycles with enough artifacts to reproduce the comparison. Performance improvements must come from the generated implementation and declared preparation policy.

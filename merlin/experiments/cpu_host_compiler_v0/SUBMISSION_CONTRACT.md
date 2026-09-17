# CPU-host compiler submission contract (v1)

Every arm produces the same self-contained package under `submission/`. The grader builds the package
once, makes it read-only, and invokes the compiler once per capsule in an isolated process with no network
and no corpus visibility. A compiler invocation sees exactly one MLIR file and a fresh output directory.

## Package manifest

`submission/manifest.yaml`:

```yaml
version: 1
build:
  command: [cmake, -S, ., -B, build, -G, Ninja, -DCMAKE_BUILD_TYPE=Release]
  then: [cmake, --build, build, -j2]
compiler:
  command:
    - build/bin/rvvhost-compile
    - --input
    - "{input_mlir}"
    - --output-dir
    - "{output_dir}"
    - --mode
    - "{mode}"
    - --harts
    - "{harts}"
    - --vlen-bits
    - "{vlen_bits}"
policy: policy.yaml
```

Commands are argument arrays, not shell strings. Paths in the arrays must stay within the package. The
only substitutions are `{input_mlir}`, `{output_dir}`, `{mode}`, `{harts}`, and `{vlen_bits}`. Allowed
modes are `scalar`, `rvv`, and `rvv_multicore`. The compiler must fail rather than silently substitute a
different mode.

## Per-capsule outputs

The compiler must create all three files below in `{output_dir}`:

- `kernel.c`: freestanding-compatible C/C++ implementing the ABI below. RVV modes may use RVV intrinsics
  or inline assembly. It must not define `main`, constructors, destructors, or process/syscall wrappers.
- `lowered.mlir`: the post-policy IR. This is evidence that a compiler transformation occurred, not an
  executable authority; the grader independently compiles and runs `kernel.c`.
- `metadata.json`:

```json
{
  "version": 1,
  "capsule_sha256": "digest copied from the MLIR module attribute",
  "requested_mode": "rvv",
  "actual_mode": "rvv",
  "fallback_used": false,
  "harts": 1,
  "vlen_bits": 256,
  "vlen_policy": "scalable_vl",
  "tail_policy": "dynamic_vl",
  "transformations": ["canonical generic transformation names"],
  "source_sha256": "sha256(kernel.c)"
}
```

For `scalar`, `vlen_policy` and `tail_policy` may be `not_applicable`. For `rvv_multicore`, `harts` must
equal the request and the kernel must partition work deterministically without overlaps or omissions.
Fixed-VLEN code must report `vlen_policy: runtime_verified_fixed` and is accepted only after the same-run
K1 probe reports the requested width. `fallback_used: true`, a requested/actual mismatch, or missing
metadata is a scored failure.

## Kernel ABI

```c
#include <stdint.h>

typedef struct {
  uint32_t version;       /* 1 */
  uint32_t family;        /* enum in capsule MLIR */
  uint32_t operation;     /* enum in capsule MLIR */
  uint32_t dtype;         /* enum in capsule MLIR */
  uint32_t layout;        /* enum in capsule MLIR */
  uint32_t harts;
  uint32_t vlen_bits;
  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  uint64_t state0;
} merlin_capsule_params_t;

int merlin_capsule_run(const merlin_capsule_params_t *params,
                       const void *input0, const void *input1, const void *input2,
                       void *output);
```

The compiler sees dimensions and operation identity only through the supplied MLIR. `input0..2` and
`output` are contiguous, non-aliasing buffers. Returning zero means success; any other value is failure.
The harness selects input seeds only after code generation, computes its reference into a private buffer,
and tests guard zones around every allocation.

## Capsule semantics

The v1 MLIR module is a typed capsule descriptor. It contains a `merlin.capsule` dictionary with the
content digest, numeric ABI enums, exact dimensions, buffer extents, and target request, plus a private
typed `func.func @capsule` declaration. `contracts/capsule_descriptor.py` is the canonical public renderer,
and `contracts/descriptor_fixtures/` contains one complete synthetic example for every family. Both are
part of the staged immutable input lock. Run the fixtures through your parser before packaging.

The exact dictionary keys and integer types are:

```text
sha256 : string
family : string                    family_code : i32
operation : string                 operation_code : i32
semantic_operation_code : i32
dtype : string                     dtype_code : i32
layout : string                    layout_code : i32
dim0, dim1, dim2, state0 : i64
input0_count, input1_count, input2_count, output_count : i64
requested_harts : i32
```

The private declaration always has four arguments in this order: `%input0`, `%input1`, `%input2`, and
`%output`. Each is `memref<count x element-type>`. An absent input is represented by `memref<0xi8>`, not
by omitting the argument. The element type is `f32`, `i8`, or `i32` according to the buffer plan.

All integer codes are stable public ABI, independent of the corpus split or the capsules present in a
particular run:

```text
family: contraction=1, elementwise_map=2, reduction=3, movement_layout=4,
        fusion_epilogue=5, runtime_parallel=6
dtype:  fp32=1, int8=2, int32=3, w8a8_i32=4, int8_i32=5
layout: contiguous=1, row_row=2, row_packed_rhs=3, transposed_rhs=4,
        operation_defined=5
operation: add=1, barrier=2, batch_matmul=3, clamp=4, concatenate=5,
  convolution_im2col=6, copy=7, gelu=8, layernorm_components=9, matmul=10,
  matmul_bias=11, matmul_bias_relu=12, matmul_requant=13, max=14, multiply=15,
  pack_rhs=16, persistent_weight_reuse=17, producer_consumer=18, relu=19,
  requant=20, residual_norm=21, silu=22, single_hart=23, softmax_components=24,
  static_partition=25, strided_slice=26, sum=27, transpose2d=28, unpack=29
semantic operation: contraction aliases=1; add=2; multiply=3; relu=4; silu=5;
  gelu=6; clamp=7; requant=8; sum=9; max=10; softmax_components=11;
  layernorm_components=12; copy=13; transpose2d=14; pack_rhs=15; unpack=16;
  strided_slice=17; concatenate=18; matmul_bias=19; matmul_bias_relu=20;
  matmul_requant=21; residual_norm=22; all runtime_parallel operations=23
```

The trusted grader is the executable semantic authority. The descriptor deliberately does not pretend to
be an independently executable standard-dialect program; `kernel.c` is always compiled and checked
against the trusted reference harness.

- `contraction`: `dim0=M`, `dim1=N`, `dim2=K`. `input0` is M×K; `input1` is K×N. `fp32`
  accumulates f32; `w8a8_i32` multiplies signed i8 and accumulates exact i32. `row_row` stores B[k,N+j],
  `transposed_rhs` stores B[j,K+k], and `row_packed_rhs` uses 8-column panels padded to 8:
  B[(j/8)*K*8 + k*8 + j%8]. All three contraction operation names have these numerical semantics.
- `elementwise_map`: `dim0=length`. `add`, `multiply`, `relu`, `silu`, `gelu`, `clamp`, and `requant`
  use the trusted expressions. Floating SiLU is `x/(1+exp(-x))`; floating GELU is the tanh approximation;
  floating clamp is `[-1,1]`; and floating requant multiplies by `0.25`. For integers, ReLU, SiLU, and GELU
  use `max(x,0)`, clamp is `[-8,8]`, and requant is signed C integer division by four. These integer
  definitions are fixed approximations for compiler scoring, not claims about a particular model's
  quantization recipe.
- `reduction`: `dim0=length`. `sum` and `max` return one value. `softmax_components` returns `[max,
  sum(exp(x-max))]`; `layernorm_components` returns `[sum(x), sum(x*x)]`. `int8_i32` outputs i32. Its
  softmax component uses the exact fixed approximation `delta <= -8 ? 1 : 1 << (8 + delta)` after
  subtracting the maximum.
- `movement_layout`: `dim0=floor(working_set_bytes / dtype_width_bytes)` (unused partial bytes do not
  name an element). `copy` is
  identity; `transpose2d` uses the rows/columns encoded in the MLIR; `pack_rhs`/`unpack` use 8-column
  panels; `strided_slice` selects even elements; `concatenate` places input0 then input1. For concatenate,
  `input0_count=state0`, `input1_count=dim0-state0`, and `output_count=dim0`. Exact input and output extents
  are present both in the dictionary and function types.
- `fusion_epilogue`: first compute the contraction. `input2` is an N-element bias for `matmul_bias` and
  `matmul_bias_relu`; ReLU follows bias. `matmul_requant` applies the integer or floating requant expression
  in the MLIR. `residual_norm` adds the M×N residual in `input2` and applies the row-wise normalization
  definition: floating point uses `(x-mean)/sqrt(variance+1e-5)` and integer scoring uses exact row mean
  centering (`x-mean`) without an integer variance approximation.
- `runtime_parallel`: `dim0=work_items`, `state0=reuse_count`. Every operation has the exact numerical
  semantics `output[i] = input0[i] + input1[i]` for `0 <= i < dim0`. Operation identity controls only the
  required scheduling and reuse evidence. `single_hart` must report one hart. Other modes must report and
  use exactly the requested count.

No capsule ID, digest, exact shape tuple, or operation instance may be a dispatch key in the policy or
source. Family, operation, dtype, layout, divisibility/tail facts, cache-fit facts, and target facts are
legal generic dispatch inputs. The held-out split is never visible during building or candidate selection.

## Scoring authorities

- **L0:** build before held-out access; read-only package; fresh networkless process per capsule; manifest,
  metadata, MLIR verifier, source policy, output-size, and source-diversity checks.
- **L1:** the scalar artifact is linked only to the trusted harness and run under ASan/UBSan for three
  random seeds generated after code generation. Goldens, input snapshots, and allocation guards are
  private to the grader.
- **L2:** the RVV artifact is cross-compiled with auto-vectorization disabled and run on Spike with
  VLEN=256 over sealed tail cases. `kernel.o` must contain substantive vector load/store/compute evidence;
  `vsetvli` alone is insufficient.
- **L3:** the same artifact is cross-built for K1 Linux, transferred with SHA verification, and run without
  network access. The monitor validates exact affinity, active CPU/task count, requested mode/harts,
  `vlenb=32`, no fallback, numeric digest, wall time, `rdtime`, and peak RSS. For a request of H>1 harts,
  the audited invocation must create exactly H-1 joinable pthread workers, singleton-pin the controller to
  hart 0 and each worker to its distinct hart 1..H-1, and join all workers before returning. The trusted
  harness independently serializes those H-1 submitted callbacks on a secret measured invocation to prove
  disjoint balanced output ownership, and performs an untimed call in which all H-1 worker callbacks are
  suppressed to prove the result depends on them. Persistent pools are outside this per-capsule ABI and
  therefore fail this authority; they are evaluated only by the separate continuous-inference session
  runtime. H=1 requires no pthread or affinity calls.

All levels must pass. An unavailable or unrun authority is `not_run`, never a pass.

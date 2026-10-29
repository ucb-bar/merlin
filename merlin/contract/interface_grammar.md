# `merlin_iface` interface grammar — v0.1 (frozen)

This is the **frozen, versioned input format** the experiment ABI hands to an out-of-tree
target-backend package. A package's job is to consume an `*.interface.mlir` file written in this
grammar and produce (a) a `command_buffer.json` and (b) lowered LLVM/RoCC, which the Merlin
runner then certifies (see `oracle_runner_contract.yaml`).

The grammar is a small, regular **MLIR module** using a custom `merlin_iface` dialect. It is
deliberately regular enough that:

- a registered **C++ MLIR dialect** parses it natively (`mlir-opt` / `gemmini-opt`), and
- a **few-line regex parser** (any language) reads it — the reference Python implementation is
  `merlin/targetgen/contract/interface_emit.py`.

It is **decoupled from xDSL**: producers emit plain text; consumers parse plain text. No xDSL is
required to satisfy the contract.

> Version is carried in the module attribute `merlin_iface.version`. A consumer **must** reject a
> version it does not implement. v0.1 is the only version today.

## Why logical names matter

Leaf tensors are materialized **deterministically by name** (`W`, `A0`, …): the same name +
shape + dtype always yields the same data on both sides. Therefore every leaf tensor and every
committed output carries a string `name`, and the runner maps outputs by that name. Producers
must preserve these names verbatim.

## Module

```mlir
module attributes {
  merlin_iface.version = "0.1",
  merlin_iface.target = "gemmini",
  merlin_iface.abi_version = "0.1"
} {
  ...ops...
}
```

`backend`/oracle choice is **not** part of this grammar — the runner selects the simulator. The
contract surface describes the *computation*, not where it runs.

## Types

| Type | Meaning |
|------|---------|
| `tensor<D0x...xDNxDT>` | a dense, nonempty ranked tensor; `DT` is a command-buffer-registered dtype (builtin MLIR tensor type) |
| `!merlin_iface.resident` | an opaque handle to a resident (packed, stationary) weight |
| `!merlin_iface.acc<i32>` | an opaque integer accumulator handle |

## Op index

The complete list of ops the grammar accepts, **generated from the reference parser's own tables**
(`interface_emit._OP_TO_OPCODE` + `_NAMED_OP_OPERAND_KEYS` + its structural ops), which the canonical
parser fails CLOSED on: an op not in this table is not in the grammar, and a module using one is
rejected naming the mnemonic. `build_tools/scripts/check_grammar_documentation.py` holds this table,
the prose sections below, `interface_dialect_contract.yaml` and the registered dialect
(`merlin_iface.irdl.mlir`) together by content, and regenerates the table (`--print-index`).

| op | command-buffer opcode | positional operands (in order) | registered dialect (IRDL) |
| --- | --- | --- | --- |
| `attention_pv` | `ATTENTION_PV` | `p`, `v` | no |
| `attention_qk` | `ATTENTION_QK` | `q`, `k` | no |
| `bias_add` | `BIAS_ADD` | `src`, `bias` | no |
| `commit` | `COMMIT` | — | yes |
| `conv2d` | `CONV2D` | `ifm`, `weight` | yes |
| `evict` | `EVICT` | — | yes |
| `matmul` | `MATMUL_RESIDENT` | — | yes |
| `matmul_batched` | `BATCHED_MATMUL` | `a`, `w` | no |
| `movement` | `MOVEMENT` | `src` | yes |
| `resident_pack` | `RES_PACK` | — | yes |
| `residual_add` | `RESIDUAL_ADD` | `lhs`, `rhs` | no |
| `rmsnorm` | `RMSNORM` | `src`, `gamma` | no |
| `rope` | `ROPE` | `src` | no |
| `softmax` | `SOFTMAX` | `src` | no |
| `tensor` | — (declares a leaf, issues no command) | — | yes |

Positional operands are given in the order the op takes them; the parser refuses a count that does not
match (truncating either side would change the ABI). Every whole-op result is recorded as
`role = "output"` under its `name` attribute, which becomes the command's `dst` operand. The residency
ops (`resident_pack`, `matmul`, `commit`, `evict`) name their operands in their own sections.

The last column says whether the dynamically registered dialect a C++ tool loads
(`mlir-opt --irdl-file=merlin_iface.irdl.mlir`, generated from the reference ODS) declares the op. An op
marked `no` is accepted by the reference parser and graded in capsules, but that registered dialect
does not yet verify it, so a C++ consumer has to accept it through its own parser.

The **semantics** of each opcode -- its attributes, defaults and arithmetic -- live in
`command_buffer_abi.yaml` under that opcode, not here: this document specifies the input SURFACE, the
ABI what the command means. Three opcodes (`RMSNORM`, `ROPE`, `SOFTMAX`) are accepted by the parser and
have no ABI entry yet; they are recorded as debt in `build_tools/scripts/undocumented_opcodes_ratchet.txt`.

## Ops

### `merlin_iface.tensor` — declare a leaf input/weight
```mlir
%W  = merlin_iface.tensor {name = "W",  role = "weight"} : tensor<16x16xi8>
%A0 = merlin_iface.tensor {name = "A0", role = "input"}  : tensor<16x16xi8>
```
`role ∈ {weight, input, bias, scale}`. Result type gives shape + dtype.

### `merlin_iface.resident_pack` — make a weight resident
```mlir
%W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} :
         (tensor<16x16xi8>) -> !merlin_iface.resident
```
Maps to command-buffer opcode `RES_PACK` (`operands: {src, dst}`, `attributes: {layout}`).

### `merlin_iface.matmul` — matmul against a resident weight

An optional attribute dictionary preserves source metadata on this operation.
For original sibling source transport, `prov.region_id` is the original root's
string identity and `prov.source_node_ids` is its ordered string list. They
carry no execution or coverage permission: the target must bind real compute
intervals to the corresponding source member and the original owner must verify
the identities against its source roster. Unannotated matmuls remain unchanged.
```mlir
%acc0 = merlin_iface.matmul %A0, %W_res :
        (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
```
Maps to `MATMUL_RESIDENT` (`operands: {lhs, rhs, dst}`). `dst` rows × resident cols = output shape.

### `merlin_iface.commit` — apply epilogue, commit accumulator to an output tensor
```mlir
%Y0 = merlin_iface.commit %acc0 {
        name = "Y0", epilogue = ["acc_scale", "relu"],
        output_dtype = "i8", acc_scale = 0.0625 : f32
      } : (!merlin_iface.acc<i32>) -> tensor<16x16xi8>
```
Maps to `COMMIT` (`operands: {src, dst}`, `attributes: {epilogue, output_dtype, acc_scale?}`).
- `epilogue` — ordered subset of `["bias_add", "requant", "acc_scale", "relu", "maxpool"]`.
- `bias = "B"` — required iff `"bias_add"` is in `epilogue`, and it names the `role = "bias"` tensor
  the stage adds. The stage was listed here and the role was listed above, but nothing said WHICH
  operand the stage consumes, so a module could declare that a bias is added without saying what to
  add. The bias is a length-`N` vector added to every row, **in the accumulator's dtype**, before any
  requant or activation — it lands on the accumulator, which is why the vector is `i32` on an
  `i8 x i8 -> i32` datapath and not `i8`.
- `output_dtype ∈ {i32, i8}` — `i32` = full-width readout, `i8` = scaled/clamped readout.
- `acc_scale : f32` — scalar multiplier for an `"acc_scale"` epilogue stage. Select exactly one of
  `acc_scale` and `acc_scale_by_column` for that stage. The `: f32` suffix is honest:
  the requant applies an **f32 multiply, round-to-nearest-even, clamp to i8**.
- `acc_scale_by_column = [<gain> : f32, ...]` — the explicit alternative, with one finite float32
  multiplier per final-axis output column. Every row uses that same vector. Integer conversion,
  multiplication, ties-to-even rounding and narrowing retain the scalar stage's order. This declares
  arithmetic; target admission and the schedule of scale changes must be proved separately.
- `"maxpool"` — the one epilogue stage that CHANGES the result extent, because the store path fuses
  pooling into the accumulator readout. It reshapes the `M` rows to `[batch, H, W]` using
  `pool_in_dims = [H, W]`, walks `pool_size` at `pool_stride` over `pool_padding`, and commits
  `batch*Ho*Wo` rows (`Ho = (H + pt + pb - ph) / sh + 1`, floor; `Wo` likewise). `pool_in_dims`,
  `pool_size` and `pool_stride` are **required** with no defaults — an `[M, N]` accumulator carries no
  spatial extent, so `25` rows is `5x5` or `25x1` and only the declaration says which. Integer-list
  attributes: `pool_size = [2, 2]`, never `["2", "2"]`.
- `pool_pad_value : i64` — required iff any `pool_padding` entry is nonzero. The identity element of a
  max over a padded cell is a datapath property (`-inf` mathematically, commonly `0` in a store path),
  so it is declared rather than assumed.

```mlir
%Y0 = merlin_iface.commit %acc0 {
        name = "Y0", epilogue = ["maxpool"], output_dtype = "i32",
        pool_in_dims = [4, 4], pool_size = [2, 2], pool_stride = [2, 2], pool_padding = [0, 0, 0, 0]
      } : (!merlin_iface.acc<i32>) -> tensor<4x16xi32>
```

### `merlin_iface.bias_add` — add a bias vector to a committed tensor
```mlir
%X  = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi32>
%B  = merlin_iface.tensor {name = "B", role = "bias"}  : tensor<16xi32>
%Y0 = merlin_iface.bias_add %X, %B {name = "Y0", output_dtype = "i32"} :
      (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16x16xi32>
```
The `"bias_add"` commit stage standing on its own, over an already-committed tensor rather than over an
accumulator. Same arithmetic, same length-`N`-vector-per-row broadcast, same dtype rule — its operands
are in the **accumulator's** dtype because that is the domain the stage runs in, so a fused capsule and
this one are performing the same addition on the same numbers.

It exists so a fused epilogue has something to be measured against: the fusion claim is that
`matmul+bias` fused costs less than the `matmul` and the `bias_add` it replaces, and that comparison
needs the unfused halves to be expressible. A backend may map it to whatever its own vector or
accumulator-readout path provides; a target with no separate vector class is expected to fold it into
the readout, and its capsule then requires no vector instruction (see `expected_instruction_coverage`).

### `merlin_iface.matmul_batched` — independent rank-N contractions

```mlir
%Y0 = merlin_iface.matmul_batched %A0, %W {
        name = "Y0", batch = 2 : i64, output_dtype = "i32"
      } : (tensor<2x16x32xi8>, tensor<2x32x16xi8>) -> tensor<2x16x16xi32>
```

Maps to `BATCHED_MATMUL` with the exact operand map `{a: A0, w: W, dst: Y0}`. For a nonempty
batch prefix `B...`, its shapes are `A[B..., M, K]`, `W[B..., K, N]`, and `Y[B..., M, N]`.
Both operands vary independently at every batch coordinate; broadcasting, shared-weight relabelling,
and flattening `B...` into `M` change the operation and are not permitted. If present, `batch` equals
the product of the batch-prefix extents. The result type declares the destination tensor's complete
rank, shape, and dtype; the reference parser records that destination as `role = "output"`, just as it
does for every named whole-op result. A fused program may later consume such a produced tensor; final
output selection is then determined by command dataflow rather than by the role spelling alone.

### `merlin_iface.evict` — release a resident weight
```mlir
merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
```
Maps to `EVICT` (`operands: {handle}`).

### `merlin_iface.conv2d` — im2col convolution
```mlir
%Y0 = merlin_iface.conv2d %IFM, %W_res {
        name = "Y0", kernel = [3, 3, 64, 64], stride = [1, 1], padding = [1, 1, 1, 1],
        dilation = [1, 1], layout = "nhwc", epilogue = [], output_dtype = "i32"
      } : (tensor<1x28x28x64xi8>, !merlin_iface.resident) -> tensor<784x64xi32>
```
Maps to `CONV2D` (`operands: {ifm, weight, dst}`). An NHWC activation contracted against a
**pre-im2col'd** weight `[Kh*Kw*Ci, Co]` that has been made resident, producing `[N*Ho*Wo, Co]`. The
geometry rides in the ATTRIBUTES -- `kernel = [kh, kw, ci, co]` (required), `stride`, `padding`,
`dilation`, `layout` -- never in extra operands, so the operand list is the activation and the weight.
There is no separate `commit`: `epilogue` is the same readout a commit applies, including a fused
`maxpool` with the same required pooling attributes, `bias` naming a `role = "bias"` tensor in the
accumulator's dtype, and `acc_scale` required iff its stage is listed. `output_dtype` is an integer
dtype. Only `nhwc` is defined; grouped convolution has no attribute and is rejected. Full attribute
rules: `command_buffer_abi.yaml` → `CONV2D`.

### `merlin_iface.movement` — identity round-trip through the accelerator
```mlir
%Y0 = merlin_iface.movement %A0 {name = "Y0", output_dtype = "i32", semantic = "mvin_mvout"} :
      (tensor<16x16xi8>) -> tensor<16x16xi32>
```
Maps to `MOVEMENT` (`operands: {src, dst}`). A load→store round trip: `dst` holds `src`'s values
UNCHANGED, only the container dtype widens. No clamp, no requantize -- the point of the capsule is that
the data survives the trip bit-for-bit. Its capsules carry exactly ONE op, so a parser that skips it
reads them as empty programs.

### `merlin_iface.attention_qk` — the first matmul of attention
```mlir
%S = merlin_iface.attention_qk %Q, %K {name = "S", output_dtype = "i32", epilogue = []} :
     (tensor<16x64xi8>, tensor<32x64xi8>) -> tensor<16x32xi32>
```
Maps to `ATTENTION_QK` (`operands: {q, k, dst}`). `dst = q @ transpose(k)`: `q` is `[m, d]`, `k` is
`[n, d]` stored ROW-PER-KEY, so the contraction is over the TRAILING head dim of BOTH operands and `dst`
is `[m, n]`. **The transpose is part of the op's definition and is not spelled as an attribute** --
there is no `transpose_rhs` / `rhs_transposed` / `transpose_b` flag, and an unrecognised attribute is
rejected rather than ignored. `epilogue` is an ordered subset of `[acc_scale, requant, relu]`.

### `merlin_iface.attention_pv` — the second matmul of attention
```mlir
%O = merlin_iface.attention_pv %P, %V {name = "O", output_dtype = "i32", epilogue = []} :
     (tensor<16x32xi8>, tensor<32x64xi8>) -> tensor<16x64xi32>
```
Maps to `ATTENTION_PV` (`operands: {p, v, dst}`). `dst = p @ v`, a plain `[m, s] x [s, d]` contraction
with **no transpose** -- the sibling of `attention_qk` and deliberately not the same shape rule. Until
this op was defined, every shipped flash-attention capsule parsed without its second matmul while the
parser reported success.

### `merlin_iface.residual_add` — two quantized tensors summed into one output domain
```mlir
%Y0 = merlin_iface.residual_add %X, %R {
        name = "Y0", lhs_scale = 0.5 : f32, rhs_scale = 0.25 : f32, bound_lsb = 1 : i64,
        epilogue = [], output_dtype = "i8"
      } : (tensor<16x16xi8>, tensor<16x16xi8>) -> tensor<16x16xi8>
```
Maps to `RESIDUAL_ADD` (`operands: {lhs, rhs, dst}`). Each operand carries its own multiplier into the
output's domain, the sum is rounded ONCE, and `bound_lsb` declares how many output steps a conforming
target may lie from that reference. `lhs_scale`, `rhs_scale` and `bound_lsb` are REQUIRED with no
defaults; `epilogue` is an ordered subset of `[relu]` and `output_dtype` an integer dtype. Exact rounding
and saturation: `command_buffer_abi.yaml` → `RESIDUAL_ADD`. Without this op a residual connection has
no command at all and can only ever be host work.

### `merlin_iface.rmsnorm` · `merlin_iface.rope` · `merlin_iface.softmax`
```mlir
%Y0 = merlin_iface.rmsnorm %X, %G {name = "Y0", output_dtype = "i8"} :
      (tensor<16x64xi8>, tensor<64xi8>) -> tensor<16x64xi8>
%Y1 = merlin_iface.rope    %X     {name = "Y1", output_dtype = "i8"} : (tensor<16x64xi8>) -> tensor<16x64xi8>
%Y2 = merlin_iface.softmax %X     {name = "Y2", output_dtype = "i8"} : (tensor<16x64xi8>) -> tensor<16x64xi8>
```
Map to `RMSNORM` (`operands: {src, gamma, dst}`), `ROPE` (`{src, dst}`) and `SOFTMAX` (`{src, dst}`).

> **These three have no `command_buffer_abi.yaml` entry.** The parser accepts them and the ABI states
> no attributes and no arithmetic for them, so a capsule using one asks you to infer the operation from
> its name. They are recorded as debt in `undocumented_opcodes_ratchet.txt`, which may only shrink; if
> you meet one and the ABI still says nothing, that is a gap in the contract you were handed, not a
> detail you were expected to know.

## Attribute encoding

- string: `k = "v"`; string list: `k = ["a", "b"]` (empty: `k = []`)
- integer: `k = 4 : i64`; float: `k = 0.0625 : f32`
- integer list (geometry — `kernel`, `stride`, `padding`, `dilation`, `pool_*`): `k = [2, 2]`,
  **unquoted**. A quoted geometry parses back as strings and fails an arity/type check far from the
  spelling that caused it.

## Worked example (g0 — matmul only, i32)

```mlir
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
```

The Gemmini test fixture for this example is
`merlin/tests/gemmini/fixtures/expected_command_buffer_g0.json`.

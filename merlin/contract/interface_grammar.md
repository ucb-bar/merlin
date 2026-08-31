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
| `tensor<RxCxDT>` | a dense 2-D tensor, `DT ∈ {i8, i32}` (builtin MLIR tensor type) |
| `!merlin_iface.resident` | an opaque handle to a resident (packed, stationary) weight |
| `!merlin_iface.acc<i32>` | an opaque integer accumulator handle |

## Ops

### `merlin_iface.tensor` — declare a leaf input/weight
```mlir
%W  = merlin_iface.tensor {name = "W",  role = "weight"} : tensor<16x16xi8>
%A0 = merlin_iface.tensor {name = "A0", role = "input"}  : tensor<16x16xi8>
```
`role ∈ {weight, input, bias}`. Result type gives shape + dtype.

### `merlin_iface.resident_pack` — make a weight resident
```mlir
%W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} :
         (tensor<16x16xi8>) -> !merlin_iface.resident
```
Maps to command-buffer opcode `RES_PACK` (`operands: {src, dst}`, `attributes: {layout}`).

### `merlin_iface.matmul` — matmul against a resident weight
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
- `output_dtype ∈ {i32, i8}` — `i32` = full-width readout, `i8` = scaled/clamped readout.
- `acc_scale : f32` — required iff `"acc_scale"` is in `epilogue`. The `: f32` suffix is honest:
  the requant applies an **f32 multiply, round-to-nearest-even, clamp to i8**.
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

### `merlin_iface.evict` — release a resident weight
```mlir
merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
```
Maps to `EVICT` (`operands: {handle}`).

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

The golden command buffer this lowers to is `examples/expected_command_buffer_g0.json`.

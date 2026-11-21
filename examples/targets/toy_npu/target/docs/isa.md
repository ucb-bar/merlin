# ToyNPU ISA (reference target)

ToyNPU is a synthetic reference target for TargetGen and conformance testing. It is
intentionally small. The ISA below is documentation only — no implementation yet.

## Instructions (eventual `toynpu` dialect ops)

| Op              | Summary                                                        |
| --------------- | -------------------------------------------------------------- |
| `toynpu.res_pack` | Pack an immutable RHS tensor and make it resident.           |
| `toynpu.matmul`   | Matmul against a resident tensor, producing an accumulator.  |
| `toynpu.commit`   | Apply the epilogue and commit an accumulator to a tensor.    |
| `toynpu.evict`    | Evict a resident tensor, freeing resident storage.           |

## Types

- `!toynpu.resident_tensor<...>` — a packed, resident weight tensor.
- `!toynpu.accumulator<...>`     — an in-place accumulator for matmul + epilogue.

## Constraints

- `res_pack` requires the source tensor to be immutable for the region.
- Total resident bytes must not exceed `resident_storage_bytes` (see target_contract).

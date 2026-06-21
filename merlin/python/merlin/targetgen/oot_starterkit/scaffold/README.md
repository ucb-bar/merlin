# OOT backend scaffold (STRUCTURE-ONLY — you write all the lowering)

This scaffold gives you the **plumbing** so you don't rebuild it; it deliberately contains **no op
lowering** (not even matmul) — authoring every lowering is your job and the thing being measured.

What the kit hands you (import from `merlin.targetgen.oot_starterkit`):
- `parse_interface(mlir) -> {tensors, commands, params, target, abi_version}` — the input parser (don't
  rewrite it).
- `CommandBufferBuilder(target, backend)` — `.tensor(...)`, `.command(opcode, operands, attributes)`,
  `.params(...)`, `.write(path)`; raises unless the buffer is **schema-valid** (kills the schema failure
  plane).
- `transforms.im2col(ifm_nhwc, weight_khwc, ...)` — generic conv→2D-matmul shapes + recipe.
- `transforms.tile_to_dim(m, n, k, dim)` — standard systolic tiling into DIM×DIM tiles.

Your 4 CLI entrypoints (manifest `commands`): `parse`, `lower_interface_to_target`, `emit_command_buffer`,
`lower_target_to_llvm`. A minimal driver shape:

```python
from merlin.targetgen.oot_starterkit import parse_interface, CommandBufferBuilder, transforms
model = parse_interface(open(input_mlir).read())
b = CommandBufferBuilder(target=model["target"], backend="my_oot")
# ... YOUR lowering: for each command in model["commands"], decide opcodes/operands and b.command(...)
#     use transforms.im2col / tile_to_dim where helpful; map tiles to YOUR target's instructions.
b.write(output_json)
```

Recommended (no-CIRCT structural pre-screen): give your dialect ops **strong IRDL verifiers** (operand
ranks, tile-to-DIM, legal attrs) so malformed IR is caught at construction, before you run a sim.

The kit is identical for every arm that has it; it encodes nothing about any specific accelerator's ISA.

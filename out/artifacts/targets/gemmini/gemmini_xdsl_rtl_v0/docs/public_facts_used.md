# Public facts used

- `gemmini/isa_include/gemmini.h`: command funct values; CONFIG, DMA, PRELOAD, COMPUTE, loop-conv, pooling, and address field layouts.
- `gemmini/isa_include/gemmini_params.h`: DIM 16, `ADDR_LEN` 32, i8 element/i32 accumulator datapaths, IEEE-f32 scale fields, and round-to-nearest-even scaling semantics.
- RTL facts API: custom opcode 0x7b, funct3 3, legal funct set, 16x16 mesh, 262144-byte operand store, and 65536-byte accumulator store.
- `contract/interface_grammar.md` and `contract/command_buffer_abi.yaml`: operation semantics, tensor roles, epilogues, conv/im2col, and pooling geometry.
- CCA/action menus: all seven RTL-derived lever axes route to the package's tile/command emitter, with no orphan fields or routes.


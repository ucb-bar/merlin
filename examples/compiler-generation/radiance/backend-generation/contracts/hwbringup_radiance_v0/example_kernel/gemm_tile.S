# gemm_tile.S -- worked radiance/Muon example: a tiny f32 GEMM tile (C[MxN] += A[MxK].B[KxN]).
#
# THIS FILE IS REAL, ASSEMBLABLE, DERIVED CODE -- not a mnemonic sketch. It is what merlin's own
# fork-free radiance backend emits, and it is the canonical answer to "how do I encode COMPUTE for
# this target": the FP multiply-add, loads, and stores below are all here, in the target's real
# 64-bit words. There is NO Muon-LLVM fork in the loop and NONE is needed.
#
# HOW IT WAS PRODUCED (the fork-free codegen path you should mirror -- see
# merlin.runtime.backends.muon.build_forkfree_elf + merlin.targetgen.isa_transcode):
#   1. compile the tile with STOCK clang for the base substrate: -march=rv32im_zfinx -mabi=ilp32
#      (the -march + FP mode are DERIVED from this target's opcode table via
#      isa_transcode.derive_march -- Muon does FP on GPRs, so it is zfinx, not a separate FP file).
#   2. objcopy the rv32 .text, then FixedFormatTranscoder(isa_model_for_target("radiance"))
#      structurally RE-ENCODES each 32-bit rv32 instruction into this target's 64-bit fixed-format
#      word -- same opcode/funct VALUES, re-laid-out into the field positions the RTL decoder uses
#      (opcode[6:0] rd[16:9] f3[19:17] rs1[27:20] rs2[35:28] rs3[43:36] f7[58:52] ext2[8:7] ...).
#   3. stock llvm-mc assembles these .quad words; the derived disassembler decodes every one.
#
# KEY POINT for a raw-ISA backend author: you do NOT invent a compute encoding and you do NOT fall
# back to standard RV64F mnemonics. Compute IS encodable here -- MADD/OP_FP/LOAD/STORE all live in
# the derived 24-opcode table. A memory op's address space is the `ext2` field (global=0, shared=1);
# this tile is global-only, so ext2=0 throughout -- a scratchpad access would set ext2=1.
#
# VERIFY IT YOURSELF (oracle-free, instant):
#   python isa_tools.py disasm example_kernel/gemm_tile.S   # every .quad decodes; none illegal
#   python isa_tools.py lint   example_kernel/gemm_tile.S --op matmul   # clean; MADD present
#
# Source (compiled verbatim by step 1):
#   void gemm_tile(const float *A, const float *B, float *C, int M, int N, int K) {
#       for (int m = 0; m < M; ++m)
#           for (int n = 0; n < N; ++n) {
#               float acc = 0.0f;
#               for (int k = 0; k < K; ++k)
#                   acc += A[m * K + k] * B[k * N + n];
#               C[m * N + n] = acc;
#           }
#   }

.section .text
.global gemm_tile
.type gemm_tile,@function
gemm_tile:
    .quad 0x00001500d00a0063   # BRANCH
    .quad 0x0fffff0ff0200413   # OP_IMM
    .quad 0x000000c080240023   # STORE
    .quad 0x0000008090240023   # STORE
    .quad 0x0000004120240023   # STORE
    .quad 0x0000000000002013   # OP_IMM
    .quad 0x0000002000e22213   # OP_IMM
    .quad 0x0000002000f20a13   # OP_IMM
    .quad 0x000002000000006f   # JAL
    .quad 0x0000001001002013   # OP_IMM
    .quad 0x0000000050a01433   # OP
    .quad 0x00000d80d1000063   # BRANCH
    .quad 0x0ffffe80e00bfe63   # BRANCH
    .quad 0x0000000000000c13   # OP_IMM
    .quad 0x00100000e1000e33   # OP
    .quad 0x0000002000720e13   # OP_IMM
    .quad 0x0000000070c00e33   # OP
    .quad 0x0000000000b03813   # OP_IMM
    .quad 0x000003800000006f   # JAL
    .quad 0x0000002000623c13   # OP_IMM
    .quad 0x0000001000600c13   # OP_IMM
    .quad 0x00000001e0703c33   # OP
    .quad 0x00000001d1e40023   # STORE
    .quad 0x0000004001c03813   # OP_IMM
    .quad 0x0ffff880e061fe63   # BRANCH
    .quad 0x0000000000003a13   # OP_IMM
    .quad 0x0ffffc80f00bfe63   # BRANCH
    .quad 0x0000000000a03c13   # OP_IMM
    .quad 0x0000000001c03e13   # OP_IMM
    .quad 0x0000000000f01013   # OP_IMM
    .quad 0x0000000001e41203   # LOAD
    .quad 0x0000000001f42403   # LOAD
    .quad 0x0ffffffff0801013   # OP_IMM
    .quad 0x0000000111f03e33   # OP
    .quad 0x000001d1209e3a43   # MADD
    .quad 0x0000004001e03c13   # OP_IMM
    .quad 0x0ffffd000083fe63   # BRANCH
    .quad 0x0ffff70ff000006f   # JAL
    .quad 0x000000c000241003   # LOAD
    .quad 0x0000008000241203   # LOAD
    .quad 0x0000004000242403   # LOAD
    .quad 0x0000010000200413   # OP_IMM
    .quad 0x0000000000100067   # JALR
.size gemm_tile, .-gemm_tile

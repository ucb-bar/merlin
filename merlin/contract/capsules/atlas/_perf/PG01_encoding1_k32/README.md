# PG01_encoding1_k32

PG01_encoding1_k32: matmul over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, authored from shared operand-encoding pair: one identical contraction computed in two of the target's own declared encodings, so the cost of the encoding choice is isolated (tile=32; K=32, M=32, N=32; op=matmul, operand_dtype=encoding1).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

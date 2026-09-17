# PG03_encoding1_k32

PG03_encoding1_k32: matmul over W[32, 16]:fp16, A0[16, 32]:fp16, authored from shared operand-encoding pair: one identical contraction computed in two of the target's own declared encodings, so the cost of the encoding choice is isolated (tile=16; K=32, M=16, N=16; op=matmul, operand_dtype=encoding1).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

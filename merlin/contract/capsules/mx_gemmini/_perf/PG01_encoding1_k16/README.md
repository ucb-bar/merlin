# PG01_encoding1_k16

PG01_encoding1_k16: matmul over W[16, 16]:i8, A0[16, 16]:i8, authored from shared operand-encoding pair: one identical contraction computed in two of the target's own declared encodings, so the cost of the encoding choice is isolated (tile=16; K=16, M=16, N=16; op=matmul, operand_dtype=encoding1).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

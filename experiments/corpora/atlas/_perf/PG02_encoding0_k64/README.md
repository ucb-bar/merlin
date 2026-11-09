# PG02_encoding0_k64

PG02_encoding0_k64: matmul over W[64, 32]:bf16, A0[32, 64]:bf16, authored from shared operand-encoding pair: one identical contraction computed in two of the target's own declared encodings, so the cost of the encoding choice is isolated (tile=32; K=64, M=32, N=32; op=matmul, operand_dtype=encoding0).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

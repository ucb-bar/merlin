# PF01_matmul_m32k32n32

PF01_matmul_m32k32n32: matmul over W[32, 32]:bf16, A0[32, 32]:bf16, authored from a fusion triple at one shape, twice: X@W+B fused, against X@W and X+B as the two ops it replaces. The three members are generated from ONE shape statement, so they cannot drift apart (tile=32; K=32, M=32, N=32; op=matmul).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

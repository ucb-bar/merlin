# PF04_matmul_m32k64n32

PF04_matmul_m32k64n32: matmul over W[64, 32]:bf16, A0[32, 64]:bf16, authored from a fusion triple at one shape, twice: X@W+B fused, against X@W and X+B as the two ops it replaces. The three members are generated from ONE shape statement, so they cannot drift apart (tile=32; K=64, M=32, N=32; op=matmul).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

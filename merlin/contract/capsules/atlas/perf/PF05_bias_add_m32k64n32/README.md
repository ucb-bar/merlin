# PF05_bias_add_m32k64n32

PF05_bias_add_m32k64n32: bias_add over X[32, 32]:bf16, B[32]:bf16, authored from a fusion triple at one shape, twice: X@W+B fused, against X@W and X+B as the two ops it replaces. The three members are generated from ONE shape statement, so they cannot drift apart (tile=32; K=64, M=32, N=32; op=bias_add).

kind=model_slice label=dev op=bias_add modes={}

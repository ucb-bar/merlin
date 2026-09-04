# PM01_m16n32

PM01_m16n32: matmul over W[16, 32]:i8, A0[16, 16]:i8, authored from output-tile sweep at a fixed reduction depth, over exact mesh multiples; the aligned counterpart of the held-out PKG landmark tails (tile=16; M=16, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

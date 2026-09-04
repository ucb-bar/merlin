# PM02_m32n16

PM02_m32n16: matmul over W[16, 16]:i8, A0[32, 16]:i8, authored from output-tile sweep at a fixed reduction depth, over exact mesh multiples; the aligned counterpart of the held-out PKG landmark tails (tile=16; M=32, N=16).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

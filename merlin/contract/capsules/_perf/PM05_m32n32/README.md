# PM05_m32n32

PM05_m32n32: matmul over W[16, 32]:i8, A0[32, 16]:i8, authored from output-tile sweep at a fixed reduction depth, over exact mesh multiples from one to four tiles on each parallel extent; the aligned counterpart of the held-out PKG landmark tails. Four points per axis rather than two because this family fits a law over M and N, and a two-point fit cannot be refuted. Extents stay symbolic multiples of the derived mesh dimension, never literals. (tile=16; M=32, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

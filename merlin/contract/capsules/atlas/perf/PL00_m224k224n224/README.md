# PL00_m224k224n224

PL00_m224k224n224: matmul over W[224, 224]:bf16, A0[224, 224]:bf16, authored from intra-layer square and rectangular contractions (343 and 686 tile passes) -- large enough that the fixed per-tile transfer cost is amortized, and inside the program runner's DRAM window with room (tile=32; K=224, M=224, N=224).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

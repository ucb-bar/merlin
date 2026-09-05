# PV02_c96

PV02_c96: conv2d over W[864, 32]:fp8_e4m3, IFM[1, 8, 8, 96]:fp8_e4m3, authored from window-reuse amplification: convolution at a fixed 3x3 window and 8x8 image over four input channel depths. Conv is the family with headroom on this machine -- measured at 3.3% of the achievable ceiling against deep-K matmul's ~100% -- so it is where a schedule change has room to show. Channel depth is varied and everything else held fixed, so the fitted rate is cycles per unit of contraction depth and nothing else moves under it. (tile=32; ci=96).

kind=model_slice label=dev op=conv2d modes={'conv2d': True, 'k_accumulate': True}

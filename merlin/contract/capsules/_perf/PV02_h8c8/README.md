# PV02_h8c8

PV02_h8c8: conv2d over W[72, 8]:i8, IFM[1, 8, 8, 8]:i8, authored from im2col convolution at two contraction depths and two spatial extents; the regime phase 1 measures at 3.3% of the achievable ceiling and no other performance family exercises (tile=16; Himg=8, ci=8).

kind=model_slice label=dev op=conv2d modes={'conv2d': True, 'k_accumulate': True}

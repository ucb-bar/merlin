# PV03_h16c8

PV03_h16c8: conv2d over W[72, 8]:i8, IFM[1, 16, 8, 8]:i8, authored from im2col convolution at two contraction depths and two spatial extents; the regime phase 1 measures at 3.3% of the achievable ceiling and no other performance family exercises (tile=16; Himg=16, ci=8).

kind=model_slice label=dev op=conv2d modes={'conv2d': True, 'k_accumulate': True}

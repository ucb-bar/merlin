# PV00_h8c4

PV00_h8c4: conv2d over W[36, 8]:i8, IFM[1, 8, 8, 4]:i8, authored from im2col convolution at two contraction depths and two spatial extents; the regime phase 1 measures at 3.3% of the achievable ceiling and no other performance family exercises (tile=16; Himg=8, ci=4).

kind=model_slice label=dev op=conv2d modes={'conv2d': True, 'k_accumulate': True}

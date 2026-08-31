# M2_microvit_gemmini

M2_microvit_gemmini: model over I0[1, 1, 16, 16]:i8, I1[1, 16]:i8, authored from microvit int8 — tile-sized vision+recurrent control net (overlapping patch-merge, pooled-reduction attention, depthwise MixFFN, pixel-shuffle/max-pool fusion, LSTM cell), composed across the systolic mesh and the host lane.

kind=model label=public op=model modes={}

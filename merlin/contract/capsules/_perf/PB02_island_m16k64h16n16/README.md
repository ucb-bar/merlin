# PB02_island_m16k64h16n16

PB02_island_m16k64h16n16: host_island_seam over A0[16, 64]:i8, W0[64, 16]:i8, W1[16, 16]:i8, authored from shared host-island differential: the same two integer contractions and saturating requant epilogue, with one explicit scalar XOR map present only in the island member. Both execute as one submitted whole-program kernel after one unmeasured warm call; only the next complete call is measured. (tile=16; H=16, K=64, M=16, N=16; comparison_role=island, host_transform=xor_low_bit).

kind=model_slice label=dev op=host_island_seam modes={'matmul': True, 'mixed_lane_whole_program': True}

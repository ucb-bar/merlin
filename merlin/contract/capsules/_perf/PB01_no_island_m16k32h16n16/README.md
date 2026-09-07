# PB01_no_island_m16k32h16n16

PB01_no_island_m16k32h16n16: host_island_seam over A0[16, 32]:i8, W0[32, 16]:i8, W1[16, 16]:i8, authored from shared host-island differential: the same two integer contractions and saturating requant epilogue, with one explicit scalar XOR map present only in the island member. Both execute as one submitted whole-program kernel after one unmeasured warm call; only the next complete call is measured. (tile=16; H=16, K=32, M=16, N=16; comparison_role=no_island, host_transform=none).

kind=model_slice label=dev op=host_island_seam modes={'matmul': True, 'mixed_lane_whole_program': False}

# PQ06_j2_k32

PQ06_j2_k32: resident_reuse over W[32, 16]:i8, A0[16, 32]:i8, A1[16, 32]:i8, authored from barrier-count sweep: the same arithmetic issued as N independent jobs, once with a barrier after every job and once with the minimum pair. Job counts span 1..16 rather than 1..4 because the lever's cost is per barrier and real kernels issue thousands of them -- a three-point sweep over 1,2,4 cannot separate a saving that grows with the removed barrier count from noise. (tile=16; K=32; jobs=2, matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}, {'lhs': 'A1', 'out': 'Y1', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

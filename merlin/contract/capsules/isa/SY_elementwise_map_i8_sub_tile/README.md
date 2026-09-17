# SY_elementwise_map_i8_sub_tile

SY_elementwise_map_i8_sub_tile: matmul over W[8, 4]:i8, A0[4, 8]:i8, authored from synthesized for conformance cell elementwise_map/i8/sub_tile (basis=observed, admitted_by=['systolic_mesh']); extents from boundaries.extent_probes; elementwise_map is admitted only in composition with ['contraction'], so it is carried as a fused epilogue on matmul -- a standalone capsule for it would be refused by the eligibility oracle as a false fallback.

kind=isa label=public op=matmul modes={}

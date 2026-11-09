# SY_reduction_i8_partial

SY_reduction_i8_partial: matmul over W[32, 15]:i8, A0[16, 32]:i8, authored from synthesized for conformance cell reduction/i8/partial (basis=observed, admitted_by=['systolic_mesh']); extents from boundaries.extent_probes; reduction is admitted only in composition with ['contraction', 'movement'], so it is carried as a fused epilogue on matmul -- a standalone capsule for it would be refused by the eligibility oracle as a false fallback.

kind=isa label=public op=matmul modes={}

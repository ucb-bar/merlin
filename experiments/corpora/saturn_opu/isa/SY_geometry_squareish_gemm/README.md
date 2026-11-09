# SY_geometry_squareish_gemm

SY_geometry_squareish_gemm: matmul over W[1024, 256]:i8, A0[196, 1024]:i8, authored from synthesized for geometry class 'squareish_gemm': real captures present 37 contraction region(s) of this aspect ratio carrying 0.000509 of all contraction MAC work, and the heaviest of them is M=196 K=1024 N=256. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 50176 written output elements exceeds the 862 a 300.0s certification budget affords, and this target declares no tier cheaper than its cert tier (none resolved) to fall back to, so the capsule is emitted UNCAPPED and is expected to be expensive.

kind=isa label=public op=matmul modes={}

# SY_geometry_gemv_like

SY_geometry_gemv_like: matmul over W[256, 1000]:i8, A0[1, 256]:i8, authored from synthesized for geometry class 'gemv_like': real captures present 1 contraction region(s) of this aspect ratio carrying 0.0 of all contraction MAC work, and the heaviest of them is M=1 K=256 N=1000. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 1000 written output elements exceeds the 862 a 300.0s certification budget affords, and this target declares no tier cheaper than its cert tier (none resolved) to fall back to, so the capsule is emitted UNCAPPED and is expected to be expensive.

kind=isa label=public op=matmul modes={}

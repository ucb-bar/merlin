# SY_geometry_tall_skinny

SY_geometry_tall_skinny: matmul over W[256, 128]:i8, A0[1024, 256]:i8, authored from synthesized for geometry class 'tall_skinny': real captures present 377 contraction region(s) of this aspect ratio carrying 0.004318 of all contraction MAC work, and the heaviest of them is M=1024 K=256 N=128. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 131072 written output elements exceeds the 862 a 300.0s certification budget affords, and this target declares no tier cheaper than its cert tier (none resolved) to fall back to, so the capsule is emitted UNCAPPED and is expected to be expensive.

kind=isa label=public op=matmul modes={}

# SY_geometry_tall_skinny

SY_geometry_tall_skinny: matmul over W[256, 64]:f32, A0[512, 256]:f32, authored from synthesized for geometry class 'tall_skinny': real captures present 373 contraction region(s) of this aspect ratio carrying 0.004318 of all contraction MAC work, and the heaviest of them is M=512 K=256 N=64. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 32768 operand elements exceeds the 742 a 300.0s certification budget affords on this target, so it is graded at the loop tier and rests on 'SY_contraction_f32_aligned'.

kind=isa label=public op=matmul modes={}

# SY_geometry_odd_tail_heavy

SY_geometry_odd_tail_heavy: matmul over W[85, 256]:f32, A0[65, 85]:f32, authored from synthesized for geometry class 'odd_tail_heavy': real captures present 8 contraction region(s) of this aspect ratio carrying 0.000207 of all contraction MAC work, and the heaviest of them is M=65 K=85 N=256. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 16640 operand elements exceeds the 742 a 300.0s certification budget affords on this target, so it is graded at the loop tier and rests on 'SY_contraction_f32_aligned'.

kind=isa label=public op=matmul modes={}

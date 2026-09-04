# SY_geometry_odd_tail_heavy

SY_geometry_odd_tail_heavy: matmul over W[256, 768]:i8, A0[196, 256]:i8, authored from synthesized for geometry class 'odd_tail_heavy': real captures present 8 contraction region(s) of this aspect ratio carrying 0.000207 of all contraction MAC work, and the heaviest of them is M=196 K=256 N=768. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 150528 operand elements exceeds the 862 a 300.0s certification budget affords on this target, so it is graded at the loop tier and rests on 'SY_contraction_i8_aligned'.

kind=isa label=public op=matmul modes={}

# SY_geometry_squareish_gemm

SY_geometry_squareish_gemm: matmul over W[1024, 256]:i8, A0[196, 1024]:i8, authored from synthesized for geometry class 'squareish_gemm': real captures present 33 contraction region(s) of this aspect ratio carrying 0.000509 of all contraction MAC work, and the heaviest of them is M=196 K=1024 N=256. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 50176 operand elements exceeds the 875 a 300.0s certification budget affords on this target, so it is graded at the loop tier and rests on 'SY_contraction_i8_aligned'.

kind=isa label=public op=matmul modes={}

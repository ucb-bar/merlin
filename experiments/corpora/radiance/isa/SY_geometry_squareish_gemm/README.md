# SY_geometry_squareish_gemm

SY_geometry_squareish_gemm: matmul over W[1024, 128]:f32, A0[98, 1024]:f32, authored from synthesized for geometry class 'squareish_gemm': real captures present 33 contraction region(s) of this aspect ratio carrying 0.000509 of all contraction MAC work, and the heaviest of them is M=98 K=1024 N=128. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 12544 operand elements exceeds the 742 a 300.0s certification budget affords on this target, so it is graded at the loop tier and rests on 'SY_contraction_f32_aligned'.

kind=isa label=public op=matmul modes={}

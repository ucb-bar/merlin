# SY_geometry_gemv_like

SY_geometry_gemv_like: matmul over W[227, 1024]:i8, A0[1, 227]:i8, authored from synthesized for geometry class 'gemv_like': real captures present 71 contraction region(s) of this aspect ratio carrying 3.3e-05 of all contraction MAC work, and the heaviest of them is M=1 K=227 N=1024. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 1024 written output elements exceeds the 923 a 300.0s certification budget affords on verilator on this target, so it is graded at L2 and rests on 'SY_contraction_i8_aligned'.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

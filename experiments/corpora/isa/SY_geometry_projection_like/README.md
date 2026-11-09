# SY_geometry_projection_like

SY_geometry_projection_like: matmul over W[480, 160]:i8, A0[56, 480]:i8, authored from synthesized for geometry class 'projection_like': real captures present 148 contraction region(s) of this aspect ratio carrying 0.002062 of all contraction MAC work, and the heaviest of them is M=56 K=480 N=160. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 8960 written output elements exceeds the 875 a 300.0s certification budget affords on this target, so it is graded at L2 and rests on 'SY_contraction_i8_aligned'.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

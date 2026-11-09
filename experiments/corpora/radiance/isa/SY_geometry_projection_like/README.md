# SY_geometry_projection_like

SY_geometry_projection_like: matmul over W[240, 80]:fp16, A0[28, 240]:fp16, authored from synthesized for geometry class 'projection_like': real captures present 148 contraction region(s) of this aspect ratio carrying 0.002062 of all contraction MAC work, and the heaviest of them is M=28 K=240 N=80. Every other synthesized capsule is square, so without this the corpus cannot tell a compiler that tiles this ratio well from one that does not. 2240 written output elements exceeds the 742 a 300.0s certification budget affords on this target, so it is graded at L2 and rests on 'SY_contraction_fp16_aligned'.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False}

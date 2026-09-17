# SY_app_elementwise_mul_f32_rank1_32_l2

SY_app_elementwise_mul_f32_rank1_32_l2 is an exact equal-shape multiply over
`A[32]:f32` and `B[32]:f32`. It is anchored to structural region 904
(`prov.region_id=mul_22`, `prov.aten=aten.mul.Tensor`) of the byte-pinned
`smolvla_fp32_consistent` capture. This exact program shape occurs 56 times in
that capture and 168 times across Radiance's six declared application captures.

The capsule requires L0/L1/L2 only. Any later L3 result must come from the frozen
derived-GSIM stage; this source capsule makes no GSIM or physical-RTL claim.

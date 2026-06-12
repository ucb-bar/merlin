# merlin_rvv_matmul_i8(A, B, C, M, K, N, row_lo, row_hi)
#   C[i][j] = sum_k A[i][k] * B[k][j]   for i in [row_lo, row_hi)
#   A: i8 row-major (M x K), B: i8 row-major (K x N), C: i32 row-major (M x N)
#
# Hand-written RVV (not intrinsics) on purpose: GCC 13.2's vsetvl fusion pass
# miscompiles mixed-EEW intrinsic sequences (it rewrote our e32m4 config to
# e8m1, making vsext.vf4 illegal). Explicit vsetvli toggling between e8m1
# (loads) and e32m4 (widen + accumulate) is deterministic across toolchains.
#
# Register use: a0=A a1=B a2=C a3=scratch a4=K a5=N a6=i a7=row_hi
#               t0=&A[i*K] t1=&C[i*N] t2=j t3=vl t4=&B[k*N+j] t5=k t6=A[i*K+k]
  .text
  .globl merlin_rvv_matmul_i8
merlin_rvv_matmul_i8:
.Lrow:
  bge   a6, a7, .Ldone
  mul   t0, a6, a4
  add   t0, a0, t0
  mul   t1, a6, a5
  slli  t1, t1, 2
  add   t1, a2, t1
  li    t2, 0
.Lcol:
  sub   t3, a5, t2
  vsetvli t3, t3, e32, m4, ta, ma
  vmv.v.i v8, 0
  add   t4, a1, t2
  li    t5, 0
.Lk:
  bge   t5, a4, .Lkdone
  add   a3, t0, t5
  lb    t6, 0(a3)
  vsetvli zero, t3, e8, m1, ta, ma
  vle8.v  v4, (t4)
  vsetvli zero, t3, e32, m4, ta, ma
  vsext.vf4 v16, v4
  vmacc.vx  v8, t6, v16
  add   t4, t4, a5
  addi  t5, t5, 1
  j     .Lk
.Lkdone:
  slli  a3, t2, 2
  add   a3, t1, a3
  vse32.v v8, (a3)
  add   t2, t2, t3
  blt   t2, a5, .Lcol
  addi  a6, a6, 1
  j     .Lrow
.Ldone:
  ret

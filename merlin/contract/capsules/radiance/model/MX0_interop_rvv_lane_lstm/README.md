# MX0_interop_rvv_lane_lstm

MX0_interop_rvv_lane_lstm: model over I0[1, 1, 60, 90]:f32, I1[1, 1]:f32, I2[1, 4]:f32, I3[3, 128]:i64, I4[3, 128]:i64, authored from lstmnetvit composed across the SIMT accelerator and the scalar/RVV host lane — a conv/recurrent op mix, so a different split from the decode stack.

kind=model label=public op=model modes={}

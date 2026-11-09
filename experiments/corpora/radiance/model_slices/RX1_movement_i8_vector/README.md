# RX1_movement_i8_vector

RX1_movement_i8_vector: movement over X[64, 256]:i8, authored from int8 movement, vector-shaped (64x256) — same missing datapath at a size where a scalar loop is not the answer; reuse the tuned RVV package rather than regenerating one.

kind=model_slice label=public op=movement modes={'movement': True}

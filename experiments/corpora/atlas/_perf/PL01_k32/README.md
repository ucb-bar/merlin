# PL01_k32

PL01_k32: resident_reuse over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, A1[32, 32]:fp8_e4m3, A2[32, 32]:fp8_e4m3, A3[32, 32]:fp8_e4m3, authored from shared cross-regime amortization pair: one pushed weight reused by one tile versus by four, at identical per-tile work (tile=32; K=32; matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}, {'lhs': 'A1', 'out': 'Y1', 'M_tiles': 1}, {'lhs': 'A2', 'out': 'Y2', 'M_tiles': 1}, {'lhs': 'A3', 'out': 'Y3', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

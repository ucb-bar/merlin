# PL03_k32

PL03_k32: resident_reuse over W[16, 16]:i8, A0[16, 16]:i8, A1[16, 16]:i8, A2[16, 16]:i8, A3[16, 16]:i8, authored from shared cross-regime amortization pair: one pushed weight reused by one tile versus by four, at identical per-tile work (tile=16; K=32; matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}, {'lhs': 'A1', 'out': 'Y1', 'M_tiles': 1}, {'lhs': 'A2', 'out': 'Y2', 'M_tiles': 1}, {'lhs': 'A3', 'out': 'Y3', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

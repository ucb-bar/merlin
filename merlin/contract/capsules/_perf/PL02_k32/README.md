# PL02_k32

PL02_k32: resident_reuse over W[16, 16]:i8, A0[16, 16]:i8, authored from shared cross-regime amortization pair: one pushed weight reused by one tile versus by four, at identical per-tile work (tile=16; K=32; matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

# PL00_k16

PL00_k16: resident_reuse over W[16, 16]:i8, A0[16, 16]:i8, authored from shared cross-regime amortization pair: one pushed weight reused by one tile versus by four, at identical per-tile work (tile=16; K=16; matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': False}

This single-tile control uses its resident weight once. It retains resident-pack
semantics but does not claim repeated reuse; the paired layer-scale member
provides that performance witness.

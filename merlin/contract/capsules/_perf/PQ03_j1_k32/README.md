# PQ03_j1_k32

PQ03_j1_k32: resident_reuse over W[32, 16]:i8, A0[16, 32]:i8, authored from shared barrier-removal pair: one resident weight reused by one, two and four jobs at two reduction depths, where each member's A/B is that one capsule emitted at two settings of the target emitter's retire knob (tile=16; K=32; jobs=1, matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

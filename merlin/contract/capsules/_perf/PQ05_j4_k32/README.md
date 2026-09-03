# PQ05_j4_k32

PQ05_j4_k32: resident_reuse over W[32, 16]:i8, A0[16, 32]:i8, A1[16, 32]:i8, A2[16, 32]:i8, A3[16, 32]:i8, authored from shared barrier-removal pair: one resident weight reused by one, two and four jobs at two reduction depths, where each member's A/B is that one capsule emitted at two settings of the target emitter's retire knob (tile=16; K=32; jobs=4, matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}, {'lhs': 'A1', 'out': 'Y1', 'M_tiles': 1}, {'lhs': 'A2', 'out': 'Y2', 'M_tiles': 1}, {'lhs': 'A3', 'out': 'Y3', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

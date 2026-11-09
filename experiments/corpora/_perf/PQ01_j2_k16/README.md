# PQ01_j2_k16

PQ01_j2_k16: resident_reuse over W[16, 16]:i8, A0[16, 16]:i8, A1[16, 16]:i8, authored from shared barrier-removal pair: one resident weight reused by one, two and four jobs at two reduction depths, where each member's A/B is that one capsule emitted at two settings of the target emitter's retire knob (tile=16; K=16; jobs=2, matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}, {'lhs': 'A1', 'out': 'Y1', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

# PC00_k64

PC00_k64: resident_reuse over W[16, 16]:i8, A0[16, 16]:i8, A1[16, 16]:i8, authored from shared inter-layer declaration: one resident weight, two matmuls, and the A/B is the command list of THIS capsule permuted by command_stream_gen at measurement time (tile=16; K=64, N=64; matmuls=[{'lhs': 'A0', 'out': 'Y0', 'M_tiles': 1}, {'lhs': 'A1', 'out': 'Y1', 'M_tiles': 1}]).

kind=model_slice label=dev op=resident_reuse modes={'resident_reuse': True}

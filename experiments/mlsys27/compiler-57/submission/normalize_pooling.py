"""Refuse captured max reductions until a modeled accumulator program exists.

The hardware readout supports pooling and the contraction backend implements
raw MATMUL/COMMIT pooling. A captured standalone max reduction cannot be
represented by a COMMIT whose source is an ordinary tensor: the public command
interpreter requires a real accumulator producer. Keep this compiler gap
explicit instead of persisting an uninterpretable command buffer.
"""


def normalize(cb):
    if cb.get('declined'):
        return cb
    for command in cb.get('commands', []):
        if command['opcode'] == 'WINDOW_SUM' and command.get('attributes', {}).get('mode') == 'max':
            cb['declined'] = {
                'op': 'captured_max_reduction',
                'reason': 'Compiler has not yet lowered this captured maximum reduction to a modeled accumulator-producing program; COMMIT from an ordinary tensor is not supported by the selected ABI',
            }
            cb['commands'] = []
            return cb
    return cb

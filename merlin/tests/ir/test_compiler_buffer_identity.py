"""Identity prefixes retain complete memory proofs and precise coverage reports."""

from buffer_reuse_qualification_cases import BufferCase

from merlin.transforms import loops, normalization, planning
from merlin.transforms.certificates import ir_text, parse_ir


def chain(*, normalized, epochs):
    module = parse_ir(BufferCase("identity-prefix", 2, 2, 2, epochs=epochs).source)
    if normalized:
        normalization.normalize_shapes(module)
    buffers = [ir_text(module)]
    normalization.normalize_shapes(module)
    buffers.append(ir_text(module))
    planning.reuse_storage(module, planning.plan_storage(module))
    buffers.append(ir_text(module))
    loops.segment_loops(module, (2,), index_bits=64)
    buffers.append(ir_text(module))
    return tuple(buffers)

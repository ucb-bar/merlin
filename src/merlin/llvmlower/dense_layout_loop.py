"""Emit a bounded dense transpose through an existing scalar loop builder.

The caller owns immutable source storage and a disjoint private destination for
the complete synchronous copy. This helper grants no device or alias authority.
"""

from __future__ import annotations


def emit_dense_transpose(builder, source, destination, rows, cols, *, max_elements):
    """Copy dense i8 ``source[rows, cols]`` to ``destination[cols, rows]``.

    Validate the complete static work bound before emitting any operation. The
    builder supplies ordinary host loops, address arithmetic and byte accesses;
    no input value, numerical recipe or accelerator command is inspected.
    """
    if any(type(value) is not int or value <= 0 for value in (rows, cols, max_elements)):
        raise ValueError("dense transpose needs positive integer extents and work bound")
    if rows * cols > max_elements or rows * cols >= 1 << 63:
        raise ValueError("dense transpose exceeds its selected work/address bound")
    if source == destination:
        raise ValueError("dense transpose requires distinct private destination storage")

    def row(ri):
        def column(ci):
            src = builder.op("add", builder.op("mul", ri, cols), ci)
            dst = builder.op("add", builder.op("mul", ci, rows), ri)
            value = builder.load(builder.gep(source, src), "i8")
            builder.store(value, builder.gep(destination, dst), "i8")

        builder.loop(0, cols, 1, column)

    builder.loop(0, rows, 1, row)

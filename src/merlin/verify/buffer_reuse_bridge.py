"""Independent complete raw-buffer parsing for bounded memory candidates.

Parser/projection fidelity is trusted. This bridge performs no optimization,
allocator simulation, Lean invocation, runtime selection or receipt admission.
"""

from __future__ import annotations

import hashlib

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser

from .lean_graph_input_admission import parse_admitted
from .lean_graph_loops_shape_capture import export


def parse(raw):
    def parser(text):
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
            context.load_dialect(dialect)
        return Parser(context, text).parse_module()

    module = parse_admitted(raw, parser, calls="shape-direct")
    module.verify()
    return module


def decode_pair(source, final):
    """Decode both complete inputs independently; no source-plan matching."""
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "final_sha256": hashlib.sha256(final.encode()).hexdigest(),
        "source_export": export(parse(source)),
        "final_export": export(parse(final)),
        "projection_trusted": True,
        "producer_attribution": "not_checked_by_parser",
        "authority": False,
    }


def reconstruct_chain(buffers, *, index_bits=64):
    """Recompute untrusted plan hints from FOUR complete independently parsed IRs.

    Shape output and reuse input share the SAME raw normalized buffer, not just
    an asserted join/hash. No retained plan, receipt or execution outcome is read.
    """
    from .lean_graph_loops_reuse_capture import export as memory_export
    from .lean_graph_loops_shape_after_projection import decode
    from .lean_graph_loops_shape_epoch import project_chain

    if (
        type(buffers) is not tuple
        or len(buffers) != 4
        or any(type(x) is not str or len(x.encode()) > 256 * 1024 for x in buffers)
    ):
        raise ValueError("four bounded complete original/normalized/reused/segmented buffers required")
    original, normalized, reused, segmented = map(parse, buffers)
    shape = {"before_export": export(original), "after_export": export(normalized)}
    memory = {
        key: memory_export(module, calls="shape-direct")
        for key, module in zip(
            ("before_export", "middle_export", "after_export"), (normalized, reused, segmented), strict=True
        )
    }
    adapter = project_chain(shape, memory, index_bits=index_bits, max_nodes=5000)
    frame = decode(shape["after_export"], index_bits=index_bits, max_nodes=5000)["frame"]
    return {"adapter": adapter, "pair": {"afterFrame": frame}}

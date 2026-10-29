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
    """Recompute untrusted plan hints from actual complete independently parsed IRs.

    Shape output and reuse input share the SAME raw normalized buffer, not just
    an asserted join/hash. The three-buffer route proves the untiled target.
    Its auxiliary segmentation supplies only a checked epoch-model hint; it is
    never a submitted boundary or delivered compiler result.
    """
    from .lean_graph_loops_reuse_capture import export as memory_export
    from .lean_graph_loops_shape_after_projection import decode
    from .lean_graph_loops_shape_epoch import project_chain

    if (
        type(buffers) is not tuple
        or len(buffers) not in (3, 4)
        or any(type(x) is not str or len(x.encode()) > 256 * 1024 for x in buffers)
    ):
        raise ValueError("three or four bounded complete original/normalized/reused[/segmented] buffers required")
    parsed = tuple(map(parse, buffers))
    original, normalized, reused = parsed[:3]
    if len(parsed) == 4:
        segmented = parsed[3]
    else:
        from merlin.transforms.loops import segment_loops

        segmented = reused.clone()
        segment_loops(segmented, (2,), index_bits=index_bits)
        segmented.verify()
    shape = {"before_export": export(original), "after_export": export(normalized)}
    memory = {
        key: memory_export(module, calls="shape-direct")
        for key, module in zip(
            ("before_export", "middle_export", "after_export"), (normalized, reused, segmented), strict=True
        )
    }
    adapter = project_chain(shape, memory, index_bits=index_bits, max_nodes=5000)
    if len(parsed) == 3:
        adapter["terminal_initializer"] = "untiled"
    frame = decode(shape["after_export"], index_bits=index_bits, max_nodes=5000)["frame"]
    return {"adapter": adapter, "pair": {"afterFrame": frame}}

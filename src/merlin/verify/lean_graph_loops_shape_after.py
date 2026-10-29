"""Prepare an independently decoded shape pair without native proof execution.

This API performs fresh shared admission, parser verification and independent
term projection. A preparation result is explicitly unqualified: it cannot
issue a Lean or ordinary pipeline acceptance receipt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .lean_graph_input_admission import parse_admitted, source_reference
from .lean_graph_loops_shape_after_model import obligation
from .lean_graph_loops_shape_after_projection import project_pair
from .lean_graph_loops_shape_capture import export


def prepare_pair(before_text, after_text, *, parser, index_bits, max_nodes):
    before = parse_admitted(before_text, parser, calls="shape-direct")
    after = parse_admitted(after_text, parser, calls="shape-direct")
    first, second = export(before), export(after)
    pair = project_pair(first, second, index_bits=index_bits, max_nodes=max_nodes)
    query, roots = obligation(pair)
    return dict(
        before_export=first,
        after_export=second,
        model=pair,
        query=query,
        expected_roots=list(roots),
        admission_source=source_reference(),
        independent_after_term=True,
        lean_invoked=False,
        lean_accepted=False,
        qualified=False,
        pipeline_receipt_issued=False,
    )


def prepare_files(before, after, *, parser, output, index_bits, max_nodes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(before).resolve(), Path(after).resolve()]
    payloads = [path.read_bytes() for path in paths]
    result = prepare_pair(
        *(payload.decode() for payload in payloads), parser=parser, index_bits=index_bits, max_nodes=max_nodes
    )
    refs = []
    for path, data, name in zip(paths, payloads, ("before.mlir", "after.mlir"), strict=True):
        (output / name).write_bytes(data)
        refs.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    query = result.pop("query")
    (output / "Obligation.lean").write_text(query)
    if payloads != [path.read_bytes() for path in paths]:
        raise ValueError("original pair changed during preparation")
    result.update(originals=refs, query_sha256=hashlib.sha256(query.encode()).hexdigest())
    (output / "PREPARATION.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result

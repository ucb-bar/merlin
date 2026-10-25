"""Reopen both actual modules using the pinned producer's parser and full SSA export."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__:
    from .lean_graph_loops_capture import _producer_sources, _reference
    from .lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree
else:
    from lean_graph_loops_capture import _producer_sources, _reference
    from lean_graph_loops_segment_capture import loaded_xdsl_sources, operation_tree


def project(*, source: Path, revision: str, before: Path, after: Path, output: Path):
    sys.dont_write_bytecode = True
    from merlin.transforms.certificates import parse_ir

    source = source.resolve(strict=True)
    observed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if observed != revision:
        raise ValueError("actual parser/producer revision changed")
    first, second = before.read_bytes(), after.read_bytes()
    before_tree = operation_tree(parse_ir(first.decode()))
    after_tree = operation_tree(parse_ir(second.decode()))
    closure = _producer_sources(source, revision, ())
    parser_sources = loaded_xdsl_sources()
    result = {
        "schema": "merlin.graph.scf_actual_tree_projection.v1",
        "producer_revision": revision,
        "before": _reference(before),
        "after": _reference(after),
        "before_tree": before_tree,
        "after_tree": after_tree,
        "producer_source": closure,
        "xdsl_loaded_source": parser_sources,
        "python_projection_trusted": True,
        "pipeline_receipt_issued": False,
    }
    if before.read_bytes() != first or after.read_bytes() != second:
        raise ValueError("actual input module changed during parsing/projection")
    if closure != _producer_sources(source, revision, ()) or parser_sources != loaded_xdsl_sources():
        raise ValueError("selected parser/source dependencies changed during projection")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "after", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    result = project(**vars(parser.parse_args()))
    print(json.dumps({"producer_revision": result["producer_revision"], "python_projection_trusted": True}))


if __name__ == "__main__":
    main()

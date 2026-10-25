"""Fresh exact triple export or actual pinned reuse→segment production.

Parser-derived full trees and source identities are evidence, not formal parser
correctness. This leaf does not modify production or depend on saved witnesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__:
    from .lean_graph_loops_capture import _producer_sources, _reference
    from .lean_graph_loops_reuse_capture import export
    from .lean_graph_loops_segment_capture import loaded_xdsl_sources
else:
    from lean_graph_loops_capture import _producer_sources, _reference
    from lean_graph_loops_reuse_capture import export
    from lean_graph_loops_segment_capture import loaded_xdsl_sources


def buffer_reference(path, data):
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def run(*, source, revision, before, output, middle=None, after=None, tile=2):
    sys.dont_write_bytecode = True
    from merlin.transforms import loops, planning
    from merlin.transforms.certificates import ir_text, parse_ir

    observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if observed != revision or (middle is None) != (after is None):
        raise ValueError("exact pinned provider and either production or full triple required")
    if type(tile) is not int or tile <= 1:
        raise ValueError("actual changed positive segmentation tile required")
    original = before.read_bytes()
    first = parse_ir(original.decode())
    first.verify()
    before_export = export(first)
    if middle is None:
        output.mkdir(parents=True, exist_ok=False)
        plan = planning.plan_storage(first)
        reuse = planning.reuse_storage(first, plan)
        first.verify()
        middle_bytes = ir_text(first).encode()
        middle = output / "middle.mlir"
        middle.write_bytes(middle_bytes)
        middle_export = export(first)
        segmented = loops.segment_loops(first, tile=tile)
        first.verify()
        after_bytes = ir_text(first).encode()
        after = output / "after.mlir"
        after.write_bytes(after_bytes)
        after_export = export(first)
        destination = output / "CAPTURE.json"
        changed = dict(reuse=len(reuse), segmentation=len(segmented))
    else:
        middle_bytes, after_bytes = middle.read_bytes(), after.read_bytes()
        center, final = parse_ir(middle_bytes.decode()), parse_ir(after_bytes.decode())
        center.verify()
        final.verify()
        middle_export, after_export = export(center), export(final)
        destination, changed = output, None
    identities = _producer_sources(source, revision, (Path(loops.__file__).resolve(), Path(planning.__file__).resolve()))
    xdsl = loaded_xdsl_sources()
    result = dict(schema="merlin.graph.memory_scf.actual_triple.v1", producer_revision=revision,
                  producer_source=identities, xdsl_loaded_source=xdsl,
                  before=buffer_reference(before, original), middle=buffer_reference(middle, middle_bytes), after=buffer_reference(after, after_bytes),
                  before_export=before_export, middle_export=middle_export, after_export=after_export,
                  changed=changed, python_projection_trusted=True, untrusted_plan_used_as_authority=False,
                  raw_parser_formally_verified=False, native_or_hardware_proved=False)
    if before.read_bytes() != original or middle.read_bytes() != middle_bytes or after.read_bytes() != after_bytes:
        raise ValueError("input buffer writer during fresh actual triple capture")
    if identities != _producer_sources(source, revision, ()) or xdsl != loaded_xdsl_sources():
        raise ValueError("selected provider/parser source changed during capture")
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "before", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("middle", "after"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tile", type=int, default=2)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(dict(schema=result["schema"], producer_revision=result["producer_revision"], changed=result["changed"])))


if __name__ == "__main__":
    main()

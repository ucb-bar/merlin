"""Checked pointwise destination forwarding for ordinary serial host lowering.

The source's scalar arithmetic is retained. Decisions use static tensor
geometry, source effects, the selected index width and bounded IR growth.
Physical ownership and allocation remain the upstream bufferizer's job.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from xdsl.utils.exceptions import ParseError


def prepare_automatic_destinations(
    source, *, serial, index_bits=64, data_layout=None, max_added_operations=4096, max_metadata_items=65536,
):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms import certificates, pipeline_state
    from merlin.transforms.certificates import ir_digest
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
    from merlin.xdsl_dialects._common import text

    from . import destination_check, insert_slice_destination, optional_passes, quant_round_check

    if type(serial) is not bool or type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("destination forwarding requires the selected lowering family and index width")
    for value in (max_added_operations, max_metadata_items):
        if type(value) is not int or value < 0:
            raise ValueError("destination growth budgets must be nonnegative integers")
    if data_layout is not None and (not isinstance(data_layout, str) or not data_layout):
        raise ValueError("destination forwarding requires its original selected data layout")
    digest = hashlib.sha256(source.encode()).hexdigest()
    owners = [Path(__file__), *(Path(owner.__file__) for owner in (
        destination_check, insert_slice_destination, optional_passes, quant_round_check, pipeline_state, certificates)),
        *(Path(__file__).with_name(name) for name in ("lower.py", "impr_features.py", "selfcopy.py", "toolchain.py"))]
    library = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in owners}
    context = PipelineContext(
        hardware=None if data_layout is None else "llvm-datalayout:" + hashlib.sha256(data_layout.encode()).hexdigest(),
        numerical_policy="typed_source:" + digest,
        resources=f"index_bits={index_bits};added_ops={max_added_operations};metadata_items={max_metadata_items}",
        execution="serial_host_tensor;upstream_bufferization_ownership",
        library=hashlib.sha256(json.dumps(library, sort_keys=True).encode()).hexdigest(),
    )
    report = {
        "schema": "merlin.automatic_host_destinations.v1", "rewrites": 0,
        "source_file_sha256": digest, "output_file_sha256": digest,
        "context": asdict(context), "library_sources_sha256": library, "checks": [],
        "machine_conformance": "not_established",
        "cost_policy": (
            "reduce logical interior initialization/copy work under explicit IR growth budgets; no cycle estimate"
        ),
    }
    if not serial or not optional_passes.switched(
        "automatic-host-destinations", os.environ.get("MERLIN_AUTOMATIC_HOST_DESTINATIONS", "1") != "0",
    ):
        report["diagnostic"] = "disabled or selected lowering family has no qualified destination route"
        return source, report
    # A conservative lexical prefilter saves an additional parse for complete
    # models without insertions. Unusual escaped spellings may retain the legal
    # original route; this filter never authorizes a rewrite.
    if "tensor.insert_slice" not in source or max_added_operations < 2:
        report["diagnostic"] = "no canonical insertion token or insufficient growth budget; original lowering retained"
        return source, report
    try:
        original = parse_mlir_text(source)
    except ParseError:
        report["diagnostic"] = "source is outside the typed destination parser; original lowering retained"
        return source, report
    current = original.clone()
    session = PipelineSession(current, context=context, ir_level="host_tensor", max_repairs=0)
    session.run(
        PassSpec("destination_source_legality", 0, mutates_ir=False, ir_level="host_tensor", produces=("source",)),
        lambda: {"source": not any(op.successors or any(len(r.blocks) > 1 for r in op.regions)
                                   for op in current.walk())},
    )
    if not session.analysis("source"):
        report["diagnostic"] = "unsupported control flow; original lowering retained"
        return source, report
    session.run(
        PassSpec("destination_forwarding", 1, ir_level="host_tensor", requires=("source",),
                 invalidates=("*",), produces=("emission",)),
        lambda: {"emission": insert_slice_destination.rewrite_module(
            current, initialize_border_only=True, index_bits=index_bits,
            max_added_operations=max_added_operations, max_metadata_items=max_metadata_items)},
    )
    session.run(
        PassSpec("destination_conformance", 2, mutates_ir=False, ir_level="host_tensor",
                 requires=("emission",), produces=("realized_regions",)),
        lambda: {"realized_regions": destination_check.verify_destination_rewrite(original, current)},
    )
    regions = session.analysis("realized_regions")
    if len(regions) != len(session.analysis("emission")):
        raise ValueError("destination rewrite accounting differs from actual emitted geometry")
    if any(region["exterior_elements"] <= 0 for region in regions):
        raise ValueError("destination selection has no nonempty padding exterior")
    if {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in owners} != library:
        raise ValueError("destination source/checker library changed during optimization")
    output = text(current, generic=True) if regions else source
    report.update({
        "rewrites": len(regions), "realized_regions": regions,
        "source_sha256": ir_digest(original), "output_sha256": ir_digest(current),
        "output_file_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "checks": [asdict(record) for record in session.records],
    })
    if not regions:
        report["diagnostic"] = "no closed static pointwise/splat insertion meets effect, index and growth checks"
    return output, report

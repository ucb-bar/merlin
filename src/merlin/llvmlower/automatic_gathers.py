"""Explicit checked host packing from static logically zero-padded images.

Selection removes a source materialization, not a predicted latency. The
ordinary target call's existing logical tensor ABI and permissions are retained.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from xdsl.utils.exceptions import ParseError


def implementation_paths():
    """Existing pass-owner closure, also retained through ordinary final linking."""
    from merlin.common import dense_float_ir, mlir_query
    from merlin.frontends import linalg_mlir
    from merlin.transforms import certificates, pipeline_state
    from merlin.xdsl_dialects import _common

    from . import im2col_pack, im2col_padding, im2col_padding_check, optional_passes

    return (
        Path(__file__),
        Path(__file__).with_name("lower.py"),
        *(
            Path(owner.__file__)
            for owner in (
                im2col_pack,
                im2col_padding,
                im2col_padding_check,
                optional_passes,
                certificates,
                pipeline_state,
                mlir_query,
                dense_float_ir,
                linalg_mlir,
                _common,
            )
        ),
        Path(__file__).with_name("source_optimization_request.py"),
        Path(__file__).with_name("source_optimization_selection.py"),
    )


def prepare_automatic_gathers(
    source, *, serial, index_bits=64, data_layout=None, max_added_operations=4096, enabled=None
):
    from merlin.frontends import linalg_mlir
    from merlin.transforms.certificates import ir_digest
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order
    from merlin.xdsl_dialects import _common

    from . import im2col_padding, im2col_padding_check, optional_passes

    if type(serial) is not bool or type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("padding/gather selection requires original lowering family and index width")
    if type(max_added_operations) is not int or not 0 <= max_added_operations <= 1 << 20:
        raise ValueError("padding/gather operation budget must be bounded and nonnegative")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("padding/gather explicit selection must be a boolean")
    if data_layout is not None and (type(data_layout) is not str or not data_layout):
        raise ValueError("padding/gather selection requires its original selected data layout")
    owners = implementation_paths()

    def library():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in owners}

    identities, digest = library(), hashlib.sha256(source.encode()).hexdigest()
    context = PipelineContext(
        hardware=None if data_layout is None else "llvm-datalayout:" + hashlib.sha256(data_layout.encode()).hexdigest(),
        numerical_policy="typed_source:" + digest,
        resources=f"index_bits={index_bits};added_ops={max_added_operations}",
        execution="serial_host_tensor;guarded_reads;upstream_bufferization_ownership",
        library=hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
    )
    report = dict(
        schema="merlin.automatic_host_gathers.v1",
        rewrites=0,
        source_file_sha256=digest,
        output_file_sha256=digest,
        context=asdict(context),
        library_sources_sha256=identities,
        machine_conformance="not_established",
        timing="UNKNOWN",
        cost_policy="explicit storage/traffic-only selection; guarded read latency unknown",
        checks=[],
    )
    if enabled is None:
        enabled = optional_passes.switched(
            "automatic-host-gathers", os.environ.get("MERLIN_AUTOMATIC_HOST_GATHERS", "0") == "1"
        )
    if not serial or not enabled or index_bits != 64:
        report["diagnostic"] = "disabled or selected lowering family lacks the checked serial index route"
        return source, report
    if "tensor.insert_slice" not in source or "linalg.generic" not in source or max_added_operations < 35:
        report["diagnostic"] = "no canonical padding/gather tokens or insufficient construction budget"
        return source, report
    try:
        before = linalg_mlir.parse_mlir_text(source)
    except ParseError:
        report["diagnostic"] = "source is outside the typed padding/gather parser; original lowering retained"
        return source, report
    before.verify()
    original_identity = ir_digest(before)
    current = before.clone()
    specs = (
        PassSpec("padding_gather_source", 0, mutates_ir=False, ir_level="host_tensor", produces=("source",)),
        PassSpec(
            "padding_gather_rewrite",
            1,
            ir_level="host_tensor",
            requires=("source",),
            invalidates=("*",),
            produces=("emission",),
        ),
        PassSpec(
            "padding_gather_conformance",
            2,
            mutates_ir=False,
            ir_level="host_tensor",
            requires=("emission",),
            produces=("realization",),
        ),
    )
    validate_pass_order(specs)
    session = PipelineSession(current, context=context, ir_level="host_tensor", max_repairs=0)
    session.run(
        specs[0],
        lambda: {
            "source": tuple(
                op
                for op in current.walk()
                if im2col_padding.match_padding_gather(op, index_bits=index_bits) is not None
            )
        },
    )
    session.run(
        specs[1],
        lambda: {
            "emission": im2col_padding.rewrite_padding_gathers(
                current, index_bits=index_bits, max_added_operations=max_added_operations
            )
        },
    )
    session.run(
        specs[2],
        lambda: {
            "realization": im2col_padding_check.verify_padding_gather_rewrite(before, current, index_bits=index_bits)
        },
    )
    checked = session.analysis("realization")
    if len(checked) != len(session.analysis("emission")):
        raise ValueError("padding/gather emitter accounting differs from independent realized source")
    if ir_digest(before) != original_identity:
        raise ValueError("padding/gather original source changed during selection")
    if library() != identities:
        raise ValueError("padding/gather source/checker implementation changed during selection")
    output = _common.text(current, generic=True) if checked else source
    report.update(
        rewrites=len(checked),
        source_sha256=original_identity,
        output_sha256=ir_digest(current),
        output_file_sha256=hashlib.sha256(output.encode()).hexdigest(),
        realized_regions=checked,
        checks=[asdict(record) for record in session.records],
    )
    if not checked:
        report["diagnostic"] = "no closed padding/gather meets source, index and construction checks"
    return output, report

"""Select checked row scheduling only for the ordinary serial CPU pipeline.

Source arithmetic is the numerical algorithm; no framework exponential, sum
tree, approximation or target capability is inferred. Unknown source syntax or
unsupported lowering modes retain the original program with a diagnostic.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from xdsl.utils.exceptions import ParseError


def prepare_automatic_rows(source, *, serial):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms import pipeline as source_pipeline
    from merlin.transforms import row_normalization, row_normalization_check
    from merlin.xdsl_dialects._common import text

    from . import pipeline
    from .optional_passes import switched

    if type(serial) is not bool:
        raise ValueError("row source scheduling requires an explicit selected lowering family")
    identities = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            (__name__, Path(__file__)),
            (row_normalization.__name__, Path(row_normalization.__file__)),
            (row_normalization_check.__name__, Path(row_normalization_check.__file__)),
            (pipeline.__name__, Path(pipeline.__file__)),
            (source_pipeline.__name__, Path(source_pipeline.__file__)),
            ("merlin.llvmlower.optional_passes", Path(__file__).with_name("optional_passes.py")),
            ("merlin.llvmlower.lower", Path(__file__).with_name("lower.py")),
        )
    }
    digest = hashlib.sha256(source.encode()).hexdigest()
    report = {
        "schema": "merlin.automatic_host_rows.v1",
        "rewrites": 0,
        "source_file_sha256": digest,
        "output_file_sha256": digest,
        "library_sources_sha256": identities,
        "machine_conformance": "not_established",
    }
    enabled = switched("automatic-host-rows", os.environ.get("MERLIN_AUTOMATIC_HOST_ROWS", "1") != "0")
    if not serial or not enabled:
        report["diagnostic"] = (
            "disabled" if not enabled else "selected lowering family lacks qualified row storage forwarding"
        )
        return source, report
    try:
        module = parse_mlir_text(source)
    except ParseError:
        report["diagnostic"] = "source is outside the typed row parser; original lowering retained"
        return source, report
    decisions = {}
    for op in module.walk():
        if (group := row_normalization.match_row_normalization(op)) is not None:
            reason = row_normalization.row_state_cost_reason(group) or "eligible"
            decisions[reason] = decisions.get(reason, 0) + 1
    report["source_candidates"] = decisions
    if not decisions.get("eligible"):
        report["diagnostic"] = (
            "no closed normalization passes the conservative graph cost checks; original lowering retained"
        )
        return source, report
    modified, checked = row_normalization.fuse_row_normalizations(module, numerical_policy="typed_source:" + digest)
    report["realization"] = checked
    report["rewrites"] = checked["rewrites"]
    output = text(modified, generic=True) if checked["rewrites"] else source
    report["output_file_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    return output, report

"""Independent compiler-byte custody for original CFG structuralization.

This imports no producer or evaluator. Byte links and recorded block edges
grant no arithmetic, effect, alias or source-equivalence authority.
"""

from __future__ import annotations

import hashlib
import json

_FILES = frozenset(
    {
        "cfg-input.mlir",
        "cfg-lifted.mlir",
        "cfg-output.ttir",
        "cfg-source-graph.json",
        "cfg-stage.json",
        "cfg-stderr.log",
    }
)
_FLAGS = [
    "--mlir-disable-threading",
    "--verify-each",
    "--mlir-print-op-generic",
    "--mlir-print-debuginfo",
    "--mlir-print-local-scope",
    "--canonicalize",
    "--lift-cf-to-scf",
    "--canonicalize",
]


def _record(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def checked_control_flow_sources(checked, provenance, source):
    report = provenance.get("control_flow_normalization")
    if report is None:
        return {}
    if (
        not isinstance(report, dict)
        or set(report) != {"schema", "entry", "source_sha256", "artifacts", "grants_semantics"}
        or report.get("schema") != "t2m.control-flow-normalization.v1"
        or report.get("grants_semantics") is not False
        or set(report.get("artifacts", {})) != _FILES
    ):
        raise ValueError("CFG source custody lacks its explicit complete artifact roster")
    copied = {name: checked(name) for name in _FILES}
    if (
        report["entry"] != provenance.get("spec", {}).get("entry")
        or report["source_sha256"] != _record(source)["sha256"]
        or any(report["artifacts"][name] != _record(copied[name]) for name in _FILES)
    ):
        raise ValueError("CFG source custody differs from original source or retained compiler bytes")
    graph = json.loads(copied["cfg-source-graph.json"])
    if (
        graph.get("operations") != provenance.get("source_operations")
        or graph.get("blocks") != provenance.get("original_blocks")
        or graph.get("attributes") != provenance.get("original_attributes")
    ):
        raise ValueError("CFG source graph differs from its complete original block/operation inventory")
    stage = json.loads(copied["cfg-stage.json"])
    if (
        set(stage) != {"schema", "flags", "artifacts", "stage_completed", "optimizer", "grants_semantics"}
        or stage.get("schema") != "t2m.control-flow-stage.v1"
        or stage.get("flags") != _FLAGS
        or stage.get("stage_completed") is not True
        or stage.get("grants_semantics") is not False
        or stage.get("artifacts")
        != {
            "source.ttir": _record(copied["cfg-input.mlir"]),
            "lowered.mlir": _record(copied["cfg-lifted.mlir"]),
            "stderr.log": _record(copied["cfg-stderr.log"]),
        }
    ):
        raise ValueError("CFG native stage does not bind its fixed pass and input/output bytes")
    optimizer = stage.get("optimizer", {})
    digest = optimizer.get("sha256")
    if (
        set(optimizer) != {"sha256", "bytes"}
        or type(optimizer.get("bytes")) is not int
        or optimizer["bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("CFG native stage lacks its actual optimizer fingerprint")
    return copied

"""Ordinary captured-policy host legalization through shared pass contracts.

The source language's canonical pure arithmetic is the numerical contract here.
This route changes no selected quantization model, floating environment policy,
device code, reduction order, or approximation. Unknown scalar effects decline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from xdsl.dialects import builtin
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.rewriter import Rewriter

from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline import PassSpec, PipelineContext, PipelineSession

from .quant_round import fuse_round_clamp_convert
from .quant_round_check import pure_scalar_body, verify_quantization_rewrite

PASS_NAME = "automatic-host-numerics"


def _tensor_body(operation):
    if (
        not isinstance(operation, GenericOp)
        or not operation.results
        or any(not isinstance(value.type, builtin.TensorType) for value in operation.results)
        or len(operation.regions) != 1
        or len(operation.regions[0].blocks) != 1
        or not pure_scalar_body(operation.regions[0].block)
    ):
        return False
    for value in (*operation.operands, *operation.results):
        typ = value.type
        if isinstance(typ, builtin.TensorType):
            if not isinstance(typ.encoding, builtin.NoneAttr):
                return False
        elif not isinstance(typ, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)):
            return False
    return True


def optimize_host_quantization(module, *, numerical_policy: str, max_added_operations: int = 4096):
    """Return private, independently checked IR and observations from that IR.

    The supplied identity binds the selected original policy; it grants no new
    semantics. Eligibility is restricted to canonical tensor scalar bodies with
    no unknown effects, attributes or floating-environment operations.
    """
    if type(max_added_operations) is not int or max_added_operations < 0 or not numerical_policy:
        raise ValueError("host legalization requires its numerical policy and nonnegative code budget")
    original = module
    current = module.clone()
    from merlin.transforms import pipeline, pipeline_state

    library_sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path(__file__).with_name("quant_round.py"),
            Path(__file__).with_name("quant_round_check.py"),
            Path(pipeline.__file__),
            Path(pipeline_state.__file__),
        )
    }
    library = hashlib.sha256(json.dumps(library_sources, sort_keys=True).encode()).hexdigest()
    session = PipelineSession(
        current,
        context=PipelineContext(
            numerical_policy=numerical_policy, resources=f"added_operations={max_added_operations}", library=library
        ),
        ir_level="host_tensor",
    )
    counts = {"rewrites": 0, "refused_effect_bodies": 0, "refused_code_budget": 0, "refused_control_flow": 0}
    before_count = sum(1 for _ in current.walk())

    def rewrite():
        if any(op.successors or any(len(region.blocks) > 1 for region in op.regions) for op in current.walk()):
            counts["refused_control_flow"] += 1
            return {}
        growth = 0
        for op in tuple(current.walk()):
            if op.name != "linalg.generic" or not any(child.name == "math.roundeven" for child in op.walk()):
                continue
            if not _tensor_body(op):
                counts["refused_effect_bodies"] += 1
                continue
            replacement = op.clone()
            count = fuse_round_clamp_convert(replacement)
            if not count:
                continue
            added = sum(1 for _ in replacement.walk()) - sum(1 for _ in op.walk())
            if growth + max(0, added) > max_added_operations:
                counts["refused_code_budget"] += 1
                continue
            Rewriter.replace_op(op, replacement)
            counts["rewrites"] += count
            growth += max(0, added)
        return {}

    session.run(PassSpec("bounded_quantization", 0, ir_level="host_tensor", invalidates=("*",)), rewrite)
    session.run(
        PassSpec("arithmetic_conformance", 1, mutates_ir=False, ir_level="host_tensor"),
        lambda: (verify_quantization_rewrite(original, current), {})[1],
    )

    def rounds(module):
        return sum(op.name == "math.roundeven" for op in module.walk())

    observed = {
        "round_sites_before": rounds(original),
        "round_sites_after": rounds(current),
        "operations_before": before_count,
        "operations_after": sum(1 for _ in current.walk()),
    }
    if observed["round_sites_before"] - observed["round_sites_after"] != counts["rewrites"]:
        raise ValueError("actual rounding removal differs from legalization accounting")
    return current, {
        "schema": "merlin.automatic_host_numerics.v1",
        "source_sha256": ir_digest(original),
        "output_sha256": ir_digest(current),
        "context": asdict(session.context),
        **counts,
        **observed,
        "library_sources_sha256": library_sources,
        "checks": [asdict(row) for row in session.records],
        "machine_conformance": "not_established",
        "formal_coverage": "outside_existing_integer_transform_domains",
    }


def prepare_automatic_host(prepared: Path, directory: Path, *, policy_path: Path):
    """Consume the ordinary frozen-policy selection; preserve the input file."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    from .optional_passes import switched

    if (
        not switched(PASS_NAME, os.environ.get("MERLIN_AUTOMATIC_HOST_NUMERICS", "1") != "0")
        or not policy_path.is_file()
    ):
        return prepared
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    if not isinstance(policy, dict):
        raise ValueError("automatic host numerical policy must be a JSON object")
    if policy.get("schema") != "merlin.quantization_policy.v1" or policy.get("selection") != "captured_scope":
        return prepared
    source_bytes = prepared.read_bytes()
    source = source_bytes.decode("utf-8")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if policy.get("prepared_mlir_sha256") != source_sha256:
        # A policy recorded before another preparation rewrite does not bind
        # this input. Retain the already legal preparation path and explain
        # why this optional optimization could not consume that policy.
        directory.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "merlin.automatic_host_numerics.v1",
            "rewrites": 0,
            "refused_policy_binding": 1,
            "source_file_sha256": source_sha256,
            "output_file_sha256": source_sha256,
            "policy_file_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "diagnostic": "captured numerical policy does not bind the current prepared source bytes",
            "scope": "unchanged source; no optimization executed",
            "checks": [],
            "machine_conformance": "not_established",
            "formal_coverage": "not_run",
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return prepared
    module, report = optimize_host_quantization(
        parse_mlir_text(source), numerical_policy=hashlib.sha256(policy_bytes).hexdigest()
    )
    directory.mkdir(parents=True, exist_ok=True)
    output = prepared
    if report["rewrites"]:
        output = directory / "model.mlir"
        output.write_text(text(module, generic=True), encoding="utf-8")
    report["source_file_sha256"] = source_sha256
    report["output_file_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output

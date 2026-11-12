"""Explicit current-source observer discovery after ordinary scalar fusion.

The native worker retains one exact generic tensor checkpoint and continues
the original lowering. The installed parent analyzes that checkpoint using the
existing typed observation theorem. Discovery grants no rewrite or cost policy.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .source_expression_interval import (
    IntervalEffectContract,
    find_closed_scalar_i8_observers,
    validate_closed_scalar_observer,
)

FEATURE = "source_observation_closed_scalar_i8"
MARKER = "__merlin_source_observation_closed_scalar_i8__"
CHECKPOINT = "source_observation.mlir"
REPORT = "source_observation_report.json"
TOKEN = "OK source_observation_stage"
_FUSE = "func.func(linalg-fuse-elementwise-ops)"
_GENERALIZE = "func.func(linalg-generalize-named-ops)"


def _split_passes(text):
    """Preserve top-level native passes, including nested lists and options."""
    passes, depth, start = [], 0, 0
    for index, char in enumerate(text + ","):
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced source-observation pipeline")
        elif char == "," and depth == 0:
            value = text[start:index].strip()
            if value:
                passes.append(value)
            start = index + 1
    if depth:
        raise ValueError("unbalanced source-observation pipeline")
    return passes


def _stage_position(passes):
    if passes.count(_FUSE) != 1 or passes.count(_GENERALIZE) != 1:
        raise ValueError("source observation requires one scalar fusion/generalization stage")
    fused, generalized = passes.index(_FUSE), passes.index(_GENERALIZE)
    buffers = [index for index, value in enumerate(passes) if value.split("{", 1)[0] == "one-shot-bufferize"]
    if len(buffers) != 1 or not fused < generalized < buffers[0]:
        raise ValueError("source observation requires fused/generalized tensors before bufferization")
    earlier = passes[: generalized + 1]
    if any("pointwise" in value or "broadcast_source_rsqrt" in value for value in earlier):
        raise ValueError("source observation must precede pointwise scheduling")
    return generalized + 1


def _edit_pipeline(passes):
    if MARKER in passes:
        raise ValueError("duplicate source-observation stage")
    index = _stage_position(passes)
    return [*passes[:index], MARKER, *passes[index:]]


def validate_pipeline(pipeline):
    passes = _split_passes(pipeline)
    if passes.count(MARKER) != 1:
        raise ValueError("selected source observation requires one explicit stage marker")
    plain = [value for value in passes if value != MARKER]
    if passes.index(MARKER) != _stage_position(plain):
        raise ValueError("source observation marker is not at the fused tensor boundary")


def ensure_registered():
    from merlin.compiler.selection.shared import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Discover closed scalar i8 observers from current fused tensor source "
                    "under explicit effects; preserve original code."
                ),
                edit_pipeline=_edit_pipeline,
            )
        )
    return FEATURE


STAGE_RUNNER = r"""
_SO_ORIGINAL_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    from pathlib import Path
    import hashlib
    passes = _split_passes(pipeline)
    marker = "__merlin_source_observation_closed_scalar_i8__"
    if marker not in passes:
        return _SO_ORIGINAL_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise ValueError("source observation requires one stage")
    index = passes.index(marker)
    head, tail = passes[:index], passes[index + 1:]
    hoist_head = any("buffer-loop-hoisting" in value for value in head)
    late_head = any("convert-scf-to-openmp" in value for value in head)
    _SO_ORIGINAL_RUN_STAGES(ctx, module, ",".join(head),
        erase if hoist_head else 0, mid if hoist_head else (),
        late if late_head else (), post_openmp if late_head else (), pre_generalize)
    payload = module.operation.get_asm(print_generic_op_form=True).encode("utf-8")
    (Path(sys.argv[2]).parent / "source_observation.mlir").write_bytes(payload)
    print("OK source_observation_stage", hashlib.sha256(payload).hexdigest())
    return _SO_ORIGINAL_RUN_STAGES(ctx, module, ",".join(tail),
        0 if hoist_head else erase, () if hoist_head else mid,
        () if late_head else late, () if late_head else post_openmp, ())
"""


def bind_runner(source, *, selected):
    """Bind a separate stage wrapper at an explicit top-level runner call."""
    if not selected:
        return source
    tree = ast.parse(source)
    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_run_stages"
    ]
    if len(calls) != 1:
        raise ValueError("source observation requires one staged native runner invocation")
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_stages"]
    if not definitions or any(node.lineno >= calls[0].lineno for node in definitions):
        raise ValueError("source observation requires an already defined native stage runner")
    lines = source.splitlines(keepends=True)
    index = calls[0].lineno - 1
    return (
        "".join(lines[:index]) + inspect.getsource(_split_passes) + "\n" + STAGE_RUNNER + "\n" + "".join(lines[index:])
    )


@dataclass(frozen=True)
class SourceObservations:
    module: object
    proofs: tuple
    refusals: tuple
    source_sha256: str


def _analyze(payload, *, effects):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if not isinstance(effects, IntervalEffectContract):
        raise ValueError("explicit IntervalEffectContract required for source observation")
    effects.validate()
    module = parse_mlir_text(payload.decode("utf-8"))
    module.verify()
    proofs, refusals = find_closed_scalar_i8_observers(module, effects=effects)
    for proof in proofs:
        validate_closed_scalar_observer(proof)
    return SourceObservations(module, proofs, refusals, hashlib.sha256(payload).hexdigest())


def require_report(stdout, workdir, *, effects):
    work = Path(workdir)
    payload = (work / CHECKPOINT).read_bytes()
    tokens = [line.split() for line in stdout.splitlines() if line.startswith(TOKEN + " ")]
    digest = hashlib.sha256(payload).hexdigest()
    if len(tokens) != 1 or tokens[0] != ["OK", "source_observation_stage", digest]:
        raise ValueError("native source observation report is absent, ambiguous or stale")
    observation = _analyze(payload, effects=effects)
    positions = {operation: index for index, operation in enumerate(observation.module.walk())}
    record = {
        "schema": "merlin.source_observation_stage.v1",
        "status": "OBSERVED_CURRENT_TYPED_SOURCE",
        "feature": FEATURE,
        "source": {"path": str((work / CHECKPOINT).resolve()), "sha256": digest, "bytes": len(payload)},
        "effect_permissions": asdict(effects),
        "observers": [
            {
                "expression_sha256": proof.expression.canonical_sha256,
                "quant_factor_bits": proof.quant_factor_bits,
                "source_operation_ordinal": positions[proof.endpoint.owner],
            }
            for proof in observation.proofs
        ],
        "refusals": [
            {"source_operation_ordinal": positions[op], "reason": reason} for op, reason in observation.refusals
        ],
        "scope": (
            "Read-only current-source analysis. Ordinals authenticate source binding; "
            "no numerical rewrite, target capability, storage ownership or profitability permission is granted."
        ),
    }
    (work / REPORT).write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_source_observations(workdir, *, effects):
    """Reclose current checkpoint bytes and effects before exposing typed proofs."""
    work = Path(workdir)
    record = json.loads((work / REPORT).read_text())
    payload = (work / CHECKPOINT).read_bytes()
    if (
        record.get("schema") != "merlin.source_observation_stage.v1"
        or record.get("status") != "OBSERVED_CURRENT_TYPED_SOURCE"
        or record.get("feature") != FEATURE
        or record.get("source")
        != {
            "path": str((work / CHECKPOINT).resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        or not isinstance(effects, IntervalEffectContract)
        or record.get("effect_permissions") != asdict(effects)
    ):
        raise ValueError("source observation checkpoint or effects changed")
    return _analyze(payload, effects=effects)

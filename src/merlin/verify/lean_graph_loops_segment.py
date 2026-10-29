"""Kernel-checked complete ordered SCF segmentation on freshly reopened actual IR.

The full SSA/body/context decoder and parser remain trusted Python. The selected
model proves complete ordered carry/state trace equivalence, not a universal SCF,
countedKernel, numerical, renderer, pipeline or hardware correspondence theorem.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .lean_graph_loops import LeanSelection, _audit, _name, _reference
from .lean_graph_loops_segment_projection import project_trees


@dataclass(frozen=True)
class SegmentSelection:
    proof: LeanSelection
    source: Path
    revision: str
    producer_manifest: Path
    producer_manifest_sha256: str

    def dependencies(self) -> dict:
        data = self.producer_manifest.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.producer_manifest_sha256:
            raise ValueError("originally selected segmentation producer/parser manifest changed")
        selected = json.loads(data)
        if selected["producer_revision"] != self.revision:
            raise ValueError("producer revision differs from originally selected source")
        if (
            subprocess.run(
                ["git", "-C", str(self.source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            != self.revision
        ):
            raise ValueError("actual selected source revision changed")
        source_rows, parser_rows = selected["producer_source"], selected["xdsl_loaded_source"]
        for row in (*source_rows, *parser_rows):
            actual = _reference(Path(row["path"]))
            if actual != {key: row[key] for key in ("path", "sha256", "bytes")}:
                raise ValueError("originally selected producer/parser code changed")
        if (
            not source_rows
            or not parser_rows
            or any(not Path(row["path"]).resolve().is_relative_to(self.source.resolve() / "src") for row in source_rows)
        ):
            raise ValueError("complete producer/parser closure is unavailable")
        build = json.loads(self.proof.build_record.read_bytes())
        if {row["module"] for row in build["attempts"]} != {
            "Merlin.StateEffects",
            "Merlin.ResourceGeometry",
            "Merlin.GraphLoopsSegment",
            "Merlin.GraphLoopsSegmentBounds",
            "Merlin.GraphLoopsSegmentModel",
            "Merlin.GraphLoopsSegmentState",
        }:
            raise ValueError("select the complete separate SCF proof/model closure")
        return {
            "proof": self.proof.dependencies(),
            "producer_manifest": _reference(self.producer_manifest, data),
            "producer_source": source_rows,
            "xdsl_loaded_source": parser_rows,
            "revision": self.revision,
        }


def _nat(value):
    if type(value) is not int or value < 0:
        raise ValueError("index/control/SSA identities require actual nonnegative integers")
    return str(value)


def _list(values, render):
    return "[" + ", ".join(render(value) for value in values) + "]"


SEGMENT_ROOTS = (
    "GraphSegmentObligation.accepted",
    "GraphSegmentObligation.complete",
    "GraphSegmentObligation.contextual",
    "GraphSegmentObligation.signed_source",
    "GraphSegmentObligation.signed_target",
    "GraphSegmentObligation.carried_state",
    "Merlin.GraphLoopsSegment.Model.checked_carry_state",
    "Merlin.GraphLoopsSegment.Execution.state_actions",
    "Merlin.GraphLoopsSegment.Model.check_sound",
    "Merlin.GraphLoopsSegment.Model.checked_complete_trace",
    "Merlin.GraphLoopsSegment.Model.checked_context",
    "Merlin.GraphLoopsSegment.complete_trace_iff",
    "Merlin.GraphLoopsSegment.Bounds.constants",
    "Merlin.GraphLoopsSegment.Bounds.outer_control",
    "Merlin.GraphLoopsSegment.Bounds.inner_control",
    "Merlin.GraphLoopsSegment.Execution.signed_indices",
)


def obligation(projection: dict) -> str:
    strings = {"sourceBody", "targetBody", "sourceContext", "targetContext"}
    string_lists = {"carryTypes", "sourceResults", "outerResults", "innerResults", "outerArgumentTypes"}
    natural_lists = {
        "sourceInitial",
        "outerInitial",
        "outerCarryArguments",
        "addOperands",
        "minOperands",
        "innerOperands",
        "innerResultValues",
        "outerYield",
    }
    naturals = {
        "targetLower",
        "targetUpper",
        "targetStride",
        "targetInnerStep",
        "outerIV",
        "strideValue",
        "upperValue",
        "originalStepValue",
        "addResult",
        "minResult",
        "sourceYieldArity",
    }
    if set(projection) != strings | string_lists | natural_lists | naturals | {"bounds", "structural_nodes"}:
        raise ValueError("unknown model fields require an explicit checker extension")
    bounds = projection["bounds"]
    if set(bounds) != {"bits", "lower", "upper", "step", "tile"}:
        raise ValueError("complete actual control bounds are required")
    fields = ["bounds := ⟨" + ", ".join(_nat(bounds[key]) for key in ("bits", "lower", "upper", "step", "tile")) + "⟩"]
    for name in sorted(strings):
        fields.append(name + " := " + _name(projection[name]))
    for name in sorted(string_lists):
        fields.append(name + " := " + _list(projection[name], _name))
    for name in sorted(natural_lists):
        fields.append(name + " := " + _list(projection[name], _nat))
    for name in sorted(naturals):
        fields.append(name + " := " + _nat(projection[name]))
    literal = "{\n  " + "\n  ".join(fields) + "\n}"
    roots = "\n".join("#print axioms " + name for name in SEGMENT_ROOTS)
    return f"""import Merlin.GraphLoopsSegmentState
set_option autoImplicit false
open Merlin GraphLoopsSegment
namespace GraphSegmentObligation
def actual : Model.Projection := {literal}
theorem accepted : Model.check actual = true := by cbv
theorem complete {{σ : Type}} (interpret : String → Nat → σ → σ → Prop)
    (initial final : σ) (trace : List Nat) :
    Model.original actual interpret initial trace final ↔ Model.transformed actual interpret initial trace final :=
  Model.checked_complete_trace accepted interpret initial final trace
theorem contextual {{σ Outcome : Type}} (interpret : String → Nat → σ → σ → Prop)
    (context : String → (σ → List Nat → σ → Prop) → Outcome) :
    context actual.sourceContext (Model.original actual interpret) =
      context actual.targetContext (Model.transformed actual interpret) :=
  Model.checked_context accepted interpret context
theorem signed_source {{σ : Type}} (body : Nat → σ → σ → Prop) (initial final : σ) (trace : List Nat)
    (run : Execution body actual.bounds.upper actual.bounds.step actual.bounds.lower initial trace final) :
    ∀ index ∈ trace, FitsSigned actual.bounds.bits index ∧ FitsSigned actual.bounds.bits (index + actual.bounds.step) :=
  Execution.signed_indices (Model.check_sound accepted).1.1 run
theorem signed_target {{σ : Type}} (interpret : String → Nat → σ → σ → Prop)
    (initial final : σ) (trace : List Nat) (run : Model.transformed actual interpret initial trace final) :
    ∀ index ∈ trace, FitsSigned actual.bounds.bits index ∧ FitsSigned actual.bounds.bits (index + actual.bounds.step) :=
  Execution.signed_indices (Model.check_sound accepted).1.1
    ((Model.checked_complete_trace accepted interpret initial final trace).mpr run)
theorem carried_state {{Value : Type}}
    (interpret : String → Nat → CarriedState Value actual.carryTypes.length →
      CarriedState Value actual.carryTypes.length → Prop)
    (initial final : CarriedState Value actual.carryTypes.length) (trace : List Nat) :
    Model.original actual interpret initial trace final ↔ Model.transformed actual interpret initial trace final :=
  Model.checked_carry_state accepted interpret initial final trace
end GraphSegmentObligation
{roots}
"""


def verify_segment_files(
    before_path: Path,
    after_path: Path,
    *,
    selection: SegmentSelection,
    output: Path,
    tile: int,
    index_bits: int,
    max_nodes: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify(
            before_path,
            after_path,
            selection=selection,
            output=output,
            tile=tile,
            index_bits=index_bits,
            max_nodes=max_nodes,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "type": type(error).__name__,
                    "lean_complete_relation_checked": False,
                    "pipeline_receipt_issued": False,
                    "python_projection_trusted": True,
                },
                indent=2,
            )
            + "\n"
        )
        raise


def _verify(before_path, after_path, *, selection, output, tile, index_bits, max_nodes):
    before_path, after_path = before_path.resolve(strict=True), after_path.resolve(strict=True)
    first, second = before_path.read_bytes(), after_path.read_bytes()
    originals = [_reference(before_path, first), _reference(after_path, second)]
    dependencies = selection.dependencies()
    from . import (
        lean_graph_loops,
        lean_graph_loops_capture,
        lean_graph_loops_segment_capture,
        lean_graph_loops_segment_project,
        lean_graph_loops_segment_projection,
    )

    owner_paths = [
        Path(__file__).with_name("lean_graph_input_admission.py").resolve(),
        Path(__file__).resolve(),
        Path(lean_graph_loops.__file__).resolve(),
        Path(lean_graph_loops_capture.__file__).resolve(),
        Path(lean_graph_loops_segment_capture.__file__).resolve(),
        Path(lean_graph_loops_segment_project.__file__).resolve(),
        Path(lean_graph_loops_segment_projection.__file__).resolve(),
        Path(sys.executable).resolve(),
    ]
    owners = [_reference(path) for path in owner_paths]
    before_copy, after_copy = output / "before.mlir", output / "after.mlir"
    before_copy.write_bytes(first)
    after_copy.write_bytes(second)
    fresh = output / "actual-trees.json"
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(selection.source / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(Path(lean_graph_loops_segment_project.__file__).resolve()),
        "--source",
        str(selection.source.resolve()),
        "--revision",
        selection.revision,
        "--before",
        str(before_copy.resolve()),
        "--after",
        str(after_copy.resolve()),
        "--output",
        str(fresh.resolve()),
    ]
    parsed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment, timeout=120)
    (output / "projection.stdout").write_text(parsed.stdout)
    (output / "projection.stderr").write_text(parsed.stderr)
    if parsed.returncode:
        raise ValueError("actual module parsing/SSA export refused; retained logs distinguish compiler-invalid inputs")
    tree_bytes = fresh.read_bytes()
    tree_ref = _reference(fresh, tree_bytes)
    trees = json.loads(tree_bytes)
    if trees["before"] != _reference(before_copy, first) or trees["after"] != _reference(after_copy, second):
        raise ValueError("fresh full trees are not bound to the same actual input buffers")
    for field in ("producer_source", "xdsl_loaded_source"):
        known = {row["path"]: row for row in dependencies[field]}
        if not trees[field] or any(known.get(row["path"]) != row for row in trees[field]):
            raise ValueError("fresh parser/source closure differs from the original selected context")
    projection = project_trees(
        trees["before_tree"], trees["after_tree"], tile=tile, index_bits=index_bits, max_nodes=max_nodes
    )
    model_path = output / "model.json"
    model_bytes = (json.dumps(projection, sort_keys=True, indent=2) + "\n").encode()
    model_path.write_bytes(model_bytes)
    model_ref = _reference(model_path, model_bytes)
    query = output / "Obligation.lean"
    query_bytes = obligation(projection).encode()
    query.write_bytes(query_bytes)
    query_ref = _reference(query, query_bytes)
    proof = selection.proof
    environment.update(
        LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2"
    )
    lean_command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query.resolve())]
    checked = subprocess.run(lean_command, capture_output=True, text=True, check=False, env=environment, timeout=120)
    (output / "stdout.log").write_text(checked.stdout)
    (output / "stderr.log").write_text(checked.stderr)
    record = {
        "schema": "merlin.graph.actual_ordered_scf_segmentation_check.v1",
        "requested_configuration": {"tile": tile, "index_bits": index_bits, "max_nodes": max_nodes},
        "compiler_index_width_is_trusted_context": True,
        "originals": originals,
        "dependencies": dependencies,
        "trusted_decoder_sources": owners,
        "fresh_trees": tree_ref,
        "projection": projection,
        "model": model_ref,
        "query": query_ref,
        "command": lean_command,
        "projection_command": command,
        "returncode": checked.returncode,
        "lean_complete_relation_checked": False,
        "python_projection_trusted": True,
        "body_and_context_interpretation_explicit": True,
        "source_success_inferred": False,
        "untrusted_witness_used": False,
        "LeanMLIR_selected": False,
        "SCF_countedKernel_bridge_proved": False,
        "pipeline_receipt_issued": False,
        "numerical_gates_discharged": False,
        "renderer_or_hardware_proved": False,
    }
    if (
        dependencies != selection.dependencies()
        or originals != [_reference(before_path), _reference(after_path)]
        or owners != [_reference(path) for path in owner_paths]
        or tree_ref != _reference(fresh)
        or query_ref != _reference(query)
        or model_ref != _reference(model_path)
        or before_copy.read_bytes() != first
        or after_copy.read_bytes() != second
    ):
        record["refusal"] = "actual IR/tree/query or selected source/parser/model/tool changed during checking"
    elif checked.returncode:
        record["refusal"] = "Lean refused the actual complete SSA/carry/control/trace relation"
    else:
        record["transitive_axioms"] = _audit(checked.stdout, SEGMENT_ROOTS)
        record["lean_complete_relation_checked"] = True
    (output / "CHECK.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record

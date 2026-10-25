"""Explicit actual SCF unroll→segment join with a logged grouped-body bridge.

Requires exact common middle bytes and both complete-boundary checker results.
No counted-Kernel, native MLIR, numerical, renderer or hardware bridge is made.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .lean_graph_loops import _audit, _reference
from .lean_graph_loops_segment import obligation as segment_obligation
from .lean_graph_loops_segment import verify_segment_files
from .lean_graph_loops_unroll import obligation as unroll_obligation
from .lean_graph_loops_unroll import verify_unroll_files

COMPOSITION_ROOTS = (
    "GraphUnrollJoin.compatible",
    "GraphUnrollJoin.complete",
    "Merlin.GraphLoopsUnroll.Composition.no_tail_bulk_iff",
    "Merlin.GraphLoopsUnroll.Composition.checked_composed",
)


def obligation(unroll, segment):
    def definitions(text):
        return "\n".join(line for line in text.splitlines() if not line.startswith(("import ", "#print axioms ")))

    first = (
        definitions(unroll_obligation(unroll))
        .replace("GraphUnrollObligation", "GraphJoinUnroll")
        .replace("Model.", "GraphLoopsUnroll.Model.")
    )
    second = (
        definitions(segment_obligation(segment))
        .replace("GraphSegmentObligation", "GraphJoinSegment")
        .replace("Model.", "GraphLoopsSegment.Model.")
    )
    roots = "\n".join("#print axioms " + root for root in COMPOSITION_ROOTS)
    return f"""import Merlin.GraphLoopsUnrollComposition
{first}
{second}
namespace GraphUnrollJoin
open Merlin GraphLoopsUnroll
theorem compatible : Composition.Compatible GraphJoinUnroll.actual GraphJoinSegment.actual := by cbv
theorem complete {{σ : Type}} (interpret : String → Nat → σ → σ → Prop)
    (initial final : σ) (trace : List Nat) :
    GraphLoopsUnroll.Model.original GraphJoinUnroll.actual interpret initial trace final ↔
      ∃ bases, GraphLoopsSegment.Model.transformed GraphJoinSegment.actual
        (Composition.groupInterpretation GraphJoinUnroll.actual interpret) (initial, []) bases (final, trace) :=
  Composition.checked_composed GraphJoinUnroll.accepted GraphJoinSegment.accepted compatible
    interpret initial final trace
end GraphUnrollJoin
{roots}
"""


def verify_composition_files(
    before: Path,
    middle: Path,
    after: Path,
    *,
    unroll_selection,
    segment_selection,
    output: Path,
    factor: int,
    tile: int,
    index_bits: int,
    max_nodes: int,
):
    output.mkdir(parents=True, exist_ok=False)
    originals = [_reference(path) for path in (before, middle, after)]
    code = _reference(Path(__file__))
    dependencies = [unroll_selection.dependencies(), segment_selection.dependencies()]
    try:
        first = verify_unroll_files(
            before,
            middle,
            selection=unroll_selection,
            output=output / "unroll",
            factor=factor,
            index_bits=index_bits,
            max_nodes=max_nodes,
        )
        second = verify_segment_files(
            middle,
            after,
            selection=segment_selection,
            output=output / "segment",
            tile=tile,
            index_bits=index_bits,
            max_nodes=max_nodes,
        )
        if not first["lean_complete_relation_checked"] or not second["lean_complete_relation_checked"]:
            raise ValueError("both complete actual same-SCF boundaries must pass before joining")
        if first["originals"][1] != second["originals"][0] or first["originals"][1] != originals[1]:
            raise ValueError("adjacent boundaries do not share exact actual middle bytes")
        query = output / "Composition.lean"
        query.write_text(obligation(first["projection"], second["projection"]))
        query_ref = _reference(query)
        proof = unroll_selection.proof
        env = os.environ.copy()
        env.update(LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2")
        command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query.resolve())]
        checks = [_reference(output / name / "CHECK.json") for name in ("unroll", "segment")]
        owners = first["trusted_decoder_sources"] + second["trusted_decoder_sources"]
        result = subprocess.run(command, capture_output=True, text=True, check=False, env=env, timeout=120)
        (output / "stdout.log").write_text(result.stdout)
        (output / "stderr.log").write_text(result.stderr)
        record = dict(
            schema="merlin.graph.actual_scf_unroll_segment_join.v1",
            originals=originals,
            middle=originals[1],
            boundary_checks=checks,
            query=query_ref,
            command=command,
            returncode=result.returncode,
            dependencies=dependencies,
            checker=code,
            lean_composition_checked=False,
            python_projection_trusted=True,
            group_body_interpretation_explicit=True,
            original_body_occurrence_log_preserved=True,
            numerical_gates_discharged=False,
            pipeline_receipt_issued=False,
            SCF_countedKernel_bridge_proved=False,
            renderer_or_hardware_proved=False,
        )
        if (
            originals != [_reference(path) for path in (before, middle, after)]
            or code != _reference(Path(__file__))
            or dependencies != [unroll_selection.dependencies(), segment_selection.dependencies()]
            or query_ref != _reference(query)
            or checks != [_reference(output / name / "CHECK.json") for name in ("unroll", "segment")]
            or owners != [_reference(Path(row["path"])) for row in owners]
        ):
            record["refusal"] = "composition inputs/code/query or proof/source dependencies changed"
        elif result.returncode:
            record["refusal"] = "Lean refused the actual explicit same-SCF middle/body/geometry bridge"
        else:
            record["transitive_axioms"] = _audit(result.stdout, COMPOSITION_ROOTS)
            record["lean_composition_checked"] = True
        (output / "COMPOSITION.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        return record
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                dict(
                    error=str(error),
                    type=type(error).__name__,
                    lean_composition_checked=False,
                    python_projection_trusted=True,
                ),
                indent=2,
            )
            + "\n"
        )
        raise

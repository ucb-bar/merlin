"""Bind supported actual typed state reindexing to the common-primitive theorem."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root

from . import lean_graph_state as stable
from .lean_graph_state_chain_kernel import check_query


def lean_string(value):
    if type(value) is not str:
        raise ValueError("exact actual typed primitive string required")
    return json.dumps(value, ensure_ascii=False)


def lean_list(values, emit):
    if type(values) is not list:
        raise ValueError("exact complete primitive list required")
    return "[" + ",".join(emit(value) for value in values) + "]"


def lean_naturals(values):
    return str(stable._naturals(values))


def primitive(value):
    if type(value) is not dict or set(value) != {"core", "inputs", "outputs"}:
        raise ValueError("complete actual primitive descriptor required")
    core = value["core"]
    if type(core) is not dict or set(core) != {
        "operation",
        "operandTypes",
        "resultTypes",
        "attributes",
        "physicalInputs",
        "physicalOutputs",
    }:
        raise ValueError("complete typed opcode/metadata/physical slot description required")
    fields = []
    for key in ("operation", "operandTypes", "resultTypes", "attributes", "physicalInputs", "physicalOutputs"):
        text = lean_string(core[key]) if key in ("operation", "attributes") else lean_list(core[key], lean_string)
        fields.append(key + " := " + text)
    return (
        "{core := {"
        + ",".join(fields)
        + "},inputs := "
        + lean_naturals(value["inputs"])
        + ",outputs := "
        + lean_naturals(value["outputs"])
        + "}"
    )


def cells(value):
    if value["kind"] == "physical" and set(value) == {"kind", "identity"}:
        return ".physical " + lean_string(value["identity"])
    if value["kind"] in ("temporary", "argument") and set(value) == {"kind", "logical_identity", "type"}:
        return "." + value["kind"] + " " + lean_string(value["logical_identity"]) + " " + lean_string(value["type"])
    raise ValueError("actual source-relative SSA type or fixed physical identity required")


def footprint(value):
    if type(value) is not dict or set(value) != {"reads", "writes"}:
        raise ValueError("complete actual footprint required")
    return "⟨" + lean_naturals(value["reads"]) + "," + lean_naturals(value["writes"]) + "⟩"


def definitions(projection, name):
    if len(projection["source_primitives"]) > 8192 or len(projection["source_cells"]) > 65536:
        raise ValueError("explicit transport source/cell budget exceeded")
    mapping = projection["renaming"]
    return (
        f"namespace {name}\n"
        "def mapping : Renaming := {"
        + ",".join(key + " := " + lean_naturals(mapping[key]) for key in ("forward", "backward", "physical"))
        + "}\n"
        "def source : List Primitive := " + lean_list(projection["source_primitives"], primitive) + "\n"
        "def target : List Primitive := " + lean_list(projection["target_primitives"], primitive) + "\n"
        "def sourceCells : List Cell := " + lean_list(projection["source_cells"], cells) + "\n"
        "def targetCells : List Cell := " + lean_list(projection["target_cells"], cells) + "\n"
        "def sourceFootprints : List Footprint := " + lean_list(projection["source_footprints"], footprint) + "\n"
        "def targetFootprints : List Footprint := " + lean_list(projection["target_footprints"], footprint) + "\n"
        "def order : List Nat := " + lean_naturals(projection["order"]) + "\n"
        "theorem accepted : checkTransport mapping source target order sourceCells targetCells = true := by decide\n"
        "theorem source_supported : checkPrimitives sourceFootprints source = true := by decide\n"
        "theorem target_supported : checkPrimitives targetFootprints target = true := by decide\n"
        "theorem source_models {Value : Type} (interpret : Core → List Value → Nat → Value) :\n"
        "    Merlin.GraphState.Models sourceFootprints (run interpret source) :=\n"
        "  checkPrimitives_models interpret source_supported\n"
        "theorem target_models {Value : Type} (interpret : Core → List Value → Nat → Value) :\n"
        "    Merlin.GraphState.Models targetFootprints (run interpret target) :=\n"
        "  checkPrimitives_models interpret target_supported\n"
        "theorem actual_transport {Value : Type} (interpret : Core → List Value → Nat → Value)\n"
        "    (initial : State Value) :\n"
        "    pull (rename mapping.forward) (execute (target.map (step interpret)) initial) =\n"
        "    execute ((permute source order).map (step interpret)) (pull (rename mapping.forward) initial) :=\n"
        "  checkTransport_sound accepted interpret initial\n"
        f"end {name}\n"
    )


def emit_query(projection):
    return (
        "import Merlin.GraphStateChain\nnamespace Merlin.GraphStateChainTransportInstance\n"
        "open Merlin.StateEffects Merlin.GraphState.Chain\n"
        + definitions(projection, "Actual")
        + "#print axioms Actual.actual_transport\nend Merlin.GraphStateChainTransportInstance\n"
    )


def emit_composition(first, second, first_stage, second_stage):
    """Two independently checked schedules, joined by checked semantic transport."""
    text = "import Merlin.GraphStateChain\nnamespace Merlin.GraphStateChainCompositionInstance\n"
    text += "open Merlin.StateEffects Merlin.GraphState.Chain\n"
    for name, projection, stage in (("First", first, first_stage), ("Second", second, second_stage)):
        text += definitions(projection, name)
        moves = []
        for move in stage["projection"]["exchanges"]:
            stable._naturals([move[key] for key in ("position", "first", "second")])
            moves.append("⟨" + ",".join(str(move[key]) for key in ("position", "first", "second")) + "⟩")
        text += f"namespace {name}\n"
        text += "def exchanges : List Merlin.GraphState.Exchange := [" + ",".join(moves) + "]\n"
        text += "theorem schedule : Merlin.GraphState.check sourceFootprints order exchanges = true := by decide\n"
        text += "theorem mapping_checked : checkRenaming mapping = true := by decide\n"
        text += (
            "theorem independently_preserved {Value : Type} (interpret : Core → List Value → Nat → Value)\n"
            "    (initial : State Value) :\n"
            "    execute (source.map (step interpret)) (pull (rename mapping.forward) initial) =\n"
            "      pull (rename mapping.forward) (execute (target.map (step interpret)) initial) := by\n"
            "  have preserved := Merlin.GraphState.check_sound schedule Value (run interpret source)\n"
            "    (source_models interpret) [] [] (pull (rename mapping.forward) initial)\n"
            "  change execute (source.map (step interpret)) (pull (rename mapping.forward) initial) =\n"
            "    execute ((permute source order).map (step interpret))\n"
            "      (pull (rename mapping.forward) initial) at preserved\n"
            "  exact preserved.trans (actual_transport interpret initial).symm\n"
        )
        text += f"end {name}\n"
    text += "theorem middle_primitives : First.target = Second.source := by decide\n"
    text += "def firstMiddleNames : List String := " + lean_list(first["target_cell_names"], lean_string) + "\n"
    text += "def secondMiddleNames : List String := " + lean_list(second["source_cell_names"], lean_string) + "\n"
    text += "theorem middle_cell_domain : firstMiddleNames = secondMiddleNames := by decide\n"
    text += (
        "theorem independently_composed {Value : Type} (interpret : Core → List Value → Nat → Value)\n"
        "    (initial : State Value) :\n"
        "    execute (First.source.map (step interpret))\n"
        "      (pull (rename First.mapping.forward) (pull (rename Second.mapping.forward) initial)) =\n"
        "    pull (rename First.mapping.forward) (pull (rename Second.mapping.forward)\n"
        "      (execute (Second.target.map (step interpret)) initial)) := by\n"
        "  have first := First.independently_preserved interpret (pull (rename Second.mapping.forward) initial)\n"
        "  have second := Second.independently_preserved interpret initial\n"
        "  rw [←middle_primitives] at second\n"
        "  exact first.trans (congrArg (pull (rename First.mapping.forward)) second)\n"
        "theorem fixed_observations {Value : Type} (state : State Value) (key : Nat)\n"
        "    (first : key ∈ First.mapping.physical) (second : key ∈ Second.mapping.physical) :\n"
        "    pull (rename First.mapping.forward) (pull (rename Second.mapping.forward) state) key = state key := by\n"
        "  rw [physical_observation First.mapping_checked _ key first,\n"
        "    physical_observation Second.mapping_checked state key second]\n"
        "#print axioms independently_composed\n#print axioms fixed_observations\n"
        "end Merlin.GraphStateChainCompositionInstance\n"
    )
    return text


def check_composition(boundaries, destination, *, reference_root, contract):
    if len(boundaries) != 2:
        raise ValueError("supported independent composition needs exactly two actual boundaries")
    middle_out, middle_in = (stable.file_ref(path) for path in (boundaries[0][1], boundaries[1][0]))
    if any(middle_out[key] != middle_in[key] for key in ("sha256", "bytes")):
        raise ValueError("independent composition needs byte-identical actual middle artifacts")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    stages, projections, actual = [], [], []
    for index, (before, after) in enumerate(boundaries):
        path = destination / f"transport-{index}"
        report = check_transport(before, after, path, reference_root=reference_root, contract=contract)
        stages.append(report)
        if report["status"] != "kernel_model_checked":
            result = {
                "schema": "merlin.graph-state-chain.independent.v1",
                "status": report["status"],
                "failed_stage": index,
                "reason": report.get("reason", "transport kernel or audit unavailable"),
                "public_acceptance_issued": False,
            }
            stable._write(destination / "report.json", result)
            return result
        projections.append(json.loads((path / "projection/transport.json").read_text()))
        actual.append(json.loads((path / "projection/actual/capture.json").read_text()))
    kernel = check_query(
        emit_composition(*projections, *actual),
        destination / "kernel",
        module="GraphStateChainCompositionInstance",
        roots=[
            "Merlin.GraphStateChainCompositionInstance.First.independently_preserved",
            "Merlin.GraphStateChainCompositionInstance.Second.independently_preserved",
            "Merlin.GraphStateChainCompositionInstance.independently_composed",
            "Merlin.GraphStateChainCompositionInstance.fixed_observations",
        ],
    )
    result = {
        "schema": "merlin.graph-state-chain.independent.v1",
        "status": kernel["status"],
        "theorem": "Merlin.GraphStateChainCompositionInstance.independently_composed",
        "kernel": stable.file_ref(destination / "kernel/report.json"),
        "stages": [stable.file_ref(destination / f"transport-{index}/report.json") for index in range(2)],
        "middle_artifacts": [middle_out, middle_in],
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "stage_state_numberings_identified": False,
        "mapping": "explicit checked finite bijections with fixed physical cells, "
        "common total primitive interpretation",
        "operational_fidelity": "external premise; no xDSL/Kernel bridge, poison, runtime/native or numerical claim",
    }
    stable._write(destination / "report.json", result)
    return result


def check_transport(before, after, destination, *, reference_root, contract):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    originals = [stable.file_ref(path) for path in (before, after, contract)]
    retained = []
    for record, name in zip(originals, ("before.mlir", "after.mlir", "contract.json"), strict=True):
        path = destination / name
        path.write_bytes(Path(record["path"]).read_bytes())
        identity = stable.file_ref(path)
        if any(identity[key] != record[key] for key in ("sha256", "bytes")):
            raise ValueError("transport original/snapshot bytes differ")
        retained.append(identity)
    capture = repo_root() / "src/merlin/verify/lean_graph_state_chain_capture.py"
    own = [stable.file_ref(path) for path in (__file__, capture)]
    command = [
        sys.executable,
        str(capture),
        "--reference",
        str(reference_root),
        "--source",
        retained[0]["path"],
        "--target",
        retained[1]["path"],
        "--contract",
        retained[2]["path"],
        "--destination",
        str(destination / "projection"),
    ]
    with (destination / "projection.stdout").open("x") as out, (destination / "projection.stderr").open("x") as err:
        produced = run_owned(
            command,
            cwd=reference_root,
            env={**os.environ, "PYTHONPATH": str(Path(reference_root) / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    result = {
        "schema": "merlin.graph-state-chain.transport.v1",
        "inputs": originals,
        "snapshots": retained,
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "primitive_interpreter": "one common arbitrary total pure interpretation of exact typed "
        "opcode/metadata and ordered slots",
        "operational_fidelity": "explicit external premise; no source-definedness/poison or hardware claim",
    }
    if produced.returncode:
        result.update(status="unavailable", reason="actual transport projection process failed")
    else:
        projection = json.loads((destination / "projection/transport.json").read_text())
        if not projection["projection_checked"]:
            result.update(status="unsupported", reason=projection["reason"])
        else:
            measured = check_query(
                emit_query(projection),
                destination / "kernel",
                module="GraphStateChainTransportInstance",
                roots=[
                    "Merlin.GraphState.Chain.checkTransport_sound",
                    "Merlin.GraphState.Chain.checkPrimitives_models",
                    "Merlin.GraphStateChainTransportInstance.Actual.actual_transport",
                ],
            )
            actual = json.loads((destination / "projection/actual/capture.json").read_text())
            dependencies = [
                *own,
                *originals,
                *retained,
                *actual["producer_sources"],
                actual["owned_generator"],
                *projection["external_runtime"]["loaded_sources"],
                projection["external_runtime"]["python"],
            ]
            if not stable.fresh(dependencies):
                raise ValueError("transport actual IR, model projection or selected source/runtime changed")
            result.update(
                status=measured["status"],
                projection=stable.file_ref(destination / "projection/transport.json"),
                kernel=stable.file_ref(destination / "kernel/report.json"),
                dependencies=dependencies,
                renumbered_cells=sum(index != mapped for index, mapped in enumerate(projection["renaming"]["forward"])),
            )
    stable._write(destination / "report.json", result)
    return result

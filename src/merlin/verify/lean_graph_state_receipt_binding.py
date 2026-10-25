"""Exact retained buffers for the separately proved Receipt binding checker."""

from __future__ import annotations

import json
from pathlib import Path

from . import lean_graph_state as stable
from .lean_graph_state_chain_kernel import check_query
from .lean_graph_state_chain_transport import footprint, lean_list

STAGE_NAMES = ("stage-000-before.mlir", "stage-000-after.mlir", "stage-001-before.mlir", "stage-001-after.mlir")


def buffer_text(data):
    # The Lean encode/decode round-trip proves this representation is injective
    # on legal byte buffers. The explicit length preserves leading zero bytes.
    return "⟨" + str(len(data)) + ",0x" + format(int.from_bytes(data, "little"), "x") + "⟩"


def buffers_text(values):
    return "[" + ",".join(buffer_text(value) for value in values) + "]"


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def emit_query(binding, projection):
    fields = ["version := 1"]
    for name, value in binding.items():
        if name in ("originalStages", "snapshotStages", "originalProofSources", "snapshotProofSources"):
            text = buffers_text(value)
        else:
            text = buffer_text(value)
        fields.append(name + " := " + text)
    fields += ["approved := false", "nativeQualified := false", "numericalGatesDischarged := false"]
    effects = [{"reads": node["reads"], "writes": node["writes"]} for node in projection["nodes"]]
    moves = [
        "⟨" + ",".join(str(move[key]) for key in ("position", "first", "second")) + "⟩"
        for move in projection["exchanges"]
    ]
    return (
        "import Merlin.GraphStateReceipt\nnamespace Merlin.GraphStateReceiptInstance\n"
        "open Merlin.StateEffects Merlin.GraphState.Receipt\n"
        "set_option maxRecDepth 2000000\nset_option maxHeartbeats 8000000\n"
        "def binding : Binding := {" + ",".join(fields) + "}\n"
        "def footprints : List Footprint := " + lean_list(effects, footprint) + "\n"
        "def target : List Nat := " + str(stable._naturals(projection["order"])) + "\n"
        "def exchanges : List Merlin.GraphState.Exchange := [" + ",".join(moves) + "]\n"
        "theorem accepted : check binding footprints target exchanges = true := by decide\n"
        "theorem exactly_bound : valid binding := (check_sound accepted).1\n"
        "theorem raw_stage_bytes : binding.originalStages.map Bytes.decode =\n"
        "    binding.snapshotStages.map Bytes.decode := by\n"
        "  have checks : checkBinding binding = true ∧\n"
        "      Merlin.GraphState.check footprints target exchanges = true := by\n"
        "    simpa only [check,Bool.and_eq_true] using accepted\n"
        "  exact exact_stage_bytes checks.1\n"
        "theorem direct_composite : Merlin.GraphState.Preserves footprints target := (check_sound accepted).2\n"
        "#print axioms exactly_bound\n#print axioms direct_composite\nend Merlin.GraphStateReceiptInstance\n"
    )


def check_binding(root, original_paths, contract_path, source_pairs, preceding_queries):
    root = Path(root).resolve()
    snapshots = root / "snapshots"
    direct_capture = json.loads((root / "checks/direct/complete/projection/capture.json").read_text())
    semantics = canonical(
        {
            "effect_semantics_sha256": direct_capture["effect_semantics_sha256"],
            "contract_identity": direct_capture["contract_sha256"],
            "primitive_model": "checked common typed total interpretation; operational fidelity remains external",
        }
    )
    projection = canonical(direct_capture["projection"])
    tool = Path(stable.LEAN).read_bytes()
    for name, data in (("semantics.json", semantics), ("projection.json", projection), ("selected-tool.bytes", tool)):
        (snapshots / name).write_bytes(data)
    source_originals = [
        Path(row["original"]["path"]).read_bytes() for row in source_pairs if row["original"]["path"].endswith(".lean")
    ]
    source_snapshots = [
        Path(row["snapshot"]["path"]).read_bytes() for row in source_pairs if row["original"]["path"].endswith(".lean")
    ]
    # This query cannot recursively embed its own source. It binds all preceding
    # query sources and mathematical source bytes; its own source/object are
    # separately recompiled and inventoried by the strict live verifier.
    originals = source_originals + [Path(path).read_bytes() for path in preceding_queries]
    copies = source_snapshots + [Path(path).read_bytes() for path in preceding_queries]
    binding = {
        "originalStages": [Path(path).read_bytes() for path in original_paths],
        "snapshotStages": [(snapshots / name).read_bytes() for name in STAGE_NAMES],
        "originalContract": Path(contract_path).read_bytes(),
        "snapshotContract": (snapshots / "contract.json").read_bytes(),
        "originalSemantics": semantics,
        "snapshotSemantics": (snapshots / "semantics.json").read_bytes(),
        "originalTool": tool,
        "snapshotTool": (snapshots / "selected-tool.bytes").read_bytes(),
        "originalProofSources": originals,
        "snapshotProofSources": copies,
        "originalProjection": projection,
        "snapshotProjection": (snapshots / "projection.json").read_bytes(),
    }
    return check_query(
        emit_query(binding, direct_capture["projection"]),
        root / "checks/binding",
        module="GraphStateReceiptInstance",
        roots=[
            "Merlin.GraphState.Receipt.check_sound",
            "Merlin.GraphStateReceiptInstance.exactly_bound",
            "Merlin.GraphStateReceiptInstance.raw_stage_bytes",
            "Merlin.GraphStateReceiptInstance.direct_composite",
        ],
    )

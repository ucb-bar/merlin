"""Finite, initialized actual-loop witness CANDIDATE; never gate authority."""
import json
from dataclasses import asdict

from .lean_graph_pipeline_gate_projection import GateUnsupported, project_lean_sources


def witness(prepared):
    projection = json.loads(prepared.projections[-1])
    functions = [(i, f) for i, f in enumerate(projection["functions"]) if not f["declaration"]
                 and f["inputs"] == [{"bits": 8}, {"bits": 8}] and f["outputs"] == [{"bits": 8}]]
    if len(functions) != 1:
        raise GateUnsupported("one concrete two-input i8 loop witness required")
    index, function = functions[0]
    constants = []
    for row in projection["registry"]:
        if row["index_constant"] is not None and row["inputs"] == [] and row["outputs"] == [{"index": True}]:
            constants.append("some (" + str(row["index_constant"]) + ")")
        elif (row["descriptor"]["name"] == "func.call" and row["inputs"] == [{"bits": 8}]
              and row["outputs"] == [{"bits": 8}]):
            constants.append("none")
        else:
            raise GateUnsupported("witness supports actual index constants and unary i8 external observers only")
    if not any("loop" in row for row in function["before"]):
        raise GateUnsupported("actual source loop required")
    # Proposed outcomes are intentionally fixed to this bounded worked example.
    # rfl must independently derive them from actual source, not a host evaluator
    # or a source-success hypothesis. A different example must prove new values.
    query = "import Merlin.GraphStateIntegerOperations\n" + prepared.query
    query += '''namespace Merlin.GraphGateChain.Witness
open Merlin.GraphState.Scalar
abbrev registry := Merlin.GraphGateChain.Endpoint.Entries.registry
'''
    query += "def constants : List (Option Int) := [" + ",".join(constants) + "]\n"
    query += '''def chosen (fail : Bool) : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun lower upper stride =>
    if 0 < stride then
      (List.range ((upper - lower + stride - 1) / stride).toNat).map
        (fun k => lower + (k : Int) * stride)
    else []
  effect := fun identity arguments world =>
    if (constants[identity]?.getD none).isSome then
      .ok ((signature registry identity).outputs.map
        (fun _ => (constants[identity]?.getD none).getD 0),world)
    else if fail then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success; simp only [List.length_map]
    · split at success
      · cases success
      · cases success; simp only [List.length_map]
'''
    roots = list(prepared.roots)
    base = f"Merlin.GraphGateChain.Endpoint.Scalar.F{index}"
    query += "theorem finite_control : (chosen false).iterations 0 3 1 = [0,1,2] := rfl\n"
    roots.append("Merlin.GraphGateChain.Witness.finite_control")
    for label, args, result in (("success", "[3,5]", ".ok ([195],11)"),
                                ("success_other", "[4,5]", ".ok ([247],11)"),
                                ("error", "[3,5]", ".error (.external 17,8)")):
        fail = "true" if label == "error" else "false"
        query += f"theorem source_{label} : evalProgram (chosen {fail}) {base}.before {args} 7 = {result} := rfl\n"
        query += f"theorem final_{label} : evalProgram (chosen {fail}) {base}.after {args} 7 = {result} := by\n"
        query += f"  rw [← {base}.actual Int Nat Int (chosen {fail}) {args} 7 rfl]\n  exact source_{label}\n"
        for side, component in (("source", "1"), ("final", "2")):
            program = "before" if side == "source" else "after"
            query += f"theorem {side}_{label}_initialized : Initialized (chosen {fail}) {base}.{program} {args} 7 :=\n"
            query += f"  ({base}.operands_initialized (chosen {fail}) {args} 7 rfl).{component}\n"
        roots += ["Merlin.GraphGateChain.Witness." + name for name in (
            "source_" + label, "final_" + label, "source_" + label + "_initialized", "final_" + label + "_initialized")]
    query += "end Merlin.GraphGateChain.Witness\n"
    return query, tuple(roots), {
        "state": "UNCOMPILED", "native": "UNQUALIFIED", "model_approved": False,
        "source_success_assumed": False, "finite_control": "range(ceil((upper-lower)/positive-stride)), empty otherwise",
        "input": [3, 5], "success_proposal": {"values": [195], "world": 11},
        "second_input": [4, 5], "second_proposal": {"values": [247], "world": 11},
        "failure_proposal": {"external_error": 17, "world": 8},
        "observer": "every external unary i8 call echoes its operand or fails; world increments once per call",
        "initialization": "candidate existing Scalar initialized theorem; not poison freedom or numerical fidelity",
        "additional_sources": [asdict(r) for r in project_lean_sources(("Merlin.GraphStateIntegerOperations",))],
    }

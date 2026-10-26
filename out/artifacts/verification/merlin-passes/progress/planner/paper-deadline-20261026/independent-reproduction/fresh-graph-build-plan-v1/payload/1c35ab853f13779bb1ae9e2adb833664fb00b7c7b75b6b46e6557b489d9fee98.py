"""Arithmetic-sensitive witness candidates on actual projected scalar programs.

The host computes proposed literal results; Lean must check them independently.
These are neither approval of the primitive model nor native execution evidence.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .lean_graph_pipeline_gate_projection import GateUnsupported, project_lean_sources


def evaluate(records, arguments, *, fail=False):
    """Small example evaluator; never consulted by the compiler authority gate."""
    env, world = list(arguments), 7
    for row in records:
        if "ret" in row:
            return {"values": [env[i] for i in row["ret"]], "world": world}
        args = [env[i] for i in row["arguments"]]
        if "effect" in row:
            world += 1
            if fail:
                return {"error": 17, "world": world}
            env = [args[0]] + env
            continue
        op = row["pure"]
        kind = op["kind"]
        if kind == "constant":
            value = op["value"]
        elif kind == "add":
            value = args[0] + args[1]
        elif kind == "sub":
            value = args[0] - args[1]
        elif kind == "mul":
            value = args[0] * args[1]
        else:
            raise GateUnsupported("example mode supports scalar constants/add/sub/mul only")
        env.insert(0, value % (1 << op["width"]))
    raise GateUnsupported("example has no complete return")


def witness(prepared):
    projection = json.loads(prepared.projection)
    bits = {"bits": 8}
    if any(r["inputs"] != [bits] or r["outputs"] != [bits] for r in projection["registry"]):
        raise GateUnsupported("echo witness needs only i8-to-i8 opaque descriptors")
    functions = [(i, f) for i, f in enumerate(projection["functions"]) if not f["declaration"]
                 and f["inputs"] == [bits, bits] and f["outputs"] == [bits]]
    if len(functions) != 1:
        raise GateUnsupported("echo witness needs one two-input i8 entry")
    index, function = functions[0]
    for side in ("before", "after"):
        rows = function[side]
        if any("loop" in r for r in rows) or sum("effect" in r for r in rows) != 1:
            raise GateUnsupported("echo witness needs loop-free arithmetic and one retained observer")
    candidates = [(arguments, evaluate(function["before"], arguments)) for arguments in ([3, 5], [4, 5])]
    if candidates[0][1] == candidates[1][1] or any(not r["values"] or r["values"] == [0] for _, r in candidates):
        raise GateUnsupported("example must demonstrate nonzero arithmetic-sensitive observations")
    query = "import Merlin.GraphStateIntegerOperations\n" + prepared.query
    query += '''namespace Merlin.GraphPipelineGateWitness
open Merlin.GraphState.Scalar
abbrev registry := Merlin.GraphStateScalarInstance.registry
def chosen (fail : Bool) : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun identity arguments world =>
    if fail then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
'''
    roots = list(prepared.roots)
    for n, (arguments, result) in enumerate(candidates):
        values = str(arguments).replace(" ", "")
        expected = ".ok (" + str(result["values"]).replace(" ", "") + "," + str(result["world"]) + ")"
        query += f"theorem source_success{n} : evalProgram (chosen false) Merlin.GraphStateScalarInstance.F{index}.before {values} 7 = {expected} := rfl\n"
        query += f"theorem target_success{n} : evalProgram (chosen false) Merlin.GraphStateScalarInstance.F{index}.after {values} 7 = {expected} := by\n"
        query += f"  rw [← Merlin.GraphStateScalarInstance.F{index}.actual Int Nat Int (chosen false) {values} 7 rfl]\n  exact source_success{n}\n"
        roots += [f"Merlin.GraphPipelineGateWitness.{side}_success{n}" for side in ("source", "target")]
    query += f'''theorem source_failure : evalProgram (chosen true) Merlin.GraphStateScalarInstance.F{index}.before [3,5] 7 = .error (.external 17,8) := rfl
theorem target_failure : evalProgram (chosen true) Merlin.GraphStateScalarInstance.F{index}.after [3,5] 7 = .error (.external 17,8) := by
  rw [← Merlin.GraphStateScalarInstance.F{index}.actual Int Nat Int (chosen true) [3,5] 7 rfl]
  exact source_failure
theorem source_initialized : Initialized (chosen false) Merlin.GraphStateScalarInstance.F{index}.before [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F{index}.operands_initialized (chosen false) [3,5] 7 rfl).1
theorem target_initialized : Initialized (chosen false) Merlin.GraphStateScalarInstance.F{index}.after [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F{index}.operands_initialized (chosen false) [3,5] 7 rfl).2
end Merlin.GraphPipelineGateWitness
'''
    roots += ["Merlin.GraphPipelineGateWitness." + label for label in (
        "source_failure", "target_failure", "source_initialized", "target_initialized")]
    return query, tuple(roots), {
        "state": "UNCOMPILED_CANDIDATE", "function_roster_index": index,
        "query_contract": "pure_declarations_separate_root_audit_v1",
        "model": "exact wrapping integer primitive; observer echoes its argument and advances world",
        "success_proposals": [{"input": arguments, "result": result} for arguments, result in candidates],
        "failure_proposal": {"input": [3, 5], "error": 17, "world": 8},
        "source_success_assumed": False, "proof_checked": False,
        "additional_project_sources": [asdict(ref) for ref in project_lean_sources(("Merlin.GraphStateIntegerOperations",))],
        "model_approved": False, "native_or_poison_fidelity_established": False,
    }

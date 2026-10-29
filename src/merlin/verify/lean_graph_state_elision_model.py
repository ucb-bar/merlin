"""Actual-roster initialized Int fixture witness; no physical-runtime admission."""

from __future__ import annotations

import json

from .lean_graph_state_elision import emit_query, integer


def operands(atom):
    record = json.loads(atom["token"])
    if record[0] != "merlin.sched.ir.nodes.Call":
        # The pinned dataclass module is inspected, never guessed by an opcode.
        if not record[0].endswith(".Call"):
            raise ValueError("actual complete Call token required")
    fields = dict(record[1])
    name = fields["instr"][1]
    result = {}
    for entry in fields["args"][1]:
        key, value = entry[1]
        encoded = dict(value[1])["value"]
        if encoded[0] != "int":
            raise ValueError("initialized model witness supports actual integer operands only")
        result[key[1]] = int(encoded[1])
    return name, result


def emit_witness(projection):
    if projection["configuration_fields"] != ["mode"]:
        raise ValueError("this fixture witness requires one actual mode field")
    rows = [operands(atom) for atom in projection["before"]]
    if any(name not in ("setup", "work") for name, _ in rows):
        raise ValueError("fixture witness supports actual setup/work instructions only")
    definitions = []
    cases = []
    for atom, (name, args) in zip(projection["before"], rows, strict=True):
        ident = atom["id"]
        if name == "setup":
            body = f".ok (assign (fun x => x) [(0,{integer(args['value'])})] state)"
        else:
            body = (
                f"if state 0 = {integer(args['required'])} then .ok (fun key =>\n"
                f"    if key = 1 then state 1 + {integer(args['add'])} else state key)\n"
                "  else .error (17,state)"
            )
        definitions.append(f"def step{ident} (state : State Int) : Except (Int × State Int) (State Int) := {body}\n")
        cases.append(f"| {ident} => step{ident} state")
    core = emit_query(projection).removesuffix("end Merlin.GraphStateElisionInstance\n")
    steps = ",".join("step" + str(atom["id"]) for atom in projection["before"])
    mode, observation, failed = -13, 2, False
    for name, args in rows:
        if name == "setup":
            mode = args["value"]
        elif mode == args["required"]:
            observation += args["add"]
        else:
            failed = True
            break
    outcome = (
        "theorem actual_failure : ∃ terminal, execute run before initial = .error (17,terminal) ∧ "
        f"terminal 0 = {integer(mode)} ∧ terminal 1 = {integer(observation)} := by\n"
        "  exact ⟨_,rfl,by decide,by decide⟩\n#print axioms actual_failure\n"
        if failed
        else "theorem actual_success : ∃ final, execute run before initial = .ok final ∧ "
        f"final 0 = {integer(mode)} ∧ final 1 = {integer(observation)} := by\n"
        "  exact ⟨_,rfl,by decide,by decide⟩\n#print axioms actual_success\n"
    )
    return (
        core
        + "\n"
        + "".join(definitions)
        + (
            "def run (index : Nat) (state : State Int) : Except (Int × State Int) (State Int) :=\n"
            "  match index with\n  " + "\n  ".join(cases) + "\n  | _ => .error (19,state)\n"
            "theorem model_admitted : ∀ atom ∈ before, Admitted fields (fun x : Int => x) run atom := by\n"
            "  intro atom member\n"
            "  simp only [before,List.mem_cons,List.not_mem_nil,or_false] at member\n"
            "  rcases member with " + " | ".join("h" + str(i) for i in range(len(rows))) + "\n"
            "  all_goals subst atom\n"
            "  all_goals simp [Admitted,Merlin.GraphState.Elision.pure,fields,run," + steps + ",assign]\n"
            "  all_goals first\n"
            "    | (intro initial key value found; simp only [found])\n"
            "    | (intro initial final success; split at success <;> simp_all; subst final; simp_all)\n"
            "def initial : State Int := fun key => if key = 0 then -13 else if key = 1 then 2 else 19\n"
            "theorem joint_model : execute run before initial = execute run after initial :=\n"
            "  actual_elision Int Int (fun x => x) run model_admitted initial\n"
            + outcome
            + "#print axioms model_admitted\n#print axioms joint_model\n"
            "end Merlin.GraphStateElisionInstance\n"
        )
    )

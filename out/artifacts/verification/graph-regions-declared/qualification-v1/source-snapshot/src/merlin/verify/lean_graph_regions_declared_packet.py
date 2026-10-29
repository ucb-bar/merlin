"""Propose complete actual scalar/source/call binding queries.

Host code is a named literal serializer, not an execution proof. Lean recomputes
SSA graph projection, scalar eligibility, layouts and physical nonaliasing.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from merlin.verify import lean_graph_regions_declared_ast as ast
from merlin.verify.lean_graph_regions_declared_check import propose


def load(root, name):
    return json.loads((root / name).read_text())


def strings(values):
    return ast.array(values, ast.text)


def type_tag(value):
    if value["kind"] == "int":
        return "i" + str(value["width"])
    if value["kind"] == "tensor" and value["element"]["kind"] == "int" and value["encoding"] is None:
        return (
            "tensor["
            + ",".join("?" if d is None else str(d) for d in value["shape"])
            + "]i"
            + str(value["element"]["width"])
        )
    return "unsupported-context-type"


def window(value, backings):
    return f"⟨{backings[value['space']]},{ast.natural(value['start'])},{ast.natural(value['size'])}⟩"


def pairs(values, encode=ast.natural):
    return ast.array(values, lambda pair: f"({ast.text(pair[0])},{encode(pair[1])})")


def raw_call(value):
    if set(value) != {"instr", "args", "assumes", "awaits", "produces", "sets", "stages", "unit"}:
        raise ValueError("complete raw call fields required")
    arguments = []
    for key, val in value["args"]:
        if set(val) != {"value"}:
            raise ValueError("only original constant raw call arguments supported")
        arguments.append([key, val["value"]])
    return (
        f"⟨{ast.text(value['instr'])},{pairs(arguments)},{strings(value['assumes'])},{strings(value['awaits'])},"
        f"{ast.optional(value['produces'], ast.text)},{strings(value['sets'])},{strings(value['stages'])},"
        f"{ast.optional(value['unit'], ast.text)}⟩"
    )


def raw_kernel(value):
    if set(value) != {"name", "args", "attrs", "body"}:
        raise ValueError("complete raw Kernel fields required")
    args = []
    for arg in value["args"]:
        if set(arg) != {"name", "shape", "dtype", "access"}:
            raise ValueError("complete tensor argument required")
        args.append(
            f"⟨{ast.text(arg['name'])},{ast.array(arg['shape'], ast.natural)},"
            f"{ast.text(arg['dtype'])},{ast.text(arg['access'])}⟩"
        )
    return (
        f"⟨{ast.text(value['name'])},[{','.join(args)}],{pairs(value['attrs'], ast.text)},"
        f"{ast.array(value['body'], raw_call)}⟩"
    )


def packet(capture: Path, documents: dict):
    before = documents["before"]["module"]["ops"][0]
    block = before["regions"][0]["blocks"][0]
    operations = block["ops"]
    generic_indices = [index for index, op in enumerate(operations) if op["name"] == "linalg.generic"][:3]
    if len(generic_indices) != 3:
        raise ValueError("requires actual producer and both consumers")
    software = yaml.safe_load((capture / "selected/software.yaml").read_text())
    storage = software["scheduling"]["storage"]
    resources = storage["banks"] | storage["external"]
    backing_ids = {name: index for index, name in enumerate(sorted({row["backing"] for row in resources.values()}))}
    backings = {name: backing_ids[row["backing"]] for name, row in resources.items()}
    resolved_resources = {row["resource"]: row for row in load(capture, "selected-storage-windows.json")}
    facts = load(capture, "selected/facts.json")["facts"]
    effects = load(capture, "actual-output-effects.json")
    kernel = load(capture, "actual-output-kernel.json")
    original = load(capture, "original-provider-kernel.json")
    selected = load(capture, "selected-storage-declarations.json")
    fusion = load(capture, "selected-fusion-declarations.json")["elementwise"]
    case = load(capture, "CAPTURE.json")["case"]
    elements = 1
    for dimension in case["shape"]:
        elements *= dimension
    types = {value["id"]: value["type"] for value in block["args"]}
    for op in operations:
        types.update({value["id"]: value["type"] for value in op["results"]})
    declarations = [
        "import Merlin.GraphRegionsDeclaredBridge",
        "import Merlin.GraphRegionsDeclaredRaw",
        "import Merlin.GraphRegionsDeclaredProjection",
        "open Merlin MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared",
        f"def actualBefore : Module := {ast.module(documents['before'])}",
        f"def actualRetained : Module := {ast.module(documents['retained'])}",
    ]
    addresses, known = {}, []
    site_terms = []
    for i, ordinal in enumerate(generic_indices):
        op, effect, call = operations[ordinal], effects[i], kernel["body"][i]
        program, source_name, source_visibility = propose(documents[f"stage-{i}-source"])
        scalar = op["regions"][0]["blocks"][0]
        inputs = op["operands"][:-1]
        output = op["results"][0]["id"]
        names = [arg["id"] for arg in scalar["args"][:-1]] + [
            value["id"] for local in scalar["ops"][:-1] for value in local["results"]
        ]
        stage = (
            f"⟨{ordinal},{ast.text(effect['semantic_key'])},{ast.array(inputs, ast.natural)},{output},"
            f"{strings([type_tag(types[slot]) for slot in inputs])},{ast.text(type_tag(types[output]))},{program}⟩"
        )
        declarations.append(f"def stage{i} : ScalarStage := {stage}")
        declarations.append(f"def program{i} : Program := {program}")
        for role in ("source", "adapter", "selected"):
            declarations.append(f"def scalar{i}_{role} : Module := {ast.module(documents[f'stage-{i}-{role}'])}")
        for role in ("adapter", "selected"):
            (func,) = documents[f"stage-{i}-{role}"]["module"]["ops"]
            name = func["attrs"]["sym_name"]["value"]
            visibility = func["attrs"].get("sym_visibility", {}).get("value")
            declarations.append(
                f"theorem scalar{i}_{role}_accepted : checkScalar scalar{i}_source scalar{i}_{role} "
                f"{ast.text(source_name)} {ast.text(name)} {ast.optional(source_visibility, ast.text)} "
                f"{ast.optional(visibility, ast.text)} program{i} = true := by\n"
                "  apply checkScalar_iff.mpr\n  exact ⟨rfl, rfl, by decide⟩"
            )
        site_terms.append(
            f"⟨stage{i},{ast.array(case['shape'], ast.natural)},{ast.array(names, ast.natural)},"
            f"{op['operands'][-1]},{scalar['args'][-1]['id']}⟩"
        )
        accesses = effect["accesses"]
        for slot, access in zip(inputs, accesses[:-1], strict=True):
            win = window(access, backings)
            if slot in addresses and addresses[slot] != win:
                raise ValueError("logical SSA value has inconsistent actual physical windows")
            if slot not in addresses:
                known.append(slot)
                addresses[slot] = win
        addresses[output] = window(accesses[-1], backings)
        args = [[key, val["value"]] for key, val in call["args"]]
        bound = (
            f"⟨{ast.text(call['instr'])},{ast.text(effect['semantic_key'])},{pairs(args)},"
            f"{ast.array(accesses[:-1], lambda access: window(access, backings))},{window(accesses[-1], backings)},"
            f"{strings(effect['elementwise']['byte_orders'])},{ast.array(case['shape'], ast.natural)},"
            f"{8 * case['width']},{ast.text(effect['completion'])},{strings(effect['waits'])},"
            f"{ast.optional(effect['token'], ast.text)},[]⟩"
        )
        declarations.append(f"def bound{i} : BoundCall := {bound}")
        contract = selected[call["instr"]]
        schemas = contract["accesses"]
        choices = schemas[0]["space"]["choices"]
        selectors = (
            "["
            + ",".join(
                f"({c['value']},⟨{backings[c['resource']]},"
                f"{resolved_resources[c['resource']]['start']},{resolved_resources[c['resource']]['size']}⟩)"
                for c in choices
            )
            + "]"
        )
        read_fields = [[a["space"]["argument"], a["start"]["argument"]] for a in schemas[:-1]]
        write = schemas[-1]
        (count_field,) = schemas[0]["size"]["product"]
        schema = (
            f"⟨{ast.text(original['body'][i]['instr'])},"
            f"{ast.text(load(capture, f'stage-{i}-effect.json')['semantic_key'])},"
            f"{strings([key for key, _ in original['body'][i]['args']])},"
            f"{ast.array([facts['argument_bits']] * len(original['body'][i]['args']), ast.natural)},"
            f"{selectors},{pairs(read_fields, ast.text)},"
            f"{ast.text(write['space']['argument'])},{ast.text(write['start']['argument'])},"
            f"{ast.text(count_field['argument'])},{ast.text('width')},"
            f"{strings(fusion[call['instr']]['byte_orders'])},{ast.array(case['shape'], ast.natural)}⟩"
        )
        declarations.append(f"def schema{i} : CallSchema := {schema}")
    declarations.extend(
        [
            f"def sites : List GenericSite := [{','.join(site_terms)}]",
            "def items : List CallStage := [⟨stage0,bound0⟩,⟨stage1,bound1⟩,⟨stage2,bound2⟩]",
            f"def initialKnown : List Nat := {ast.array(known, ast.natural)}",
            "def addresses : Nat → Window := fun slot => "
            + "".join(f"if slot = {slot} then {win} else " for slot, win in addresses.items())
            + "⟨0,0,0⟩",
            f"def actualKernel : RawKernel := {raw_kernel(kernel)}",
            f"def originalKernel : RawKernel := {raw_kernel(original)}",
            "def actualGraph : Graph := (projectModule actualBefore sites).getD ⟨[],[],[],[]⟩",
            "theorem actual_kernel_decoded : checkKernel actualKernel originalKernel.name originalKernel.arguments "
            "[schema0,schema1,schema2] [bound0,bound1,bound2] = true := by decide",
            f"theorem actual_calls_valid : checkSequence {elements} addresses initialKnown items = true := by decide",
            "theorem actual_generic_binding : SourceBinding actualBefore actualRetained sites actualGraph := by\n"
            "  refine ⟨rfl, rfl, ?_⟩\n"
            "  constructor\n"
            "  · simp [SitesMatch, findSite, sites, stage0, stage1, stage2, checkGeneric, genericOperation,\n"
            "      genericWellFormed, renamedOperations, renameSlot, identityMap,\n"
            "      typedIds, typeTag, Op.results, Program.Valid, instructionsValid, finalWidths,\n"
            "      Primitive.Valid, Primitive.inputs, Primitive.width, Primitive.name, Primitive.attrs]\n"
            "    all_goals repeat first | apply And.intro | rfl | decide\n"
            "  · intro site member\n"
            "    simp only [sites, List.mem_cons, List.not_mem_nil, or_false] at member\n"
            "    rcases member with rfl | rfl | rfl <;>\n"
            "      refine ⟨_, rfl, rfl, by decide⟩",
            "#print axioms actual_kernel_decoded",
            "#print axioms actual_calls_valid",
            "#print axioms actual_generic_binding",
        ]
    )
    if (capture / "original-connected-plan.json").exists():
        native_plan = load(capture, "original-connected-plan.json")
        intake = load(capture, "CAPTURE.json")

        def locator_key(value):
            return value["kind"], value["owner_ordinal"], value["index"]

        value_ids = {}
        for row, ordinal in zip(intake["stages"], generic_indices, strict=True):
            op = operations[ordinal]
            for loc, slot in zip(row["source_operands"], op["operands"], strict=True):
                value_ids[locator_key(loc)] = slot
            for loc, result in zip(row["source_results"], op["results"], strict=True):
                value_ids[locator_key(loc)] = result["id"]
        ids = [value_ids[locator_key(value)] for value in load(capture, "original-connected-plan-values.json")]
        plan_stages = []
        for i, row in enumerate(native_plan["stages"]):
            plan_stages.append(
                f"⟨stage{i}.ordinal,stage{i}.role,{ast.array(row['inputs'], ast.natural)},"
                f"[{row['output']}],stage{i}.inputTypes,[stage{i}.outputType]⟩"
            )
        lifetimes = (
            "["
            + ",".join(f"({first},{last})" for first, last in [row["lifetime"] for row in native_plan["values"]])
            + "]"
        )
        plan = (
            f"⟨{ast.array(ids, ast.natural)},[{','.join(plan_stages)}],"
            f"{ast.array(native_plan['inputs'], ast.natural)},"
            f"{ast.array(native_plan['outputs'], ast.natural)},{lifetimes}⟩"
        )
        start = generic_indices[0]
        declarations.extend(
            [
                f"def actualPlan : Plan := {plan}",
                "def actualFused : FusedBody := ⟨items.map (fun item => item.stage.node),"
                "initialKnown,[stage1.output,stage2.output]⟩",
                f"theorem actual_realization : checkRealization actualGraph actualGraph {start} 3 "
                "[stage0.ordinal,stage1.ordinal,stage2.ordinal] actualPlan actualFused = true := by decide",
                "theorem actual_boundary : items.map (fun item => item.stage.node) = "
                f"(actualGraph.nodes.drop {start}).take 3 := by rfl",
                "theorem actual_nodes_unique : ((items.map CallStage.stage).map ScalarStage.node).Nodup := by decide",
                "theorem actual_output_coverage : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node)) "
                f"(actualGraph.nodes.drop {start + 3}) actualGraph.observations, "
                "slot ∈ finalKnown initialKnown items := by decide",
                f"""theorem actual_complete_observers {{α M : Type}} [Inhabited α]
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor {elements} α) (fun _ _ => True))
    (read : M → Window → Option (Tensor {elements} α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor {elements} α) → M)
    (initialized : ∀ entry, Refines read addresses initialKnown entry (pack entry))
    (native : StateEffects.Action
      ⟨liveIns ((actualGraph.nodes.drop {start}).take 3),
        liveOuts ((actualGraph.nodes.drop {start}).take 3)
          (actualGraph.nodes.drop {start + 3}) actualGraph.observations⟩
      (Tensor {elements} α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts ((actualGraph.nodes.drop {start}).take 3)
          (actualGraph.nodes.drop {start + 3}) actualGraph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor {elements} α)) :
    SourceBinding actualBefore actualRetained sites actualGraph ∧
    checkKernel actualKernel originalKernel.name originalKernel.arguments
      [schema0,schema1,schema2] [bound0,bound1,bound2] = true ∧
    (∀ arguments, ScalarExecution primitive scalar0_source arguments (program0.evaluate primitive arguments) ∧
      ScalarExecution primitive scalar0_selected arguments (program0.evaluate primitive arguments)) ∧
    (∀ arguments, ScalarExecution primitive scalar1_source arguments (program1.evaluate primitive arguments) ∧
      ScalarExecution primitive scalar1_selected arguments (program1.evaluate primitive arguments)) ∧
    (∀ arguments, ScalarExecution primitive scalar2_source arguments (program2.evaluate primitive arguments) ∧
      ScalarExecution primitive scalar2_selected arguments (program2.evaluate primitive arguments)) ∧
    (let meaning := registry primitive fallback (items.map CallStage.stage)
     actualGraph.observations.map (executeNodes meaning (actualGraph.nodes.drop {start + 3})
       (native.step (executeNodes meaning (actualGraph.nodes.take {start}) initial))) =
       actualGraph.observations.map (executeNodes meaning actualGraph.nodes initial)) := by
  refine ⟨actual_generic_binding, actual_kernel_decoded,
    (fun arguments => checkScalar_sound scalar0_selected_accepted primitive arguments),
    (fun arguments => checkScalar_sound scalar1_selected_accepted primitive arguments),
    (fun arguments => checkScalar_sound scalar2_selected_accepted primitive arguments), ?_⟩
  exact checked_calls_preserve_observers primitive fallback actual_realization items actual_boundary
    addresses initialKnown
    (checkSequence_sound actual_calls_valid) actual_nodes_unique read invoke contracts pack initialized
    actual_output_coverage native publication initial
#print axioms actual_complete_observers

-- A mathematical nonzero initialized witness. Its total interpretation is
-- illustrative; no target primitive fidelity follows from this computation.
def witnessPrimitive (primitive : Primitive) (values : List Nat) : Nat :=
  match primitive with
  | .constant _ literal => literal.toNat
  | .binary _ .add => values[0]?.getD 0 + values[1]?.getD 0
  | .binary _ .sub => values[0]?.getD 0 - values[1]?.getD 0
  | .binary _ .mul => values[0]?.getD 0 * values[1]?.getD 0
  | .cast .. => values[0]?.getD 0
def witnessInitial : StateEffects.State (Tensor {elements} Nat) := fun slot _ =>
  if slot = stage0.inputs[0]?.getD 0 then 3 else
  if slot = stage0.inputs[1]?.getD 0 then 5 else
  if slot = stage0.inputs[2]?.getD 0 then 7 else
  if slot = stage1.inputs[1]?.getD 0 then 11 else 1
theorem actual_nonzero_witness : ∀ index : Fin {elements},
    executeStages witnessPrimitive (items.map CallStage.stage) witnessInitial stage1.output index = 33 ∧
    executeStages witnessPrimitive (items.map CallStage.stage) witnessInitial stage2.output index = 242 := by
  intro index
  simp [executeStages, ScalarStage.step, tensorEvaluate, Program.evaluate,
    Merlin.GraphRegionsDeclared.run, Merlin.GraphRegionsDeclared.bind,
    witnessPrimitive, witnessInitial, items, stage0, stage1, stage2]
#print axioms actual_nonzero_witness
""",
            ]
        )
    return "\n\n".join(declarations) + "\n", dict(
        elements=elements,
        known=known,
        addresses={str(slot): value for slot, value in addresses.items()},
        source_generic_indices=generic_indices,
        backings=backings,
        source_operational_bridge_qualified=False,
    )

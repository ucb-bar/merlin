"""Source-bound Unit execution candidates for literal/binary leaf entries.

This demonstrates an inhabited abstract interpretation, not actual i32 numerics.
The entire function body is bound by a finite equality before the semantic root.
"""

from .lean_graph_loops_shape_model import binding
from .lean_graph_loops_unroll import _string


def literal_witnesses(source, bits):
    texts, roots, refusals = [], [], []
    for number, function in enumerate(source["functions"]):
        code = function["code"]
        if code["tag"] != "action" or len(code["results"]) != 1 or code["operands"]:
            continue
        binary = code["rest"]
        if binary["tag"] != "action" or len(binary["results"]) != 1 or len(function["parameters"]) != 1:
            continue
        input_value, literal_result, result = function["parameters"][0], code["results"][0], binary["results"][0]
        if (binary["operands"] != [input_value, literal_result]
                or binary["rest"] != dict(tag="returned", values=[result])
                or input_value["kind"]["tag"] != "scalar"):
            refusals.append(dict(entry=function["name"], obligation="nonvacuity_literal_binary_body_outside_witness_subset"))
            continue
        element = _string(input_value["kind"]["type"])
        name, literal_key, binary_key = "entry" + str(number), _string(code["key"]), _string(binary["key"])
        meaning = (
            f"(entryMeaning (Value := Unit) (User := Unit) {bits} indices plan.opcodes "
            "(fun _ _ _ _ => True) literals (fun _ _ => True))"
        )
        texts.append(f"""
theorem {name}_literal_selection :
    (withCalls {meaning} independentSource
      (Merlin.GraphLoopsShape.invoke {meaning} independentSource false 0)).action {literal_key} =
    LiteralAction {element} {literal_key} (fun _ _ => True) := by rfl
theorem {name}_binary_selection :
    (withCalls {meaning} independentSource
      (Merlin.GraphLoopsShape.invoke {meaning} independentSource false 0)).action {binary_key} =
    Epoch.Binary {element} (fun _ _ _ => True) := by rfl
theorem {name}_source_execution :
    complete {meaning} independentSource false {name}.name
      [.scalar {element} ()] ⟨⟨0, ()⟩, sampleHeap⟩ [.scalar {element} ()] ⟨⟨0, ()⟩, sampleHeap⟩ :=
  literal_binary_complete {meaning} independentSource {name} {name}_found
    {literal_key} {binary_key} {element} {binding(input_value)} {binding(literal_result)} {binding(result)}
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    {name}_literal_selection {name}_binary_selection ⟨⟨0, ()⟩, sampleHeap⟩
theorem {name}_source_outcome :
    Epoch.graphOutcomes plan.objects.length {meaning} independentSource {name}.name
      {name}Inputs sampleHeap ()
      (Epoch.observeWorld plan.objects.length [.scalar {element} ()] ⟨⟨0, ()⟩, sampleHeap⟩) :=
  ⟨[.scalar {element} ()], ⟨⟨0, ()⟩, sampleHeap⟩, {name}_source_execution, rfl⟩
theorem {name}_final_outcome :
    Epoch.graphOutcomes plan.objects.length {meaning} GraphMemoryAfterMaterialization.actualFinal {name}.name
      {name}Inputs sampleHeap ()
      (Epoch.observeWorld plan.objects.length [.scalar {element} ()] ⟨⟨0, ()⟩, sampleHeap⟩) :=
  (complete_roster (fun _ _ _ _ => True) (fun _ _ => True) {name}Row {name}_member {name}Inputs {name}_domain
    sampleHeap () _).mp {name}_source_outcome
""")
        roots.extend(name + suffix for suffix in (
            "_literal_selection", "_binary_selection", "_source_execution", "_source_outcome", "_final_outcome",
        ))
    return "".join(texts), roots, refusals

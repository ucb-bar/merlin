"""Finite inhabited common interpretations bound to an actual complete model.

This generates an explicit abstract witness, not native arithmetic conformance.
It supports uniform rank2 bodies, logs all private loads as public observations,
and gives explicit common meanings to retained keys. Separate xDSL tests check
real scalar operations, caller updates and returned values on finite inputs.
"""

from __future__ import annotations

from .lean_graph_loops_memory_scf_model import obligation
from .lean_graph_loops_unroll import _nat, _string

WITNESS_ROOTS = (
    "GraphMemorySCFWitness.source_inhabited",
    "GraphMemorySCFWitness.final_inhabited",
)


def witness_obligation(projection):
    configs = projection["configurations"]
    instructions = projection["reuse"]["instructions"]
    if len(configs) > 4:
        raise ValueError("finite witness supports at most four epochs; general theorem is not bounded here")
    seeds, values = {}, {}
    declarations = []
    for index, row in enumerate(configs):
        program = row["program"]
        if len(program["shape"]) != 2 or program["outerPrefix"] or program["leaf"]["ops"]:
            raise ValueError("inhabited witness requires explicit uniform rank2 bodies")
        seed = program["leaf"]["value"]
        if seed in seeds or seed in program["ivs"]:
            raise ValueError("distinct captured scalar seeds and IVs required")
        seeds[seed] = 17 + 12 * index
        values[row["object"]] = seeds[seed]
        declarations.append(f"def cfg{index} : Configuration := actual.boundaries[{index}].config")
    captures = "none"
    for seed, value in reversed(list(seeds.items())):
        captures = f"if name = {_nat(seed)} then some (.scalar {value}) else ({captures})"
    stores = [row for row in instructions if row["kind"] == "store"]
    stored = {row["key"]: values[row["object"]] * 2 for row in stores}
    choices = " ∨ ".join(f"(key = {_string(key)} ∧ value = {value})" for key, value in stored.items()) or "False"
    old = obligation(projection).replace(
        "import Merlin.GraphLoopsMemorySCFModel", "import Merlin.GraphLoopsMemorySCFWitness", 1
    )
    pieces = [
        old,
        "namespace GraphMemorySCFWitness",
        "open Merlin Merlin.GraphLoopsMemorySCF",
        "abbrev actual := GraphMemorySCFObligation.actual",
        "abbrev objects := actual.reuse.objects",
        "abbrev configs := Model.configs actual",
        "abbrev instructions := actual.reuse.instructions",
        *declarations,
        f"def captures : List Nat → Env Nat := fun _ name => {captures}",
        f"""def retained : GraphLoopsReuse.Primitives Nat (List Nat) where
  initialization := fun _ _ _ => False
  address := fun _ _ index => index = 0
  load := fun _ value before after => after = before ++ [value]
  store := fun key _ index value => index = 0 ∧ ({choices})
  dimension := fun _ _ before after => after = before
  retained := fun _ before after => after = before
def meaning : Interpretation Nat (List Nat) where
  pure := fun _ _ _ => False
  captures := captures
  retained := retained
def source : GraphLoopsReuse.Memory Nat (List Nat) := ⟨[], fun _ _ => none⟩
""",
        "theorem source_inhabited : ∃ after, "
        "Execution objects configs meaning false false instructions source after ∧ after.observed = ",
    ]
    observation = []
    current_values = dict(values)
    for row in instructions:
        if row["kind"] == "load":
            observation.append(current_values[row["object"]])
        if row["kind"] == "store":
            current_values[row["object"]] = stored[row["key"]]
    outcome = "[" + ", ".join(map(str, observation)) + "]"
    pieces[-1] += outcome + " := by"
    lines = []
    current_values = dict(values)
    heap_proofs = {}
    step_names = []
    for index, row in enumerate(instructions):
        prior = "source" if index == 0 else f"state{index - 1}"
        state, step = f"state{index}", f"step{index}"
        kind = row["kind"]
        obj = row.get("object")
        key = _string(row["key"]) if "key" in row else None
        if kind in ("allocate", "release"):
            keep = str(row["retained"]).lower()
            lines.append(f"  let {state} := GraphLoopsReuse.clearMemory objects false {obj} {keep} {prior}")
            constructor = "allocate" if kind == "allocate" else "release"
            instruction = f"(.{kind} {obj} {keep})"
            proof = f".ordinary True.intro (.{constructor} {obj} {keep} {prior})"
        elif kind == "initialize":
            cfg = next(i for i, config in enumerate(configs) if config["object"] == obj and config["key"] == row["key"])
            program = configs[cfg]["program"]
            rows, columns = program["shape"]
            outer, inner = program["ivs"]
            seed, value = program["leaf"]["value"], values[obj]
            lines.append(
                f"  obtain ⟨heap{index}, run{index}, initialized{index}⟩ := Witness.uniform_heap "
                f"meaning.pure {rows} {columns} {outer} {inner} {seed} {configs[cfg]['tile']} {obj} "
                "(by decide) (by decide) (by decide) (by decide) (by decide) "
                f"(meaning.captures {prior}.observed) {value} (by rfl) {prior}.heap"
            )
            lines.append(f"  let {state} : GraphLoopsReuse.Memory Nat (List Nat) := ⟨{prior}.observed, heap{index}⟩")
            instruction = f"(.initialize {obj} {key})"
            proof = (
                f".initialized (config := cfg{cfg}) (by simp [Model.configs, configs, actual, "
                f"GraphMemorySCFObligation.actual, cfg{cfg}]) {prior} run{index}"
            )
            heap_proofs[obj] = f"initialized{index} 0 (by decide)"
            current_values[obj] = value
        elif kind == "load":
            value = current_values[obj]
            lines.append(
                f"  let {state} : GraphLoopsReuse.Memory Nat (List Nat) := "
                f"⟨{prior}.observed ++ [{value}], {prior}.heap⟩"
            )
            instruction = f"(.load {obj} {key})"
            proof = (
                f".ordinary True.intro (.load {obj} 0 {key} {prior} {value} _ rfl (by decide) "
                "(by simpa only [GraphLoopsReuse.addressOf, Bool.false_eq_true, ↓reduceIte] "
                f"using {heap_proofs[obj]}) rfl)"
            )
        elif kind == "store":
            value = stored[row["key"]]
            lines.append(f"  let {state} := GraphLoopsReuse.storeMemory objects false {obj} 0 {value} {prior}")
            instruction = f"(.store {obj} {key})"
            proof = (
                f".ordinary True.intro (.store {obj} 0 {key} {prior} {value} "
                "(by simp [inducedPrimitives, meaning, retained]) (by decide))"
            )
            lines.append(
                f"  have cell{index} : {state}.heap {obj} 0 = some {value} := by simp [{state}, "
                "GraphLoopsReuse.storeMemory, GraphLoopsReuse.addressOf, "
                "GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject]"
            )
            heap_proofs[obj] = f"cell{index}"
            current_values[obj] = value
        else:
            lines.append(f"  let {state} := {prior}")
            instruction = f"(.{kind} {obj} {key})" if obj is not None else f"(.retained {key})"
            constructor = (
                f".dimension {obj} {key} {prior} {prior}.observed rfl"
                if kind == "dimension"
                else f".retained {key} {prior} {prior}.observed rfl"
            )
            proof = f".ordinary True.intro ({constructor})"
        lines.append(
            f"  have {step} : Step objects configs meaning false false {instruction} {prior} {state} := {proof}"
        )
        step_names.append(step)
    lines.append(f"  refine ⟨state{len(instructions) - 1}, ?_, rfl⟩")
    nested = ".done _"
    for step in reversed(step_names):
        nested = f".more {step} ({nested})"
    lines.append("  exact " + nested)
    pieces.extend(lines)
    pieces.append(f"""theorem final_inhabited : ∃ after,
    Execution objects configs meaning true true instructions source after ∧ after.observed = {outcome} :=
  (GraphMemorySCFObligation.complete meaning source source rfl {outcome}).mp source_inhabited
end GraphMemorySCFWitness
""")
    pieces.extend("#print axioms " + root for root in WITNESS_ROOTS)
    return "\n".join(pieces) + "\n"

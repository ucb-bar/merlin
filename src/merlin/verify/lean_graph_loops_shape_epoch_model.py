"""Kernel obligations for complete typed shape/epoch/segmentation correspondence.

The query checks actual syntax, every scoped step, entry and whole observations.
Receipt issuance and fixed-context qualification belong to the separate checker.
"""

from .lean_graph_loops_shape_model import binding, function
from .lean_graph_loops_shape_model import obligation as shape_obligation
from .lean_graph_loops_unroll import _list, _nat, _string

ADAPTER_ROOTS = (
    "Merlin.GraphLoopsShape.Epoch.offset",
    "Merlin.GraphLoopsShape.Epoch.address",
    "Merlin.GraphLoopsShape.Epoch.erase_read",
    "Merlin.GraphLoopsShape.Epoch.erase_assign",
    "Merlin.GraphLoopsShape.Epoch.store_iff",
    "Merlin.GraphLoopsShape.Epoch.load_iff",
    "Merlin.GraphLoopsShape.Epoch.normalized_eval",
    "Merlin.GraphLoopsShape.Epoch.normalized_invoke",
    "Merlin.GraphLoopsShape.Epoch.normalized_complete",
    "Merlin.GraphLoopsShape.Epoch.heap_execution_forward",
    "Merlin.GraphLoopsShape.Epoch.heap_execution_backward",
    "Merlin.GraphLoopsShape.Epoch.scf_heap_iff",
    "Merlin.GraphLoopsShape.Epoch.Cell.eval",
    "Merlin.GraphLoopsShape.Epoch.uniform_cell_inputs",
    "Merlin.GraphLoopsShape.Epoch.Uniform.inner_corresponds",
    "Merlin.GraphLoopsShape.Epoch.Uniform.outer_corresponds",
    "Merlin.GraphLoopsShape.Epoch.Uniform.eval_iff",
    "Merlin.GraphLoopsShape.Epoch.Uniform.checked_eval",
    "GraphShapeEpochObligation.normalized_syntax",
    "GraphShapeEpochObligation.complete_plain",
)


def module_literal(module):
    calls = _list(module["calls"], lambda pair: "(" + _string(pair[0]) + ", " + _string(pair[1]) + ")")
    return "⟨" + _list(module["functions"], function) + ", " + calls + "⟩"


def uniform_literal(program):
    return (
        "⟨"
        + ", ".join(
            [_string(program[key]) for key in ("storeKey", "loopKey")]
            + [binding(program[key]) for key in ("seed", "buffer", "outer", "inner", "zero", "rows", "columns", "one")]
        )
        + "⟩"
    )


def obligation(adapter):
    source = shape_obligation(adapter["shape"]).replace(
        "import Merlin.GraphLoopsShapeCountermodels\n",
        "import Merlin.GraphLoopsShapeEpochLink\nimport Merlin.GraphLoopsShapeEpochAllocation\n"
        "import Merlin.GraphLoopsShapeEpochObserver\nimport Merlin.GraphLoopsShapeEpochSequence\n"
        "import Merlin.GraphLoopsShapeEpochChain\n",
        1,
    )
    imports, body = source.split("set_option autoImplicit false", 1)
    source = imports + "section ShapeBoundary\nset_option autoImplicit false" + body + "\nend ShapeBoundary\n"
    roots = list(ADAPTER_ROOTS) + [
        "Merlin.GraphLoopsShape.Epoch.uniform_heap_alpha",
        "Merlin.GraphLoopsShape.Epoch.eval_free_reads",
        "Merlin.GraphLoopsShape.Epoch.merge_heap_writes",
        "Merlin.GraphLoopsShape.Epoch.heap_run_partition",
        "Merlin.GraphLoopsShape.Epoch.initializer_world_partition",
        "Merlin.GraphLoopsShape.Epoch.InitializerLink.checked_execution",
        "Merlin.GraphLoopsShape.Epoch.allocation_epoch",
        "Merlin.GraphLoopsShape.Epoch.release_epoch",
        "Merlin.GraphLoopsShape.Epoch.preseed_fresh_view",
        "Merlin.GraphLoopsShape.Epoch.Touch.eval",
        "Merlin.GraphLoopsShape.Epoch.Touch.caller_effect",
        "Merlin.GraphLoopsShape.Epoch.Touch.invoke",
        "Merlin.GraphLoopsShape.Epoch.Touch.call",
        "Merlin.GraphLoopsShape.Epoch.Atom.eval_suffix",
        "Merlin.GraphLoopsShape.Epoch.sequence_complete",
        "Merlin.GraphLoopsShape.Epoch.constant_atom",
        "Merlin.GraphLoopsShape.Epoch.binary_atom",
        "Merlin.GraphLoopsShape.Epoch.allocate_atom",
        "Merlin.GraphLoopsShape.Epoch.release_atom",
        "Merlin.GraphLoopsShape.Epoch.load_atom",
        "Merlin.GraphLoopsShape.Epoch.store_atom",
        "Merlin.GraphLoopsShape.Epoch.initializer_atom",
        "Merlin.GraphLoopsShape.Epoch.observer_atom",
        "Merlin.GraphLoopsShape.Epoch.InitializerLink.from_registers",
        "Merlin.GraphLoopsShape.Epoch.InitializerLink.clocked",
        "Merlin.GraphLoopsShape.Epoch.registers_assign",
        "Merlin.GraphLoopsShape.Epoch.callers_assign",
        "Merlin.GraphLoopsShape.Epoch.indices_of_typed",
        "Merlin.GraphLoopsShape.Epoch.calls_with_memory",
        "Merlin.GraphLoopsShape.Epoch.returned_step_iff",
        "Merlin.GraphLoopsShape.Epoch.sequence_refused",
        "Merlin.GraphLoopsShape.Epoch.PlanContext.memory_nonCall",
        "Merlin.GraphLoopsShape.Epoch.PlanContext.meaning_memory",
        "Merlin.GraphLoopsShape.Epoch.Command.correspondence",
        "Merlin.GraphLoopsShape.Epoch.Command.assignment",
        "Merlin.GraphLoopsShape.Epoch.Command.pins_preserved",
        "Merlin.GraphLoopsShape.Epoch.Command.preserves",
        "Merlin.GraphLoopsShape.Epoch.commands_forward",
        "Merlin.GraphLoopsShape.Epoch.commands_backward",
        "Merlin.GraphLoopsShape.Epoch.commands_equivalent",
        "Merlin.GraphLoopsShape.Epoch.execution_append",
        "Merlin.GraphLoopsShape.Epoch.execution_single",
        "Merlin.GraphLoopsShape.Epoch.block_equivalent",
        "Merlin.GraphLoopsShape.Epoch.seed_other",
        "Merlin.GraphLoopsShape.Epoch.seed_pinned",
        "Merlin.GraphLoopsShape.Epoch.seed_eval",
        "Merlin.GraphLoopsShape.Epoch.entry_world",
        "Merlin.GraphLoopsShape.Epoch.entry_registers",
        "Merlin.GraphLoopsShape.Epoch.entry_callers",
        "Merlin.GraphLoopsShape.Epoch.zero_call_refused",
        "Merlin.GraphLoopsShape.Epoch.Kernel.depth_one_refused",
        "Merlin.GraphLoopsShape.Epoch.Kernel.depth_equivalent",
        "Merlin.GraphLoopsShape.Epoch.Kernel.complete_equivalent",
        "Merlin.GraphLoopsShape.Epoch.visible_merge",
        "Merlin.GraphLoopsShape.Epoch.observe_clock",
        "Merlin.GraphLoopsShape.Epoch.Kernel.observations",
        "Merlin.GraphLoopsShape.Epoch.epoch_chain",
        "Merlin.GraphLoopsShape.Epoch.shape_outcomes",
        "Merlin.GraphLoopsShape.Epoch.Kernel.checked_chain",
    ]
    if adapter["shape"].get("identity_normalization") is True:
        roots.extend(
            "Merlin.GraphLoopsShape.Identity." + name
            for name in (
                "check_sound",
                "checked_complete",
                "checked_memory",
                "checked_context",
            )
        )
    extra = []
    for index, row in enumerate(adapter["initializers"]):
        name = "initializer" + str(index)
        extra.append(f"""def {name} : Epoch.Uniform := {uniform_literal(row["program"])}
theorem {name}_syntax :
    (functionAt actualNormalized {_string(row["function"])}).map
      (fun function => (Epoch.topRegions function.code)[{_nat(row["position"])}]?) = some (some {name}.code) := by rfl
theorem {name}_facts :
    decide ({name}.Facts {_nat(row["shape"][0])} {_nat(row["shape"][1])} {_string(row["element"])}) = true := by decide
""")
        roots += ["GraphShapeEpochObligation." + name + suffix for suffix in ("_syntax", "_facts")]
    for index, link in enumerate(adapter.get("initializer_links", [])):
        name = "link" + str(index)
        extra.append(f"""def {name} : Epoch.InitializerLink := {link_literal(link)}
theorem {name}_checked : decide {name}.Facts = true := by decide
theorem {name}_canonical : {name}.canonical = initializer{index} := by rfl
theorem {name}_old_configuration :
    {name}.config ∈ GraphLoopsMemorySCF.Model.configs GraphMemorySCFObligation.actual := by decide
""")
        roots += [
            "GraphShapeEpochObligation." + name + suffix for suffix in ("_checked", "_canonical", "_old_configuration")
        ]
    if "observers" in adapter:
        from .lean_graph_loops_shape_epoch_observer import clauses

        observer_clauses, observer_roots = clauses(adapter)
        extra.append(observer_clauses)
        roots += list(observer_roots)
    if "step_binding" in adapter:
        from .lean_graph_loops_shape_epoch_plan import clauses as plan_clauses

        plan_source, plan_roots = plan_clauses(adapter)
        extra.append(plan_source)
        roots += list(plan_roots)
    if "memory" in adapter:
        from .lean_graph_loops_memory_scf_model import MEMORY_SCF_ROOTS
        from .lean_graph_loops_memory_scf_model import obligation as memory_obligation

        source += "\n" + memory_obligation(adapter["memory"]).partition("\n")[2]
        roots = list(MEMORY_SCF_ROOTS) + roots
    query = (
        source
        + f"""
namespace GraphShapeEpochObligation
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 4000000
open Merlin Merlin.GraphLoopsShape
def actualNormalized : Module := {module_literal(adapter["normalized_candidate"])}
theorem normalized_syntax : actualNormalized = Epoch.normalizedModule GraphShapeObligation.actual.module := by rfl
theorem complete_plain {{Value World : Type}} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    complete meaning GraphShapeObligation.actual.module false name inputs before outputs after ↔
      complete meaning actualNormalized false name inputs before outputs after := by
  rw [normalized_syntax]
  exact (GraphShapeObligation.complete meaning name inputs before outputs after).trans
    (Epoch.normalized_complete meaning GraphShapeObligation.actual.module name inputs before after outputs)
{chr(10).join(extra)}
end GraphShapeEpochObligation
"""
    )
    already_printed = {
        line.removeprefix("#print axioms ") for line in query.splitlines() if line.startswith("#print axioms ")
    }
    query += "\n".join("#print axioms " + root for root in roots if root not in already_printed) + "\n"
    return query, tuple(roots)


def link_literal(link):
    from .lean_graph_loops_memory_scf_model import configuration

    fields = [uniform_literal(link["raw"]), uniform_literal(link["canonical"]), configuration(link["config"])]
    fields.append(_list(link["mapping"], lambda pair: "(" + _nat(pair[0]) + ", " + _nat(pair[1]) + ")"))
    fields.extend((_nat(link["rows"]), _nat(link["columns"]), _string(link["element"]), _nat(link["sourceBuffer"])))
    return "⟨" + ", ".join(fields) + "⟩"

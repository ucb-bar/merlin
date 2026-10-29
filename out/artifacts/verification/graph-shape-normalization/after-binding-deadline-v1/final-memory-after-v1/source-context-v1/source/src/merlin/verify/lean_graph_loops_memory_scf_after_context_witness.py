"""Construct a nonzero abstract execution for the retained two-epoch wrapper.

Reuses the immutable actual epoch witness builder. Nat addition is an inhabited
shared relation used for this example, not a numerical/native MLIR fidelity law.
The complete actual continuation is inspected and bound to emitted proof terms.
"""

from .lean_graph_loops_shape_epoch_witness import witness_obligation
from .lean_graph_loops_unroll import _string


def _shape(wrapper):
    """Recognize a witness shape only; semantic correspondence remains in Lean."""
    parameters = wrapper['function']['parameters']
    call_results = wrapper['results']
    first = wrapper['suffix']
    if len(parameters) != 4 or len(call_results) != 2 or first['tag'] != 'action':
        return None
    zero = first['rest']
    if zero['tag'] != 'constant' or zero['value'] != 0:
        return None
    load = zero['rest']
    if load['tag'] != 'action':
        return None
    second = load['rest']
    if second['tag'] != 'action':
        return None
    store = second['rest']
    if store['tag'] != 'action' or store['rest']['tag'] != 'returned':
        return None
    if (any(len(node['results']) != 1 for node in (first, load, second))
            or store['results']
            or wrapper['arguments'] != parameters[:3]
            or first['operands'] != [call_results[0], parameters[3]]
            or load['operands'] != [parameters[2], zero['result']]
            or second['operands'] != [load['results'][0], first['results'][0]]
            or store['operands'] != [second['results'][0], parameters[2], zero['result']]
            or store['rest']['values'] != [second['results'][0], call_results[1]]):
        return None
    return dict(first=first, zero=zero, load=load, second=second, store=store)


def witness(adapter, data):
    if len(data['wrappers']) != 1 or data['refusals']:
        return '', [], dict(status='NOT_EMITTED', reason='requires one supported wrapper')
    wrapper = data['wrappers'][0]
    original = next((row for row in adapter['shape']['module']['functions'] if row['name'] == wrapper['function']['name']), None)
    if original != wrapper['function']:
        return '', [], dict(status='NOT_EMITTED', reason='concrete execution witness requires original literal body; normalized aliases retain generic context theorem')
    shape = _shape(wrapper)
    if shape is None:
        return '', [], dict(status='NOT_EMITTED', reason='actual continuation outside concrete witness grammar')
    if [row['kind'] for row in wrapper['function']['parameters']] != [
        {'tag': 'scalar', 'type': 'i32'}, {'tag': 'scalar', 'type': 'i32'},
        {'tag': 'view', 'type': {'shape': [1], 'offset': 0, 'strides': [1], 'element': 'i32', 'space': 0}},
        {'tag': 'scalar', 'type': 'i32'},
    ]:
        return '', [], dict(status='NOT_EMITTED', reason='concrete example requires original i32/singleton caller interface')
    old, _ = witness_obligation(adapter)
    begin = old.index('namespace GraphShapeEpochWitness\n')
    end = old.index('end GraphShapeEpochWitness', begin) + len('end GraphShapeEpochWitness')
    # Keep only the old proof declarations; the complete adapter query already exists.
    old = old[begin:end] + '\n'
    key = {name: _string(shape[name]['key']) for name in ('first', 'load', 'second', 'store')}
    text = old + f'''
namespace GraphMemoryAfterContextWitness
open Merlin Merlin.GraphLoopsShape Merlin.GraphLoopsMemorySCF.After
open GraphShapeEpochObligation GraphMemoryAfterExpansion GraphMemoryAfterWhole GraphMemoryAfterContext
abbrev scalar := GraphShapeEpochWitness.scalar
def literal : LiteralMeaning Nat := fun _ value => value = 1
abbrev meaning : Interpretation Nat (CallerWorld Nat Unit) := boundary.meaning scalar literal
theorem source_wrapper_syntax : wrapper0.Facts independentSource := by decide
def inputs : List (Runtime Nat) :=
  [.scalar "i32" 17, .scalar "i32" 29, Epoch.view 2 [1] "i32", .scalar "i32" 7]
def outputs : List (Runtime Nat) := [.scalar "i32" 498, .index 2]
def middleOutputs : List (Runtime Nat) := [.scalar "i32" 308, .index 2]
def env : Merlin.GraphLoopsShape.Env Nat :=
  assign (assign emptyEnv wrapper0.function.parameters inputs) wrapper0.results middleOutputs
theorem typed : ValuesHold wrapper0.function.parameters inputs := by
  exact .cons ⟨rfl, by intro index impossible; cases impossible⟩
    (.cons ⟨rfl, by intro index impossible; cases impossible⟩
      (.cons ⟨rfl, by intro index impossible; cases impossible⟩
        (.cons ⟨rfl, by intro index impossible; cases impossible⟩ .nil)))
theorem domain : wrapper0.Domain plan kernel inputs := by
  refine ⟨typed, ?_, ?_⟩
  · intro value member
    simp only [inputs, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl
    all_goals first | trivial | decide
  · intro caller member
    simp only [plan, List.mem_cons, List.not_mem_nil, or_false] at member
    subst caller
    exact ⟨2, by decide, rfl⟩
theorem call_arguments :
    readArguments false [] (assign emptyEnv wrapper0.function.parameters inputs) wrapper0.arguments =
      GraphShapeEpochWitness.inputs := by rfl
theorem return_typed : ValuesHold wrapper0.results middleOutputs := by
  exact .cons ⟨rfl, by intro index impossible; cases impossible⟩
    (.cons ⟨rfl, by intro index impossible; cases impossible⟩ .nil)
theorem first_selected : meaning.action {key['first']} = Epoch.Binary "i32" (fun a b result => result = a + b) := by rfl
theorem second_selected : meaning.action {key['second']} = Epoch.Binary "i32" (fun a b result => result = a + b) := by rfl
theorem load_selected : meaning.action {key['load']} = Load := by rfl
theorem store_selected : meaning.action {key['store']} = Store := by rfl

/-- Both kernel initialization and caller cell91 are constructed by the frozen
witness, then transported through derived conservative primitive extensions. -/
theorem kernel_execution : ∃ world,
    complete meaning independentSource false kernel.function.name GraphShapeEpochWitness.inputs
      ⟨⟨0, ()⟩, GraphShapeEpochWitness.initialHeap⟩ middleOutputs world ∧ world.heap 2 0 = some 183 := by
  obtain ⟨world, run, cell⟩ := GraphShapeEpochWitness.typed_source_inhabited
  have original : complete (Epoch.graphMeaning plan.opcodes scalar) independentSource false kernel.function.name
      GraphShapeEpochWitness.inputs ⟨⟨0, ()⟩, GraphShapeEpochWitness.initialHeap⟩ middleOutputs world := by
    rw [source_term_exact]
    exact run
  have indexed := (source_complete_extension boundary.bits indices plan.opcodes scalar independentSource
    source_index_absent kernel.function.name GraphShapeEpochWitness.inputs _ middleOutputs world).mp original
  have extended := (closed_literal_extension (finalMeaning boundary.bits indices plan.opcodes scalar)
    literals literal independentSource reachable (by decide) (by decide) kernel.function.name (by decide)
    GraphShapeEpochWitness.inputs _ middleOutputs world).mp indexed
  exact ⟨world, extended, cell⟩

/-- Actual post-call scalar continuation, initialized public load, scalar
observer update, public store and return. No whole-suffix execution premise. -/
theorem continuation_execution (world : CallerWorld Nat Unit) (initialized : world.heap 2 0 = some 183) :
    Eval meaning false (extendAliases (extendAliases [] wrapper0.function.parameters) wrapper0.results)
      wrapper0.suffix env world outputs
      ⟨world.observed, GraphLoopsReuse.replaceCell world.heap 2 0 498⟩ := by
  apply Eval.source_action [.scalar "i32" 315] world
  · rw [first_selected]
    exact ⟨308, 7, 315, rfl, rfl, rfl, rfl⟩
  · exact .cons ⟨rfl, by intro index impossible; cases impossible⟩ .nil
  · apply Eval.source_constant
    apply Eval.source_action [.scalar "i32" 183] world
    · rw [load_selected]
      exact Load.ready (Epoch.type [1] "i32") (Epoch.descriptor 2 [1] "i32") (Epoch.conforms 2 [1] "i32")
        [0] 183 world (by decide) initialized
    · exact .cons ⟨rfl, by intro index impossible; cases impossible⟩ .nil
    · apply Eval.source_action [.scalar "i32" 498] world
      · rw [second_selected]
        exact ⟨183, 315, 498, rfl, rfl, rfl, rfl⟩
      · exact .cons ⟨rfl, by intro index impossible; cases impossible⟩ .nil
      · apply Eval.source_action [] ⟨world.observed, GraphLoopsReuse.replaceCell world.heap 2 0 498⟩
        · rw [store_selected]
          exact Store.ready (Epoch.type [1] "i32") (Epoch.descriptor 2 [1] "i32") (Epoch.conforms 2 [1] "i32")
            [0] 498 world (by decide)
        · exact .nil
        · exact ⟨rfl, rfl⟩

theorem source_execution : ∃ world,
    complete meaning independentSource false wrapper0.function.name inputs
      ⟨⟨0, ()⟩, GraphShapeEpochWitness.initialHeap⟩ outputs world ∧ world.heap 2 0 = some 498 := by
  obtain ⟨world, call, initialized⟩ := kernel_execution
  refine ⟨⟨world.observed, GraphLoopsReuse.replaceCell world.heap 2 0 498⟩, ?_, rfl⟩
  apply (wrapper0.complete_iff independentSource source_wrapper_syntax meaning inputs _ outputs _).mpr
  refine ⟨typed, middleOutputs, world, ?_, return_typed, continuation_execution world initialized⟩
  rw [call_arguments]
  exact call

theorem source_observation : ∃ observed,
    Epoch.graphOutcomes plan.objects.length meaning independentSource wrapper0.function.name inputs
      GraphShapeEpochWitness.initialHeap () observed ∧ observed.returned = outputs ∧ observed.heap 2 0 = some 498 := by
  obtain ⟨world, run, initialized⟩ := source_execution
  refine ⟨Epoch.observeWorld plan.objects.length outputs world, ⟨outputs, world, run, rfl⟩, rfl, ?_⟩
  exact initialized

theorem final_observation : ∃ observed,
    Epoch.graphOutcomes plan.objects.length meaning boundary.actualFinal wrapper0.function.name inputs
      GraphShapeEpochWitness.initialHeap () observed ∧ observed.returned = outputs ∧ observed.heap 2 0 = some 498 := by
  obtain ⟨observed, run, returned, initialized⟩ := source_observation
  exact ⟨observed, (wrapper0_complete scalar literal inputs domain GraphShapeEpochWitness.initialHeap () observed).mp run,
    returned, initialized⟩
end GraphMemoryAfterContextWitness
'''
    roots = ['GraphShapeEpochWitness.' + name for name in ('epoch_source_inhabited', 'typed_source_inhabited', 'segmented_inhabited')]
    roots += ['GraphMemoryAfterContextWitness.' + name for name in (
        'source_wrapper_syntax', 'typed', 'domain', 'call_arguments', 'return_typed', 'first_selected', 'second_selected', 'load_selected', 'store_selected',
        'kernel_execution', 'continuation_execution', 'source_execution', 'source_observation', 'final_observation',
    )]
    return text, roots, dict(status='UNCOMPILED', interpretation='existing Nat-addition witness; no native i32 fidelity law',
                            seeds=[17,29], initial_caller=91, bias=7, callee_returns=[308,2],
                            callee_caller=183, wrapper_returns=[498,2], final_caller=498,
                            initial_private_cells='none; kernel initializers construct cells', source_success_assumed=False)

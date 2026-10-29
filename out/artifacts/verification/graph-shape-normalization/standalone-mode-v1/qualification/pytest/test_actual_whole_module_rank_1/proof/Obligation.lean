import Merlin.GraphLoopsShapeCountermodels
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 4000000
open Merlin Merlin.GraphLoopsShape
namespace GraphShapeObligation
def actual : Model.Projection := ⟨64, ⟨[⟨"probe", [⟨0, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨1, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨2, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩], [.index, .index, (.scalar "i32")], (.constant ⟨3, .index, (some 0)⟩ 0 (.constant ⟨4, .index, (some 1)⟩ 1 (.constant ⟨5, .index, (some 1)⟩ 1 (.constant ⟨6, .index, (some 0)⟩ 0 (.cast ⟨0, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨7, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ (.cast ⟨7, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨8, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ (.dimension ⟨8, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨5, .index, (some 1)⟩ ⟨9, .index, (some 3)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 1 3 (.dimension ⟨1, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨6, .index, (some 0)⟩ ⟨10, .index, (some 2)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 0 2 (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operand_arity\":3,\"properties\":[],\"result_types\":[\"i32\"]}" [⟨8, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [⟨11, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operand_arity\":3,\"properties\":[],\"result_types\":[]}" [⟨11, (.scalar "i32"), none⟩, ⟨2, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩] [] (.returned [⟨9, .index, (some 3)⟩, ⟨10, .index, (some 2)⟩, ⟨11, (.scalar "i32"), none⟩])))))))))))⟩, ⟨"untouched", [⟨12, (.scalar "i32"), none⟩], [(.scalar "i32")], (.returned [⟨12, (.scalar "i32"), none⟩])⟩], []⟩, "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"builtin.module\",\"operands\":[],\"properties\":[],\"regions\":[[{\"arguments\":[],\"operations\":[{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.func\",\"operands\":[],\"properties\":[[\"function_type\",\"(memref<2x3xi32>, memref<2x3xi32>, memref<1xi32>) -> (index, index, i32)\"],[\"sym_name\",\"\\\"probe\\\"\"]],\"regions\":[[{\"arguments\":[[0,\"memref<2x3xi32>\"],[1,\"memref<2x3xi32>\"],[2,\"memref<1xi32>\"]],\"operations\":[{\"attributes\":[],\"index_constant\":0,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"0 : index\"]],\"regions\":[],\"results\":[[3,\"index\"]]},{\"attributes\":[],\"index_constant\":1,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"1 : index\"]],\"regions\":[],\"results\":[[4,\"index\"]]},{\"attributes\":[],\"index_constant\":1,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"1 : index\"]],\"regions\":[],\"results\":[[5,\"index\"]]},{\"attributes\":[],\"index_constant\":0,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"0 : index\"]],\"regions\":[],\"results\":[[6,\"index\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.cast\",\"operands\":[0],\"properties\":[],\"regions\":[],\"results\":[[7,\"memref<2x3xi32>\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.cast\",\"operands\":[7],\"properties\":[],\"regions\":[],\"results\":[[8,\"memref<2x3xi32>\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.dim\",\"operands\":[8,5],\"properties\":[],\"regions\":[],\"results\":[[9,\"index\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.dim\",\"operands\":[1,6],\"properties\":[],\"regions\":[],\"results\":[[10,\"index\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operands\":[8,3,3],\"properties\":[],\"regions\":[],\"results\":[[11,\"i32\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operands\":[11,2,3],\"properties\":[],\"regions\":[],\"results\":[]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.return\",\"operands\":[9,10,11],\"properties\":[],\"regions\":[],\"results\":[]}]}]],\"results\":[]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.func\",\"operands\":[],\"properties\":[[\"function_type\",\"(i32) -> i32\"],[\"sym_name\",\"\\\"untouched\\\"\"]],\"regions\":[[{\"arguments\":[[12,\"i32\"]],\"operations\":[{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.return\",\"operands\":[12],\"properties\":[],\"regions\":[],\"results\":[]}]}]],\"results\":[]}]}]],\"results\":[]}", "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"builtin.module\",\"operands\":[],\"properties\":[],\"regions\":[[{\"arguments\":[],\"operations\":[{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.func\",\"operands\":[],\"properties\":[[\"function_type\",\"(memref<2x3xi32>, memref<2x3xi32>, memref<1xi32>) -> (index, index, i32)\"],[\"sym_name\",\"\\\"probe\\\"\"]],\"regions\":[[{\"arguments\":[[0,\"memref<2x3xi32>\"],[1,\"memref<2x3xi32>\"],[2,\"memref<1xi32>\"]],\"operations\":[{\"attributes\":[],\"index_constant\":0,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"0 : index\"]],\"regions\":[],\"results\":[[3,\"index\"]]},{\"attributes\":[],\"index_constant\":1,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"1 : index\"]],\"regions\":[],\"results\":[[4,\"index\"]]},{\"attributes\":[],\"index_constant\":1,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"1 : index\"]],\"regions\":[],\"results\":[[5,\"index\"]]},{\"attributes\":[],\"index_constant\":0,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"0 : index\"]],\"regions\":[],\"results\":[[6,\"index\"]]},{\"attributes\":[],\"index_constant\":3,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"3 : index\"]],\"regions\":[],\"results\":[[7,\"index\"]]},{\"attributes\":[],\"index_constant\":2,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operands\":[],\"properties\":[[\"value\",\"2 : index\"]],\"regions\":[],\"results\":[[8,\"index\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operands\":[0,3,3],\"properties\":[],\"regions\":[],\"results\":[[9,\"i32\"]]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operands\":[9,2,3],\"properties\":[],\"regions\":[],\"results\":[]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.return\",\"operands\":[7,8,9],\"properties\":[],\"regions\":[],\"results\":[]}]}]],\"results\":[]},{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.func\",\"operands\":[],\"properties\":[[\"function_type\",\"(i32) -> i32\"],[\"sym_name\",\"\\\"untouched\\\"\"]],\"regions\":[[{\"arguments\":[[10,\"i32\"]],\"operations\":[{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.return\",\"operands\":[10],\"properties\":[],\"regions\":[],\"results\":[]}]}]],\"results\":[]}]}]],\"results\":[]}"⟩
theorem valid : Model.Valid actual := by
  simp [Model.Valid, Module.Valid, Function.Valid, GraphLoopsShape.Valid, Scope,
    CastFacts, DimensionFacts, ConstantFacts, AllBound, Plain, Fresh, names,
    extendContext, extendAliases, resolve, ValidDimension, actual]
  all_goals decide
theorem accepted : Model.check actual = true := by
  simp only [Model.check, valid, decide_true]
theorem complete {Value World : Type} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    GraphLoopsShape.complete meaning actual.module false name inputs before outputs after ↔
      GraphLoopsShape.complete meaning actual.module true name inputs before outputs after :=
  Model.checked_complete accepted meaning name inputs before outputs after
theorem memory {Value Public : Type} (meaning : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (name : String) (inputs : List (Runtime Value))
    (before : MemoryWorld Value Public) (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    GraphLoopsShape.complete (withMemory meaning classify) actual.module false name inputs before outputs after ↔
      GraphLoopsShape.complete (withMemory meaning classify) actual.module true name inputs before outputs after :=
  Model.checked_memory accepted meaning classify name inputs before outputs after
theorem contextual {Value World Outcome : Type} (meaning : Interpretation Value World)
    (context : (String → RegionMeaning Value World) → Outcome) :
    context (GraphLoopsShape.complete meaning actual.module false) =
      context (GraphLoopsShape.complete meaning actual.module true) := Model.checked_context accepted meaning context
theorem call_depth {Value World : Type} (meaning : Interpretation Value World) (depth : Nat) :
    invoke meaning actual.module false depth = invoke meaning actual.module true depth :=
  Module.checked_calls actual.module actual.bits (Model.check_sound accepted).1 meaning depth
end GraphShapeObligation
#print axioms GraphShapeObligation.valid
#print axioms GraphShapeObligation.accepted
#print axioms GraphShapeObligation.complete
#print axioms GraphShapeObligation.memory
#print axioms GraphShapeObligation.contextual
#print axioms GraphShapeObligation.call_depth
#print axioms Merlin.GraphLoopsShape.identical_cast
#print axioms Merlin.GraphLoopsShape.cast_descriptor
#print axioms Merlin.GraphLoopsShape.cast_address
#print axioms Merlin.GraphLoopsShape.static_dimension
#print axioms Merlin.GraphLoopsShape.Scope.extend
#print axioms Merlin.GraphLoopsShape.Agreement.define
#print axioms Merlin.GraphLoopsShape.Agreement.alias
#print axioms Merlin.GraphLoopsShape.Agreement.define_many
#print axioms Merlin.GraphLoopsShape.cast_result
#print axioms Merlin.GraphLoopsShape.dimension_result
#print axioms Merlin.GraphLoopsShape.preservation
#print axioms Merlin.GraphLoopsShape.entry_equivalence
#print axioms Merlin.GraphLoopsShape.Function.preservation
#print axioms Merlin.GraphLoopsShape.Module.checked_calls
#print axioms Merlin.GraphLoopsShape.Module.checked_complete
#print axioms Merlin.GraphLoopsShape.Module.checked_memory
#print axioms Merlin.GraphLoopsShape.Model.check_sound
#print axioms Merlin.GraphLoopsShape.Model.checked_complete
#print axioms Merlin.GraphLoopsShape.Model.checked_memory
#print axioms Merlin.GraphLoopsShape.Model.checked_context
#print axioms Merlin.GraphLoopsShape.zero_has_no_coordinate
#print axioms Merlin.GraphLoopsShape.inconsistent_dimension
#print axioms Merlin.GraphLoopsShape.observable_cast_countermodel
#print axioms Merlin.GraphLoopsShape.Countermodels.same_type_different_allocation
#print axioms Merlin.GraphLoopsShape.Countermodels.same_type_cannot_change_buffer
#print axioms Merlin.GraphLoopsShape.Countermodels.logging_deletion
#print axioms Merlin.GraphLoopsShape.Countermodels.trapping_deletion
#print axioms Merlin.GraphLoopsShape.Countermodels.inconsistent_metadata
#print axioms Merlin.GraphLoopsShape.Countermodels.same_type_different_base

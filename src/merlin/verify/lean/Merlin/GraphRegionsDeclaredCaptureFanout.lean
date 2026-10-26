/- Inhabited complete-parent fan-out: captured producer, two consumers, then an
unchanged observer reading BOTH outputs. This is independently written synthetic
syntax, not an authoritative provider capture or native qualification. -/
import Merlin.GraphRegionsDeclaredCaptureParentExamples

namespace Merlin.GraphRegionsDeclared.Capture.Fanout
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared
open Examples (tensor2)

def addLiteral : Program := ⟨[8], [⟨.constant 8 6, []⟩, ⟨.binary 8 .add, [0, 1]⟩], 2, 8⟩
def mulLiteral : Program := ⟨[8], [⟨.constant 8 6, []⟩, ⟨.binary 8 .mul, [0, 1]⟩], 2, 8⟩
def addOutputs : Program := ⟨[8, 8], [⟨.binary 8 .add, [0, 1]⟩], 2, 8⟩
def producer : ScalarStage := ⟨5, "producer", [0], 6, ["tensor[2]i8"], "tensor[2]i8", addLiteral⟩
def firstConsumer : ScalarStage := ⟨6, "first", [6], 7, ["tensor[2]i8"], "tensor[2]i8", addLiteral⟩
def secondConsumer : ScalarStage := ⟨7, "second", [6], 8, ["tensor[2]i8"], "tensor[2]i8", mulLiteral⟩
def observer : ScalarStage :=
  ⟨8, "observer", [7, 8], 9, ["tensor[2]i8", "tensor[2]i8"], "tensor[2]i8", addOutputs⟩
def captures : List Reference := [⟨1, 1, 0⟩]
def sourceSites : List Site :=
  [⟨⟨producer, [2], [10, 1, 12], 2, 11⟩, captures⟩,
   ⟨⟨firstConsumer, [2], [13, 1, 15], 3, 14⟩, captures⟩,
   ⟨⟨secondConsumer, [2], [16, 1, 18], 4, 17⟩, captures⟩,
   ⟨⟨observer, [2], [19, 20, 22], 5, 21⟩, []⟩]
def parent : GraphParent :=
  ⟨"captured_fanout", none, 1, [2], 8,
    [.literal 1 8 6, .empty 2 [2] 8, .empty 3 [2] 8, .empty 4 [2] 8, .empty 5 [2] 8],
    sourceSites, [7, 8, 9]⟩

/-- Independent low-level source syntax. The templates spell the source body
operations/attributes directly; they do not call the expected generic emitter. -/
def sourceUnary (result input initializer argument unused value : Nat) (operation : String) : Op :=
  .mk "linalg.generic" [⟨result, tensor2⟩] [input, initializer]
    [("indexing_maps", .array [.affineMap ⟨1, 0, [.dim 0]⟩, .affineMap ⟨1, 0, [.dim 0]⟩]),
     ("iterator_types", .array [.enum "linalg" "iterator_type" "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])]
    [] [.mk [.mk [⟨argument, .int 8⟩, ⟨unused, .int 8⟩]
      [.mk operation [⟨value, .int 8⟩] [argument, 1]
        [("overflowFlags", .enum "arith" "overflow" "none")] [] [] [],
       .mk "linalg.yield" [] [value] [] [] [] []]]] []
def sourceObserver : Op :=
  .mk "linalg.generic" [⟨9, tensor2⟩] [7, 8, 5]
    [("indexing_maps", .array [.affineMap ⟨1, 0, [.dim 0]⟩, .affineMap ⟨1, 0, [.dim 0]⟩,
       .affineMap ⟨1, 0, [.dim 0]⟩]),
     ("iterator_types", .array [.enum "linalg" "iterator_type" "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [2, 1])]
    [] [.mk [.mk [⟨19, .int 8⟩, ⟨20, .int 8⟩, ⟨21, .int 8⟩]
      [.mk "arith.addi" [⟨22, .int 8⟩] [19, 20]
        [("overflowFlags", .enum "arith" "overflow" "none")] [] [] [],
       .mk "linalg.yield" [] [22] [] [] [] []]]] []
def actual : Module :=
  ⟨[.mk "func.func" [] []
    [("function_type", .type (.function [tensor2] [tensor2, tensor2, tensor2])),
     ("sym_name", .str "captured_fanout")]
    [] [.mk [.mk [⟨0, tensor2⟩]
      [.mk "arith.constant" [⟨1, .int 8⟩] [] [("value", .int 6 (.int 8))] [] [] [],
       .mk "tensor.empty" [⟨2, tensor2⟩] [] [] [] [] [],
       .mk "tensor.empty" [⟨3, tensor2⟩] [] [] [] [] [],
       .mk "tensor.empty" [⟨4, tensor2⟩] [] [] [] [] [],
       .mk "tensor.empty" [⟨5, tensor2⟩] [] [] [] [] [],
       sourceUnary 6 0 2 10 11 12 "arith.addi", sourceUnary 7 6 3 13 14 15 "arith.addi",
       sourceUnary 8 6 4 16 17 18 "arith.muli", sourceObserver,
       .mk "func.return" [] [7, 8, 9] [] [] [] []]]] []], [], []⟩

theorem all_sites_bound : SitesBound (parent.leading.map Prefix.Instruction.emit) parent.sites := by
  intro site member
  simp only [parent, sourceSites, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl <;>
    refine ⟨⟨rfl, by decide, ?_⟩, by decide⟩
  all_goals simp [checkReferences, checkReference, isLiteral, parent, captures, producer,
    firstConsumer, secondConsumer, observer, addLiteral, mulLiteral, addOutputs,
    Prefix.Instruction.emit, instructionMatches, Instruction.operation,
    Primitive.name, Primitive.attrs, Primitive.width]

theorem parent_valid : parent.Valid := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, all_sites_bound, ?_,
    by decide, by decide⟩
  intro index site found
  cases index with
  | zero =>
    simp only [parent, sourceSites, List.getElem?_cons_zero, Option.some.injEq] at found
    subst site
    exact ⟨rfl, rfl, rfl, by decide, rfl, by decide, 1, rfl⟩
  | succ index =>
    cases index with
    | zero =>
      simp only [parent, sourceSites, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at found
      subst site
      exact ⟨rfl, rfl, rfl, by decide, rfl, by decide, 2, rfl⟩
    | succ index =>
      cases index with
      | zero =>
        simp only [parent, sourceSites, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at found
        subst site
        exact ⟨rfl, rfl, rfl, by decide, rfl, by decide, 3, rfl⟩
      | succ index =>
        cases index with
        | zero =>
          simp only [parent, sourceSites, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at found
          subst site
          exact ⟨rfl, rfl, rfl, by decide, rfl, by decide, 4, rfl⟩
        | succ index => simp [parent, sourceSites] at found

theorem complete_binding : FullSourceBinding parent actual actual :=
  ⟨⟨rfl, rfl, parent_valid⟩, rfl, by decide⟩

def selected : List Node := [producer.node, firstConsumer.node, secondConsumer.node]
def selectedPlan : Plan :=
  ⟨[0, 6, 7, 8],
    [⟨5, "producer", [0], [1], ["tensor[2]i8"], ["tensor[2]i8"]⟩,
     ⟨6, "first", [1], [2], ["tensor[2]i8"], ["tensor[2]i8"]⟩,
     ⟨7, "second", [1], [3], ["tensor[2]i8"], ["tensor[2]i8"]⟩],
    [0], [2, 3], [(-1, 3), (0, 2), (1, 3), (2, 3)]⟩
def fused : FusedBody := ⟨selected, [0], [7, 8]⟩

theorem complete_realization_checked :
    checkRealization parent.fullGraph parent.fullGraph 5 3 [5, 6, 7] selectedPlan fused = true := by decide

theorem dropped_second_output_refused :
    checkRealization parent.fullGraph parent.fullGraph 5 3 [5, 6, 7] selectedPlan
      { fused with outputs := [7] } = false := by decide

theorem dropped_observer_refused :
    checkRealization parent.fullGraph { parent.fullGraph with nodes := parent.fullGraph.nodes.take 8 }
      5 3 [5, 6, 7] selectedPlan fused = false := by decide

theorem observed_producer_requires_publication :
    checkRealization { parent.fullGraph with observations := [6, 7, 8, 9] }
      { parent.fullGraph with observations := [6, 7, 8, 9] } 5 3 [5, 6, 7] selectedPlan fused = false := by decide

def meaning : Primitive → List Nat → Nat
  | .constant _ literal, _ => literal.toNat
  | .binary _ .add, arguments => (arguments[0]?).getD 0 + (arguments[1]?).getD 0
  | .binary _ .mul, arguments => (arguments[0]?).getD 0 * (arguments[1]?).getD 0
  | _, _ => 0
def initial : StateEffects.State (Tensor 2 Nat) := fun slot index => if slot = 0 then index.val + 3 else 0

/-- Initialized mathematical input [3,4] yields producer [9,10], first [15,16],
second [54,60], and unchanged observer [69,76], preserving ordered arithmetic. -/
theorem nonzero_fanout_witness :
    executeSource meaning parent.leading (fun _ => none) parent.sites initial 7 0 = 15 ∧
    executeSource meaning parent.leading (fun _ => none) parent.sites initial 8 0 = 54 ∧
    executeSource meaning parent.leading (fun _ => none) parent.sites initial 9 0 = 69 ∧
    executeSource meaning parent.leading (fun _ => none) parent.sites initial 9 1 = 76 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem complete_parent_execution (initialCells : Prefix.Cells Nat) :
    GraphParentExecution meaning actual initial
      (parent.observations.map (executeSource meaning parent.leading initialCells parent.sites initial)) :=
  .evaluated parent parent_valid initialCells initial rfl

end Merlin.GraphRegionsDeclared.Capture.Fanout

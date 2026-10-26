/- These independently written AST examples are NOT authoritative compiler
exports or actual-pass qualification. They demonstrate the restricted syntax
and closure relation under a common pure mathematical interpretation. -/
import Merlin.GraphRegionsDeclaredCaptureContract

namespace Merlin.GraphRegionsDeclared.Capture.Examples
open MerlinVerify Merlin.GraphRegionsDeclared

def tensor2 : Ty := .tensor [some 2] (.int 8) none

def literalAdd : Program := ⟨[8], [⟨.constant 8 6, []⟩, ⟨.binary 8 .add, [0, 1]⟩], 2, 8⟩
def literalAddStage : ScalarStage :=
  ⟨2, "captured-add", [0], 3, ["tensor[2]i8"], "tensor[2]i8", literalAdd⟩
def literalAddSite : GenericSite := ⟨literalAddStage, [2], [4, 1, 6], 2, 5⟩
def literalAddReferences : List Reference := [⟨1, 1, 0⟩]

def actualOuter : Op :=
  .mk "arith.constant" [⟨1, .int 8⟩] [] [("value", .int 6 (.int 8))] [] [] []

def actualGeneric : Op :=
  .mk "linalg.generic" [⟨3, tensor2⟩] [0, 2]
    [("indexing_maps", .array [.affineMap ⟨1, 0, [.dim 0]⟩, .affineMap ⟨1, 0, [.dim 0]⟩]),
     ("iterator_types", .array [.enum "linalg" "iterator_type" "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])]
    [] [.mk [.mk [⟨4, .int 8⟩, ⟨5, .int 8⟩]
      [.mk "arith.addi" [⟨6, .int 8⟩] [4, 1] [("overflowFlags", .enum "arith" "overflow" "none")] [] [] [],
       .mk "linalg.yield" [] [6] [] [] [] []]]] []

def actualParent : Module :=
  ⟨[.mk "func.func" [] []
    [("function_type", .type (.function [tensor2] [tensor2])), ("sym_name", .str "captured_add")]
    [] [.mk [.mk [⟨0, tensor2⟩]
      [actualOuter, .mk "tensor.empty" [⟨2, tensor2⟩] [] [] [] [] [], actualGeneric,
       .mk "func.return" [] [3] [] [] [] []]]] []], [], []⟩

def declaredClosed : Module :=
  ⟨[.mk "func.func" [] []
    [("function_type", .type (.function [.int 8] [.int 8])), ("sym_name", .str "forward")]
    [] [.mk [.mk [⟨0, .int 8⟩]
      [.mk "arith.constant" [⟨1, .int 8⟩] [] [("value", .int 6 (.int 8))] [] [] [],
       .mk "arith.addi" [⟨2, .int 8⟩] [0, 1] [("overflowFlags", .enum "arith" "overflow" "none")] [] [] [],
       .mk "func.return" [] [2] [] [] [] []]]] []], [], []⟩

theorem literalAdd_region : RegionBinding [actualOuter, .mk "tensor.empty" [⟨2, tensor2⟩] [] [] [] [] []]
    actualGeneric literalAddSite literalAddReferences := by
  refine ⟨rfl, by decide, ?_⟩
  simp [checkReferences, checkReference, isLiteral, literalAddReferences, literalAddSite, literalAddStage,
    literalAdd, actualOuter, instructionMatches, Instruction.operation, Primitive.name, Primitive.attrs, Primitive.width]

theorem literalAdd_parent : ProgramBinding actualParent actualParent literalAddSite literalAddReferences := by
  exact ⟨rfl, actualGeneric, rfl, literalAdd_region⟩

theorem literalAdd_closed (meaning : Primitive → List α → α) (arguments : Environment α) :
    sourceRun meaning [1] 1 literalAdd.instructions
      (cacheLiterals meaning [1] 1 literalAdd.instructions arguments) literalAdd.returned =
        literalAdd.evaluate meaning arguments :=
  initialized_closure_result meaning [1] literalAdd (by decide) (by decide) arguments

/-- First-use materialization is interleaved: multiplication precedes the
captured literal, followed by addition. No reassociation, CSE or DCE law is used. -/
def interleaved : Program :=
  ⟨[8, 8], [⟨.binary 8 .mul, [0, 1]⟩, ⟨.constant 8 6, []⟩, ⟨.binary 8 .add, [2, 3]⟩], 4, 8⟩

theorem interleaved_closed (meaning : Primitive → List α → α) (arguments : Environment α) :
    sourceRun meaning [3] 2 interleaved.instructions
      (cacheLiterals meaning [3] 2 interleaved.instructions arguments) interleaved.returned =
        interleaved.evaluate meaning arguments :=
  initialized_closure_result meaning [3] interleaved (by decide) (by decide) arguments

/-- The wrong dominating definition is refused structurally even though it has
the same integer type. Compiler validity/native execution remain unmeasured. -/
theorem wrong_prefix_refused : checkReference
    [actualOuter, .mk "tensor.empty" [⟨2, tensor2⟩] [] [] [] [] []]
    literalAdd [4, 1, 6] ⟨1, 1, 1⟩ = false := by decide

end Merlin.GraphRegionsDeclared.Capture.Examples

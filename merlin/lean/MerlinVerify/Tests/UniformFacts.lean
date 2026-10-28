/- Producer identity and dependency invalidation for the splat SSA bridge. -/
import MerlinVerify.Check.UniformFacts

namespace MerlinVerify.Tests.UniformFacts

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.UniformFacts MerlinVerify.Sem.Ops

def i32 : Ty := .int 32
def producer (operands : List ValueId := [0])
    (results : List Value := [⟨1, .tensor [some 2, some 3] i32 none⟩])
    (name : String := "tensor.splat") (props : List (String × Attr) := []) : Op :=
  .mk name results operands [] props [] []
def splat : Op := producer
def fact : Fact := ⟨1, 0, [2, 3], i32⟩
def write (id : ValueId) : Op := .mk "arith.constant" [⟨id, i32⟩] []
  [("value", .int 7 i32)] [] [] []

#guard learn splat == some fact
#guard scan [] [splat] == [fact]
#guard scan [] [splat, write 2] == [fact]
#guard scan [] [splat, write 0] == []
#guard scan [] [splat, write 1] == []
#guard scan [] [splat, write 0, write 2] == []
#guard scan [] [splat, write 0, splat] == [fact]
#guard BindingFacts.forget model [0, 0] [fact] == []
#guard advance [fact, ⟨5, 4, [3], i32⟩] (write 0) == [⟨5, 4, [3], i32⟩]

-- Rebinding the old tensor through a new splat replaces its payload fact.
#guard advance [fact] (producer (operands := [3])) == [⟨1, 3, [2, 3], i32⟩]
-- A scalar/result collision would destroy the payload binding used by the fact.
#guard learn (producer (operands := [1])) == none
#guard learn (producer (name := "tensor.cast")) == none
#guard learn (producer (operands := [0, 2])) == none
#guard learn (producer (operands := [])) == none
#guard learn (producer (results := [])) == none
#guard learn (producer (results := [⟨1, i32⟩])) == none
#guard learn (producer (results := [⟨1, .tensor [none] i32 none⟩])) == none
#guard learn (producer (results := [⟨1, .tensor [some 2] i32 (some .unit)⟩])) == none
#guard learn (producer (props := [("unknown_semantics", .unit)])) == none
#guard learn (producer (results := [⟨1, .tensor [] i32 none⟩])) == some ⟨1, 0, [], i32⟩
#guard learn (producer (results := [⟨1, .tensor [some 0] i32 none⟩])) == some ⟨1, 0, [0], i32⟩

-- The fact theorem uses actual execution and does not assume initialized input values.
example (ctx : Ctx) (call : CallSem) :
    Holds fact [(1, .tensor (Tensor.full [2, 3] i32 .poison)), (0, .scalar .poison)] := by
  have hs : stepOp opSem ctx call [(0, .scalar .poison)] splat =
      .ok [(1, .tensor (Tensor.full [2, 3] i32 .poison)), (0, .scalar .poison)] := by cbv
  exact learn_sound (by cbv) hs

end MerlinVerify.Tests.UniformFacts

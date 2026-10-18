/- Transitive provenance guards include malformed SSA and stale dependency bindings. -/
import MerlinVerify.Check.ReshapeFacts
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ReshapeChain

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def first : ReshapeFacts.Fact := ⟨1, 0, [2, 3, 2, 2], [6, 4], .int 32⟩
def second : ReshapeFacts.Fact := ⟨2, 1, [6, 4], [2, 3, 4], .int 32⟩
def third : ReshapeFacts.Fact := ⟨3, 2, [2, 3, 4], [24], .int 32⟩
def secondFacts := ReshapeFacts.enrich [first] second ++ [first]
def ultimate : ReshapeFacts.Fact := ⟨3, 0, [2, 3, 2, 2], [24], .int 32⟩
def thirdFacts := ReshapeFacts.enrich secondFacts third ++ secondFacts

#guard secondFacts.length == 3
#guard thirdFacts.length == 5
#guard thirdFacts.head? == some (⟨3, 0, [2, 3, 2, 2], [24], .int 32⟩ : ReshapeFacts.Fact)
#guard (thirdFacts.drop 1).head? == some third
#guard !(decide (ReshapeFacts.Composable first {second with original := [3, 8]}))
#guard !(decide (ReshapeFacts.Composable first {second with shape := [23]}))
#guard !(decide (ReshapeFacts.Composable first {second with element := .int 64}))
#guard !(decide (ReshapeFacts.Composable first {second with root := 99}))

-- The ultimate relation no longer reads former intermediates. Its direct fact still does.
#guard (ReshapeFacts.advance [] thirdFacts (mkOp "unknown.write" [v 1 i32] [])).contains ultimate
#guard (ReshapeFacts.advance [] thirdFacts (mkOp "unknown.write" [v 2 i32] [])).contains ultimate
#guard !(ReshapeFacts.advance [] thirdFacts (mkOp "unknown.write" [v 0 i32] [])).contains ultimate
#guard !(ReshapeFacts.advance [] thirdFacts (mkOp "unknown.write" [v 3 i32] [])).contains ultimate
#guard (ReshapeFacts.advance [] thirdFacts (mkOp "unknown.write" [v 99 i32] [])) == thirdFacts

-- Rebinding the ultimate source root may retain only an independently live intermediate root.
def shadowRoot : Op := mkOp "tensor.collapse_shape" [v 0 (tensorTy [24] i32)] [2]
  [("reassociation", .array [.array [.int 0 (.int 64), .int 1 (.int 64), .int 2 (.int 64)]])]
def afterRootWrite := ReshapeFacts.advance [(2, .tensor [2, 3, 4] (.int 32))] secondFacts shadowRoot
#guard afterRootWrite.head? == some (⟨0, 1, [6, 4], [24], .int 32⟩ : ReshapeFacts.Fact)
#guard !(afterRootWrite.contains (⟨0, 0, [2, 3, 2, 2], [24], .int 32⟩ : ReshapeFacts.Fact))

/-- Even long chains add only immediate and ultimate facts, rather than every ancestral path. -/
def chain (count : Nat) : List ReshapeFacts.Fact :=
  (List.range count).foldl (fun facts n =>
    let last : ReshapeFacts.Fact := ⟨n + 1, n, [6], [6], .int 32⟩
    ReshapeFacts.enrich facts last ++ facts) []

#guard (chain 128).length == 255
#guard ((chain 128).head?).map ReshapeFacts.Fact.root == some 0
#guard ((chain 128).head?).map ReshapeFacts.Fact.result == some 128

end MerlinVerify.Tests.ReshapeChain

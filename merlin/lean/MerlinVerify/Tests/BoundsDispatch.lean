/- Dispatch reads actual destination facts once. A sound but deliberately nonmonotone body
book exposes the conservative coverage difference from retrying initialized outs as dirty. -/
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.BoundsDispatch

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.Definedness
open IndexRanges

def generic : Op := mkOp "linalg.generic" [v 20 (tensorTy [] i32)] [0]
  [("indexing_maps", .array [amap 0 []]), ("iterator_types", .array []),
   ("operandSegmentSizes", .denseArray (.int 32) [0, 1])] []
  [.mk [.mk [v 10 i32]
    [mkOp "arith.constant" [v 11 i32] [] [("value", .int 7 i32)], mkOp "linalg.yield" [] [11]]]]

def destination (initialized : Bool) : List IndexRanges.Fact :=
  [Refined.emptyFact model ⟨.tensor [] i32, initialized⟩]

/-- Filtering a proved book remains sound. This filter intentionally distinguishes stronger
initialized facts from weaker facts, so it must not silently trigger another recursive route. -/
def guardedBody (initialized : Bool) : Refined.Book model Sem.Ops.opSem 64 where
  rule Γ op fs :=
    if (Γ.lookup 10).map (·.base.initialized) = some initialized then
      (Bounds.leafBook 64).rule Γ op fs else none
  sound Γ op fs rs h ctx call env args hi he ha := by
    split at h
    · exact (Bounds.leafBook 64).sound Γ op fs rs h ctx call env args hi he ha
    · cases h

def oldRoute (initialized : Bool) : Option (List IndexRanges.Fact) :=
  (Bounds.initializedRule 64 (guardedBody false) [] generic (destination initialized)).orElse fun _ =>
    Bounds.overwriteRule 64 (guardedBody false) [] generic (destination initialized)

#guard ((Bounds.genericBook 64 (Bounds.leafBook 64)).rule [] generic (destination true)).isSome
#guard ((Bounds.genericBook 64 (Bounds.leafBook 64)).rule [] generic (destination false)).isSome
#guard ((Bounds.genericBook 64 (guardedBody true)).rule [] generic (destination true)).isSome
#guard ((Bounds.genericBook 64 (guardedBody true)).rule [] generic (destination false)).isNone
#guard ((Bounds.genericBook 64 (guardedBody false)).rule [] generic (destination false)).isSome
#guard ((Bounds.genericBook 64 (guardedBody false)).rule [] generic (destination true)).isNone
#guard (oldRoute true).isSome -- Explicit lost fallback for an arbitrary nonmonotone body book.
#guard (oldRoute false).isSome
#guard ((Bounds.genericBook 64 (Bounds.leafBook 64)).rule []
  (mkOp "linalg.generic" [v 20 (tensorTy [] i32)] [0]) (destination true)).isNone
#guard ((Bounds.genericBook 64 (Bounds.leafBook 64)).rule [] generic []).isNone

end MerlinVerify.Tests.BoundsDispatch

/- Shared fact-preserving views under the additive source interpretation. The same
operation proofs serve finite, literal-aware and bounded W8A8 input domains. -/
import MerlinVerify.Check.DefinedStaticSlice
import MerlinVerify.Check.DefinedConcat
import MerlinVerify.Check.DefinedReshapeView
import MerlinVerify.Check.DefinedRefinedSplat
import MerlinVerify.Check.W8A8SourceBook

namespace MerlinVerify.Check.PolicyW8A8ViewBook

open MerlinVerify Definedness

def book {Info : Type} {M : Refined.Model Info} (elementwise : Refined.Elementwise M) (ib : Nat)
    (rules : Contextual.Structured.Invariants M ib) : Refined.Book M Rules.W8A8SourceSemantics.opSem ib :=
  W8A8SourceBook.lift ((Refined.Splat.book elementwise ib).orElse
    ((Refined.ReshapeView.book elementwise ib).orElse
      ((Refined.StaticSlice.book elementwise ib).orElse (Refined.Concat.book elementwise ib rules))))

#print axioms book

end MerlinVerify.Check.PolicyW8A8ViewBook

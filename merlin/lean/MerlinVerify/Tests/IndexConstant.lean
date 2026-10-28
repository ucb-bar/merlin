/- Exact literal normalization at actual index width, independent of hints and SSA names. -/
import MerlinVerify.Rules.IndexConstant
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IndexConstant

open MerlinVerify MerlinVerify.Rules.IndexConstant

def constant (id : Nat) (n : Int) : Op :=
  .mk "arith.constant" [⟨id,.index⟩] [] [("value",.int n .index)] [] [] []
def normalized (ib : Nat) (n : Int) := replacement? ib (constant 7 n)
def payload (s : Option Op) : Option Attr := s.bind (fun op => op.attr? "value")

#guard payload (normalized 32 (-9223372036854775805)) == some (.int 3 .index)
#guard payload (normalized 32 3) == some (.int 3 .index)
#guard payload (normalized 64 (-9223372036854775805)) == some (.int (-9223372036854775805) .index)
#guard payload (normalized 32 4294967295) == some (.int (-1) .index)
#guard payload (normalized 64 4294967295) == some (.int 4294967295 .index)
#guard payload (normalized 32 2147483648) == some (.int (-2147483648) .index)
#guard payload (normalized 64 2147483648) == some (.int 2147483648 .index)
#guard payload (normalized 1 1) == some (.int (-1) .index)
#guard payload (normalized 0 123) == some (.int 0 .index)
#guard (normalized 32 3).map (·.results.map (·.id)) == some [7]
#guard payload ((normalized 32 3).bind (replacement? 32)) == payload (normalized 32 3)
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.int 32⟩] []
  [("value",.int 3 (.int 32))] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.int 3 .index),("unknown.meaning",.unit)] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] [99]
  [("value",.int 3 .index)] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [] []
  [("value",.int 3 .index)] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩,⟨8,.index⟩] []
  [("value",.int 3 .index)] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.int 3 .index)] [] [] [0])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.int 3 .index)] [] [.mk []] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] [] [] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.addi" [⟨7,.index⟩] [] [] [] [] [])).isNone

-- Tensor payloads are refused before dense decoding, including a malformed scalar header.
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.tensor [some 1] .index none⟩] []
  [("value",.dense (.tensor [some 1] .index none) true [3,0,0,0])] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.dense (.tensor [some 1] .index none) true [3,0,0,0])] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.opaque "poison")] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.str "poison")] [] [] [])).isNone
#guard (replacement? 32 (.mk "arith.constant" [⟨7,.index⟩] []
  [("value",.float 0 (.float .f32))] [] [] [])).isNone

end MerlinVerify.Tests.IndexConstant

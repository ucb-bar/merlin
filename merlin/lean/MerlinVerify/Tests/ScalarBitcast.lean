/- Raw-bit reinterpretation, promise refusal and the existing float-to-float support boundary. -/
import MerlinVerify.Check.RulesBitcast
import MerlinVerify.Check.DefinedBitcast
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ScalarBitcast

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.ScalarBitcast

def op (t : Ty) (attrs : List (String × Attr) := []) : Op := .mk "arith.bitcast" [⟨1,t⟩] [0] attrs [] [] []
def clean (t : Ty) : List Definedness.Fact := [⟨.scalar t,true⟩]
def supports (src dst : Ty) := (Bitcast.rule 0 (op dst) [.scalar src] [.scalar dst]).isSome

#guard supports (.int 1) (.int 1)
#guard supports (.int 13) (.int 13)
#guard supports (.int 16) (.float .f16)
#guard supports (.int 16) (.float .bf16)
#guard supports (.float .f16) (.int 16)
#guard supports (.float .bf16) (.int 16)
#guard supports (.int 32) (.float .f32)
#guard supports (.float .f64) (.int 64)
#guard !supports (.int 0) (.int 0)
#guard !supports (.int 32) (.int 64)
#guard !supports .index (.int 64)
#guard !supports (.float .f16) (.float .bf16)
#guard !supports (.float .f32) (.float .f32)
#guard (DefinedBitcast.rule (op (.float .f32)) (clean (.int 32))).isSome
#guard (DefinedBitcast.rule (op (.int 64)) (clean (.float .f64))).isSome
#guard (DefinedBitcast.rule (op (.float .f32)) [⟨.scalar (.int 32),false⟩]).isNone
#guard (DefinedBitcast.rule (op (.float .f32)) [⟨.tensor [] (.int 32),true⟩]).isNone
#guard (DefinedBitcast.rule (op (.float .f32)) (clean (.int 64))).isNone
#guard (DefinedBitcast.rule (op (.float .f32) [("nonNeg",.unit)]) (clean (.int 32))).isNone
#guard (DefinedBitcast.rule (op (.float .f32) [("fastmath",.enum "arith" "fastmath" "nnan")]) (clean (.int 32))).isNone
#guard ((DefinedBitcast.book 64).rule [] (op (.float .f32) [("unknown.promise",.unit)]) (clean (.int 32))).isNone

def bits : Outcome (List RVal) → Option Nat
  | .ok [.scalar (.some (.bits _ x))] => some x.toNat
  | .ok [.scalar (.some (.flt _ x))] => some x.toNat
  | _ => none
def run (t : Ty) (v : Val) := Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls") (op t) [.scalar v] []
#guard bits (run (.float .f32) (.some (.bits 32 0x7fa01234))) == some 0x7fa01234
#guard bits (run (.int 64) (.some (.flt .f64 0x8000000000000000))) == some 0x8000000000000000
#guard (match run (.float .f32) .poison with | .ok [.scalar .poison] => true | _ => false)
#guard (match run (.float .f32) (.some (.flt .f32 0)) with | .stuck _ => true | _ => false)

end MerlinVerify.Tests.ScalarBitcast

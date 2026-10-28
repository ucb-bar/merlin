/- Parametric scalar floating source totality, without numerical laws. -/
import MerlinVerify.Check.DefinedFloating
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.DefinedFloating

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Tests.Semantics

def unary (name : String) (kind : FKind) : Op :=
  mkOp name [v 1 (.float kind)] [0]

#guard (["arith.negf"] ++ mathUnaryNames).all fun name =>
  [FKind.f16, .bf16, .f32, .f64].all fun kind =>
    (floatingBook 64).rule [] (unary name kind) [⟨.scalar (.float kind), true⟩] ==
      some [⟨.scalar (.float kind), true⟩]
#guard ((floatingBook 64).rule [] (unary "math.exp" .f32) [⟨.scalar (.float .f32), false⟩]).isNone
#guard ((floatingBook 64).rule [] (unary "math.exp" .f64) [⟨.scalar (.float .f32), true⟩]).isNone
#guard ((floatingBook 64).rule [] (unary "math.unreviewed" .f32) [⟨.scalar (.float .f32), true⟩]).isNone
#guard ((floatingBook 64).rule [] (unary "math.fma" .f32) [⟨.scalar (.float .f32), true⟩]).isNone

def fast : Op := mkOp "math.sqrt" [v 1 (.float .f32)] [0] [("fastmath", .enum "arith" "fastmath" "nnan")]
#guard ((floatingBook 64).rule [] fast [⟨.scalar (.float .f32), true⟩]).isNone

def fma : Op := mkOp "math.fma" [v 3 (.float .f32)] [0,1,2]
#guard (floatingBook 64).rule [] fma (List.replicate 3 ⟨.scalar (.float .f32), true⟩) ==
  some [⟨.scalar (.float .f32), true⟩]
#guard ((floatingBook 64).rule [] fma
  [⟨.scalar (.float .f32), true⟩, ⟨.scalar (.float .f32), false⟩, ⟨.scalar (.float .f32), true⟩]).isNone

end MerlinVerify.Tests.DefinedFloating

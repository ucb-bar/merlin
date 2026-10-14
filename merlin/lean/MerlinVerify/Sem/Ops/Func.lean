/-
Op family: `func.call` (design §4.3). A call to a defined function denotes its body (the evaluator
supplies `call`); a call to a declaration denotes `ctx.ext` (design D8). `func.return` is a
terminator handled by the evaluator. `func.call_indirect`/`func.constant` are `stuck`.
-/
import MerlinVerify.Sem.Support

namespace MerlinVerify.Sem.Ops.Func

open MerlinVerify

def sem : OpSem := fun _ call op args _ =>
  match op.name with
  | "func.call" =>
    match attrSymbol? op "callee" with
    | some f => (call f args).bind fun rs =>
        if rs.length = op.results.length then .ok rs else .stuck s!"call @{f}: result count"
    | none => .stuck "func.call without a flat callee symbol"
  | name => .stuck s!"unsupported op {name}"

end MerlinVerify.Sem.Ops.Func

/- Typed bitcast identities retain exact payloads, promises, escapes, captures and stale SSA. -/
import MerlinVerify.Check.BitcastCancel
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.BitcastCancel

open MerlinVerify MerlinVerify.Check
open MerlinVerify.Check.BitcastCancel

def Γ (src : Ty) : TEnv := [(0, .scalar src), (1, .scalar (.int 32))]
def cast (input result : Nat) (dst : Ty) : Op := .mk "arith.bitcast" [⟨result, dst⟩] [input] [] [] [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def source (src mid : Ty) (escape : Bool := false) : List Op :=
  [cast 0 2 mid, cast 2 3 src, ret (if escape then [3, 2, 0, 1] else [3, 0, 1])]
def normalized (src mid : Ty) (escape : Bool := false) := normalize 64 (Γ src) [] (source src mid escape)

def pairs : List (Ty × Ty) :=
  [(.int 1, .int 1), (.int 13, .int 13), (.int 128, .int 128)] ++
  ([FKind.f16, .bf16, .f32, .f64].flatMap fun k => [(.int k.width, .float k), (.float k, .int k.width)])

#guard pairs.all fun (src, mid) =>
  (normalized src mid).map (fun ops => ops.map (·.operands)) == some [[0, 0, 1]]
#guard pairs.all fun (src, mid) =>
  (normalized src mid true).map (fun ops => ops.map (·.name)) ==
    some (if src == mid then ["func.return"] else ["arith.bitcast", "func.return"])
#guard [1, 8, 13, 32, 64, 128].all fun w =>
  (normalize 32 (Γ (.int w)) [] [cast 0 2 (.int w), ret [2, 0, 1]]).map
    (fun ops => ops.map (·.operands)) == some [[0, 0, 1]]
#guard (normalize 64 [] [] (source (.int 32) (.float .f32))).isNone

-- A legal-but-unmodelled float-to-float bitcast never becomes an identity proof.
#guard ([FKind.f16, .bf16, .f32, .f64].all fun k =>
  (read (Γ (.float k)) (cast 0 2 (.float k))).isNone)
#guard (read (Γ (.float .f16)) (cast 0 2 (.float .bf16))).isNone
#guard (read (Γ .index) (cast 0 2 (.int 64))).isNone
#guard (read (Γ (.int 0)) (cast 0 2 (.int 0))).isNone
#guard (read (Γ (.int 32)) (cast 0 2 (.float .f64))).isNone
#guard (read [(0, .tensor [] (.int 32))] (cast 0 2 (.float .f32))).isNone

def flagged (key : String) (value : Attr) : Op :=
  .mk "arith.bitcast" [⟨2, .float .f32⟩] [0] [(key, value)] [] [] []
#guard ["nonNeg", "isExact", "disjoint", "unreviewed.meaning"].all fun key =>
  (read (Γ (.int 32)) (flagged key .unit)).isNone
#guard (read (Γ (.int 32)) (flagged "fastmath" (.enum "arith" "fastmath" "nnan"))).isNone
#guard (read (Γ (.int 32)) (flagged "overflowFlags" (.enum "arith" "overflow" "nsw"))).isNone
#guard (read (Γ (.int 32)) (flagged "roundingmode" (.int 0 (.int 32)))).isNone
#guard (read (Γ (.int 32)) (flagged "fastmath" (.enum "arith" "fastmath" "none"))).isSome
#guard (read (Γ (.int 32)) (flagged "prov.fqn" (.str "example"))).isSome
#guard (read (Γ (.int 32)) (.mk "arith.bitcast" [⟨2, .float .f32⟩] [0] [] [] [.mk []] [])).isNone
#guard (read (Γ (.int 32)) (.mk "arith.bitcast" [⟨2, .float .f32⟩] [0] [] [] [] [4])).isNone

-- Facts retain only current root/result bindings; overwrites invalidate either dependency.
def overwrite (id : Nat) (ty : Ty) : Op := .mk "arith.constant" [⟨id, ty⟩] [] [("value", .int 7 ty)] [] [] []
#guard (normalize 64 (Γ (.int 32)) []
  [cast 0 2 (.float .f32), overwrite 0 (.int 32), cast 2 3 (.int 32), ret [3, 0, 1]]).map List.length == some 4
#guard (normalize 64 (Γ (.int 32)) []
  [cast 0 2 (.float .f32), cast 0 2 (.float .f32), cast 2 3 (.int 32), ret [3, 0, 1]]).map
  (fun ops => ops.any (fun op => op.name == "arith.bitcast")) == some true
#guard (learn (Γ (.int 32)) (cast 0 0 (.float .f32))).isNone
#guard !(drop? (Γ (.int 32)) (cast 0 0 (.float .f32)) [ret [1]])
def captures (id : Nat) : Op := .mk "test.region" [] [] [] [] [.mk [.mk [] [ret [id]]]] []
#guard !(drop? (Γ (.int 32)) (cast 0 2 (.float .f32)) [captures 2, ret [1]])
#guard (normalized (.int 32) (.float .f32) true).map (fun ops => ops.map (·.operands)) ==
  some [[0], [0, 2, 0, 1]]
#guard (normalize 64 (Γ (.int 32)) []
  [cast 0 2 (.float .f32), cast 2 3 (.int 32), captures 3, ret [0, 1]]).map
  (fun ops => ops.map (·.name)) == some ["test.region", "func.return"]

-- Exact observations include float format and raw payload, never floating-point numeric equality.
def word : Val → Option (String × Nat)
  | .poison => none
  | .some (.bits _ bits) => some ("integer", bits.toNat)
  | .some (.flt format bits) => some (reprStr format, bits.toNat)
def observe : Outcome (List RVal) → Option (List (Option (String × Nat)))
  | .ok values => values.mapM fun | .scalar value => some (word value) | .tensor _ => none
  | _ => none
def run (ops : List Op) (value : Val) :=
  observe (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0, .scalar value), (1, .scalar (.some (.bits 32 37)))] ops)
#guard pairs.all fun (src, mid) =>
  run (source src mid true) .poison == run ((normalized src mid true).getD []) .poison
#guard (List.range 256).all fun n =>
  let value := Val.some (.bits 8 (BitVec.ofNat 8 n))
  run (source (.int 8) (.int 8)) value == run ((normalized (.int 8) (.int 8)).getD []) value
#guard [FKind.f16, .bf16, .f32, .f64].all fun k =>
  [0, 1, 2^k.width-1, 2^(k.width-1), 32705, 2139095041, 9221120237041095220].all fun n =>
    let bits := BitVec.ofNat k.width n
    (run (source (.int k.width) (.float k) true) (.some (.bits k.width bits)) ==
      run ((normalized (.int k.width) (.float k) true).getD []) (.some (.bits k.width bits))) &&
    (run (source (.float k) (.int k.width) true) (.some (.flt k bits)) ==
      run ((normalized (.float k) (.int k.width) true).getD []) (.some (.flt k bits)))

end MerlinVerify.Tests.BitcastCancel

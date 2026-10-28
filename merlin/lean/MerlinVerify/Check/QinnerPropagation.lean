/- Independently reconstruct consumer-to-empty requests. The production walk traverses arbitrary
operand-zero chains; this checker admits only explicit tensor view wrappers, up to eight owners.
Conflicts and duplicate archive keys fail closed rather than endorsing last-writer binding. -/
import MerlinVerify.Rules.QinnerPropagation
namespace MerlinVerify.Check.QinnerPropagation
open MerlinVerify
abbrev Tags := List (Nat × String)
mutual
  def flatten : List Op → List Op
    | [] => []
    | op@(.mk _ _ _ _ _ rs _)::rest => op :: flatRegions rs ++ flatten rest
  def flatRegions : List Region → List Op
    | [] => []
    | .mk bs::rs => flatBlocks bs ++ flatRegions rs
  def flatBlocks : List Block → List Op
    | [] => []
    | .mk _ ops::bs => flatten ops ++ flatBlocks bs
end

def views : List String := ["tensor.cast","tensor.collapse_shape","tensor.expand_shape",
  "tensor.reshape","tensor.extract_slice","tensor.bitcast","linalg.transpose"]

def origin (ops : List Op) : Nat → Nat → Option Nat
  | 0,_ => none
  | fuel+1,id => do
    let producer ← ops.find? (fun op => op.results.any (fun r => r.id == id))
    if producer.name == "tensor.empty" then
      if producer.results.length == 1 then some id else none
    else if views.contains producer.name then
      origin ops fuel (← producer.operands.head?)
    else none

def add (tags : Tags) (id : Nat) (key : String) : Option Tags :=
  if key.isEmpty then none else
  match tags.lookup id with
  | some previous => if previous == key then some tags else none
  | none => if tags.any (fun p => p.2 == key) then none else some (tags ++ [(id,key)])

def direct (ops : List Op) : Option Tags := ops.foldlM (fun tags op =>
  match op.attr? "prov.quant_inner" with
  | none => some tags
  | some (.str key) => match op.name,op.results with
    | "tensor.empty",[r] => add tags r.id key
    | _,_ => none
  | _ => none) []

def index? (key : String) : Option Nat :=
  let suffix := (key.drop "prov.quant_inner_".length).toString
  if suffix == "w" then some 0 else if suffix == "s" then some 1 else suffix.toNat?

def requests (ops : List Op) (initial : Tags) : Option Tags :=
  ops.foldlM (fun tags op => op.attrs.foldlM (fun tags pair =>
    if pair.1.startsWith "prov.quant_inner_" then do
      let index ← index? pair.1
      let .str key := pair.2 | none
      let operand ← op.operands[index]?
      let id ← origin ops 8 operand
      add tags id key
    else some tags) tags) initial

mutual
  def attach (tags : Tags) : List Op → List Op
    | [] => []
    | .mk n r o a p rs s :: rest =>
      let attrs := if n == "tensor.empty" then
        match r with
        | [value] => match tags.lookup value.id with
          | some key => ((a.filter (fun pair => pair.1 != "prov.quant_inner")) ++
              [("prov.quant_inner",Attr.str key)]).mergeSort (fun a b => a.1 ≤ b.1)
          | none => a
        | _ => a
        else a
      .mk n r o attrs p (attachRegions tags rs) s :: attach tags rest
  def attachRegions (tags : Tags) : List Region → List Region
    | [] => []
    | .mk bs::rs => .mk (attachBlocks tags bs) :: attachRegions tags rs
  def attachBlocks (tags : Tags) : List Block → List Block
    | [] => []
    | .mk args ops::bs => .mk args (attach tags ops) :: attachBlocks tags bs
end

def expected? (m : Module) : Option Module := do
  let ops := flatten m.ops
  let tags ← requests ops (← direct ops)
  return {m with ops := attach tags m.ops}

def check (s t : Module) : Bool :=
  match expected? s with
  | none => false
  | some expected => ExactSyntax.modules expected t && Rules.QinnerPropagation.check s t

theorem expected_output {s t : Module} (h : check s t = true) : expected? s = some t := by
  unfold check at h
  split at h
  · cases h
  · next expected found =>
    simp only [Bool.and_eq_true] at h
    rw [ExactSyntax.modules_sound h.1] at found
    exact found

theorem check_sound {s t : Module} (h : check s t = true)
    (ctx : Ctx) (name : String) (args : List RVal) :
    denote Sem.Ops.opSem ctx s name args = denote Sem.Ops.opSem ctx t name args := by
  unfold check at h
  split at h
  · cases h
  · simp only [Bool.and_eq_true] at h
    exact Rules.QinnerPropagation.complete_program h.2 ctx name args
#print axioms expected_output
#print axioms check_sound
end MerlinVerify.Check.QinnerPropagation

/- Complete boundary checker. Census, ABI order, types, function signatures, every operation and
all outputs participate. Binding intent remains provisional; no frozen specification is changed. -/
import MerlinVerify.Check.QinnerProgram
import MerlinVerify.Check.QinnerPropagation
namespace MerlinVerify.Check.Qinner
open MerlinVerify MerlinVerify.Check Rules.QinnerBinding

mutual
  def sitesOps : List Op → List Site
    | [] => []
    | op@(.mk _ _ _ _ _ rs _) :: rest => (site? op).toList ++ sitesRegions rs ++ sitesOps rest
  def sitesRegions : List Region → List Site
    | [] => []
    | .mk bs :: rest => sitesBlocks bs ++ sitesRegions rest
  def sitesBlocks : List Block → List Site
    | [] => []
    | .mk _ ops :: rest => sitesOps ops ++ sitesBlocks rest
end

def signature (op : Op) : Bool :=
  if op.name != "func.func" then true else
  match op.attr? "function_type",op.regions with
  | some (.type (.function ins _)), [.mk [.mk args _]] => decide (ins = args.map (·.ty))
  | some (.type (.function _ _)), [.mk []] => true
  | _,_ => false

def skeleton (m : Module) : Module :=
  { m with ops := m.ops.map fun op => match op with
    | .mk n r o a p rs s => .mk n r o
      (if n == "func.func" && op.attr? "sym_name" == some (.str "forward") then
        a.filter (fun pair => pair.1 != "function_type") else a)
      (if n == "func.func" && op.attr? "sym_name" == some (.str "forward") then
        p.filter (fun pair => pair.1 != "function_type") else p)
      (if n == "func.func" then [] else rs) s }

def outputs? (m : Module) : Option (List Ty) := do
  let op ← m.ops.find? (fun op => op.name == "func.func" && op.attr? "sym_name" == some (.str "forward"))
  let .type (.function _ outs) ← op.attr? "function_type" | none
  return outs

def slots (s t : Module) : List Slot :=
  let found := sitesOps s.ops
  let args := ((t.func? "forward").map fun f => f.body.blocks.head!.args).getD []
  let old := ((s.func? "forward").map fun f => f.body.blocks.head!.args.length).getD 0
  (found.zip (args.drop old)).map fun (site,arg) => ⟨site.key,arg⟩

def wellFormedSites (s : Module) : Bool :=
  let found := sitesOps s.ops
  decide (found.map (·.key)).Nodup &&
  found.all (fun site => (site.result.ty.staticTensor?).isSome &&
    match site.result.ty with
    | .tensor _ elt none => [.int 1,.int 8,.int 16,.int 32,.int 64,.float .f16,.float .f32,.float .f64].contains elt
    | _ => false)

def planCheck (s t : Module) : Bool :=
  let found := sitesOps s.ops
  let abi := slots s t
  found.length == abi.length &&
  (found.zip abi).all (fun (site,slot) => site.key == slot.key && site.result.ty == slot.arg.ty)

def check (s t : Module) : Bool :=
  if (sitesOps s.ops).isEmpty then ExactSyntax.modules s t else
  QinnerPropagation.check s s && wellFormedSites s && planCheck s t && s.ops.all signature && t.ops.all signature &&
  outputs? s == outputs? t && (outputs? s).isSome &&
  ExactSyntax.modules (skeleton s) (skeleton t) &&
  QinnerProgram.functionsCheck (slots s t) s.funcs t.funcs

theorem check_sound {s t : Module} (checked : check s t = true)
    (changed : (sitesOps s.ops).isEmpty = false)
    (binding : String → RVal) (ctx : Ctx) {a : Func} (source : s.func? "forward" = some a)
    (args : List RVal) (arity : a.body.blocks.head!.args.length = args.length) :
    denote (sem binding) ctx s "forward" args =
      denote Sem.Ops.opSem ctx t "forward" (args ++ Rules.QinnerABI.values (slots s t) binding) := by
  simp only [check, changed, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true] at checked
  exact QinnerProgram.program_equal checked.2 binding ctx source args arity

theorem unchanged_sound {s t : Module} (checked : check s t = true)
    (unchanged : (sitesOps s.ops).isEmpty = true) : s = t := by
  simp only [check,unchanged,↓reduceIte] at checked
  exact ExactSyntax.modules_sound checked

theorem slot_type_and_key {s t : Module} (checked : planCheck s t = true)
    {site : Site} {slot : Slot} (member : (site,slot) ∈ (sitesOps s.ops).zip (slots s t)) :
    site.key = slot.key ∧ site.result.ty = slot.arg.ty := by
  simp only [planCheck,Bool.and_eq_true] at checked
  simpa only [Bool.and_eq_true, beq_iff_eq] using List.all_eq_true.mp checked.2 (site,slot) member
#print axioms check_sound
#print axioms unchanged_sound
#print axioms slot_type_and_key
end MerlinVerify.Check.Qinner

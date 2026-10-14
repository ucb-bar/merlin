/-
Typing rules and safety proofs for the `arith` and `math` families (design §4.5, progress).

The family rule is a list of small rules, one per element signature (integer binary ops,
comparisons, casts, float binary ops, constants, `select`, ...). Each comes with a proof that on
typed operands the op is never `stuck` and its result has the declared type. An op or a typing
that no rule accepts is refused.
-/
import MerlinVerify.Check.RulesElementwise

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-! ### Scalar typing facts -/

/-- Width of an integer-like scalar type (`iN` or `index`). -/
def intWidth? (ib : Nat) : Ty → Option Nat
  | .int w => some w
  | .index => some ib
  | _ => none

def isInt (ib : Nat) (t : Ty) : Bool := (intWidth? ib t).isSome

def isFlt : Ty → Bool
  | .float _ => true
  | _ => false

theorem isInt_eq {ib : Nat} {t : Ty} (h : isInt ib t = true) : ∃ w, intWidth? ib t = some w :=
  Option.isSome_iff_exists.1 h

theorem isFlt_eq {t : Ty} (h : isFlt t = true) : ∃ k, t = .float k := by
  cases t <;> simp_all [isFlt]

theorem hasTy_intWidth {ib w : Nat} {t : Ty} {v : Val} (ht : intWidth? ib t = some w)
    (h : Val.HasTy ib t v) : v = .poison ∨ ∃ a : BitVec w, v = .some (.bits w a) := by
  cases v with
  | poison => exact Or.inl rfl
  | some s =>
    cases s with
    | bits w' a =>
      right
      cases t <;> simp [intWidth?, Val.HasTy, Scalar.conforms] at ht h
      · subst ht; subst h; exact ⟨a, rfl⟩
      · subst ht; subst h; exact ⟨a, rfl⟩
    | flt k a => cases t <;> simp [intWidth?, Val.HasTy, Scalar.conforms] at ht h

theorem hasTy_float {ib : Nat} {k : FKind} {v : Val} (h : Val.HasTy ib (.float k) v) :
    v = .poison ∨ ∃ x : BitVec k.width, v = .some (.flt k x) := by
  cases v with
  | poison => exact Or.inl rfl
  | some s =>
    cases s with
    | bits w a => simp [Val.HasTy, Scalar.conforms] at h
    | flt k' x =>
      simp only [Val.HasTy, Scalar.conforms, beq_iff_eq] at h
      subst h
      exact Or.inr ⟨x, rfl⟩

theorem hasTy_bits {ib w : Nat} {t : Ty} (ht : intWidth? ib t = some w) (a : BitVec w) :
    Val.HasTy ib t (.some (.bits w a)) := by
  cases t <;> simp_all [intWidth?, Val.HasTy, Scalar.conforms]

theorem hasTy_flt (ib : Nat) (k : FKind) (x : BitVec k.width) : Val.HasTy ib (.float k) (.some (.flt k x)) := by
  simp [Val.HasTy, Scalar.conforms]

theorem hasTy_poison (ib : Nat) (t : Ty) : Val.HasTy ib t .poison := trivial

theorem hasTy_bool (ib : Nat) (b : Bool) : Val.HasTy ib (.int 1) (Arith.boolVal b) := by
  simp [Arith.boolVal, Val.HasTy, Scalar.conforms]

/-- Typed values of element types that all equal `re` have type `re`. -/
theorem ValsTyped.of_all {ib : Nat} {re : Ty} {vs : List Val} {elts : List Ty} (h : ValsTyped ib vs elts)
    (hall : ∀ t ∈ elts, t = re) : ∀ v ∈ vs, Val.HasTy ib re v := by
  unfold ValsTyped at h
  induction h with
  | nil => simp
  | cons hv _ ih =>
    intro u hu
    rcases List.mem_cons.1 hu with rfl | hu
    · rw [← hall _ (List.mem_cons_self ..)]; exact hv
    · exact ih (fun t ht => hall t (List.mem_cons_of_mem _ ht)) u hu

/-- Close `Val.HasTy ib t v` where `v` is built from `poison` and `Arith.bv w _` by `if`s, given
`ht : intWidth? ib t = some w`. -/
macro "int_res " ht:term : tactic =>
  `(tactic| ((try dsimp only [Arith.arithWithFlags]) <;> (repeat' split) <;>
    first | exact hasTy_poison _ _ | exact hasTy_bits $ht _))

/-! ### Element functions -/

theorem intBin_safe {ib w : Nat} {tIn tOut : Ty} (htIn : intWidth? ib tIn = some w)
    (f : (w : Nat) → BitVec w → BitVec w → Val) (hf : ∀ a b : BitVec w, Val.HasTy ib tOut (f w a b))
    {x y : Val} (hx : Val.HasTy ib tIn x) (hy : Val.HasTy ib tIn y) :
    Safe (Val.HasTy ib tOut) (Arith.intBin f x y) := by
  rcases hasTy_intWidth htIn hx with rfl | ⟨a, rfl⟩ <;>
    rcases hasTy_intWidth htIn hy with rfl | ⟨b, rfl⟩ <;>
    simp only [Arith.intBin, dite_true] <;>
    first | exact Safe.of_ok (hasTy_poison _ _) | exact Safe.of_ok (hf _ _)

theorem intDiv_safe {ib w : Nat} {tIn tOut : Ty} (htIn : intWidth? ib tIn = some w) (signed : Bool)
    (f : (w : Nat) → BitVec w → BitVec w → Val) (hf : ∀ a b : BitVec w, Val.HasTy ib tOut (f w a b))
    {x y : Val} (hx : Val.HasTy ib tIn x) (hy : Val.HasTy ib tIn y) :
    Safe (Val.HasTy ib tOut) (Arith.intDiv signed f x y) := by
  rcases hasTy_intWidth htIn hx with rfl | ⟨a, rfl⟩ <;>
    rcases hasTy_intWidth htIn hy with rfl | ⟨b, rfl⟩ <;>
    simp only [Arith.intDiv, dite_true] <;> (repeat' split) <;>
    first | exact Safe.of_ub | exact Safe.of_ok (hasTy_poison _ _) | exact Safe.of_ok (hf _ _)

theorem intUn_safe {ib w : Nat} {tIn tOut : Ty} (htIn : intWidth? ib tIn = some w)
    (f : (w : Nat) → BitVec w → Val) (hf : ∀ a : BitVec w, Val.HasTy ib tOut (f w a))
    {x : Val} (hx : Val.HasTy ib tIn x) : Safe (Val.HasTy ib tOut) (Arith.intUn f x) := by
  rcases hasTy_intWidth htIn hx with rfl | ⟨a, rfl⟩
  · exact Safe.of_ok (hasTy_poison _ _)
  · exact Safe.of_ok (hf a)

theorem fltBin_safe (M : FloatModel) (key : FloatOpKey) {ib : Nat} {k : FKind} {x y : Val}
    (hx : Val.HasTy ib (.float k) x) (hy : Val.HasTy ib (.float k) y) :
    Safe (Val.HasTy ib (.float k)) (Arith.fltBin M key x y) := by
  rcases hasTy_float hx with rfl | ⟨a, rfl⟩ <;>
    rcases hasTy_float hy with rfl | ⟨b, rfl⟩ <;>
    simp only [Arith.fltBin, dite_true] <;>
    first | exact Safe.of_ok (hasTy_poison _ _) | exact Safe.of_ok (hasTy_flt _ _ _)

theorem fltN_safe (M : FloatModel) (key : FloatOpKey) {ib : Nat} {k : FKind} (vs : List Val)
    (hne : vs ≠ []) (hvs : ∀ v ∈ vs, Val.HasTy ib (.float k) v) :
    Safe (Val.HasTy ib (.float k)) (Arith.fltN M key vs) := by
  unfold Arith.fltN
  by_cases hp : vs.contains .poison = true
  · simp only [hp, ↓reduceIte, Outcome.pure_eq]
    exact Safe.of_ok (hasTy_poison _ _)
  · have hnp : ∀ v ∈ vs, ∃ x : BitVec k.width, v = .some (.flt k x) := by
      intro v hv
      rcases hasTy_float (hvs v hv) with rfl | hx
      · exact absurd (List.contains_iff_mem.2 hv) hp
      · exact hx
    obtain ⟨v, vs', rfl⟩ := List.exists_cons_of_ne_nil hne
    obtain ⟨x, rfl⟩ := hnp _ (List.mem_cons_self ..)
    simp only [hp, Bool.false_eq_true, ↓reduceIte, List.head?_cons, Option.bind_some,
      Outcome.bindOp_eq, Outcome.pure_eq]
    refine Safe.bind (P := fun _ => True) ?_ (fun _ _ => Safe.of_ok (hasTy_flt _ _ _))
    refine ⟨(listMapM_safe _ (fun _ => True) _ fun u hu => ?_).1, fun _ _ => trivial⟩
    obtain ⟨y, rfl⟩ := hnp u hu
    exact ⟨by simp [Outcome.NotStuck], fun _ _ => trivial⟩

/-! ### Lifted unary and binary ops -/

theorem unOp_safe {ib : Nat} {op : Op} {args : List RVal} {f : Val → Outcome Val}
    {sh : Option (List Nat)} {t re : Ty} (hres : Arith.resultElt op = some re)
    (hargs : HasTys ib args ([t].map (liftTy sh)))
    (hf : ∀ x, Val.HasTy ib t x → Safe (Val.HasTy ib re) (f x)) :
    Safe (HasTys ib · [liftTy sh re]) (Arith.unOp op args f) := by
  obtain _ | ⟨ha, hnil⟩ := hargs
  cases hnil
  rename_i a
  simp only [Arith.unOp]
  refine pureN_safe (elts := [t]) (by simp) hres (List.Forall₂.cons ha List.Forall₂.nil) ?_
  intro vs hvs
  obtain _ | ⟨hx, hnil'⟩ := hvs
  cases hnil'
  exact hf _ hx

theorem binOp_safe {ib : Nat} {op : Op} {args : List RVal} {f : Val → Val → Outcome Val}
    {sh : Option (List Nat)} {t1 t2 re : Ty} (hres : Arith.resultElt op = some re)
    (hargs : HasTys ib args ([t1, t2].map (liftTy sh)))
    (hf : ∀ x y, Val.HasTy ib t1 x → Val.HasTy ib t2 y → Safe (Val.HasTy ib re) (f x y)) :
    Safe (HasTys ib · [liftTy sh re]) (Arith.binOp op args f) := by
  obtain _ | ⟨ha, _ | ⟨hb, hnil⟩⟩ := hargs
  cases hnil
  rename_i a b
  simp only [Arith.binOp]
  refine pureN_safe (elts := [t1, t2]) (by simp) hres
    (List.Forall₂.cons ha (List.Forall₂.cons hb List.Forall₂.nil)) ?_
  intro vs hvs
  obtain _ | ⟨hx, _ | ⟨hy, hnil'⟩⟩ := hvs
  cases hnil'
  exact hf _ _ hx hy

/-! ### The blinded op -/

theorem resultElt_blind (op : Op) : Arith.resultElt op.blind = Arith.resultElt op := by
  unfold Arith.resultElt
  rw [Op.blind_results]
  cases op.results <;> rfl

theorem attrInt?_blind (op : Op) (k : String) : attrInt? op.blind k = attrInt? op k := by
  simp only [attrInt?, Op.blind_attr?]

theorem widthOf_eq {ib w : Nat} {op : Op} {re : Ty} (hres : Arith.resultElt op = some re)
    (hw : intWidth? ib re = some w) : Arith.widthOf ib op = .ok w := by
  unfold Arith.widthOf
  rw [hres]
  cases re <;> simp_all [intWidth?, Ty.bitWidth?, Outcome.ofOption]

/-- The `math.*` ops reach the fallback branch of `Arith.sem`. -/
theorem sem_math {ctx : Ctx} {call : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (hs : op.name.startsWith "math." = true) :
    Arith.sem ctx call op args regions =
      match Arith.resultElt op with
      | some (.float _) => Arith.pureN op args (Arith.fltN ctx.M (Arith.floatKey op))
      | some (.int _) =>
        if op.name == "math.absi" then
          Arith.unOp op args (Arith.intUn fun w x => Arith.bv w (if x.msb then -x else x))
        else .stuck s!"unsupported integer math op {op.name}"
      | _ => .stuck s!"unsupported math op {op.name}" := by
  unfold Arith.sem
  split
  all_goals first
    | (rename_i heq; rw [heq] at hs; simp at hs)
    | (simp only [hs, ↓reduceIte]; rfl)

/-! ### Rules -/

/-- Integer ops on two operands of the result type: `intBin` (no UB) and the division family. -/
def intBinNames : List String :=
  ["arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori",
   "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui", "arith.shli", "arith.shrui",
   "arith.shrsi", "arith.divui", "arith.divsi", "arith.remui", "arith.remsi", "arith.ceildivui",
   "arith.ceildivsi", "arith.floordivsi"]

/-- Same-format float ops on two operands. -/
def fltBinNames : List String :=
  ["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.remf", "arith.maximumf",
   "arith.minimumf", "arith.maxnumf", "arith.minnumf"]

def intBinOk (ib : Nat) (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  decide (dialectOf op.name = "arith" ∧ op.name ∈ intBinNames ∧ elts = [re, re]) && isInt ib re

/-- `arith.cmpi`: two integers of one type, a known predicate, an `i1` result. -/
def cmpiOk (ib : Nat) (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  match elts with
  | [t, t'] =>
    decide (dialectOf op.name = "arith" ∧ op.name = "arith.cmpi" ∧ t' = t ∧ re = .int 1) &&
      isInt ib t && (attrInt? op "predicate" >>= Arith.cmpiPred).isSome
  | _ => false

def fltBinOk (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  decide (dialectOf op.name = "arith" ∧ op.name ∈ fltBinNames ∧ elts = [re, re]) && isFlt re

def negfOk (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  decide (dialectOf op.name = "arith" ∧ op.name = "arith.negf" ∧ elts = [re]) && isFlt re

/-- `arith.cmpf`: two floats of one type, an `i1` result. -/
def cmpfOk (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  match elts with
  | [t, t'] =>
    decide (dialectOf op.name = "arith" ∧ op.name = "arith.cmpf" ∧ t' = t ∧ re = .int 1) && isFlt t
  | _ => false

/-- A cast: one operand whose element type passes `src`, a result element type passing `dst`. -/
def castOk (names : List String) (src dst : Ty → Bool) (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  match elts with
  | [t] => decide (dialectOf op.name = "arith" ∧ op.name ∈ names) && src t && dst re
  | _ => false

def intCastNames : List String :=
  ["arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast", "arith.index_castui"]

/-- `math` ops on floats: every operand has the (float) result type. -/
def mathFltOk (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  decide (dialectOf op.name = "math" ∧ ∀ t ∈ elts, t = re) && op.name.startsWith "math." && isFlt re

/-- `math.absi`: one operand of the (signless integer) result type. -/
def absiOk (op : Op) (elts : List Ty) (re : Ty) : Bool :=
  match re with
  | .int _ => decide (dialectOf op.name = "math" ∧ op.name = "math.absi" ∧ elts = [re]) &&
      op.name.startsWith "math."
  | _ => false

/-- `arith.select` operand/result types: an `i1` condition and two operands of the result type; a
tensor condition must have the shape of the (tensor) operands. -/
def selectTyped : List VTy → List VTy → Bool
  | [.scalar (.int 1), x, y], [r] => decide (x = r ∧ y = r)
  | [.tensor sh (.int 1), x, y], [.tensor sh' e] =>
    decide (sh' = sh ∧ x = .tensor sh e ∧ y = .tensor sh e)
  | _, _ => false

def selectRule : OpRule := fun op argTys resTys =>
  if dialectOf op.name = "arith" ∧ op.name = "arith.select" ∧ selectTyped argTys resTys = true then
    some []
  else none

/-- A dense attribute's bytes decode to a tensor of shape `sh` and element type `elt`. -/
def denseOk (ty : Ty) (splat : Bool) (bytes : List UInt8) (sh : List Nat) (elt : Ty) : Bool :=
  match ty.staticTensor?, denseElemBytes elt with
  | some (sh', elt'), some eb =>
    decide (sh = sh' ∧ elt = elt' ∧ eb ≠ 0 ∧ bytes.length = if splat then eb else shapeSize sh * eb)
  | _, _ => false

/-- `arith.constant`: the `value` attribute denotes a value of the result type. -/
def constTyped (ib : Nat) (value : Option Attr) : List VTy → Bool
  | [.tensor sh elt] =>
    match value with
    | some (.dense ty splat bytes) => denseOk ty splat bytes sh elt
    | _ => false
  | [.scalar t] =>
    match value.bind (scalarOfAttr ib) with
    | some s => s.conforms ib t
    | none => false
  | _ => false

def constantRule (ib : Nat) : OpRule := fun op argTys resTys =>
  if dialectOf op.name = "arith" ∧ op.name = "arith.constant" ∧ argTys = [] ∧
      op.results.length = 1 ∧ constTyped ib (op.attr? "value") resTys = true then some []
  else none

/-- The `arith`/`math` typing rule. -/
def arithRule (ib : Nat) : OpRule :=
  firstRule [constantRule ib, selectRule, ewRule (intBinOk ib), ewRule (cmpiOk ib), ewRule fltBinOk,
    ewRule negfOk, ewRule cmpfOk, ewRule (castOk intCastNames (isInt ib) (isInt ib)),
    ewRule (castOk ["arith.extf", "arith.truncf"] isFlt isFlt),
    ewRule (castOk ["arith.fptosi", "arith.fptoui"] isFlt (isInt ib)),
    ewRule (castOk ["arith.sitofp", "arith.uitofp"] (isInt ib) isFlt),
    ewRule mathFltOk, ewRule absiOk]

/-! ### Soundness -/

/-- Reduce an `arith`/`math` op's safety to the family semantics on the blinded op. -/
theorem arith_safe_of {ib : Nat} {op : Op} {argTys resTys : List VTy}
    (hm : unmodelledAttr? op.blind = none)
    (hd : dialectOf op.name = "arith" ∨ dialectOf op.name = "math")
    (h : ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args argTys →
      Safe (HasTys ib · resTys) (Arith.sem ctx call op.blind args regions)) :
    OpSafe opSem ib op argTys resTys [] := by
  intro ctx call args regions hib hargs _
  rw [opSem_modelled hm]
  rcases hd with hd | hd <;> simp only [dispatch, Op.blind_name, hd] <;> exact h ctx _ args regions hib hargs

theorem intBinRule_sound (ib : Nat) : RuleSound ib (ewRule (intBinOk ib)) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  simp only [intBinOk, Bool.and_eq_true, decide_eq_true_eq] at hok
  obtain ⟨⟨hd, hn, rfl⟩, hw⟩ := hok
  obtain ⟨w, hw⟩ := isInt_eq hw
  rw [← resultElt_blind] at hres
  refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
  subst hib
  simp only [intBinNames, List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn |
    hn | hn | hn
  all_goals simp only [Arith.sem, Op.blind_name, hn, Arith.shift]
  all_goals refine binOp_safe hres hargs fun x y hx hy => ?_
  all_goals first
    | exact intBin_safe hw _ (fun a b => by int_res hw) hx hy
    | exact intDiv_safe hw _ _ (fun a b => by int_res hw) hx hy
    | -- `arith.remsi`: `INT_MIN % -1` is defined, the rest is the division family
      rcases hasTy_intWidth hw hy with rfl | ⟨b, rfl⟩
      · exact intDiv_safe hw false _ (fun a b => by int_res hw) hx (hasTy_poison _ _)
      · dsimp only
        split
        · exact intBin_safe hw _ (fun a b => by int_res hw) hx (hasTy_bits hw b)
        · exact intDiv_safe hw false _ (fun a b => by int_res hw) hx (hasTy_bits hw b)

theorem cmpiRule_sound (ib : Nat) : RuleSound ib (ewRule (cmpiOk ib)) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold cmpiOk at hok
  split at hok
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨⟨hd, hn, rfl, rfl⟩, ht⟩, hp⟩ := hok
    obtain ⟨w, hw⟩ := isInt_eq ht
    obtain ⟨p, hp⟩ := Option.isSome_iff_exists.1 hp
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
    subst hib
    simp only [Arith.sem, Op.blind_name, hn, attrInt?_blind, hp]
    exact binOp_safe hres hargs fun x y hx hy => intBin_safe hw _ (fun _ _ => hasTy_bool _ _) hx hy
  · cases hok

theorem fltBinRule_sound (ib : Nat) : RuleSound ib (ewRule fltBinOk) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  simp only [fltBinOk, Bool.and_eq_true, decide_eq_true_eq] at hok
  obtain ⟨⟨hd, hn, rfl⟩, hf⟩ := hok
  obtain ⟨k, rfl⟩ := isFlt_eq hf
  rw [← resultElt_blind] at hres
  refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
  subst hib
  simp only [fltBinNames, List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with hn | hn | hn | hn | hn | hn | hn | hn | hn
  all_goals simp only [Arith.sem, Op.blind_name, hn]
  all_goals exact binOp_safe hres hargs fun x y hx hy => fltBin_safe _ _ hx hy

theorem negfRule_sound (ib : Nat) : RuleSound ib (ewRule negfOk) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  simp only [negfOk, Bool.and_eq_true, decide_eq_true_eq] at hok
  obtain ⟨⟨hd, hn, rfl⟩, hf⟩ := hok
  obtain ⟨k, rfl⟩ := isFlt_eq hf
  rw [← resultElt_blind] at hres
  refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
  subst hib
  simp only [Arith.sem, Op.blind_name, hn]
  exact unOp_safe hres hargs fun x hx => fltN_safe _ _ [x] (by simp) (by simpa using hx)

theorem cmpfRule_sound (ib : Nat) : RuleSound ib (ewRule cmpfOk) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold cmpfOk at hok
  split at hok
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨hd, hn, rfl, rfl⟩, hf⟩ := hok
    obtain ⟨k, rfl⟩ := isFlt_eq hf
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
    subst hib
    simp only [Arith.sem, Op.blind_name, hn]
    refine binOp_safe hres hargs fun x y hx hy => ?_
    rcases hasTy_float hx with rfl | ⟨a, rfl⟩ <;> rcases hasTy_float hy with rfl | ⟨b, rfl⟩ <;>
      simp only [dite_true] <;>
      first | exact Safe.of_ok (hasTy_poison _ _) | exact Safe.of_ok (hasTy_bool _ _)
  · cases hok

theorem intCastRule_sound (ib : Nat) :
    RuleSound ib (ewRule (castOk intCastNames (isInt ib) (isInt ib))) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold castOk at hok
  split at hok
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨⟨hd, hn⟩, ht⟩, hre⟩ := hok
    obtain ⟨w, hw⟩ := isInt_eq ht
    obtain ⟨w', hw'⟩ := isInt_eq hre
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
    subst hib
    have hwid := widthOf_eq hres hw'
    simp only [intCastNames, List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with hn | hn | hn | hn | hn
    all_goals simp only [Arith.sem, Op.blind_name, hn, hwid, Outcome.bindOp_eq, Outcome.bind_ok]
    all_goals exact unOp_safe hres hargs fun x hx => intUn_safe hw _ (fun a => by int_res hw') hx
  · cases hok

theorem fltCastRule_sound (ib : Nat) :
    RuleSound ib (ewRule (castOk ["arith.extf", "arith.truncf"] isFlt isFlt)) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold castOk at hok
  split at hok
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨⟨hd, hn⟩, ht⟩, hre⟩ := hok
    obtain ⟨k, rfl⟩ := isFlt_eq ht
    obtain ⟨k', rfl⟩ := isFlt_eq hre
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
    subst hib
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with hn | hn
    all_goals simp only [Arith.sem, Op.blind_name, hn, hres]
    all_goals refine unOp_safe hres hargs fun x hx => ?_
    all_goals rcases hasTy_float hx with rfl | ⟨a, rfl⟩
    all_goals first | exact Safe.of_ok (hasTy_poison _ _) | exact Safe.of_ok (hasTy_flt _ _ _)
  · cases hok

theorem fptoiRule_sound (ib : Nat) :
    RuleSound ib (ewRule (castOk ["arith.fptosi", "arith.fptoui"] isFlt (isInt ib))) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold castOk at hok
  split at hok
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨⟨hd, hn⟩, ht⟩, hre⟩ := hok
    obtain ⟨k, rfl⟩ := isFlt_eq ht
    obtain ⟨w', hw'⟩ := isInt_eq hre
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
    subst hib
    have hwid := widthOf_eq hres hw'
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with hn | hn
    all_goals simp only [Arith.sem, Op.blind_name, hn, hwid, Outcome.bindOp_eq, Outcome.bind_ok]
    all_goals refine unOp_safe hres hargs fun x hx => ?_
    all_goals rcases hasTy_float hx with rfl | ⟨a, rfl⟩
    all_goals first
      | exact Safe.of_ok (hasTy_poison _ _)
      | (refine Safe.of_ok ?_; split <;>
          first | exact hasTy_bits hw' _ | exact hasTy_poison _ _)
  · cases hok

theorem itofpRule_sound (ib : Nat) :
    RuleSound ib (ewRule (castOk ["arith.sitofp", "arith.uitofp"] (isInt ib) isFlt)) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold castOk at hok
  split at hok
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨⟨hd, hn⟩, ht⟩, hre⟩ := hok
    obtain ⟨w, hw⟩ := isInt_eq ht
    obtain ⟨k, rfl⟩ := isFlt_eq hre
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
    subst hib
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with hn | hn
    all_goals simp only [Arith.sem, Op.blind_name, hn, hres]
    all_goals refine unOp_safe hres hargs fun x hx => ?_
    all_goals rcases hasTy_intWidth hw hx with rfl | ⟨a, rfl⟩
    all_goals first
      | exact Safe.of_ok (hasTy_poison _ _)
      | (refine Safe.of_ok ?_; split <;>
          first | exact hasTy_poison _ _ | exact hasTy_flt _ _ _)
  · cases hok

theorem mathFltRule_sound (ib : Nat) : RuleSound ib (ewRule mathFltOk) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, hne, hres, hok, rfl⟩ := ewRule_spec h
  simp only [mathFltOk, Bool.and_eq_true, decide_eq_true_eq] at hok
  obtain ⟨⟨⟨hd, hall⟩, hs⟩, hf⟩ := hok
  obtain ⟨k, rfl⟩ := isFlt_eq hf
  rw [← resultElt_blind] at hres
  refine arith_safe_of hm (.inr hd) fun ctx call args regions hib hargs => ?_
  subst hib
  rw [sem_math (by rwa [Op.blind_name])]
  simp only [hres]
  refine pureN_safe hne hres hargs fun vs hvs => fltN_safe _ _ vs ?_ (ValsTyped.of_all hvs hall)
  rintro rfl
  exact hne (List.eq_nil_of_length_eq_zero (List.Forall₂.length_eq hvs).symm)

theorem absiRule_sound (ib : Nat) : RuleSound ib (ewRule absiOk) := by
  intro op argTys resTys sigs hm h
  obtain ⟨sh, elts, re, rfl, rfl, -, hres, hok, rfl⟩ := ewRule_spec h
  unfold absiOk at hok
  split at hok
  · next w =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨hd, hn, rfl⟩, hs⟩ := hok
    rw [← resultElt_blind] at hres
    refine arith_safe_of hm (.inr hd) fun ctx call args regions hib hargs => ?_
    subst hib
    rw [sem_math (by rwa [Op.blind_name])]
    simp only [hres, Op.blind_name, hn, beq_self_eq_true, ↓reduceIte]
    have hw : intWidth? ctx.indexBits (.int w) = some w := rfl
    exact unOp_safe hres hargs fun x hx => intUn_safe hw _ (fun a => by int_res hw) hx
  · cases hok

/-! ### `arith.select` -/

theorem hasTy_canonize {ib : Nat} {v : RVal} {t : VTy} (h : RVal.HasTy ib v t) :
    RVal.HasTy ib v.canonize t := by
  cases v with
  | scalar x => exact h
  | tensor x =>
    cases t with
    | scalar _ => exact (h : False).elim
    | tensor sh e =>
      obtain ⟨hs, he, hg⟩ := h
      refine ⟨hs, he, fun i hi => ?_⟩
      simp only [Tensor.canonize, Tensor.canon, hs, hi, ↓reduceIte]
      exact hg i hi

theorem hasTy_full {ib : Nat} {sh : List Nat} {e : Ty} {v : Val} (h : Val.HasTy ib e v) :
    RVal.HasTy ib (.tensor (Tensor.full sh e v)) (.tensor sh e) :=
  ⟨rfl, rfl, fun i hi => by simp only [Tensor.full, Tensor.canon, hi, ↓reduceIte]; exact h⟩

theorem selectRule_sound (ib : Nat) : RuleSound ib selectRule := by
  intro op argTys resTys sigs hm h
  simp only [selectRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, hty⟩ := hc
    cases h
    unfold selectTyped at hty
    split at hty
    · -- scalar condition
      next x y r =>
      simp only [decide_eq_true_eq] at hty
      obtain ⟨hx, hy⟩ := hty
      subst x y
      refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
      subst hib
      obtain _ | ⟨hc, _ | ⟨ha, _ | ⟨hb, hnil⟩⟩⟩ := hargs
      cases hnil
      rename_i c a b
      simp only [Arith.sem, Op.blind_name, hn]
      cases c with
      | tensor _ => exact (hc : False).elim
      | scalar cv =>
        rcases hasTy_intWidth (w := 1) rfl hc with rfl | ⟨bc, rfl⟩ <;> simp only
        · cases a with
          | scalar av =>
            refine Safe.of_ok (List.Forall₂.cons ?_ List.Forall₂.nil)
            cases r with
            | scalar t => exact hasTy_poison ctx.indexBits t
            | tensor _ _ => exact (ha : False).elim
          | tensor ta =>
            refine Safe.of_ok (List.Forall₂.cons ?_ List.Forall₂.nil)
            cases r with
            | scalar _ => exact (ha : False).elim
            | tensor sh e =>
              obtain ⟨hs, he, _⟩ := ha
              rw [hs, he]
              exact hasTy_full (hasTy_poison _ _)
        · refine Safe.of_ok (List.Forall₂.cons ?_ List.Forall₂.nil)
          split
          · exact hasTy_canonize ha
          · exact hasTy_canonize hb
    · -- tensor condition of the operands' shape
      next sh x y sh' e =>
      simp only [decide_eq_true_eq] at hty
      obtain ⟨hsh, hx, hy⟩ := hty
      subst sh' x y
      refine arith_safe_of hm (.inl hd) fun ctx call args regions hib hargs => ?_
      subst hib
      obtain _ | ⟨hc, _ | ⟨ha, _ | ⟨hb, hnil⟩⟩⟩ := hargs
      cases hnil
      rename_i c a b
      cases c with
      | scalar _ => exact (hc : False).elim
      | tensor tc =>
      cases a with
      | scalar _ => exact (ha : False).elim
      | tensor ta =>
      cases b with
      | scalar _ => exact (hb : False).elim
      | tensor tb =>
      obtain ⟨hcs, -, hcg⟩ := hc
      obtain ⟨has, hae, hag⟩ := ha
      obtain ⟨hbs, -, hbg⟩ := hb
      simp only [Arith.sem, Op.blind_name, hn, hcs, has, hbs, beq_self_eq_true, Bool.and_self,
        ↓reduceIte]
      refine Safe.of_ok (List.Forall₂.cons ⟨rfl, hae, fun i hi => ?_⟩ List.Forall₂.nil)
      simp only [Tensor.canon, hi, ↓reduceIte]
      rcases hasTy_intWidth (w := 1) rfl (hcg i hi) with hci | ⟨bc, hci⟩ <;> rw [hci] <;> simp only
      · exact hasTy_poison _ _
      · split
        · exact hag i hi
        · exact hbg i hi
    · cases hty
  · cases h

/-! ### `arith.constant` -/

theorem scalarOfBits_conforms {ib eb : Nat} {elt : Ty} (he : denseElemBytes elt = some eb) (n : Nat) :
    ∃ s, scalarOfBits ib elt n = some s ∧ s.conforms ib elt = true := by
  cases elt <;> simp_all [denseElemBytes, scalarOfBits, Scalar.conforms]

theorem denseTensor_safe {ib : Nat} {ty : Ty} {splat : Bool} {bytes : List UInt8} {sh : List Nat}
    {elt : Ty} (h : denseOk ty splat bytes sh elt = true) :
    Safe (fun t => RVal.HasTy ib (.tensor t) (.tensor sh elt)) (denseTensor ib ty splat bytes) := by
  unfold denseOk at h
  split at h
  · next sh' elt' eb hst heb =>
    simp only [decide_eq_true_eq] at h
    obtain ⟨rfl, rfl, heb0, hlen⟩ := h
    unfold denseTensor
    simp only [hst, heb, heb0, ↓reduceIte, Outcome.bindOp_eq, Outcome.pure_eq]
    cases splat with
    | true =>
      obtain ⟨s, hs, hc⟩ := scalarOfBits_conforms (ib := ib) heb (leNat bytes)
      simp only [↓reduceIte] at hlen ⊢
      simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte, hs]
      exact Safe.of_ok (hasTy_full hc)
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at hlen ⊢
      simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte]
      refine Safe.bind (listMapM_safe _ (fun s => s.conforms ib elt = true) _ fun k _ => ?_)
        fun elems helems => Safe.of_ok ⟨rfl, rfl, fun i hi => ?_⟩
      · obtain ⟨s, hs, hc⟩ := scalarOfBits_conforms (ib := ib) heb
          (leNat ((bytes.toArray.extract (k * eb) (k * eb + eb)).toList))
        simp only [hs, Outcome.ofOption]
        exact Safe.of_ok hc
      · simp only [Tensor.canon, hi, ↓reduceIte]
        split
        · next s hs =>
          rw [List.getElem?_toArray] at hs
          exact helems s (List.mem_of_getElem? hs)
        · exact hasTy_poison _ _
  · cases h

theorem constantRule_sound (ib : Nat) : RuleSound ib (constantRule ib) := by
  intro op argTys resTys sigs hm h
  simp only [constantRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl, hlen, hty⟩ := hc
    cases h
    obtain ⟨r, hr⟩ := List.length_eq_one_iff.1 hlen
    refine arith_safe_of hm (.inl hd) fun ctx call args regions hib _ => ?_
    subst hib
    simp only [Arith.sem, Op.blind_name, hn, Op.blind_attr?, Op.blind_results, hr, List.map_cons,
      List.map_nil]
    match resTys, hty with
    | [.tensor sh elt], hty =>
      cases hv : op.attr? "value" with
      | none => simp [constTyped, hv] at hty
      | some a =>
        cases a
        all_goals simp only [constTyped, hv, Bool.false_eq_true] at hty
        simp only [Outcome.bindOp_eq, Outcome.pure_eq]
        exact Safe.bind (denseTensor_safe hty) fun t ht => Safe.of_ok (List.Forall₂.cons ht List.Forall₂.nil)
    | [.scalar t], hty =>
      cases hv : op.attr? "value" with
      | none => simp [constTyped, hv] at hty
      | some a =>
        simp only [constTyped, hv, Option.bind_some] at hty
        split at hty
        · next s hs =>
          -- the attribute is a scalar one, so the dense branch of `arith.constant` does not apply
          cases a
          all_goals first
            | (simp [scalarOfAttr] at hs; done)
            | (simp only [hs]; exact Safe.of_ok (List.Forall₂.cons hty List.Forall₂.nil))
        · cases hty
    | [], hty => simp [constTyped] at hty
    | _ :: _ :: _, hty => simp [constTyped] at hty
  · cases h

/-! ### The family rule -/

theorem arithRule_sound (ib : Nat) : RuleSound ib (arithRule ib) := by
  apply firstRule_sound
  simp only [List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true]
  exact ⟨constantRule_sound ib, selectRule_sound ib, intBinRule_sound ib, cmpiRule_sound ib,
    fltBinRule_sound ib, negfRule_sound ib, cmpfRule_sound ib, intCastRule_sound ib,
    fltCastRule_sound ib, fptoiRule_sound ib, itofpRule_sound ib, mathFltRule_sound ib,
    absiRule_sound ib⟩

end MerlinVerify.Check

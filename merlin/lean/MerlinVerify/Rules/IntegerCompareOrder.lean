/- Exact integer-comparison operand ordering with inverse predicates. The internal canonical
header is justified by actual source attribute lookup; original source/target IR is retained. -/
import MerlinVerify.Rules.Commutative

namespace MerlinVerify.Rules.IntegerCompareOrder

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

inductive Predicate where
  | eq | ne | slt | sle | sgt | sge | ult | ule | ugt | uge
  deriving DecidableEq, Repr

def predicates : List Predicate := [.eq,.ne,.slt,.sle,.sgt,.sge,.ult,.ule,.ugt,.uge]
def Predicate.code : Predicate → Int
  | .eq => 0 | .ne => 1 | .slt => 2 | .sle => 3 | .sgt => 4 | .sge => 5
  | .ult => 6 | .ule => 7 | .ugt => 8 | .uge => 9

def Predicate.eval : Predicate → (w : Nat) → BitVec w → BitVec w → Bool
  | .eq => fun _ x y => x == y | .ne => fun _ x y => x != y
  | .slt => fun _ x y => x.slt y | .sle => fun _ x y => x.sle y
  | .sgt => fun _ x y => y.slt x | .sge => fun _ x y => y.sle x
  | .ult => fun _ x y => x.ult y | .ule => fun _ x y => x.ule y
  | .ugt => fun _ x y => y.ult x | .uge => fun _ x y => y.ule x

def Predicate.flip : Predicate → Predicate
  | .eq => .eq | .ne => .ne | .slt => .sgt | .sle => .sge | .sgt => .slt | .sge => .sle
  | .ult => .ugt | .ule => .uge | .ugt => .ult | .uge => .ule

theorem predicate_spec (p : Predicate) : Arith.cmpiPred p.code = some p.eval := by cases p <;> rfl

theorem flip_eval {w : Nat} (p : Predicate) (x y : BitVec w) : p.flip.eval w x y = p.eval w y x := by
  cases p <;> simp [Predicate.flip,Predicate.eval,eq_comm,bne_comm]

def scalar (p : Predicate) : Val → Val → Outcome Val :=
  Arith.intBin fun w x y => Arith.boolVal (p.eval w x y)

/-- The equality includes poison, malformed float values, and mismatched integer widths. -/
theorem scalar_flip (p : Predicate) (a b : Val) : scalar p.flip a b = scalar p b a := by
  cases a with
  | poison => cases b with
    | poison => rfl
    | some s => cases s <;> rfl
  | some a =>
    cases b with
    | poison => cases a <;> rfl
    | some b =>
      cases a with
      | flt _ _ => cases b <;> rfl
      | bits w x =>
        cases b with
        | flt _ _ => rfl
        | bits w' y =>
          by_cases h : w' = w
          · subst w'; simp [scalar,Arith.intBin,flip_eval]
          · simp [scalar,Arith.intBin,h,Ne.symm h]

/-- Equal scalar computations lift to the same eager tensor computation in either order.
The scalar-header checker below is narrower than this universal runtime-value lemma. -/
theorem lift_flip (p : Predicate) (e : Ty) (a b : RVal) :
    liftEager e (fun | [x,y] => scalar p.flip x y | _ => .stuck "arity") [a,b] =
      liftEager e (fun | [x,y] => scalar p x y | _ => .stuck "arity") [b,a] := by
  cases a with
  | scalar x => cases b <;> simp [liftEager,scalar_flip]
  | tensor a =>
    cases b with
    | scalar _ => simp [liftEager]
    | tensor b =>
      by_cases hs : a.shape = b.shape
      · simp only [liftEager,List.mapM_cons,List.mapM_nil,bind,Option.bind,Option.pure_def]
        simp only [List.all_cons,List.all_nil,Bool.and_true,beq_iff_eq,hs,↓reduceIte,
          List.map_cons,List.map_nil]
        have he : (fun i => scalar p.flip (a.get i) (b.get i)) =
            (fun i => scalar p (b.get i) (a.get i)) := funext fun _ => scalar_flip p _ _
        rw [he]
      · simp [liftEager,hs,Ne.symm hs]

theorem binOp_flip (p : Predicate) (s : Op) (a b : RVal) :
    Arith.binOp s [a,b] (scalar p.flip) = Arith.binOp s [b,a] (scalar p) := by
  simp only [Arith.binOp,Arith.pureN]
  cases Arith.resultElt s with
  | none => rfl
  | some e =>
    change (liftEager e _ [a,b]).bind _ = (liftEager e _ [b,a]).bind _
    exact congrArg (fun r : Outcome RVal => r.bind fun v => .ok [v]) (lift_flip p e a b)

theorem binOp_results {s t : Op} (h : s.results = t.results) (a b : RVal)
    (f : Val → Val → Outcome Val) : Arith.binOp s [a,b] f = Arith.binOp t [a,b] f := by
  simp [Arith.binOp,Arith.pureN,Arith.resultElt,h]

/-- Only the interpreted predicate and result types affect integer comparison semantics.
In particular no property of the parametric floating-point model is used. -/
theorem semantic_form {s : Op} (p : Predicate)
    (hn : s.name = "arith.cmpi") (hd : dialectOf s.name = "arith")
    (hm : unmodelledAttr? s.blind = none) (hp : attrInt? s.blind "predicate" = some p.code)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    opSem ctx call s args regions = Arith.binOp s.blind args (scalar p) := by
  rw [opSem_modelled hm]
  simp only [dispatch,Op.blind_name,hd]
  simp [Arith.sem,Op.blind_name,hn,hp,predicate_spec,scalar]

def header (s : Op) (p : Predicate) (operands : List ValueId) : Op :=
  .mk "arith.cmpi" s.results operands [("predicate",.int p.code (.int 64))] [] [] []

theorem header_predicate (s : Op) (p : Predicate) (ids : List ValueId) :
    attrInt? (header s p ids).blind "predicate" = some p.code := by
  rfl

theorem header_modelled (s : Op) (p : Predicate) (ids : List ValueId) :
    unmodelledAttr? (header s p ids).blind = none := by
  rfl

@[simp] theorem header_results (s : Op) (p : Predicate) (ids : List ValueId) :
    (header s p ids).results = s.results := rfl
@[simp] theorem header_operands (s : Op) (p : Predicate) (ids : List ValueId) :
    (header s p ids).operands = ids := rfl
@[simp] theorem header_regions (s : Op) (p : Predicate) (ids : List ValueId) :
    (header s p ids).regions = [] := rfl
@[simp] theorem header_terminator (s : Op) (p : Predicate) (ids : List ValueId) :
    (header s p ids).isTerminator = false := rfl

def scalarResult (s : Op) : Bool :=
  match s.results with | [{ty := .int 1,..}] => true | _ => false

def Eligible (s : Op) (p : Predicate) : Prop :=
  s.name = "arith.cmpi" ∧ dialectOf s.name = "arith" ∧
  unmodelledAttr? s.blind = none ∧ attrInt? s.blind "predicate" = some p.code ∧
  scalarResult s = true ∧
  s.regions = [] ∧ s.successors = [] ∧ overflowFlags s = [] ∧ Arith.fastmathKey s = "0" ∧
  hasUnit s "nonNeg" = false ∧ hasUnit s "isExact" = false ∧ hasUnit s "disjoint" = false
instance (s : Op) (p : Predicate) : Decidable (Eligible s p) := by unfold Eligible; infer_instance

/-- Operand lookup failures retain the same exact stuck reason. Successful lookups use the
actual parsed source predicate, including its property-over-attribute precedence. -/
theorem header_step {s : Op} {p : Predicate} {x y : ValueId}
    (h : Eligible s p) (ho : s.operands = [x,y]) (reverse : Bool)
    (ctx : Ctx) (call : CallSem) (env : Env) :
    stepOp opSem ctx call env
      (header s (if reverse then p.flip else p) (if reverse then [y,x] else [x,y])) =
      stepOp opSem ctx call env s := by
  obtain ⟨hn,hd,hm,hp,_,hregions,_⟩ := h
  have target_form (q : Predicate) (ids : List ValueId) (a b : RVal) :
      opSem ctx call (header s q ids) [a,b] [] = Arith.binOp s.blind [a,b] (scalar q) := by
    rw [semantic_form q rfl (by change dialectOf "arith.cmpi" = "arith"; simpa only [hn] using hd) (header_modelled s q ids) (header_predicate s q ids)]
    exact binOp_results (by rfl) a b (scalar q)
  simp only [stepOp,header_operands,header_results,header_regions,ho,hregions,evalRegions]
  cases hx : env.get? x with
  | none =>
    cases hy : env.get? y <;> cases reverse <;>
      simp [Env.getAll,Outcome.mapM,Outcome.ofOption,hx,hy]
  | some a =>
    cases hy : env.get? y with
    | none =>
      cases reverse <;> simp [Env.getAll,Outcome.mapM,Outcome.ofOption,hx,hy]
    | some b =>
      have hxy : env.getAll [x,y] = .ok [a,b] := by
        simp [Env.getAll,Outcome.mapM,Outcome.ofOption,hx,hy]
      have hyx : env.getAll [y,x] = .ok [b,a] := by
        simp [Env.getAll,Outcome.mapM,Outcome.ofOption,hx,hy]
      cases reverse <;>
        simp only [Bool.false_eq_true,↓reduceIte,hxy,hyx,Outcome.bind_ok]
      · rw [target_form p [x,y] a b,semantic_form p hn hd hm hp]
      · rw [target_form p.flip [y,x] b a,semantic_form p hn hd hm hp,binOp_flip]

def candidate (s : Op) : Option Predicate := do
  let n ← attrInt? s.blind "predicate"
  predicates.find? fun p => p.code == n

def operation (s : Op) : Op :=
  match s.operands,candidate s with
  | [x,y],some p =>
    if Eligible s p then
      if y < x then header s p.flip [y,x] else header s p [x,y]
    else s
  | _,_ => s

theorem operation_spec (s : Op) :
    (operation s).isTerminator = s.isTerminator ∧ (operation s).results = s.results ∧
    ∀ ctx call env, stepOp opSem ctx call env (operation s) = stepOp opSem ctx call env s := by
  unfold operation
  split
  · next x y p ho _ =>
    split
    · next he =>
      have hn : s.isTerminator = false := by
        simp [Op.isTerminator,terminatorNames,he.1]
      split
      · exact ⟨(header_terminator _ _ _).trans hn.symm,rfl,
          fun ctx call env => header_step he ho true ctx call env⟩
      · exact ⟨(header_terminator _ _ _).trans hn.symm,rfl,
          fun ctx call env => header_step he ho false ctx call env⟩
    · exact ⟨rfl,rfl,fun _ _ _ => rfl⟩
  · exact ⟨rfl,rfl,fun _ _ _ => rfl⟩

def operations (ops : List Op) : List Op := ops.map operation

/-- The local pass is exact on arbitrary environments, including unbound SSA, poison and
malformed runtime values. Sorting does not rename any results or captures. -/
theorem operations_equal (ctx : Ctx) (call : CallSem) :
    ∀ ops env, evalOps opSem ctx call env (operations ops) = evalOps opSem ctx call env ops
  | [],_ => rfl
  | s :: ss,env => by
    by_cases ht : s.isTerminator = true
    · have hs : operation s = s := by
        unfold operation
        split
        · next _ _ _ _ _ =>
          split
          · next he => simp [Op.isTerminator,terminatorNames,he.1] at ht
          · rfl
        · rfl
      simp only [operations,List.map_cons,hs,evalOps_cons_term _ _ _ _ _ _ ht]
      cases ss <;> rfl
    · have hn : s.isTerminator = false := by simpa using ht
      simp only [operations,List.map_cons]
      rw [evalOps_cons_step _ _ _ _ _ _ ((operation_spec s).1.trans hn),
        evalOps_cons_step _ _ _ _ _ _ hn,(operation_spec s).2.2]
      cases h : stepOp opSem ctx call env s with
      | ub => rfl
      | stuck _ => rfl
      | ok env' => exact operations_equal ctx call ss env'

end MerlinVerify.Rules.IntegerCompareOrder

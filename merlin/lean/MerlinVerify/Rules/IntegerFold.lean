/- Closed fixed-width integer evaluation. The checker replays actual SSA definitions; source
annotations never stand in for runtime values, and poison is compared exactly. -/
import MerlinVerify.Rules.Dce
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Rules.IntegerFold

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- A computational model used only after independence from every context has been proved. -/
def reference : Ctx :=
  { M := { op := fun _ _ _ => 0, cmp := fun _ _ _ _ => false,
           convert := fun _ _ _ _ => 0, toInt := fun _ _ _ _ => none,
           ofInt := fun _ _ _ _ => 0 },
    ext := fun _ _ => .stuck "closed integer expression", indexBits := 0 }

def names : List String :=
  ["arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori",
   "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui", "arith.divui", "arith.divsi",
   "arith.remui", "arith.remsi", "arith.ceildivui", "arith.ceildivsi", "arith.floordivsi",
   "arith.shli", "arith.shrui", "arith.shrsi", "arith.cmpi", "arith.select",
   "arith.extsi", "arith.extui", "arith.trunci"]

def fixedConstant (op : Op) : Bool :=
  op.name == "arith.constant" && match op.attr? "value" with
  | some (.int _ (.int _)) | some (.bool _) => true
  | _ => false

def fixedResult (op : Op) : Bool :=
  match op.results with
  | [{ ty := .int _, .. }] => true
  | _ => false

def eligible (op : Op) : Prop :=
  dialectOf op.name = "arith" ∧ op.isTerminator = false ∧
  op.regions = [] ∧ op.successors = [] ∧ unmodelledAttr? op.blind = none ∧
  fixedResult op = true ∧ (fixedConstant op = true ∨ op.name ∈ names)

instance (op : Op) : Decidable (eligible op) := by
  unfold eligible
  exact inferInstance

theorem context_independent {op : Op} (h : eligible op) (ctx : Ctx) (call : CallSem)
    (args : List RVal) :
    opSem ctx call op args [] = opSem reference reference.ext op args [] := by
  obtain ⟨hd, -, -, -, hm, hresult, hn⟩ := h
  obtain ⟨r, w, hr, ht⟩ : ∃ r w, op.results = [r] ∧ r.ty = .int w := by
    unfold fixedResult at hresult
    split at hresult
    · next id w he => exact ⟨⟨id, .int w⟩, w, he, rfl⟩
    · cases hresult
  rw [opSem_modelled hm, opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  rcases hn with hc | hn
  · simp only [fixedConstant, Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨hn, ha⟩ := hc
    split at ha <;>
      simp_all [Arith.sem, Op.blind_name, Op.blind_attr?, Op.blind_results, scalarOfAttr]
  · simp only [names, List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn |
      hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn | hn
    all_goals simp [Arith.sem, Op.blind_name, hn, Arith.widthOf, Arith.resultElt,
      Op.blind_results, hr, ht, Ty.bitWidth?]

/-- Reuse the proved operational semantics instead of maintaining a second arithmetic evaluator. -/
def run : List Op → Env → Option Env
  | [], env => some env
  | op :: ops, env =>
    if eligible op then
      match stepOp opSem reference reference.ext env op with
      | .ok next => run ops next
      | _ => none
    else none

theorem getAll_append {env : Env} {ids : List Nat} {vs : List RVal}
    (h : env.getAll ids = .ok vs) (outer : Env) :
    (env ++ outer).getAll ids = .ok vs := by
  induction ids generalizing vs with
  | nil => simpa [Env.getAll, Outcome.mapM] using h
  | cons x xs ih =>
    change (Outcome.ofOption "unbound SSA value" (env.get? x)).bind
      (fun v => (env.getAll xs).bind fun rest => .ok (v :: rest)) = .ok vs at h
    change (Outcome.ofOption "unbound SSA value" ((env ++ outer).get? x)).bind
      (fun v => ((env ++ outer).getAll xs).bind fun rest => .ok (v :: rest)) = .ok vs
    cases hx : env.get? x with
    | none => simp [Outcome.ofOption, hx] at h
    | some v =>
      cases hxs : env.getAll xs with
      | ub => simp [Outcome.ofOption, hx, hxs] at h
      | stuck r => simp [Outcome.ofOption, hx, hxs] at h
      | ok rest =>
        have hlookup : (env ++ outer).get? x = some v := by
          have hx' : List.lookup x env = some v := hx
          simp only [Env.get?, List.lookup_append, hx', Option.or]
        have htail := ih hxs
        simpa [Outcome.ofOption, hx, hlookup, hxs, htail] using h

theorem step_append {op : Op} (he : eligible op) {env next : Env}
    (h : stepOp opSem reference reference.ext env op = .ok next)
    (ctx : Ctx) (call : CallSem) (outer : Env) :
    stepOp opSem ctx call (env ++ outer) op = .ok (next ++ outer) := by
  unfold stepOp at h ⊢
  rw [he.2.2.1] at h ⊢
  simp only [evalRegions] at h ⊢
  cases hg : env.getAll op.operands with
  | ub => simp [hg] at h
  | stuck r => simp [hg] at h
  | ok args =>
    rw [getAll_append hg outer, Outcome.bind_ok, context_independent he]
    simp only [hg, Outcome.bind_ok] at h
    cases hv : opSem reference reference.ext op args [] with
    | ub => simp [hv] at h
    | stuck r => simp [hv] at h
    | ok vals =>
      simp only [hv, Outcome.bind_ok, Env.bindAll] at h ⊢
      split at h
      · next hlen =>
        have hnext := Outcome.ok.inj h
        rw [ite_eq_left hlen, ← hnext, List.append_assoc]
      · cases h

theorem run_sound : ∀ ops env next, run ops env = some next →
    (∀ op ∈ ops, op.isTerminator = false) ∧
    ∀ ctx call outer,
      execOps opSem ctx call (env ++ outer) ops = .ok (next ++ outer)
  | [], env, next, h => by
    simp only [run, Option.some.injEq] at h
    subst next
    exact ⟨by simp, fun _ _ _ => rfl⟩
  | op :: ops, env, next, h => by
    simp only [run] at h
    split at h
    · next he =>
      cases hs : stepOp opSem reference reference.ext env op with
      | ub => simp [hs] at h
      | stuck r => simp [hs] at h
      | ok mid =>
        simp only [hs] at h
        obtain ⟨hterms, ih⟩ := run_sound ops mid next h
        refine ⟨by simp only [List.mem_cons]; intro o ho; rcases ho with rfl | ho
                   exact he.2.1
                   exact hterms o ho, ?_⟩
        intro ctx call outer
        rw [execOps_cons_step _ _ _ _ _ _ he.2.1,
          step_append he hs ctx call outer, Outcome.bind_ok]
        exact ih ctx call outer
    · cases h

def scalarLookup (env : Env) (id : Nat) : Option Val :=
  match env.get? id with
  | some (.scalar v) => some v
  | _ => none

theorem scalarLookup_sound {env : Env} {id : Nat} {v : Val}
    (h : scalarLookup env id = some v) : env.get? id = some (.scalar v) := by
  unfold scalarLookup at h
  split at h
  · next value he => cases h; exact he
  · cases h

def sourceFresh (ρ : Rho) (env : Env) : Bool :=
  env.all fun p => ρ.find p.1 == none

def targetFresh (ρ : Rho) (env : Env) : Bool :=
  env.all fun p => ρ.all fun q => q.2 != p.1

/-- Local replay starts with no bindings: every operand must come from the actual fragment.
Only the last result gets a correspondence; escaping intermediate uses are therefore refused. -/
def check (ρ : Rho) (ss ts : List Op) : Option Rho := do
  let s ← ss.getLast?
  match s.results, ts with
  | [rs], [t] =>
    match t.results with
    | [rt] =>
      if fixedConstant t then
        let es ← run ss []
        let et ← run [t] []
        let vs ← scalarLookup es rs.id
        let vt ← scalarLookup et rt.id
        if vs = vt ∧ sourceFresh ρ es = true ∧ targetFresh ρ et = true then
          some ((rs.id, rt.id) :: ρ)
        else none
      else none
    | _ => none
  | _, _ => none

theorem prefix_corr {ρ : Rho} {envS envT es et : Env} (hc : Corr ρ envS envT)
    (hs : sourceFresh ρ es = true) (ht : targetFresh ρ et = true) :
    Corr ρ (es ++ envS) (et ++ envT) := by
  intro x y hxy
  have hx : List.lookup x es = none := by
    apply List.lookup_eq_none_iff.2
    intro p hp
    have hf : ρ.find p.1 = none := by simpa using List.all_eq_true.1 hs p hp
    have hne : x ≠ p.1 := by rintro rfl; rw [hxy] at hf; cases hf
    simpa using hne
  have hy : List.lookup y et = none := by
    apply List.lookup_eq_none_iff.2
    intro p hp
    obtain ⟨q, hq, hqy⟩ := lookup_mem_image hxy
    have hf := List.all_eq_true.1 (List.all_eq_true.1 ht p hp) q hq
    simpa [hqy] using hf
  have hxy' : List.lookup x envS = List.lookup y envT := hc x y hxy
  simp only [Env.get?, List.lookup_append, hx, hy, Option.or]
  exact hxy'

theorem alias_corr {ρ : Rho} {envS envT : Env} {x y : Nat} {v : RVal}
    (hc : Corr ρ envS envT) (hs : envS.get? x = some v) (ht : envT.get? y = some v) :
    Corr ((x, y) :: ρ) envS envT := by
  intro a b hab
  simp only [Rho.find, List.lookup] at hab
  by_cases ha : a = x
  · subst a
    simp only [beq_self_eq_true, Option.some.injEq] at hab
    subst b
    exact hs.trans ht.symm
  · have ha' : (a == x) = false := by simpa using ha
    simp only [ha'] at hab
    exact hc a b hab

theorem lookup_append {env outer : Env} {id : Nat} {v : RVal}
    (h : env.get? id = some v) : (env ++ outer).get? id = some v := by
  have h' : List.lookup id env = some v := h
  simp only [Env.get?, List.lookup_append, h', Option.or]

theorem check_sound : ∀ ρ ss ts ρ', check ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss) := by
  intro ρ ss ts ρ' h
  unfold check at h
  cases hl : ss.getLast? with
  | none => simp [hl] at h
  | some s =>
    simp only [hl, bind, Option.bind] at h
    split at h
    · next results targets rs t hrs =>
      split at h
      · next resultsT rt hrt =>
        split at h
        · cases hes : run ss [] with
          | none => simp [hes] at h
          | some es =>
            simp only [hes] at h
            cases het : run [t] [] with
            | none => simp [het] at h
            | some et =>
              simp only [het] at h
              cases hvs : scalarLookup es rs.id with
              | none => simp [hvs] at h
              | some vs =>
                simp only [hvs] at h
                cases hvt : scalarLookup et rt.id with
                | none => simp [hvt] at h
                | some vt =>
                  simp only [hvt] at h
                  split at h
                  · next hc =>
                    obtain ⟨rfl, hsf, htf⟩ := hc
                    cases h
                    obtain ⟨hsterms, hsrun⟩ := run_sound ss [] es hes
                    obtain ⟨htterms, htrun⟩ := run_sound [t] [] et het
                    refine ⟨hsterms, ?_, ?_⟩
                    · exact htterms
                    · intro ctx callS callT envS envT hcorr _
                      have hs := hsrun ctx callS envS
                      have ht := htrun ctx callT envT
                      simp only [List.nil_append] at hs ht
                      refine .inr (.inr ⟨es ++ envS, et ++ envT, hs, ht, ?_⟩)
                      apply alias_corr (prefix_corr hcorr hsf htf)
                      · exact lookup_append (scalarLookup_sound hvs)
                      · exact lookup_append (scalarLookup_sound hvt)
                  · cases h
        · cases h
      · cases h
    · cases h

def rule : Rule := ⟨"closed-integer-fold", check, check_sound⟩

/-- Untrusted synthesis hint. Replay each prefix once, proposing longer matches first. The C1
rule independently replays the chosen fragment, checks freshness, and compares the final value. -/
def matchingPrefixes (value : Val) : List Op → Env → Nat → List Nat → List Nat
  | [], _, _, found => found
  | op :: ops, env, count, found =>
    match run [op] env with
    | none => found
    | some next =>
      let found := match op.results with
        | [r] => if scalarLookup next r.id = some value then count :: found else found
        | _ => found
      matchingPrefixes value ops next (count + 1) found

def candidateTails (ss ts : List Op) : List Nat :=
  match ts with
  | t :: _ =>
    if fixedConstant t then
      match t.results, run [t] [] with
      | [r], some env =>
        match scalarLookup env r.id with
        | some value => matchingPrefixes value ss [] 0 []
        | none => []
      | _, _ => []
    else []
  | _ => []

end MerlinVerify.Rules.IntegerFold

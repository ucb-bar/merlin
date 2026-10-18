/- Initialized tensor reads require actual nonnegative runtime index bounds. The optional range
facts are proved at their producers and travel with the current SSA binding, including captures. -/
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.DefinedLinalg

namespace MerlinVerify.Check.Definedness.Bounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

/-- A whole interval must fit each actual dimension. Unknown or uninitialized indices refuse. -/
def within : List IndexRanges.Fact → List Nat → Bool
  | [], [] => true
  | f :: fs, size :: shape =>
    match f.base, f.info with
    | ⟨.scalar .index, true⟩, some range => range.upper ≤ size && within fs shape
    | _, _ => false
  | _, _ => false

theorem within_sound {ib : Nat} : ∀ {fs : List IndexRanges.Fact} {shape : List Nat} {args : List RVal},
    Refined.FactsHold model ib args fs → within fs shape = true →
    ∃ coords, args.mapM value? = some coords ∧ coords.length = shape.length ∧
      (coords.zip shape).all (fun (x, size) => 0 ≤ x ∧ x < size) = true
  | [], [], [], _, _ => ⟨[], rfl, rfl, rfl⟩
  | f :: fs, size :: shape, v :: args, .cons hv hargs, h => by
    unfold within at h
    split at h
    · next range hb hi =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨upper, htail⟩ := h
      have hr : range.Holds v := by simpa [model, hi] using hv.2
      obtain ⟨z, hz, hlo, hhi⟩ := hr
      obtain ⟨coords, hc, hlen, hin⟩ := within_sound hargs htail
      refine ⟨z :: coords, ?_, by simp [hlen], ?_⟩
      · simp only [List.mapM_cons, hz, hc, Option.pure_def, bind, Option.bind]
      · simp only [List.zip_cons_cons, List.all_cons, hin, Bool.and_true, decide_eq_true_eq]
        constructor <;> omega
    · cases h
  | [], _ :: _, args, hargs, h => by cases hargs; cases h
  | _ :: _, [], args, hargs, h => by cases hargs; cases h
  | [], [], _ :: _, hargs, _ => by cases hargs
  | _ :: _, _ :: _, [], hargs, _ => by cases hargs

def extractRule (_ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) :
    Option (List IndexRanges.Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧ op.name = "tensor.extract" ∧
      op.regions = [] then
    match fs, op.results with
    | ⟨⟨.tensor shape e, true⟩, _⟩ :: indices, [r] =>
      if r.ty = e ∧ within indices shape = true then
        some [Refined.emptyFact model ⟨.scalar e, true⟩]
      else none
    | _, _ => none
  else none

theorem extractRule_sound (ib : Nat) : ∀ Γ op fs rs,
    extractRule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args _ _ ha
  unfold extractRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn, hregs⟩ := hc
    split at h
    · next _ _ shape e info indices r hres =>
      split at h
      · next hc =>
        obtain ⟨_, hin⟩ := hc
        cases h
        obtain _ | ⟨ht, hi⟩ := ha
        rename_i tensor idxs
        cases tensor with
        | scalar _ => cases ht.1.1
        | tensor t =>
          obtain ⟨coords, hc, hlen, hbounds⟩ := within_sound hi hin
          have hpt : Idx.InBounds t.shape (coords.map Int.toNat) := by
            rw [ht.1.1.1]
            exact coords_inBounds coords shape hlen hbounds
          refine ⟨[.scalar (t.get (coords.map Int.toNat))], ?_,
            .cons ⟨Fact.element ht.1 hpt, trivial⟩ .nil⟩
          rw [hregs, evalRegions, tensor_sem_of hm hd]
          simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn]
          change (match idxs.mapM value? with
            | some is => if is.length = t.shape.length ∧
                (is.zip t.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) then
                Outcome.ok [RVal.scalar (t.get (is.map Int.toNat))] else Outcome.ub
            | none => Outcome.ub) = Outcome.ok [RVal.scalar (t.get (coords.map Int.toNat))]
          rw [hc]
          exact ite_eq_left ⟨by simpa only [ht.1.1.1] using hlen,
            by simpa only [ht.1.1.1] using hbounds⟩
      · cases h
    · cases h
  · cases h

def extractBook (ib : Nat) : Refined.Book model opSem ib where
  rule := extractRule
  sound := extractRule_sound ib

/-- Existing rules retain their original proof obligations and do not invent index evidence. -/
def leafBook (ib : Nat) : Refined.Book model opSem ib :=
  ((constantBook ib).orElse (extractBook ib)).orElse (Refined.Book.lift model (Definedness.book ib))

def book (ib : Nat) : Refined.Book model opSem ib := leafBook ib

def check (ib : Nat) (m : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  Refined.check (book ib) m fn argTys resTys

/-- Bounds evidence adds no input assumptions: entry values are typed and nonpoison, while every
index interval used inside the program must be learned from an actual checked producer. -/
theorem check_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib m fn argTys resTys = true) :
    FuncDefined opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m fn := Refined.check_sound (book ib) h

end MerlinVerify.Check.Definedness.Bounds

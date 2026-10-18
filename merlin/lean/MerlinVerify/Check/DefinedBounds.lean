/- Initialized tensor reads require actual nonnegative runtime index bounds. The optional range
facts are proved at their producers and travel with the current SSA binding, including captures. -/
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.DefinedLinalg

namespace MerlinVerify.Check.Definedness.Bounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges MerlinVerify.Sem.Ops.Linalg

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

/-- Structured body arguments receive their proved element type/initialization facts; captured
index ranges stay in the enclosing refined environment and block arguments shadow them normally. -/
def initializedRule (ib : Nat) (bodyBook : Refined.Book model opSem ib)
    (Γ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) : Option (List IndexRanges.Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts (α := Definedness.Fact) op (fs.map (·.base)), attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, outs), some maps, some iters, [body] =>
      if (∀ f ∈ outs, f.initialized = true) ∧ (∀ f ∈ outs, f.ty.isTensor = true) ∧
          op.results.length = outs.length ∧
          (structuredRanges? maps iters.length ((ins ++ outs).map (·.ty))).isSome ∧
          Refined.checkRegion bodyBook Γ body (((ins ++ outs).map elementFact).map (Refined.emptyFact model))
            (outs.map fun f => f.ty.eltScalar) then
        some ((clean (outs.map (·.ty))).map (Refined.emptyFact model))
      else none
    | _, _, _, _ => none
  else none

theorem initializedRule_sound (ib : Nat) (bodyBook : Refined.Book model opSem ib) : ∀ Γ op fs rs,
    initializedRule ib bodyBook Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold initializedRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn⟩ := hc
    split at h
    · next ins outs maps iters body hsplit hmaps hiters hregs =>
      split at h
      · next hc =>
        obtain ⟨hi, ht, hres, hranges, hbody⟩ := hc
        obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hranges
        cases h
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha.base
        have hclean := facts_eq_clean outs hi
        have hys' : FactsHold ib ys (clean (outs.map (·.ty))) := hclean ▸ hys
        have hb : BodyTotal ib (fun ctx => evalRegion opSem ctx call env body)
            ((ins ++ clean (outs.map (·.ty))).map elementFact)
            ((outs.map (·.ty)).map VTy.eltScalar) := by
          intro ctx hib args ha
          rw [← hclean] at ha
          simpa only [List.map_map, Function.comp_def] using
            Refined.checkRegion_sound bodyBook hbody hib hΓ (Refined.FactsHold.empty ha model)
        have hs : StructuredOk maps iters.length (ins.map (·.ty) ++ outs.map (·.ty)) ranges := by
          simpa only [List.map_append] using structuredRanges?_spec hranges
        have hout : ∀ t ∈ outs.map (·.ty), t.isTensor = true := by
          intro t ht'
          obtain ⟨f, hf, rfl⟩ := List.mem_map.1 ht'
          exact ht f hf
        obtain ⟨rs, hsem, hrf⟩ := genericSem_total hib hs hout hxs hys' hb
        refine ⟨rs, ?_, Refined.FactsHold.empty hrf model⟩
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length (clean_typed hrf), List.length_map, Op.blind_results,
            List.length_map, hres]
        rw [linalg_sem_of hm hd (.inl hn)]
        simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
          attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
          Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h

def overwriteRule (ib : Nat) (bodyBook : Refined.Book model opSem ib)
    (Γ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) : Option (List IndexRanges.Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts (α := Definedness.Fact) op (fs.map (·.base)), attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨.tensor sh e, _⟩]), some maps, some iters, [body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap sh.length] ∧
          structuredRanges? maps iters.length (ins.map (·.ty) ++ [.tensor sh e]) = some sh ∧
          Refined.checkRegion bodyBook Γ body
            (((ins ++ [(⟨.tensor sh e, false⟩ : Definedness.Fact)]).map elementFact).map (Refined.emptyFact model)) [.scalar e] then
        some ((clean [.tensor sh e]).map (Refined.emptyFact model))
      else none
    | _, _, _, _ => none
  else none

theorem overwriteRule_sound (ib : Nat) (bodyBook : Refined.Book model opSem ib) : ∀ Γ op fs rs,
    overwriteRule ib bodyBook Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold overwriteRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn⟩ := hc
    split at h
    · next ins sh e init maps iters body hsplit hmaps hiters hregs =>
      split at h
      · next hc =>
        obtain ⟨hres, hout, hranges, hbody⟩ := hc
        cases h
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha.base
        obtain _ | ⟨ht, hnil⟩ := hys
        cases hnil
        rename_i t
        cases t with
        | scalar _ => cases ht.1
        | tensor t =>
          have hb : BodyTotal ib (fun ctx => evalRegion opSem ctx call env body)
              ((ins ++ [(⟨.tensor sh e, false⟩ : Fact)]).map elementFact) [.scalar e] := by
            intro ctx hib args ha
            exact Refined.checkRegion_sound bodyBook hbody hib hΓ (Refined.FactsHold.empty ha model)
          obtain ⟨rs, hsem, hrf⟩ := overwriteSem_total hib (structuredRanges?_spec hranges) hout hxs ht.1 hb
          refine ⟨rs, ?_, Refined.FactsHold.empty hrf model⟩
          have hlen : rs.length = op.blind.results.length := by
            rw [HasTys.length (clean_typed hrf), Op.blind_results, List.length_map, hres]
            rfl
          rw [linalg_sem_of hm hd (.inl hn)]
          simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
            attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
            Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h

def genericBook (ib : Nat) (bodyBook : Refined.Book model opSem ib) : Refined.Book model opSem ib :=
  ({ rule := initializedRule ib bodyBook, sound := initializedRule_sound ib bodyBook } :
    Refined.Book model opSem ib).orElse
      { rule := overwriteRule ib bodyBook, sound := overwriteRule_sound ib bodyBook }

def nestedBook (ib : Nat) : Nat → Refined.Book model opSem ib
  | 0 => leafBook ib
  | depth + 1 => {
      rule Γ op fs := ((genericBook ib (nestedBook ib depth)).orElse (leafBook ib)).rule Γ op fs
      sound Γ op fs rs h ctx call env args hi he ha :=
        ((genericBook ib (nestedBook ib depth)).orElse (leafBook ib)).sound
          Γ op fs rs h ctx call env args hi he ha }

/-- The budget comes from actual nesting, and cannot assert a missing proof at any depth. -/
def book (ib : Nat) : Refined.Book model opSem ib where
  rule Γ op fs := (nestedBook ib (RegionDepth.op op)).rule Γ op fs
  sound Γ op fs rs h ctx call env args hi he ha :=
    (nestedBook ib (RegionDepth.op op)).sound Γ op fs rs h ctx call env args hi he ha

def check (ib : Nat) (m : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  Refined.check (book ib) m fn argTys resTys

/-- Bounds evidence adds no input assumptions: entry values are typed and nonpoison, while every
index interval used inside the program must be learned from an actual checked producer. -/
theorem check_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib m fn argTys resTys = true) :
    FuncDefined opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m fn := Refined.check_sound (book ib) h

end MerlinVerify.Check.Definedness.Bounds

/- Source-derived finite abstract execution of static structured reductions.
Every abstract update is connected to the existing generic evaluator. -/
import MerlinVerify.Rules.FrontendRangeSelect

namespace MerlinVerify.Rules.FrontendRangeReduce
open MerlinVerify MerlinVerify.Check Sem.Ops Sem.Ops.Linalg
open FrontendTensorRanges FrontendRangeFrame
open Definedness.IndexRanges (Interval)

abbrev State := Idx → Option Interval

def Represents (ib : Nat) (sh : List Nat) (e : Ty) (B : State) (t : Tensor) : Prop :=
  Definedness.Fact.Holds ib (.tensor t) ⟨.tensor sh e, true⟩ ∧
    ∀ q, Idx.InBounds sh q → model.holds (B q) (.scalar (t.get q))

def summarize (sh : List Nat) (B : State) : Option Interval :=
  (allPoints sh).foldr (fun p rest => FrontendRangeSelect.join (B p) rest) (some ⟨0,0⟩)

theorem summarize_sound (points : List Idx) (B : State) (q : Idx) (v : RVal)
    (hq : q ∈ points) (hv : model.holds (B q) v) :
    model.holds (points.foldr (fun p rest => FrontendRangeSelect.join (B p) rest) (some ⟨0,0⟩)) v := by
  induction points with
  | nil => simp at hq
  | cons p points ih =>
    rcases List.mem_cons.mp hq with rfl | hq
    · exact FrontendRangeSelect.join_left _ _ _ hv
    · exact FrontendRangeSelect.join_right _ _ _ (ih hq)

def updated (B : State) (q : Idx) (info : Option Interval) : State :=
  fun p => if p = q then info else B p

theorem updated_sound {ib : Nat} {sh : List Nat} {e : Ty} {B : State} {t : Tensor}
    {q : Idx} {info : Option Interval} {v : Val}
    (ht : Represents ib sh e B t)
    (hv : Definedness.Refined.Fact.Holds model ib (.scalar v) ⟨⟨.scalar e, true⟩, info⟩) :
    Represents ib sh e (updated B q info) (t.set q v) := by
  refine ⟨⟨⟨ht.1.1.1, ht.1.1.2.1, ?_⟩, ?_⟩, ?_⟩
  · intro p hp
    simp only [Tensor.set]
    split
    · exact hv.1.1
    · exact ht.1.1.2.2 p hp
  · intro _ p hp
    change Idx.InBounds t.shape p at hp
    simp only [Tensor.set]
    split
    · exact hv.1.2 rfl
    · exact ht.1.2 rfl p hp
  · intro p hp
    simp only [Tensor.set, updated]
    split
    · exact hv.2
    · exact ht.2 p hp

/-- One abstract step is recomputed at a real iteration coordinate and real output map. -/
def step {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges) (Γ : FEnv) (body : Region)
    (fs : List Fact) (sh : List Nat) (e : Ty) (outMap : AffineMap) (B : State) (pt : Idx) : Option State := do
  let q ← evalMap outMap pt >>= toIdx sh
  let rs ← inferRegion R Γ body ((fs.map FrontendRangeGeneric.elementFact) ++ [⟨⟨.scalar e, true⟩, B q⟩])
  match rs with
  | [⟨⟨.scalar ty, true⟩, info⟩] =>
    if ty = e then some (updated B q info) else none
  | _ => none

/-- Every accepted abstract step executes the original generic body and updates
only its actual output coordinate. All scalar contents remain universally quantified. -/
theorem step_sound {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges)
    {Γ : FEnv} {body : Region} {fs : List Fact} {sh : List Nat} {e : Ty}
    {outMap : AffineMap} {B B' : State} {pt : Idx}
    (h : step R Γ body fs sh e outMap B pt = some B')
    (hp : Idx.InBounds ranges pt) {ctx : Ctx} {call : CallSem} {env : Env}
    (hib : ctx.indexBits = ib) (hΓ : Definedness.Refined.EnvHolds model ib Γ env)
    {ins : List RVal} {inMaps : List AffineMap} {cur : Tensor}
    (hi : Definedness.Refined.FactsHold model ib ins fs)
    (hin : MapsInBounds ranges inMaps (fs.map (·.base.ty)))
    (hout : mapInBounds ranges outMap sh = true) (hc : Represents ib sh e B cur) :
    ∃ next, loopStep ctx ins inMaps [outMap] (fun inner => evalRegion opSem inner call env body) [cur] pt =
        .ok [next] ∧ Represents ib sh e B' next := by
  unfold step at h
  simp only [bind, Option.bind] at h
  split at h
  · cases h
  · next q hq =>
    dsimp only at h
    split at h
    · cases h
    · next rs hbody =>
      dsimp only at h
      split at h
      · next _ ty info hresult =>
        split at h
        · next hty =>
          cases hty
          cases h
          obtain ⟨q', hq', hqin⟩ := mapInBounds_sound hout hp
          have heq : q' = q := Option.some.inj (hq'.symm.trans hq)
          subst q'
          obtain ⟨xs, hx, hxf⟩ := FrontendRangeGeneric.mapM_read_total hp ins inMaps fs hi hin
          have hread : readAt (.tensor cur) outMap pt = .ok (cur.get q) := by
            simp only [readAt, hc.1.1.1, hq']
          have helem : Definedness.Refined.Fact.Holds model ib (.scalar (cur.get q))
              ⟨⟨.scalar e, true⟩, B q⟩ :=
            ⟨Definedness.Fact.element hc.1 (hc.1.1.1.symm ▸ hqin), hc.2 q hqin⟩
          have hargs := List.rel_append hxf (List.Forall₂.cons helem List.Forall₂.nil)
          obtain ⟨ys, hy, hyf⟩ := inferRegion_sound (ctx := {ctx with loopIndex := pt}) (call := call) R hbody hib (.inr hp) hΓ hargs
          obtain _ | ⟨hyf, hnil⟩ := hyf
          cases hnil
          rename_i y
          cases y with
          | tensor _ => cases hyf.1.1
          | scalar y =>
            have hb : bodyAt ctx ins inMaps [outMap] [cur]
                (fun inner => evalRegion opSem inner call env body) pt = .ok [y] := by
              simp only [bodyAt, List.zip_cons_cons, List.zip_nil_right, List.mapM_cons, List.mapM_nil,
                Outcome.bindOp_eq, hx, hread, Outcome.bind_ok, Outcome.pure_eq, List.map_append, List.map_cons, List.map_nil, hy,
                List.length_cons, List.length_nil, ne_eq, not_true_eq_false, ↓reduceIte]
            refine ⟨cur.set q y, ?_, updated_sound hc hyf⟩
            have hw : writeAt cur outMap pt y = .ok (cur.set q y) := by
              simp only [writeAt, hc.1.1.1, hq']
            simp only [loopStep, hb, Outcome.bindOp_eq, Outcome.bind_ok,
              List.zip_cons_cons, List.zip_nil_left, List.mapM_cons, List.mapM_nil, hw, Outcome.pure_eq]
        · cases h
      · cases h


def walk {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges) (Γ : FEnv) (body : Region)
    (fs : List Fact) (sh : List Nat) (e : Ty) (outMap : AffineMap) : List Idx → State → Option State
  | [], B => some B
  | p :: points, B => (step R Γ body fs sh e outMap B p).bind fun next =>
      walk R Γ body fs sh e outMap points next

theorem walk_sound {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges)
    {Γ : FEnv} {body : Region} {fs : List Fact} {sh : List Nat} {e : Ty} {outMap : AffineMap}
    {ctx : Ctx} {call : CallSem} {env : Env} (hib : ctx.indexBits = ib)
    (hΓ : Definedness.Refined.EnvHolds model ib Γ env) {ins : List RVal} {inMaps : List AffineMap}
    (hi : Definedness.Refined.FactsHold model ib ins fs)
    (hin : MapsInBounds ranges inMaps (fs.map (·.base.ty)))
    (hout : mapInBounds ranges outMap sh = true) :
    ∀ points B B' cur, walk R Γ body fs sh e outMap points B = some B' →
      (∀ p ∈ points, Idx.InBounds ranges p) → Represents ib sh e B cur →
      ∃ final, points.foldlM (loopStep ctx ins inMaps [outMap]
          (fun inner => evalRegion opSem inner call env body)) [cur] = .ok [final] ∧
        Represents ib sh e B' final
  | [], B, B', cur, h, _, hc => by cases h; exact ⟨cur, rfl, hc⟩
  | p :: points, B, B', cur, h, hp, hc => by
    simp only [walk] at h
    cases hs : step R Γ body fs sh e outMap B p with
    | none => simp [hs] at h
    | some next =>
      simp only [hs, Option.bind_some] at h
      obtain ⟨mid, hmid, hmr⟩ := step_sound (ctx := ctx) (call := call) R hs (hp p (List.mem_cons_self ..)) hib hΓ hi hin hout hc
      obtain ⟨final, hf, hfr⟩ := walk_sound (ctx := ctx) (call := call) R hib hΓ hi hin hout points next B' mid h
        (fun p hmem => hp p (List.mem_cons_of_mem _ hmem)) hmr
      exact ⟨final, by rw [List.foldlM_cons, hmid, Outcome.bindOp_eq, Outcome.bind_ok, hf], hfr⟩

theorem summarized {ib : Nat} {sh : List Nat} {e : Ty} {B : State} {t : Tensor}
    (ht : Represents ib sh e B t) :
    Definedness.Refined.FactsHold model ib [.tensor t.canonize] [⟨⟨.tensor sh e, true⟩, summarize sh B⟩] := by
  have hsummary : model.holds (summarize sh B) (.tensor t) := by
    cases hs : summarize sh B with
    | none => trivial
    | some r =>
      intro p hp
      have hp' : Idx.InBounds sh p := ht.1.1.1 ▸ hp
      have h := summarize_sound (allPoints sh) B p (.scalar (t.get p)) ((mem_allPoints sh p).mpr hp') (ht.2 p hp')
      change model.holds (summarize sh B) (.scalar (t.get p)) at h
      simpa only [hs, model, Holds] using h
  exact .cons ⟨⟨hasTy_canonize ht.1.1, fun _ => Definedness.defined_canonize (ht.1.2 rfl)⟩,
    FrontendRangeSelect.canonize_holds _ _ hsummary⟩ .nil

/-- A checked abstract trace follows exactly the frozen generic evaluator's
iteration space, reads and writes; its summary is a proved output postcondition. -/
theorem generic_total {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges)
    {Γ : FEnv} {body : Region} {fs : List Fact} {sh : List Nat} {e : Ty} {initInfo : Option Interval}
    {maps : List AffineMap} {iters : List IterKind} {outMap : AffineMap} {B : State}
    (hs : StructuredOk maps iters.length (fs.map (·.base.ty) ++ [.tensor sh e]) ranges)
    (hout : maps.drop fs.length = [outMap])
    (hwalk : walk R Γ body fs sh e outMap (allPoints ranges) (fun _ => initInfo) = some B)
    {ctx : Ctx} {call : CallSem} {env : Env} (hib : ctx.indexBits = ib)
    (hΓ : Definedness.Refined.EnvHolds model ib Γ env) {ins : List RVal} {t : Tensor}
    (hi : Definedness.Refined.FactsHold model ib ins fs)
    (ht : Definedness.Refined.Fact.Holds model ib (.tensor t) ⟨⟨.tensor sh e, true⟩, initInfo⟩) :
    ∃ rs, genericSem ctx maps iters ins [.tensor t] (fun inner => evalRegion opSem inner call env body) = .ok rs ∧
      Definedness.Refined.FactsHold model ib rs [⟨⟨.tensor sh e, true⟩, summarize sh B⟩] := by
  have hk : ins.length = fs.length := List.Forall₂.length_eq hi
  have hinM : MapsInBounds ranges (maps.take ins.length) (fs.map (·.base.ty)) := by
    have h := List.forall₂_take (fs.map (·.base.ty)).length hs.inBounds
    rw [List.take_left] at h
    simpa only [List.length_map, ← hk] using h
  have houtM : mapInBounds ranges outMap sh = true := by
    have h := List.forall₂_drop (fs.map (·.base.ty)).length hs.inBounds
    rw [List.drop_left] at h
    simp only [List.length_map, hout] at h
    cases h with
    | cons hm hn => exact hm
  have hinit : Represents ib sh e (fun _ => initInfo) t := by
    refine ⟨ht.1, ?_⟩
    intro p hp
    cases initInfo with
    | none => trivial
    | some r => exact ht.2 p (ht.1.1.1.symm ▸ hp)
  obtain ⟨final, hf, hfr⟩ := walk_sound (ctx := ctx) (call := call) R hib hΓ hi hinM houtM (allPoints ranges)
    (fun _ => initInfo) B t hwalk (fun p hp => (mem_allPoints ranges p).mp hp) hinit
  refine ⟨[.tensor final.canonize], ?_, summarized hfr⟩
  have hex := genericSem_execute (outs := [t]) hs hi.typed (List.Forall₂.cons ht.1.1 List.Forall₂.nil)
    ctx (fun inner => evalRegion opSem inner call env body)
  have hout' : maps.drop ins.length = [outMap] := by rw [hk, hout]
  simpa only [List.map_cons, List.map_nil, hout', hf, Outcome.bind_ok] using hex

/-- A finite, source-derived abstract execution. Exhaustion is refusal, never a
claim about unvisited coordinates. The caller chooses the explicit static budget. -/
def rule (maxPoints ib : Nat) (bodyBooks : (ranges : List Nat) → BookAt ib ranges)
    (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨⟨.tensor sh e, true⟩, initInfo⟩]), some maps, some iters, [body] =>
      match maps.drop ins.length, structuredRanges? maps iters.length (ins.map (·.base.ty) ++ [.tensor sh e]) with
      | [outMap], some ranges =>
        if shapeSize ranges ≤ maxPoints ∧ op.results.mapM (fun v => vtyOf ib v.ty) = some [.tensor sh e] then
          (walk (bodyBooks ranges) Γ body ins sh e outMap (allPoints ranges) (fun _ => initInfo)).map
            fun B => [⟨⟨.tensor sh e, true⟩, summarize sh B⟩]
        else none
      | _, _ => none
    | _, _, _, _ => none
  else none

def book (maxPoints ib : Nat) (bodyBooks : (ranges : List Nat) → BookAt ib ranges) :
    Definedness.Refined.Book model opSem ib where
  rule := rule maxPoints ib bodyBooks
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    unfold rule at h
    split at h
    · next hc =>
      obtain ⟨hm, hd, hn⟩ := hc
      split at h
      · next ins sh e initInfo maps iters body hsplit hmaps hiters hregs =>
        split at h
        · next outMap ranges hout hranges =>
          split at h
          · next hc =>
            obtain ⟨_, hres⟩ := hc
            cases hw : walk (bodyBooks ranges) Γ body ins sh e outMap (allPoints ranges) (fun _ => initInfo) with
            | none => simp [hw] at h
            | some B =>
              simp only [hw, Option.map_some, Option.some.injEq] at h
              subst rs
              obtain ⟨xs, ys, hio, hxs, hys⟩ := FrontendRangeGeneric.split_facts hsplit ha
              obtain _ | ⟨ht, hnil⟩ := hys
              cases hnil
              rename_i t
              cases t with
              | scalar _ => cases ht.1.1
              | tensor t =>
                obtain ⟨vals, hv, hvf⟩ := generic_total (ctx := ctx) (call := call) (bodyBooks ranges)
                  (structuredRanges?_spec hranges) hout hw hib hΓ hxs ht
                refine ⟨vals, ?_, hvf⟩
                have hlen : vals.length = op.blind.results.length := by
                  rw [List.Forall₂.length_eq hvf, Op.blind_results, List.length_map]
                  have hl := optionMapM_length _ _ _ hres
                  simp only [List.length_cons, List.length_nil] at hl ⊢
                  exact hl
                rw [linalg_sem_of hm hd (.inl hn)]
                simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
                  attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hv,
                  Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
          · cases h
        · cases h
      · cases h
    · cases h

#print axioms book

#print axioms step_sound
#print axioms walk_sound

end MerlinVerify.Rules.FrontendRangeReduce

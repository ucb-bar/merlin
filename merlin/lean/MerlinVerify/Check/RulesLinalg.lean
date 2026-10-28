/-
Typing rules and safety proofs for the `linalg` structured ops (design §4.5, progress).

`genericSem_safe` is the core: a structured op whose static checks pass (`structuredRanges?`: one
map per operand, loop ranges determined by the operands, every access in bounds over the
iteration box) and whose body is safe for the element types is never `stuck`, and its results
have the output types. `linalg.generic` and `linalg.fill` are rules over it; `linalg.index` is
safe at a loop depth that provides the coordinate it reads.
-/
import MerlinVerify.Check.Affine
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

/-! ### Operand types -/

/-- Shape of a value of this type (scalars have rank 0). -/
def VTy.shape : VTy → List Nat
  | .scalar _ => []
  | .tensor sh _ => sh

/-- The scalar a structured op's body receives for an operand of this type. -/
def VTy.eltScalar (t : VTy) : VTy := .scalar (eltOfVTy t)

def VTy.isTensor : VTy → Bool
  | .tensor .. => true
  | .scalar _ => false

theorem shapes_typed {ib : Nat} : ∀ {vs : List RVal} {tys : List VTy}, HasTys ib vs tys →
    vs.map shapeOf = tys.map VTy.shape
  | [], [], _ => rfl
  | v :: _, t :: _, .cons hv hrest => by
    rw [List.map_cons, List.map_cons, shapes_typed hrest]
    congr 1
    cases v <;> cases t
    · rfl
    · exact (hv : False).elim
    · exact (hv : False).elim
    · exact hv.1

theorem tensors_of_typed {ib : Nat} : ∀ {vs : List RVal} {tys : List VTy}, HasTys ib vs tys →
    (∀ t ∈ tys, t.isTensor = true) → ∃ ts : List Tensor, vs = ts.map RVal.tensor
  | [], [], _, _ => ⟨[], rfl⟩
  | v :: _, t :: _, .cons hv hrest, hall => by
    obtain ⟨us, rfl⟩ := tensors_of_typed hrest (fun u hu => hall u (List.mem_cons_of_mem _ hu))
    have ht := hall t (List.mem_cons_self ..)
    cases v with
    | tensor x => exact ⟨x :: us, rfl⟩
    | scalar _ => cases t <;> simp_all [VTy.isTensor, RVal.HasTy]

/-- Scalars typed by the element types of `tys` are typed by `tys.map VTy.eltScalar`. -/
theorem valsTyped_scalars {ib : Nat} : ∀ {vs : List Val} {tys : List VTy},
    ValsTyped ib vs (tys.map eltOfVTy) → HasTys ib (vs.map RVal.scalar) (tys.map VTy.eltScalar)
  | [], [], _ => List.Forall₂.nil
  | _ :: _, _ :: _, .cons hv hrest => List.Forall₂.cons hv (valsTyped_scalars hrest)

theorem scalars_of_typed {ib : Nat} : ∀ {ys : List RVal} {tys : List VTy},
    HasTys ib ys (tys.map VTy.eltScalar) → ∃ vs, ys = vs.map RVal.scalar ∧ ValsTyped ib vs (tys.map eltOfVTy)
  | [], [], _ => ⟨[], rfl, List.Forall₂.nil⟩
  | y :: _, _ :: _, .cons hy hrest => by
    obtain ⟨vs, rfl, hvs⟩ := scalars_of_typed hrest
    cases y with
    | scalar v => exact ⟨v :: vs, rfl, List.Forall₂.cons hy hvs⟩
    | tensor _ => exact (hy : False).elim

/-! ### Lists of outcomes -/

theorem mapM_ofTensors_safe (g : RVal → Outcome Tensor) (h : ∀ t, g (.tensor t) = .ok t) :
    ∀ ts : List Tensor, Safe (· = ts) ((ts.map RVal.tensor).mapM g)
  | [] => Safe.of_ok rfl
  | t :: ts => by
    simp only [List.map_cons, List.mapM_cons, h, Outcome.bindOp_eq, Outcome.bind_ok]
    exact Safe.bind (mapM_ofTensors_safe g h ts) fun us hus => Safe.of_ok (by rw [hus])

theorem mapM_ofScalars_safe {ib : Nat} (g : RVal → Outcome Val) (h : ∀ v, g (.scalar v) = .ok v)
    {tys : List Ty} : ∀ {vs : List Val}, ValsTyped ib vs tys →
      Safe (ValsTyped ib · tys) ((vs.map RVal.scalar).mapM g)
  | [], hvs => Safe.of_ok hvs
  | v :: vs, hvs => by
    simp only [List.map_cons, List.mapM_cons, h, Outcome.bindOp_eq, Outcome.bind_ok]
    obtain _ | ⟨hv, hrest⟩ := hvs
    exact Safe.bind (mapM_ofScalars_safe g h hrest) fun us hus => Safe.of_ok (List.Forall₂.cons hv hus)

theorem mapM_ofOption {α β : Type} (f : α → Option β) (r : α → String) :
    ∀ (l : List α) (ys : List β), l.mapM f = some ys →
      l.mapM (fun x => Outcome.ofOption (r x) (f x)) = .ok ys
  | [], ys, h => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h; subst h; rfl
  | a :: l, ys, h => by
    simp only [List.mapM_cons] at h
    cases hfa : f a with
    | none => simp [hfa] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [hfa, hl] at h
      | some bs =>
        simp only [hfa, hl, Option.pure_def, bind, Option.bind, Option.some.injEq] at h
        subst h
        have hb : Outcome.ofOption (r a) (f a) = .ok b := by rw [hfa]; rfl
        rw [List.mapM_cons, hb, mapM_ofOption f r l bs hl]
        rfl

theorem foldlM_safe {α β : Type} (f : β → α → Outcome β) (I : β → Prop) (P : α → Prop)
    (hf : ∀ b a, I b → P a → Safe I (f b a)) :
    ∀ (l : List α) (b : β), (∀ a ∈ l, P a) → I b → Safe I (l.foldlM f b)
  | [], b, _, hb => Safe.of_ok hb
  | a :: l, b, hl, hb => by
    rw [List.foldlM_cons]
    exact Safe.bind (hf b a hb (hl a (List.mem_cons_self ..)))
      fun b' hb' => foldlM_safe f I P hf l b' (fun x hx => hl x (List.mem_cons_of_mem _ hx)) hb'

/-! ### Reads and writes at a point of the box -/

/-- Maps whose accesses over the box `ranges` stay inside operands of the given types. -/
abbrev MapsInBounds (ranges : List Nat) (maps : List AffineMap) (tys : List VTy) : Prop :=
  List.Forall₂ (fun m t => mapInBounds ranges m t.shape = true) maps tys

theorem readAt_safe {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {v : RVal}
    {t : VTy} {m : AffineMap} (hv : RVal.HasTy ib v t) (hm : mapInBounds ranges m t.shape = true) :
    Safe (Val.HasTy ib (eltOfVTy t)) (readAt v m pt) := by
  cases v with
  | scalar x =>
    cases t with
    | tensor _ _ => exact (hv : False).elim
    | scalar t =>
      simp only [mapInBounds, VTy.shape, List.length_nil, beq_iff_eq, List.zip_nil_right,
        List.all_nil, Bool.and_true, List.length_eq_zero_iff] at hm
      simp only [readAt, hm, List.isEmpty_nil, ↓reduceIte]
      exact Safe.of_ok hv
  | tensor x =>
    cases t with
    | scalar _ => exact (hv : False).elim
    | tensor sh e =>
      obtain ⟨hs, he, hg⟩ := hv
      simp only [VTy.shape] at hm
      obtain ⟨i, hi, hin⟩ := mapInBounds_sound hm hpt
      simp only [readAt, hs]
      rw [hi]
      exact Safe.of_ok (hg i hin)

theorem mapM_read_safe {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {α : Type}
    (toR : α → RVal) (g : α × AffineMap → Outcome Val) (hg : ∀ a m, g (a, m) = readAt (toR a) m pt) :
    ∀ (as : List α) (ms : List AffineMap) (tys : List VTy), HasTys ib (as.map toR) tys →
      MapsInBounds ranges ms tys → Safe (ValsTyped ib · (tys.map eltOfVTy)) ((as.zip ms).mapM g)
  | [], [], [], _, _ => Safe.of_ok List.Forall₂.nil
  | a :: as, m :: ms, t :: ts, .cons ha hrest, .cons hm hms => by
    simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq, hg]
    exact Safe.bind (readAt_safe hpt ha hm) fun x hx =>
      Safe.bind (mapM_read_safe hpt toR g hg as ms ts hrest hms) fun xs hxs =>
        Safe.of_ok (List.Forall₂.cons hx hxs)
  | [], _ :: _, _, h, hm => by cases h; cases hm
  | _ :: _, [], _, h, hm => by cases hm; cases h

theorem writeAt_safe {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {x : Tensor}
    {t : VTy} {m : AffineMap} {v : Val} (hx : RVal.HasTy ib (.tensor x) t)
    (hm : mapInBounds ranges m t.shape = true) (hv : Val.HasTy ib (eltOfVTy t) v) :
    Safe (fun y => RVal.HasTy ib (.tensor y) t) (writeAt x m pt v) := by
  cases t with
  | scalar _ => exact (hx : False).elim
  | tensor sh e =>
    obtain ⟨hs, he, hg⟩ := hx
    simp only [VTy.shape] at hm
    obtain ⟨i, hi, hin⟩ := mapInBounds_sound hm hpt
    simp only [writeAt, hs]
    rw [hi]
    refine Safe.of_ok (show RVal.HasTy ib (.tensor (x.set i v)) (.tensor sh e) from
      ⟨hs, he, fun j hj => ?_⟩)
    simp only [Tensor.set]
    split
    · exact hv
    · exact hg j hj

theorem mapM_write_safe {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt)
    (g : Val × Tensor × AffineMap → Outcome Tensor) (hg : ∀ v x m, g (v, x, m) = writeAt x m pt v) :
    ∀ (vs : List Val) (cur : List Tensor) (ms : List AffineMap) (tys : List VTy),
      ValsTyped ib vs (tys.map eltOfVTy) → HasTys ib (cur.map RVal.tensor) tys →
      MapsInBounds ranges ms tys →
      Safe (fun cur' => HasTys ib (cur'.map RVal.tensor) tys) ((vs.zip (cur.zip ms)).mapM g)
  | [], [], [], [], _, _, _ => Safe.of_ok List.Forall₂.nil
  | v :: vs, x :: xs, m :: ms, t :: ts, .cons hv hvs, .cons hx hxs, .cons hm hms => by
    simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq, hg]
    exact Safe.bind (writeAt_safe hpt hx hm hv) fun y hy =>
      Safe.bind (mapM_write_safe hpt g hg vs xs ms ts hvs hxs hms) fun ys hys =>
        Safe.of_ok (List.Forall₂.cons hy hys)
  | [], _, _, _ :: _, hvs, _, _ => by cases hvs
  | _ :: _, _, _, [], hvs, _, _ => by cases hvs
  | _, [], _, _ :: _, _, hxs, _ => by cases hxs
  | _, _ :: _, _, [], _, hxs, _ => by cases hxs
  | _, _, [], _ :: _, _, _, hms => by cases hms
  | _, _, _ :: _, [], _, _, hms => by cases hms

/-! ### One iteration point -/

theorem bodyAt_safe {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {ctx : Ctx}
    (hib : ctx.indexBits = ib) {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {body : Ctx → RegionSem} {inTys outTys : List VTy}
    (hins : HasTys ib ins inTys) (hcur : HasTys ib (cur.map RVal.tensor) outTys)
    (hinM : MapsInBounds ranges inMaps inTys) (houtM : MapsInBounds ranges outMaps outTys)
    (hbody : RegionSafe ib body
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩) :
    Safe (ValsTyped ib · (outTys.map eltOfVTy)) (bodyAt ctx ins inMaps outMaps cur body pt) := by
  unfold bodyAt
  simp only [Outcome.bindOp_eq]
  refine Safe.bind (mapM_read_safe hpt id _ (by intro _ _; rfl) ins inMaps inTys
    (by simpa using hins) hinM) fun xs hxs => ?_
  refine Safe.bind (mapM_read_safe hpt RVal.tensor _ (by intro _ _; rfl) cur outMaps outTys
    hcur houtM) fun ys hys => ?_
  have hvals : HasTys ib ((xs ++ ys).map RVal.scalar) ((inTys ++ outTys).map VTy.eltScalar) := by
    rw [List.map_append, List.map_append]
    exact List.rel_append (valsTyped_scalars hxs) (valsTyped_scalars hys)
  refine Safe.bind (hbody { ctx with loopIndex := pt } hib (Nat.le_of_eq hpt.1.symm) _ hvals)
    fun zs hzs => ?_
  have hlen : zs.length = cur.length := by
    rw [HasTys.length hzs, List.length_map, ← HasTys.length hcur, List.length_map]
  simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte]
  obtain ⟨vs, rfl, hvs⟩ := scalars_of_typed hzs
  exact mapM_ofScalars_safe _ (by intro _; rfl) hvs

theorem loopStep_safe {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {ctx : Ctx}
    (hib : ctx.indexBits = ib) {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {body : Ctx → RegionSem} {inTys outTys : List VTy}
    (hins : HasTys ib ins inTys) (hcur : HasTys ib (cur.map RVal.tensor) outTys)
    (hinM : MapsInBounds ranges inMaps inTys) (houtM : MapsInBounds ranges outMaps outTys)
    (hbody : RegionSafe ib body
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩) :
    Safe (fun cur' => HasTys ib (cur'.map RVal.tensor) outTys)
      (loopStep ctx ins inMaps outMaps body cur pt) := by
  unfold loopStep
  simp only [Outcome.bindOp_eq]
  exact Safe.bind (bodyAt_safe hpt hib hins hcur hinM houtM hbody) fun vs hvs =>
    mapM_write_safe hpt _ (by intro _ _ _; rfl) vs cur outMaps outTys hvs hcur houtM

/-! ### Structured ops -/

/-- Static checks of a structured op with `n` loops over operands of types `tys`: one map per
operand, each with `n` dims and no symbols; every loop's range read off an operand; every access in
bounds over the iteration box. Returns the loop ranges. -/
def structuredRanges? (maps : List AffineMap) (n : Nat) (tys : List VTy) : Option (List Nat) :=
  if maps.length = tys.length ∧ (∀ m ∈ maps, m.numDims = n ∧ m.numSyms = 0) then
    match (List.range n).mapM (loopRange maps (tys.map VTy.shape)) with
    | some ranges =>
      if (maps.zip tys).all (fun p => mapInBounds ranges p.1 p.2.shape) then some ranges else none
    | none => none
  else none

/-- What `structuredRanges?` establishes. -/
structure StructuredOk (maps : List AffineMap) (n : Nat) (tys : List VTy) (ranges : List Nat) :
    Prop where
  len : maps.length = tys.length
  dims : ∀ m ∈ maps, m.numDims = n ∧ m.numSyms = 0
  loops : (List.range n).mapM (loopRange maps (tys.map VTy.shape)) = some ranges
  inBounds : MapsInBounds ranges maps tys

theorem structuredRanges?_spec {maps : List AffineMap} {n : Nat} {tys : List VTy} {ranges : List Nat}
    (h : structuredRanges? maps n tys = some ranges) : StructuredOk maps n tys ranges := by
  unfold structuredRanges? at h
  split at h
  · next hc =>
    split at h
    · next rs hrs =>
      split at h
      · next hall =>
        cases h
        refine ⟨hc.1, hc.2, hrs, List.forall₂_iff_zip.2 ⟨hc.1, fun {m t} hmem => ?_⟩⟩
        exact List.all_eq_true.1 hall (m, t) hmem
      · cases h
    · cases h
  · cases h

theorem hasTys_canonize {ib : Nat} : ∀ {ts : List Tensor} {tys : List VTy},
    HasTys ib (ts.map RVal.tensor) tys → HasTys ib (ts.map fun t => RVal.tensor t.canonize) tys
  | [], [], _ => List.Forall₂.nil
  | _ :: _, _ :: _, .cons ht hrest => List.Forall₂.cons (hasTy_canonize ht) (hasTys_canonize hrest)

/-- **Structured-op safety.** A structured op passing the static checks, on typed operands and with
a body safe for their element types, is never `stuck`, and its results have the output types. -/
theorem genericSem_safe {ib : Nat} {ctx : Ctx} (hib : ctx.indexBits = ib) {maps : List AffineMap}
    {iters : List IterKind} {ins outs : List RVal} {body : Ctx → RegionSem} {inTys outTys : List VTy}
    {ranges : List Nat} (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hout : ∀ t ∈ outTys, t.isTensor = true) (hins : HasTys ib ins inTys)
    (houts : HasTys ib outs outTys)
    (hbody : RegionSafe ib body
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩) :
    Safe (HasTys ib · outTys) (genericSem ctx maps iters ins outs body) := by
  obtain ⟨ts, rfl⟩ := tensors_of_typed houts hout
  have hall : HasTys ib (ins ++ ts.map RVal.tensor) (inTys ++ outTys) := List.rel_append hins houts
  have hshapes := shapes_typed hall
  have hlenR : ranges.length = iters.length := by
    rw [optionMapM_length _ _ _ hs.loops, List.length_range]
  have hL : maps.length = (ins ++ ts.map RVal.tensor).length := by rw [hs.len, HasTys.length hall]
  have hD : maps.any (fun m => m.numDims ≠ iters.length || m.numSyms ≠ 0) = false := by
    rw [List.any_eq_false]
    intro m hm
    simp [(hs.dims m hm).1, (hs.dims m hm).2]
  have hR : (maps.zip ((ins ++ ts.map RVal.tensor).map shapeOf)).any
      (fun p => p.1.results.length ≠ p.2.length) = false := by
    rw [hshapes, List.zip_map_right, List.any_map, List.any_eq_false]
    intro p hp
    have hin := List.forall₂_iff_zip.1 hs.inBounds |>.2 hp
    simp only [mapInBounds, Bool.and_eq_true, beq_iff_eq] at hin
    simp [Function.comp, hin.1]
  have hLR : loopRanges maps ((ins ++ ts.map RVal.tensor).map shapeOf) iters.length = .ok ranges := by
    rw [hshapes]
    exact mapM_ofOption _ _ _ _ hs.loops
  have hk : ins.length = inTys.length := HasTys.length hins
  have hinM : MapsInBounds ranges (maps.take ins.length) inTys := by
    have := List.forall₂_take inTys.length hs.inBounds
    rwa [List.take_left, ← hk] at this
  have houtM : MapsInBounds ranges (maps.drop ins.length) outTys := by
    have := List.forall₂_drop inTys.length hs.inBounds
    rwa [List.drop_left, ← hk] at this
  have hbody' : RegionSafe ib body
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩ := by
    rw [hlenR]; exact hbody
  unfold genericSem
  simp only [hL, ne_eq, not_true_eq_false, ↓reduceIte, hD, Bool.false_eq_true, hR, Outcome.bindOp_eq]
  refine Safe.bind (mapM_ofTensors_safe _ (by intro _; rfl) ts) fun ts' hts' => ?_
  subst hts'
  rw [hLR, Outcome.bind_ok]
  refine Safe.bind (foldlM_safe _ (fun cur => HasTys ib (cur.map RVal.tensor) outTys)
    (Idx.InBounds ranges) (fun cur pt hcur hpt => loopStep_safe hpt hib hins hcur hinM houtM hbody')
    _ _ (fun pt hp => (mem_allPoints _ _).1 hp) houts) fun final hfinal => ?_
  exact Safe.of_ok (hasTys_canonize hfinal)

/-! ### Attributes of the blinded op -/

theorem splitInsOuts_blind {α : Type} (op : Op) (xs : List α) :
    splitInsOuts op.blind xs = splitInsOuts op xs := by
  simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?]

theorem attrAffineMaps?_blind (op : Op) (k : String) : attrAffineMaps? op.blind k = attrAffineMaps? op k := by
  simp only [attrAffineMaps?, Op.blind_attr?]

theorem iterKinds?_blind (op : Op) : iterKinds? op.blind = iterKinds? op := by
  simp only [iterKinds?, Op.blind_attr?]

/-- `operandSegmentSizes` splits typed operands as it splits their types. -/
theorem splitInsOuts_typed {ib : Nat} {op : Op} {args : List RVal} {argTys inTys outTys : List VTy}
    (h : splitInsOuts op argTys = some (inTys, outTys)) (hargs : HasTys ib args argTys) :
    ∃ ins outs, splitInsOuts op args = some (ins, outs) ∧ HasTys ib ins inTys ∧ HasTys ib outs outTys := by
  have hlen := HasTys.length hargs
  unfold splitInsOuts at h ⊢
  split at h
  · next ni no _ =>
    split at h
    · next hc =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine ⟨args.take ni.toNat, args.drop ni.toNat, ?_, List.forall₂_take _ hargs,
        List.forall₂_drop _ hargs⟩
      rw [ite_eq_left ⟨hc.1, hc.2.1, by rw [hlen]; exact hc.2.2⟩]
    · cases h
  · cases h

/-- Reduce a `linalg.generic`/`fill`/`index` op's safety to `Linalg.sem` on the blinded op. -/
theorem linalg_sem_of {ctx : Ctx} {call : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (hm : unmodelledAttr? op.blind = none)
    (hd : dialectOf op.name = "linalg")
    (hn : op.name = "linalg.generic" ∨ op.name = "linalg.fill" ∨ op.name = "linalg.index") :
    opSem ctx call op args regions =
      Linalg.sem ctx (restrictCall (calleeOf op.blind) call) op.blind args regions := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  rcases hn with hn | hn | hn <;> simp [hn]

/-! ### Rules -/

/-- `linalg.generic`: tensor outputs (the results), maps and iterators accepted by the static
checks, and a body over the operands' element types at the op's loop depth. -/
def genericRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op argTys, attrAffineMaps? op "indexing_maps", iterKinds? op with
    | some (inTys, outTys), some maps, some iters =>
      if (∀ t ∈ outTys, t.isTensor = true) ∧ resTys = outTys ∧ op.results.length = outTys.length ∧
          (structuredRanges? maps iters.length (inTys ++ outTys)).isSome then
        some [⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩]
      else none
    | _, _, _ => none
  else none

theorem genericRule_sound (ib : Nat) : RuleSound ib genericRule := by
  intro d op argTys resTys sigs hm h
  simp only [genericRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next inTys outTys maps iters hsplit hmaps hiters =>
      split at h
      · next hc2 =>
        obtain ⟨hout, rfl, hres, hsr⟩ := hc2
        obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hsr
        cases h
        intro ctx call args regions hib _ hargs hregs
        obtain _ | ⟨hbody, hnil⟩ := hregs
        cases hnil
        obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
        rw [linalg_sem_of hm hd (.inl hn)]
        simp only [Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, attrAffineMaps?_blind,
          iterKinds?_blind, hio, hmaps, hiters, Outcome.bindOp_eq]
        refine Safe.bind (genericSem_safe hib (structuredRanges?_spec hranges) hout hins houts hbody)
          fun rs hrs => ?_
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length hrs, Op.blind_results, List.length_map, hres]
        simp only [hlen, ↓reduceIte, Outcome.pure_eq]
        exact Safe.of_ok hrs
      · cases h
    · cases h
  · cases h

/-- The maps `linalg.fill` runs with: the value is read at every point, the output is the identity. -/
def fillMaps (r : Nat) : List AffineMap := [{ numDims := r, numSyms := 0, results := [] }, identityMap r]

/-- `linalg.fill %v outs(%t)`: one value (a scalar, or a rank-0 tensor), one tensor output (the
result), and a body from the value's element type to the output's at the output's rank. -/
def fillRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.fill" then
    match splitInsOuts op argTys with
    | some ([vt], [.tensor sh e]) =>
      if resTys = [.tensor sh e] ∧ op.results.length = 1 ∧
          (structuredRanges? (fillMaps sh.length) sh.length [vt, .tensor sh e]).isSome then
        some [⟨[vt.eltScalar, .scalar e], [.scalar e], sh.length⟩]
      else none
    | _ => none
  else none

theorem fillRule_sound (ib : Nat) : RuleSound ib fillRule := by
  intro d op argTys resTys sigs hm h
  simp only [fillRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next vt sh e hsplit =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, hres, hsr⟩ := hc2
        obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hsr
        cases h
        intro ctx call args regions hib _ hargs hregs
        obtain _ | ⟨hbody, hnil⟩ := hregs
        cases hnil
        rename_i body
        obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
        obtain _ | ⟨hv, _ | _⟩ := hins
        obtain _ | ⟨ht, _ | _⟩ := houts
        rename_i v x
        cases x with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        have hts : t.shape = sh := ht.1
        rw [linalg_sem_of hm hd (.inr (.inl hn))]
        simp only [Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio, hts, Outcome.bindOp_eq]
        have hs : StructuredOk (fillMaps sh.length) (List.replicate sh.length IterKind.parallel).length
            ([vt] ++ [VTy.tensor sh e]) ranges := by
          rw [List.length_replicate]; exact structuredRanges?_spec hranges
        have hbody' : RegionSafe ib body
            ⟨([vt] ++ [VTy.tensor sh e]).map VTy.eltScalar, [VTy.tensor sh e].map VTy.eltScalar,
              (List.replicate sh.length IterKind.parallel).length⟩ := by
          rw [List.length_replicate]; exact hbody
        refine Safe.bind (genericSem_safe (ins := [v]) (outs := [.tensor t]) hib hs
          (by simp [VTy.isTensor]) (List.Forall₂.cons hv .nil) (List.Forall₂.cons ht .nil) hbody')
          fun rs hrs => ?_
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length hrs, Op.blind_results, List.length_map, hres]; rfl
        simp only [hlen, ↓reduceIte, Outcome.pure_eq]
        exact Safe.of_ok hrs
      · cases h
    · cases h
  · cases h

/-- `linalg.index`: reads coordinate `dim` of the iteration point, which the loop depth provides. -/
def indexRule : OpRule := fun d op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.index" ∧ argTys = [] ∧
      resTys = [.scalar .index] then
    match attrInt? op "dim" with
    | some k => if 0 ≤ k ∧ k.toNat < d then some [] else none
    | none => none
  else none

theorem indexRule_sound (ib : Nat) : RuleSound ib indexRule := by
  intro d op argTys resTys sigs hm h
  simp only [indexRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl, rfl⟩ := hc
    split at h
    · next k hk =>
      split at h
      · next hkd =>
        cases h
        intro ctx call args regions hib hdep _ _
        have hlt : k.toNat < ctx.loopIndex.length := Nat.lt_of_lt_of_le hkd.2 hdep
        rw [linalg_sem_of hm hd (.inr (.inr hn))]
        simp only [Linalg.sem, Op.blind_name, hn, attrInt?_blind, hk,
          List.getElem?_eq_getElem hlt, ge_iff_le, hkd.1, ↓reduceIte]
        refine Safe.of_ok (List.Forall₂.cons ?_ .nil)
        simp [RVal.HasTy, Val.HasTy, Scalar.conforms, hib]
      · cases h
    · cases h
  · cases h

/-! ### Named ops -/

theorem natArray?_blind (op : Op) (k : String) :
    LinalgNamed.natArray? op.blind k = LinalgNamed.natArray? op k := by
  simp only [LinalgNamed.natArray?, attrDenseArray?, Op.blind_attr?]

theorem mapsOr_blind (op : Op) (dflt : List AffineMap) :
    LinalgNamed.mapsOr op.blind dflt = LinalgNamed.mapsOr op dflt := by
  simp only [LinalgNamed.mapsOr, attrAffineMaps?_blind]

/-- Reduce a named `linalg` op's safety to `LinalgNamed.sem` on the blinded op. -/
theorem named_sem_of {ctx : Ctx} {call : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (hm : unmodelledAttr? op.blind = none)
    (hd : dialectOf op.name = "linalg")
    (hn : (op.name == "linalg.generic" || op.name == "linalg.fill" || op.name == "linalg.index") = false) :
    opSem ctx call op args regions =
      LinalgNamed.sem ctx (restrictCall (calleeOf op.blind) call) op.blind args regions := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd, hn, Bool.false_eq_true, ↓reduceIte]

/-- A named op runs its generic form and checks the result count. -/
theorem run_safe {ib : Nat} {ctx : Ctx} (hib : ctx.indexBits = ib) {op : Op} {maps : List AffineMap}
    {iters : List IterKind} {ins outs : List RVal} {body : Ctx → RegionSem} {inTys outTys : List VTy}
    {ranges : List Nat} (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hout : ∀ t ∈ outTys, t.isTensor = true) (hins : HasTys ib ins inTys)
    (houts : HasTys ib outs outTys)
    (hbody : RegionSafe ib body
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩)
    (hres : op.results.length = outTys.length) :
    Safe (HasTys ib · outTys) (LinalgNamed.run ctx op maps iters ins outs body) := by
  unfold LinalgNamed.run
  refine Safe.bind (genericSem_safe hib hs hout hins houts hbody) fun rs hrs => ?_
  rw [ite_eq_left (by rw [HasTys.length hrs, hres])]
  exact Safe.of_ok hrs

/-- A named op whose generic form runs `maps` and `iters` over inputs of types `inTys` and outputs
of types `outTys`: `genericRule`'s conditions, with the maps the op's definition fixes. -/
def namedOk (maps : List AffineMap) (iters : List IterKind) (inTys outTys : List VTy) (op : Op)
    (resTys : List VTy) : Option (List RegionSig) :=
  if (∀ t ∈ outTys, t.isTensor = true) ∧ resTys = outTys ∧ op.results.length = outTys.length ∧
      (structuredRanges? maps iters.length (inTys ++ outTys)).isSome then
    some [⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩]
  else none

theorem namedOk_safe {ib : Nat} {ctx : Ctx} (hib : ctx.indexBits = ib) {maps : List AffineMap}
    {iters : List IterKind} {inTys outTys : List VTy} {op : Op} {resTys : List VTy}
    {sigs : List RegionSig} (h : namedOk maps iters inTys outTys op resTys = some sigs)
    {ins outs : List RVal} {regions : List (Ctx → RegionSem)} (hins : HasTys ib ins inTys)
    (houts : HasTys ib outs outTys) (hregs : List.Forall₂ (RegionSafe ib) regions sigs) :
    ∃ body, regions = [body] ∧
      Safe (HasTys ib · resTys) (LinalgNamed.run ctx op.blind maps iters ins outs body) := by
  unfold namedOk at h
  split at h
  · next hc =>
    obtain ⟨hout, rfl, hres, hsr⟩ := hc
    obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hsr
    cases h
    obtain _ | ⟨hbody, hnil⟩ := hregs
    cases hnil
    exact ⟨_, rfl, run_safe hib (structuredRanges?_spec hranges) hout hins houts hbody
      (by rw [Op.blind_results, List.length_map, hres])⟩
  · cases h

theorem shapeOf_typed {ib : Nat} {v : RVal} {t : VTy} (h : RVal.HasTy ib v t) : shapeOf v = t.shape := by
  simpa using shapes_typed (vs := [v]) (tys := [t]) (List.Forall₂.cons h .nil)

theorem forall₂_reverse {α β : Type} {R : α → β → Prop} {xs : List α} {ys : List β}
    (h : List.Forall₂ R xs ys) : List.Forall₂ R xs.reverse ys.reverse := by
  induction h with
  | nil => exact .nil
  | cons h hs ih => simpa only [List.reverse_cons] using List.rel_append ih (.cons h .nil)

/-- Map has one destination, last in the operands. Its body receives that destination element
after all input elements, as required by LLVM f5e175f and the installed compiler. -/
def mapRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.map" then
    match argTys.reverse with
    | tout :: insRev =>
      namedOk (List.replicate argTys.length (identityMap tout.shape.length))
        (LinalgNamed.parallel tout.shape.length) insRev.reverse [tout] op resTys
    | [] => none
  else none

theorem mapRule_sound (ib : Nat) : RuleSound ib mapRule := by
  intro d op argTys resTys sigs hm h
  simp only [mapRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next tout insRev hrev =>
      have hlen : argTys.length = insRev.length + 1 := by
        simpa only [List.length_reverse, List.length_cons] using congrArg List.length hrev
      intro ctx call args regions hib _ hargs hregs
      have ha := forall₂_reverse hargs
      rw [hrev] at ha
      cases hv : args.reverse with
      | nil => rw [hv] at ha; cases ha
      | cons init xs =>
        rw [hv] at ha
        obtain _ | ⟨ho, hins⟩ := ha
        obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h
          (forall₂_reverse hins) (.cons ho .nil) hregs
        rw [named_sem_of hm hd (by simp [hn])]
        simp only [LinalgNamed.sem, Op.blind_name, hn, hv, shapeOf_typed ho]
        simpa only [hlen, List.map_const', List.length_append, List.length_reverse,
          List.length_singleton, HasTys.length hins] using hsafe
    · cases h
  · cases h

/-- `linalg.matmul`: `C(m,n) += A(m,k) · B(k,n)`, or the op's own `indexing_maps`. -/
def matmulRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.matmul" then
    match splitInsOuts op argTys with
    | some ([ta, tb], [tc]) =>
      namedOk (LinalgNamed.mapsOr op
          [LinalgNamed.mapOf 3 [0, 2], LinalgNamed.mapOf 3 [2, 1], LinalgNamed.mapOf 3 [0, 1]])
        [.parallel, .parallel, .reduction] [ta, tb] [tc] op resTys
    | _ => none
  else none

theorem matmulRule_sound (ib : Nat) : RuleSound ib matmulRule := by
  intro d op argTys resTys sigs hm h
  simp only [matmulRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ta tb tc hsplit =>
      intro ctx call args regions hib _ hargs hregs
      obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
      obtain _ | ⟨ha, _ | ⟨hb, _ | _⟩⟩ := hins
      obtain _ | ⟨hc', _ | _⟩ := houts
      obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons ha (.cons hb .nil)) (.cons hc' .nil) hregs
      rw [named_sem_of hm hd (by simp [hn])]
      simp only [LinalgNamed.sem, Op.blind_name, hn, splitInsOuts_blind, hio, mapsOr_blind]
      exact hsafe
    · cases h
  · cases h

/-- `linalg.batch_matmul`: `C(b,m,n) += A(b,m,k) · B(b,k,n)`, or the op's own `indexing_maps`. -/
def batchMatmulRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.batch_matmul" then
    match splitInsOuts op argTys with
    | some ([ta, tb], [tc]) =>
      namedOk (LinalgNamed.mapsOr op [LinalgNamed.mapOf 4 [0, 1, 3], LinalgNamed.mapOf 4 [0, 3, 2],
          LinalgNamed.mapOf 4 [0, 1, 2]])
        [.parallel, .parallel, .parallel, .reduction] [ta, tb] [tc] op resTys
    | _ => none
  else none

theorem batchMatmulRule_sound (ib : Nat) : RuleSound ib batchMatmulRule := by
  intro d op argTys resTys sigs hm h
  simp only [batchMatmulRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ta tb tc hsplit =>
      intro ctx call args regions hib _ hargs hregs
      obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
      obtain _ | ⟨ha, _ | ⟨hb, _ | _⟩⟩ := hins
      obtain _ | ⟨hc', _ | _⟩ := houts
      obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons ha (.cons hb .nil)) (.cons hc' .nil) hregs
      rw [named_sem_of hm hd (by simp [hn])]
      simp only [LinalgNamed.sem, Op.blind_name, hn, splitInsOuts_blind, hio, mapsOr_blind]
      exact hsafe
    · cases h
  · cases h

/-- `linalg.quantized_matmul`: `C(m,n) += (A(m,k) - azp) · (B(k,n) - bzp)`, zero points read at
every point. -/
def quantizedMatmulRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.quantized_matmul" then
    match splitInsOuts op argTys with
    | some ([ta, tb, tazp, tbzp], [tc]) =>
      namedOk [LinalgNamed.mapOf 3 [0, 2], LinalgNamed.mapOf 3 [2, 1], LinalgNamed.mapOf 3 [],
          LinalgNamed.mapOf 3 [], LinalgNamed.mapOf 3 [0, 1]]
        [.parallel, .parallel, .reduction] [ta, tb, tazp, tbzp] [tc] op resTys
    | _ => none
  else none

theorem quantizedMatmulRule_sound (ib : Nat) : RuleSound ib quantizedMatmulRule := by
  intro d op argTys resTys sigs hm h
  simp only [quantizedMatmulRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ta tb tazp tbzp tc hsplit =>
      intro ctx call args regions hib _ hargs hregs
      obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
      obtain _ | ⟨ha, _ | ⟨hb, _ | ⟨hza, _ | ⟨hzb, _ | _⟩⟩⟩⟩ := hins
      obtain _ | ⟨hc', _ | _⟩ := houts
      obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons ha (.cons hb (.cons hza (.cons hzb .nil))))
        (.cons hc' .nil) hregs
      rw [named_sem_of hm hd (by simp [hn])]
      simp only [LinalgNamed.sem, Op.blind_name, hn, splitInsOuts_blind, hio]
      exact hsafe
    · cases h
  · cases h

/-- `linalg.copy`: the input and output read and written at the same point. -/
def copyRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.copy" then
    match splitInsOuts op argTys with
    | some ([ti], [tout]) =>
      namedOk [identityMap tout.shape.length, identityMap tout.shape.length]
        (LinalgNamed.parallel tout.shape.length) [ti] [tout] op resTys
    | _ => none
  else none

theorem copyRule_sound (ib : Nat) : RuleSound ib copyRule := by
  intro d op argTys resTys sigs hm h
  simp only [copyRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ti tout hsplit =>
      intro ctx call args regions hib _ hargs hregs
      obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
      obtain _ | ⟨hi, _ | _⟩ := hins
      obtain _ | ⟨ho, _ | _⟩ := houts
      obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons hi .nil) (.cons ho .nil) hregs
      rw [named_sem_of hm hd (by simp [hn])]
      simp only [LinalgNamed.sem, Op.blind_name, hn, splitInsOuts_blind, hio, shapeOf_typed ho]
      exact hsafe
    · cases h
  · cases h

/-- `linalg.broadcast ins(%i) outs(%init) dimensions = dims`: the input reads the output dims not in
`dims`, which must be as many as the input's rank. -/
def broadcastRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.broadcast" then
    match argTys, LinalgNamed.natArray? op "dimensions" with
    | [ti, tout], some dims =>
      if ((List.range tout.shape.length).filter (fun d => !(dims.contains d))).length =
          ti.shape.length then
        namedOk [LinalgNamed.mapOf tout.shape.length
            ((List.range tout.shape.length).filter (fun d => !(dims.contains d))),
          identityMap tout.shape.length]
          (LinalgNamed.parallel tout.shape.length) [ti] [tout] op resTys
      else none
    | _, _ => none
  else none

theorem broadcastRule_sound (ib : Nat) : RuleSound ib broadcastRule := by
  intro d op argTys resTys sigs hm h
  simp only [broadcastRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ti tout dims hdims =>
      split at h
      · next hk =>
        intro ctx call args regions hib _ hargs hregs
        obtain _ | ⟨hi, _ | ⟨ho, _ | _⟩⟩ := hargs
        obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons hi .nil) (.cons ho .nil) hregs
        rw [named_sem_of hm hd (by simp [hn])]
        simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hdims, shapeOf_typed hi,
          shapeOf_typed ho, hk, ne_eq, not_true_eq_false, ↓reduceIte]
        exact hsafe
      · cases h
    · cases h
  · cases h

/-- The input map of `linalg.transpose`: input dim `p` reads the loop `i` with `permutation[i] = p`. -/
def transposeInMap (n : Nat) (perm : List Nat) : AffineMap :=
  LinalgNamed.mapOf n ((List.range n).map fun p => (perm.findIdx? (· = p)).getD 0)

/-- `linalg.transpose ins(%i) outs(%init) permutation = perm`: `perm` is a permutation of the
output's rank. -/
def transposeRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.transpose" then
    match argTys, LinalgNamed.natArray? op "permutation" with
    | [ti, tout], some perm =>
      if ¬(perm.length ≠ tout.shape.length ∨
          (List.range tout.shape.length).any (fun p => !(perm.contains p))) then
        namedOk [transposeInMap tout.shape.length perm, identityMap tout.shape.length]
          (LinalgNamed.parallel tout.shape.length) [ti] [tout] op resTys
      else none
    | _, _ => none
  else none

theorem transposeRule_sound (ib : Nat) : RuleSound ib transposeRule := by
  intro d op argTys resTys sigs hm h
  simp only [transposeRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ti tout perm hperm =>
      split at h
      · next hcond =>
        intro ctx call args regions hib _ hargs hregs
        obtain _ | ⟨hi, _ | ⟨ho, _ | _⟩⟩ := hargs
        obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons hi .nil) (.cons ho .nil) hregs
        rw [named_sem_of hm hd (by simp [hn])]
        simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hperm, shapeOf_typed ho]
        simp only [hcond, ↓reduceIte]
        exact hsafe
      · cases h
    · cases h
  · cases h

/-- Reduction's existing semantics permits either explicit segments or an equal-half split.
The implicit split is the representation exported by upstream variadic `linalg.reduce`. -/
def reductionSplit {α : Type} (op : Op) (xs : List α) : List α × List α :=
  (splitInsOuts op xs).getD (xs.take (xs.length / 2), xs.drop (xs.length / 2))

theorem splitInsOuts_isSome_length {α β : Type} (op : Op) (xs : List α) (ys : List β)
    (hlen : xs.length = ys.length) :
    (splitInsOuts op xs).isSome = (splitInsOuts op ys).isSome := by
  unfold splitInsOuts
  split
  · simp only [hlen]; split <;> rfl
  · rfl

theorem reductionSplit_typed {ib : Nat} {op : Op} {args : List RVal}
    {tys ins outs : List VTy} (h : reductionSplit op tys = (ins, outs))
    (ha : HasTys ib args tys) : ∃ xs ys, reductionSplit op args = (xs, ys) ∧
      HasTys ib xs ins ∧ HasTys ib ys outs := by
  cases hs : splitInsOuts op tys with
  | some pair =>
    simp only [reductionSplit, hs, Option.getD_some] at h
    subst pair
    obtain ⟨xs, ys, hv, hx, hy⟩ := splitInsOuts_typed hs ha
    exact ⟨xs, ys, by simp [reductionSplit, hv], hx, hy⟩
  | none =>
    have hv : splitInsOuts op args = none := by
      have his : (splitInsOuts op args).isSome = false := by
        rw [splitInsOuts_isSome_length op args tys (HasTys.length ha), hs]
        rfl
      cases hv : splitInsOuts op args <;> simp_all
    simp only [reductionSplit, hs, Option.getD_none, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨args.take (tys.length / 2), args.drop (tys.length / 2), ?_,
      List.forall₂_take _ ha, List.forall₂_drop _ ha⟩
    simp only [reductionSplit, hv, Option.getD_none, HasTys.length ha]

theorem reductionSplit_append {α : Type} (op : Op) (xs ins outs : List α)
    (h : reductionSplit op xs = (ins, outs)) : xs = ins ++ outs := by
  unfold reductionSplit splitInsOuts at h
  split at h
  · split at h
    all_goals
      simp only [Option.getD_some, Option.getD_none, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact (List.take_append_drop _ xs).symm
  · simp only [Option.getD_none, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact (List.take_append_drop _ xs).symm

/-- Variadic reduction retains input order, output order and the original scalar body. -/
def reduceRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "linalg" ∧ op.name = "linalg.reduce" then
    match reductionSplit op argTys, LinalgNamed.natArray? op "dimensions" with
    | (ti :: ins, outs), some dims =>
      let n := ti.shape.length
      let kept := (List.range n).filter (fun d => !(dims.contains d))
      namedOk (List.replicate (ins.length + 1) (identityMap n) ++
          List.replicate outs.length (LinalgNamed.mapOf n kept))
        ((List.range n).map fun d => if dims.contains d then .reduction else .parallel)
        (ti :: ins) outs op resTys
    | _, _ => none
  else none

theorem reduceRule_sound (ib : Nat) : RuleSound ib reduceRule := by
  intro d op argTys resTys sigs hm h
  simp only [reduceRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ti ins outs dims hsplit hdims =>
      intro ctx call args regions hib _ hargs hregs
      obtain ⟨xs, ys, hio, hxs, hys⟩ := reductionSplit_typed hsplit hargs
      obtain _ | ⟨hi, hrest⟩ := hxs
      obtain ⟨body, rfl, hsafe⟩ := namedOk_safe hib h (.cons hi hrest) hys hregs
      rw [named_sem_of hm hd (by simp [hn])]
      dsimp only [reductionSplit] at hio
      simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hdims,
        splitInsOuts_blind, hio, shapeOf_typed hi,
        List.map_const', List.length_cons, HasTys.length hrest, HasTys.length hys]
      exact hsafe
    · cases h
  · cases h

/-- The `linalg` typing rule: structured ops with explicit maps, `fill`, `index`, and the named ops
defined by their generic form. -/
def linalgRule : OpRule :=
  firstRule [genericRule, fillRule, indexRule, matmulRule, batchMatmulRule, quantizedMatmulRule,
    copyRule, broadcastRule, transposeRule, mapRule, reduceRule]

theorem linalgRule_sound (ib : Nat) : RuleSound ib linalgRule := by
  apply firstRule_sound
  simp only [List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true]
  exact ⟨genericRule_sound ib, fillRule_sound ib, indexRule_sound ib, matmulRule_sound ib,
    batchMatmulRule_sound ib, quantizedMatmulRule_sound ib, copyRule_sound ib, broadcastRule_sound ib,
    transposeRule_sound ib, mapRule_sound ib, reduceRule_sound ib⟩

end MerlinVerify.Check

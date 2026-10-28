/- C1 glue for the narrow dequantization lowering: per-channel i32, per-tensor i32 and i64. -/
import MerlinVerify.Rules.DequantPerTensor
import MerlinVerify.Check.C1
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Rules.DequantSite

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Sem.Ops.Linalg
open MerlinVerify.Rules.DequantScalar MerlinVerify.Rules.DequantTensor

@[simp] theorem blind_successors (op : Op) : op.blind.successors = op.successors := by
  cases op; rfl

@[simp] theorem config_blind (op : Op) : Dequant.config? op.blind = Dequant.config? op := by
  unfold Dequant.config?
  simp only [Dequant.metadataOK, Op.blind_attr?, blind_successors]
  rw [Op.blind_results]
  rcases op.results with _ | ⟨r, _ | rest⟩ <;>
    simp [Op.blind_name, Op.blind_attr?, attrInt?]

/-- Preserve correspondence while binding a target-only temporary that no mapped name uses. -/
theorem Corr.temporary {ρ : Rho} {envS envT : Env} (hc : Corr ρ envS envT)
    {id : Nat} (hf : ρ.notImage id) (v : RVal) : Corr ρ envS ((id, v) :: envT) := by
  intro x y hxy
  have hne : y ≠ id := by
    obtain ⟨p, hp, hpy⟩ := lookup_mem_image hxy
    exact hpy ▸ hf p hp
  have hb : (y == id) = false := by simpa using hne
  simpa [Env.get?, List.lookup, hb] using hc x y hxy

theorem mapM_append {α β : Type} (f : α → Outcome β) (xs ys : List α) :
    Outcome.mapM f (xs ++ ys) = (Outcome.mapM f xs).bind
      (fun a => (Outcome.mapM f ys).bind fun b => .ok (a ++ b)) := by
  induction xs with
  | nil =>
    simp only [List.nil_append, Outcome.mapM, Outcome.bind_ok, List.nil_append]
    cases Outcome.mapM f ys <;> rfl
  | cons x xs ih => simp [Outcome.mapM, ih, Outcome.bind_assoc]

/-- Fetching a list followed by one known binding. -/
theorem getAll_append_one (env : Env) (ids : List Nat) (id : Nat) (vs : List RVal) (v : RVal)
    (h : env.getAll ids = .ok vs) (hv : env.get? id = some v) :
    env.getAll (ids ++ [id]) = .ok (vs ++ [v]) := by
  simp only [Env.getAll, mapM_append]
  change (env.getAll ids).bind _ = _
  simp [h, Outcome.mapM, hv, Outcome.ofOption]

theorem source_sem (ctx : Ctx) (call : CallSem) (op : Op) (c : Dequant.Config)
    (args : List RVal) (hmodel : unmodelledAttr? op.blind = none)
    (hd : dialectOf op.name = "quant_ext") (hc : Dequant.config? op = some c) :
    opSem ctx call op args [] = Dequant.run c ctx.M args := by
  have hn : op.name ≠ "quant_ext.quantize_per_tensor" := by
    rcases DequantPerTensor.config_name hc with hn | hn <;> simp [hn]
  rw [opSem_modelled hmodel]
  simp [dispatch, hd, hn, Dequant.sem, hc]

theorem empty_sem (ctx : Ctx) (call : CallSem) (op : Op) (shape : List Nat)
    (hm : unmodelledAttr? op.blind = none) (hd : dialectOf op.name = "tensor")
    (hn : op.name = "tensor.empty")
    (ht : Sem.Ops.Tensor.resultTensorTy op = some (shape, .float .f32)) :
    opSem ctx call op [] [] = .ok [.tensor (Tensor.full shape (.float .f32) .poison)] := by
  rw [tensor_sem_of hm hd]
  simp [Sem.Ops.Tensor.sem, hn, resultTensorTy_blind, ht]

theorem generic_sem (ctx : Ctx) (call : CallSem) (op : Op) (ins : List RVal) (t : Tensor)
    (body : Ctx → RegionSem) (maps : List AffineMap) (iters : List IterKind) (out : Tensor)
    (hm : unmodelledAttr? op.blind = none) (hd : dialectOf op.name = "linalg")
    (hn : op.name = "linalg.generic") (hseg : attrDenseArray? op "operandSegmentSizes" = some [3, 1])
    (hlen : ins.length = 3) (hmaps : attrAffineMaps? op "indexing_maps" = some maps)
    (hit : iterKinds? op = some iters) (hr : op.results.length = 1)
    (hout : genericSem ctx maps iters ins [.tensor t] body = .ok [.tensor out]) :
    opSem ctx call op (ins ++ [.tensor t]) [body] = .ok [.tensor out] := by
  have htake : (ins ++ [RVal.tensor t]).take 3 = ins := by rw [← hlen]; simp
  have hdrop : (ins ++ [RVal.tensor t]).drop 3 = [.tensor t] := by rw [← hlen]; simp
  have hsplit : splitInsOuts op.blind (ins ++ [.tensor t]) = some (ins, [.tensor t]) := by
    rw [splitInsOuts_blind]
    simp [splitInsOuts, hseg, hlen, htake, hdrop]
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name]
  rw [hd]
  simp only [hn, beq_self_eq_true, Bool.true_or, ↓reduceIte]
  simp only [Linalg.sem, Op.blind_name, hn, hsplit, attrAffineMaps?_blind, hmaps,
    iterKinds?_blind, hit, hout, Outcome.bind_ok]
  simp [Op.blind_results, hr]

/-- Every syntactic and semantic precondition of a dequantization site is recomputed. -/
def Checks (ρ : Rho) (c : Dequant.Config) (src empty generic : Op)
    (rs re rg : Value) (ids : List Nat) (args : List Value) (ops : List Op)
    (maps : List AffineMap) (iters : List IterKind) : Prop :=
  src.isTerminator = false ∧ empty.isTerminator = false ∧ generic.isTerminator = false ∧
  dialectOf src.name = "quant_ext" ∧ dialectOf empty.name = "tensor" ∧
  dialectOf generic.name = "linalg" ∧ empty.name = "tensor.empty" ∧ generic.name = "linalg.generic" ∧
  unmodelledAttr? src.blind = none ∧ unmodelledAttr? empty.blind = none ∧
  unmodelledAttr? generic.blind = none ∧ Dequant.config? src = some c ∧
  empty.operands = [] ∧ generic.operands = ids ++ [re.id] ∧
  src.operands.mapM ρ.find = some ids ∧
  Sem.Ops.Tensor.resultTensorTy empty = some (c.shape, .float .f32) ∧
  rg.ty.staticTensor? = some (c.shape, .float .f32) ∧
  attrDenseArray? generic "operandSegmentSizes" = some [3, 1] ∧
  attrAffineMaps? generic "indexing_maps" = some maps ∧ iterKinds? generic = some iters ∧
  freshIds ρ [re.id, rg.id] = true ∧
  checkLayout c.channel c.axis c.shape (Dequant.paramShape c) maps iters = true ∧
  DequantPerTensor.bodyOK c.channel args ops = true ∧
  src.successors = [] ∧ empty.successors = [] ∧ generic.successors = []

instance (ρ : Rho) (c : Dequant.Config) (src empty generic : Op)
    (rs re rg : Value) (ids : List Nat) (args : List Value) (ops : List Op)
    (maps : List AffineMap) (iters : List IterKind) :
    Decidable (Checks ρ c src empty generic rs re rg ids args ops maps iters) := by
  unfold Checks
  infer_instance

def Frame (src empty generic : Op) (rs re rg : Value) (args : List Value) (ops : List Op) : Prop :=
  src.results = [rs] ∧ empty.results = [re] ∧ generic.results = [rg] ∧
  src.regions = [] ∧ empty.regions = [] ∧ generic.regions = [.mk [.mk args ops]]

/-- The core C1 obligation: replacing the source op by the checked empty/generic pair extends
SSA correspondence by precisely the externally visible result. -/
theorem site_sound {ρ : Rho} {c : Dequant.Config} {src empty generic : Op}
    {rs re rg : Value} {ids : List Nat} {args : List Value} {ops : List Op}
    {maps : List AffineMap} {iters : List IterKind}
    (frame : Frame src empty generic rs re rg args ops)
    (h : Checks ρ c src empty generic rs re rg ids args ops maps iters)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env) (hc : Corr ρ envS envT) :
    SiteRel ((rs.id, rg.id) :: ρ)
      (execOps opSem ctx callT envT [empty, generic]) (execOps opSem ctx callS envS [src]) := by
  rcases frame with ⟨hrs, hre, hrg, hsregs, heregs, hgregs⟩
  rcases h with ⟨hsterm, heterm, hgterm,
    hsd, hed, hgd, hen, hgn, hsmodel, hemodel, hgmodel, hconfig, heargs, hgargs,
    hmap, hetype, -, hseg, hmaps, hiters, hfresh, hlayout, hbody, -, -, -⟩
  rw [execOps_cons_step _ _ _ _ _ _ hsterm]
  unfold stepOp
  rw [hsregs]
  simp only [evalRegions]
  simp_rw [source_sem ctx callS src c _ hsmodel hsd hconfig]
  cases hget : envS.getAll src.operands with
  | ub => exact Or.inl rfl
  | stuck r => exact Or.inr (Or.inl ⟨r, rfl⟩)
  | ok vals =>
    simp only [Outcome.bind_ok]
    cases hrun : Dequant.run c ctx.M vals with
    | ub => exact Or.inl rfl
    | stuck r => exact Or.inr (Or.inl ⟨r, rfl⟩)
    | ok result =>
      -- A successful source operation supplies the exact runtime typing/shape facts.
      let initial := Tensor.full c.shape (.float .f32) .poison
      let mid : Env := (re.id, RVal.tensor initial) :: envT
      obtain ⟨q, s, z, out, rfl, rfl, hloop⟩ :=
        DequantPerTensor.generic_of_run ctx callT mid hrun hlayout hbody
      simp only [hrs, Env.bindAll, List.length_singleton, ↓reduceIte, List.map_cons,
        List.map_nil, List.zip_cons_cons, List.zip_nil_right, List.cons_append,
        List.nil_append, Outcome.bind_ok, execOps]
      have hfr := (freshIds_sound hfresh).2
      have hmid : Corr ρ envS mid := Corr.temporary hc (hfr _ (by simp)) _
      have hinputs : mid.getAll ids = .ok [.tensor q, .tensor s, .tensor z] := by
        rw [← Corr.getAll hmid _ _ hmap]
        exact hget
      have hvals : mid.getAll generic.operands =
          .ok [.tensor q, .tensor s, .tensor z, .tensor initial] := by
        rw [hgargs]
        exact getAll_append_one mid ids re.id _ _ hinputs (by simp [mid, Env.get?, List.lookup])
      have hgeneric : opSem ctx callT generic
          [.tensor q, .tensor s, .tensor z, .tensor initial]
          (evalRegions opSem callT mid generic.regions) = .ok [.tensor out] := by
        rw [hgregs]
        exact generic_sem ctx callT generic [.tensor q, .tensor s, .tensor z] initial _ maps iters out
          hgmodel hgd hgn hseg rfl hmaps hiters (by simp [hrg]) hloop
      have heval : execOps opSem ctx callT envT [empty, generic] =
          .ok ((rg.id, RVal.tensor out) :: mid) := by
        rw [execOps_cons_step _ _ _ _ _ _ heterm]
        unfold stepOp
        rw [heargs, heregs]
        simp only [Env.getAll, Outcome.mapM, evalRegions, Outcome.bind_ok]
        rw [empty_sem ctx callT empty c.shape hemodel hed hen hetype]
        simp only [hre, Env.bindAll, List.length_singleton, ↓reduceIte, List.map_cons,
          List.map_nil, List.zip_cons_cons, List.zip_nil_right, List.cons_append,
          List.nil_append, Outcome.bind_ok]
        change execOps opSem ctx callT mid [generic] = _
        rw [execOps_cons_step _ _ _ _ _ _ hgterm]
        unfold stepOp
        rw [hvals, Outcome.bind_ok, hgeneric]
        simp [hrg, Env.bindAll, execOps]
      refine Or.inr (Or.inr ⟨_, _, rfl, heval, ?_⟩)
      exact Corr.extend hmid [rs.id] [rg.id] [.tensor out] rfl rfl (by simp)
        (fun y hy => by simpa using hfr y (by simp_all))

/-- Instance checking consumes one source dequantization op and its two target operations. -/
def check (ρ : Rho) : List Op → List Op → Option Rho
  | [src], [empty, generic] =>
    match src.results, empty.results, generic.results, generic.regions, Dequant.config? src,
      src.operands.mapM ρ.find, attrAffineMaps? generic "indexing_maps", iterKinds? generic with
    | [rs], [re], [rg], [.mk [.mk args ops]], some c, some ids, some maps, some iters =>
      if src.regions.isEmpty && empty.regions.isEmpty then
        if Checks ρ c src empty generic rs re rg ids args ops maps iters then
        some ((rs.id, rg.id) :: ρ) else none
      else none
    | _, _, _, _, _, _, _, _ => none
  | _, _ => none

theorem check_sound : ∀ ρ siteS siteT ρ', check ρ siteS siteT = some ρ' →
    (∀ op ∈ siteS, op.isTerminator = false) ∧ (∀ op ∈ siteT, op.isTerminator = false) ∧
    ∀ (ctx : Ctx) (callS callT : CallSem) (envS envT : Env), Corr ρ envS envT →
      CallAgree callS callT (calleesOps siteS ++ calleesOps siteT) →
      SiteRel ρ' (execOps opSem ctx callT envT siteT) (execOps opSem ctx callS envS siteS)
  | ρ, [src], [empty, generic], ρ', h => by
    simp only [check] at h
    split at h
    · split at h
      · next hregs =>
        split at h
        · next hchecks =>
          simp only [Option.some.injEq] at h
          subst ρ'
          have hterms := hchecks
          rcases hterms with ⟨hst, het, hgt, _⟩
          refine ⟨by simpa using hst, ?_, fun ctx callS callT envS envT hc _ =>
            site_sound ?_ hchecks ctx callS callT envS envT hc⟩
          · intro op hop
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hop
            rcases hop with rfl | rfl <;> assumption
          · simp only [Frame]
            simp_all only [Bool.and_eq_true, List.isEmpty_iff, and_self]
        · cases h
      · cases h
    · cases h
  | _, [], _, _, h => by simp [check] at h
  | _, _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_], [], _, h => by simp [check] at h
  | _, [_], [_], _, h => by simp [check] at h
  | _, [_], _ :: _ :: _ :: _, _, h => by simp [check] at h

def rule : Rule := { name := "dequant_i8_f32", check, sound := check_sound }

end MerlinVerify.Rules.DequantSite

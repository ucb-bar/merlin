/- A partial operational rulebook: successful-result preservation plus typed callback
congruence. Known-stuck quantization contracts are identified, never given progress. -/
import MerlinVerify.Rules.ScalarSquareTypingFamilies

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open ScalarSquareComposition

abbrev Signature := List VTy × List VTy
abbrev Signatures := List (String × Signature)
def CallTyped (ib : Nat) (signatures : Signatures) (call : CallSem) : Prop :=
  ∀ name sig, signatures.lookup name = some sig → ∀ args, HasTys ib args sig.1 →
    ∀ results, call name args = .ok results → HasTys ib results sig.2
def CallAdmits (ib : Nat) (signatures : Signatures) (D : CallDomain) : Prop :=
  ∀ name sig, signatures.lookup name = some sig → ∀ args, HasTys ib args sig.1 → D name args

def flatRule (ib : Nat) : OpRule := fun d op ats rts =>
  if op.regions = [] ∧ calleeOf op.blind = none then
    match (if op.name = "linalg.index" then indexRule d op ats rts else merlinRule ib 0 op ats rts) with
    | some [] => some []
    | _ => none
  else none

theorem flatRule_related (ib : Nat) : RelatedRuleSound ib (flatRule ib) := by
  intro d op ats rts sigs hm h
  unfold flatRule at h
  split at h
  · next hc =>
    split at h
    · next hr =>
      cases h
      intro ctx callS callT args bodies hib ha hb
      cases hb
      refine ⟨opSem_call_local (fun name impossible => by rw [hc.2] at impossible; cases impossible), ?_⟩
      by_cases hn : op.name = "linalg.index"
      · simp only [hn, ↓reduceIte] at hr
        unfold indexRule at hr
        split at hr
        · next htypes =>
          obtain ⟨_, _, rfl, rfl⟩ := htypes
          split at hr
          · next k hk =>
            split at hr
            · next hkd =>
              intro results executed
              rw [linalg_sem_of hm (by rw [hn]; cbv) (.inr (.inr hn))] at executed
              simp only [Linalg.sem, Op.blind_name, hn, attrInt?_blind, hk] at executed
              cases hi : ctx.loopIndex[k.toNat]? with
              | none => simp [hi] at executed
              | some i =>
                simp only [hi, ge_iff_le, hkd.1, ↓reduceIte, Outcome.ok.injEq] at executed
                subst results
                exact .cons (by simp [RVal.HasTy, Val.HasTy, Scalar.conforms, hib]) .nil
            · cases hr
          · cases hr
        · cases hr
      · simp only [hn, ↓reduceIte] at hr
        exact (merlinRule_sound ib 0 op ats rts [] hm hr ctx callS args [] hib
          (Nat.zero_le _) ha .nil).2
    · cases h
  · cases h

def structuredRule (ib : Nat) : OpRule :=
  firstRule [flatRule ib, genericRule, fillRule, matmulRule, transposeRule, reduceRule]

theorem structuredRule_related (ib : Nat) : RelatedRuleSound ib (structuredRule ib) := by
  intro d op ats rts sigs hm h
  obtain ⟨rule, member, accepted⟩ := List.exists_of_findSome?_eq_some h
  have sound : RelatedRuleSound ib rule := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl
    · exact flatRule_related ib
    · exact genericRule_related ib
    · exact fillRule_related ib
    · exact matmulRule_related ib
    · exact transposeRule_related ib
    · exact reduceRule_related ib
  exact sound d op ats rts sigs hm accepted

/-- Static type obligations for the captured per-tensor quantization operations. The shared
semantics rejects these contracts (including their i64 zero points). This rule proves only
that they cannot return an ill-typed successful result; it does not implement quantization. -/
def rejectedQuantRule : OpRule := fun _ op ats rts =>
  if dialectOf op.name = "quant_ext" ∧ op.regions = [] then
    match ats, rts with
    | [.tensor sh (.int 8), .tensor [] (.float .f32), .tensor [] (.int 64)],
        [.tensor sh' (.float .f32)] =>
      if op.name = "quant_ext.dequantize_per_tensor" ∧ sh = sh' then some [] else none
    | [.tensor sh (.float .f32), .tensor [] (.float .f32), .tensor [] (.int 64)],
        [.tensor sh' (.int 8)] =>
      if op.name = "quant_ext.quantize_per_tensor" ∧ sh = sh' then some [] else none
    | _, _ => none
  else none

def callRule (signatures : Signatures) : OpRule := fun _ op ats rts =>
  if op.name = "func.call" ∧ op.regions = [] ∧ op.successors = [] then
    match attrSymbol? op.blind "callee" with
    | some name => if signatures.lookup name = some (ats, rts) then some [] else none
    | none => none
  else none

def typingRule (ib : Nat) (signatures : Signatures) : OpRule :=
  firstRule [structuredRule ib, rejectedQuantRule, callRule signatures]

theorem dequant_i64_stuck {ib : Nat} {a b : VTy} {shape : List Nat} {args : List RVal}
    (ha : HasTys ib args [a, b, .tensor shape (.int 64)]) (ctx : Ctx) (call : CallSem)
    (op : Op) (regions : List (Ctx → RegionSem)) :
    ∃ reason, Dequant.sem ctx call op args regions = .stuck reason := by
  cases hc : Dequant.config? op with
  | none => exact ⟨"unsupported quant_ext contract (requires signed i8/i32/f32 dequantization)", by simp [Dequant.sem, hc]⟩
  | some c =>
    simp only [Dequant.sem, hc]
    by_cases empty : regions.isEmpty = true
    · simp only [empty, ↓reduceIte]
      cases ha with
      | cons hq tail => cases tail with
        | cons hs tail => cases tail with
          | cons hz tail =>
            cases tail
            rename_i qv sv zv
            cases qv with
            | scalar _ => exact ⟨_, rfl⟩
            | tensor q =>
              cases sv with
              | scalar _ => exact ⟨_, rfl⟩
              | tensor s =>
                cases zv with
                | scalar _ => exact (hz : False).elim
                | tensor z =>
                  have invalid : Dequant.validInputs c q s z = false := by
                    simp [Dequant.validInputs, Dequant.tensorTyped, hz.2.1]
                  exact ⟨"dequantization: unsupported runtime types or parameter shapes", by simp [Dequant.run, invalid]⟩
    · exact ⟨"dequantization: unexpected regions", by simp [empty]⟩

theorem rejectedQuant_stuck {ib d : Nat} {op : Op} {ats rts : List VTy} {sigs : List RegionSig}
    (h : rejectedQuantRule d op ats rts = some sigs) (ctx : Ctx) (call : CallSem)
    (args : List RVal) (rs : List (Ctx → RegionSem)) (ha : HasTys ib args ats) :
    ∃ reason, opSem ctx call op args rs = .stuck reason := by
  unfold rejectedQuantRule at h
  split at h
  · next hc =>
    cases hu : unmodelledAttr? op.blind with
    | some reason => exact ⟨s!"{op.blind.name}: unmodelled attribute {reason}", by simp [opSem, hu]⟩
    | none =>
      simp only [opSem_modelled hu, dispatch, Op.blind_name, hc.1]
      split at h <;> try cases h
      all_goals exact dequant_i64_stuck ha ctx _ _ rs
  · cases h

/-- Every acceptance has a semantic proof, with external result contracts explicit. -/
theorem typingRule_sound {ib : Nat} {signatures : Signatures} {d : Nat} {op : Op}
    {ats rts : List VTy} {sigs : List RegionSig}
    (h : typingRule ib signatures d op ats rts = some sigs)
    (ctx : Ctx) (left right : CallSem) (args : List RVal) (bodies : List BodyPair)
    (hib : ctx.indexBits = ib) (ha : HasTys ib args ats)
    (hb : List.Forall₂ (RegionRelated ctx.M ib) bodies sigs)
    (D : CallDomain) (admit : CallAdmits ib signatures D) (calls : CallsAgree D left right)
    (typed : CallTyped ib signatures left) :
    Related (HasTys ib · rts) (opSem ctx left op args (bodies.map Prod.fst))
      (opSem ctx right op args (bodies.map Prod.snd)) := by
  by_cases hm : unmodelledAttr? op.blind = none
  · obtain ⟨rule, member, accepted⟩ := List.exists_of_findSome?_eq_some h
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl
    · exact structuredRule_related ib d op ats rts sigs hm accepted
        ctx left right args bodies hib ha hb
    · obtain ⟨reason, stuck⟩ := rejectedQuant_stuck accepted ctx left args (bodies.map Prod.fst) ha
      have hdialect : dialectOf op.name = "quant_ext" := by
        unfold rejectedQuantRule at accepted
        split at accepted
        · next hc => exact hc.1
        · cases accepted
      refine ⟨?_, fun _ impossible => by rw [stuck] at impossible; cases impossible⟩
      simp only [opSem_modelled hm, dispatch, Op.blind_name, hdialect, Dequant.sem]
      cases bodies <;> rfl
    · unfold callRule at accepted
      split at accepted
      · next hc =>
        split at accepted
        · next name hn =>
          split at accepted
          · next hs =>
            cases accepted
            cases hb
            have hargs := admit name (ats, rts) hs args ha
            have hcName : calleeOf op.blind = some name := by
              exact hn
            simp only [opSem_modelled hm, dispatch, Op.blind_name,
              show dialectOf op.name = "func" by rw [hc.1]; cbv,
              Func.sem, show op.blind.name = "func.call" from hc.1, hn,
              restrictCall, hcName, ↓reduceIte]
            simp only [show dialectOf "func.call" = "func" from by cbv]
            refine Related.bind ⟨calls name args hargs, typed name (ats, rts) hs args ha⟩ fun rs hrs => ?_
            split
            · exact Related.ok hrs
            · exact ⟨rfl, fun _ impossible => nomatch impossible⟩
          · cases accepted
        · cases accepted
      · cases accepted
  · cases hu : unmodelledAttr? op.blind with
    | none => exact (hm hu).elim
    | some _ =>
      simp only [opSem, hu]
      exact ⟨rfl, fun _ impossible => nomatch impossible⟩

end MerlinVerify.Rules.ScalarSquareTyping

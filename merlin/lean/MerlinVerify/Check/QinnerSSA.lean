/- Soundness statement: acceptance equates the source's explicit named-binding execution with
ordinary target execution, for arbitrary primitive computations and equal invoked callees.
The proof handles lexical SSA environments and nested regions, including captured values. -/
import MerlinVerify.Rules.QinnerBinding
set_option maxHeartbeats 2000000
namespace MerlinVerify.Check.QinnerSSA
open MerlinVerify MerlinVerify.Check Rules.QinnerBinding Sem.Ops

theorem regions_size (op : Op) : sizeOf op.regions < sizeOf op := by
  cases op
  simp only [Op.regions, Op.mk.sizeOf_spec]
  omega

def slot? (slots : List Slot) (s : Site) : Option Slot :=
  (slots.find? (fun t => t.key == s.key)).filter (fun t => t.arg.ty == s.result.ty)

mutual
  def operations (slots : List Slot) (rho : Rho) : List Op → List Op → Bool
    | [], ts => ts.isEmpty
    | s :: ss, ts => match site? s with
      | some site => match slot? slots site with
        | some slot => operations slots ((site.result.id,slot.arg.id)::rho) ss ts
        | none => false
      | none => match ts with
        | [] => false
        | t :: tail =>
          opHeadEq s t && s.operands.mapM rho.find == some t.operands &&
          regions slots rho s.regions t.regions && freshIds rho (t.results.map (·.id)) &&
          protects slots (t.results.map (·.id)) && (ss.isEmpty == tail.isEmpty) &&
          operations slots ((s.results.map (·.id)).zip (t.results.map (·.id)) ++ rho) ss tail
    termination_by ss _ => sizeOf ss
    decreasing_by
      all_goals simp_wf
      all_goals (try simp_all only [Op.regions])
      all_goals first | omega | (have hs := regions_size s; simp only [Op.regions] at hs; omega)
  def regions (slots : List Slot) (rho : Rho) : List Region → List Region → Bool
    | [], [] => true
    | s :: ss, t :: ts => region slots rho s t && regions slots rho ss ts
    | _, _ => false
    termination_by ss _ => sizeOf ss
    decreasing_by
      all_goals simp_wf
      all_goals (try simp_all only [Op.regions])
      all_goals omega
  def region (slots : List Slot) (rho : Rho) : Region → Region → Bool
    | .mk [.mk a ops], .mk [.mk b tail] =>
      a.map (·.ty) == b.map (·.ty) && freshIds rho (b.map (·.id)) &&
      protects slots (b.map (·.id)) &&
      operations slots ((a.map (·.id)).zip (b.map (·.id)) ++ rho) ops tail
    | .mk [], .mk [] => true
    | _, _ => false
    termination_by s _ => sizeOf s
    decreasing_by
      all_goals simp_wf
      all_goals (try simp_all only [Op.regions])
      all_goals omega
end

theorem slot_mem {slots : List Slot} {s : Site} {slot : Slot}
    (h : slot? slots s = some slot) : slot ∈ slots ∧ slot.key = s.key ∧ slot.arg.ty = s.result.ty := by
  simp only [slot?, Option.filter_eq_some_iff, beq_iff_eq] at h
  exact ⟨List.mem_of_find?_eq_some h.1, by simpa using List.find?_some h.1, h.2⟩

theorem regions_empty {slots : List Slot} {rho : Rho} {s t : List Region}
    (h : regions slots rho s t = true) : s.isEmpty = t.isEmpty := by
  cases s <;> cases t <;> simp_all [regions]

theorem eval_cons (interpret : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op) (rest : List Op) :
    evalOps interpret ctx call env (op::rest) =
    if op.isTerminator then
      if rest.isEmpty then env.getAll op.operands else .stuck "ops after terminator"
    else (env.getAll op.operands).bind fun args =>
      (interpret ctx call op args (evalRegions interpret call env op.regions)).bind fun values =>
      (env.bindAll op.results values).bind fun next => evalOps interpret ctx call next rest := by
  cases op; rfl
@[simp] theorem callees_cons (op : Op) (rest : List Op) :
    calleesOps (op::rest) = (calleeOf op.blind).toList ++ calleesRegions op.regions ++ calleesOps rest := by
  cases op; simp [calleesOps, Op.regions]

mutual
  theorem operations_sound (binding : String → RVal) (slots : List Slot) (callS callT : CallSem) :
      ∀ (rho : Rho) (ss ts : List Op) (ctx : Ctx) (envS envT : Env),
      operations slots rho ss ts = true → Corr rho envS envT → Supplies slots binding envT →
      CallAgree callS callT (calleesOps ss) →
      evalOps (sem binding) ctx callS envS ss = evalOps opSem ctx callT envT ts
    | _, [], [], _, _, _, _, _, _, _ => rfl
    | _, [], _ :: _, _, _, _, h, _, _, _ => by simp [operations] at h
    | rho, s :: ss, ts, ctx, envS, envT, h, hc, hb, hcall => by
      have hrest : CallAgree callS callT (calleesOps ss) := fun c hm =>
        hcall c (by simp only [callees_cons]; exact List.mem_append_right _ hm)
      cases hs : site? s with
      | some site =>
        cases ht : slot? slots site with
        | none => cases ts <;> simp [operations, hs, ht] at h
        | some slot =>
          have tail : operations slots ((site.result.id,slot.arg.id)::rho) ss ts = true := by
            cases ts <;> simpa only [operations, hs, ht] using h
          obtain ⟨empty, result, nonterm⟩ := site_shape hs
          have slotFacts := slot_mem ht
          have bound := hb slot slotFacts.1
          rw [slotFacts.2.1] at bound
          rw [eval_cons, nonterm]
          simp only [Bool.false_eq_true, ↓reduceIte, empty, Env.getAll, Outcome.mapM,
            Outcome.bind_ok, sem, hs, result, Env.bindAll, List.length_cons, List.length_nil,
            ↓reduceIte, List.map_cons, List.map_nil, List.zip_cons_cons, List.zip_nil_left,
            List.cons_append, List.nil_append]
          exact operations_sound binding slots callS callT _ ss ts ctx _ _ tail
            (corr_bind_left hc _ _ _ bound) hb hrest
      | none =>
        cases ts with
        | nil => simp [operations, hs] at h
        | cons t tail =>
          simp only [operations, hs, Bool.and_eq_true, beq_iff_eq] at h
          obtain ⟨⟨⟨⟨⟨⟨hhead,hmap⟩,hregs⟩,hfresh⟩,hprotect⟩,hend⟩,htail⟩ := h
          obtain ⟨hnd,hfr⟩ := freshIds_sound hfresh
          have hga := Corr.getAll hc s.operands t.operands hmap
          have hcallR : CallAgree callS callT (calleesRegions s.regions) := fun c hm =>
            hcall c (by simp only [callees_cons]; exact List.mem_append_left _ (List.mem_append_right _ hm))
          have hcallOp : ∀ c, calleeOf s.blind = some c → callS c = callT c := fun c hm =>
            hcall c (by simp only [callees_cons]; exact List.mem_append_left _ (List.mem_append_left _ (by simp [hm])))
          have hterm := isTerminator_of_opHeadEq hhead
          have hlen := results_length_of_opHeadEq hhead
          rw [eval_cons, eval_cons, hterm]
          by_cases term : t.isTerminator = true
          · rw [ite_eq_left term, ite_eq_left term, hend, hga]
          · rw [ite_eq_right term, ite_eq_right term, hga,
              regions_sound binding slots callS callT rho s.regions t.regions envS envT hregs hc hb hcallR]
            cases envT.getAll t.operands with
            | ub => rfl
            | stuck reason => rfl
            | ok args =>
              simp only [Outcome.bind_ok, sem, hs]
              rw [opSem_call_local (call' := callT) hcallOp, opSem_blind (blind_of_opHeadEq hhead)]
              cases opSem ctx callT t args (evalRegions opSem callT envT t.regions) with
              | ub => rfl
              | stuck reason => rfl
              | ok values =>
                simp only [Outcome.bind_ok]
                by_cases arity : s.results.length = values.length
                · rw [bindAll_eq_ok arity, bindAll_eq_ok (hlen ▸ arity)]
                  simp only [Outcome.bind_ok]
                  exact operations_sound binding slots callS callT _ ss tail ctx _ _ htail
                    (Corr.extend hc _ _ values (by simpa using arity) (by simpa [← hlen] using arity) hnd hfr)
                    (supplies_extend hb values hprotect) hrest
                · rw [bindAll_eq_stuck arity, bindAll_eq_stuck (fun e => arity (hlen ▸ e))]; rfl
    termination_by _ ss _ _ _ _ _ _ _ _ => sizeOf ss
    decreasing_by
      all_goals simp_wf
      all_goals (try simp_all only [Op.regions])
      all_goals first | omega | (have hs := regions_size s; simp only [Op.regions] at hs; omega)
  theorem regions_sound (binding : String → RVal) (slots : List Slot) (callS callT : CallSem) :
      ∀ (rho : Rho) (ss ts : List Region) (envS envT : Env),
      regions slots rho ss ts = true → Corr rho envS envT → Supplies slots binding envT →
      CallAgree callS callT (calleesRegions ss) →
      evalRegions (sem binding) callS envS ss = evalRegions opSem callT envT ts
    | _, [], [], _, _, _, _, _, _ => rfl
    | rho, (.mk bs) :: ss, t :: ts, envS, envT, h, hc, hb, hcall => by
      simp only [regions, Bool.and_eq_true] at h
      have hr : CallAgree callS callT (calleesBlocks bs) := fun c hm =>
        hcall c (by simp only [calleesRegions]; exact List.mem_append_left _ hm)
      have ht : CallAgree callS callT (calleesRegions ss) := fun c hm =>
        hcall c (by simp only [calleesRegions]; exact List.mem_append_right _ hm)
      simp only [evalRegions, List.cons.injEq]
      exact ⟨funext fun ctx => region_sound binding slots callS callT rho (.mk bs) t ctx envS envT h.1 hc hb hr,
        regions_sound binding slots callS callT rho ss ts envS envT h.2 hc hb ht⟩
    | _, [], _ :: _, _, _, h, _, _, _ => by simp [regions] at h
    | _, _ :: _, [], _, _, h, _, _, _ => by simp [regions] at h
    termination_by _ ss _ _ _ _ _ _ _ => sizeOf ss
    decreasing_by
      all_goals simp_wf
      all_goals (try simp_all only [Op.regions])
      all_goals omega
  theorem region_sound (binding : String → RVal) (slots : List Slot) (callS callT : CallSem) :
      ∀ (rho : Rho) (s t : Region) (ctx : Ctx) (envS envT : Env),
      region slots rho s t = true → Corr rho envS envT → Supplies slots binding envT →
      CallAgree callS callT (calleesBlocks s.blocks) →
      evalRegion (sem binding) ctx callS envS s = evalRegion opSem ctx callT envT t
    | rho, .mk [.mk a ops], .mk [.mk b tail], ctx, envS, envT, h, hc, hb, hcall => by
      simp only [region, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨hty,hfresh⟩,hprotect⟩,hops⟩ := h
      obtain ⟨hnd,hfr⟩ := freshIds_sound hfresh
      have hl : a.length = b.length := by simpa using congrArg List.length hty
      have calls : CallAgree callS callT (calleesOps ops) := fun c hm =>
        hcall c (by simp only [Region.blocks, calleesBlocks]; exact List.mem_append_left _ hm)
      simp only [evalRegion, evalBlocks, evalBlock]
      funext values
      by_cases arity : a.length = values.length
      · rw [bindAll_eq_ok arity, bindAll_eq_ok (hl ▸ arity)]
        simp only [Outcome.bind_ok]
        exact operations_sound binding slots callS callT _ ops tail ctx _ _ hops
          (Corr.extend hc _ _ values (by simpa using arity) (by simpa [← hl] using arity) hnd hfr)
          (supplies_extend hb values hprotect) calls
      · rw [bindAll_eq_stuck arity, bindAll_eq_stuck (fun e => arity (hl ▸ e))]; rfl
    | _, .mk [], .mk [], _, _, _, _, _, _, _ => rfl
    | _, .mk [], .mk (_ :: _ :: _), _, _, _, h, _, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk [], _, _, _, h, _, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk (_ :: _ :: _), _, _, _, h, _, _, _ => by simp [region] at h
    | _, .mk [], .mk [_], _, _, _, h, _, _, _ => by simp [region] at h
    | _, .mk [_], .mk [], _, _, _, h, _, _, _ => by simp [region] at h
    | _, .mk [_], .mk (_ :: _ :: _), _, _, _, h, _, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk [_], _, _, _, h, _, _, _ => by simp [region] at h
    termination_by _ s _ _ _ _ _ _ _ _ => sizeOf s
    decreasing_by
      all_goals simp_wf
      all_goals (try simp_all only [Op.regions])
      all_goals omega
end
#print axioms operations_sound
#print axioms region_sound
end MerlinVerify.Check.QinnerSSA

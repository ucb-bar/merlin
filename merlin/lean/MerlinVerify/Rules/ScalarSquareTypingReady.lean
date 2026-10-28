/- Discharge the operational premises of scalar-square composition from the complete
static checker. Every result fact is derived, including the lexical constant fact. -/
import MerlinVerify.Rules.ScalarSquareTypingFacts

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Check.ScalarSquare ScalarSquareSSA ScalarSquareComposition ScalarSquareClaims

theorem region_arguments {ib : Nat} {table : Signatures} {Γ : TEnv} {region : Region}
    {sig : RegionSig} (checked : regionCheck ib table Γ region sig = true) (args : List RVal) :
    BlockDomain ib region.blocks args ↔ HasTys ib args sig.args := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => cases checked
    | cons block rest =>
      cases block with
      | mk formals ops =>
        cases rest with
        | cons _ _ => cases checked
        | nil =>
          simp only [regionCheck, Bool.and_eq_true, beq_iff_eq] at checked
          exact arguments_types formals sig.args checked.1 args

theorem related_callbacks {ib : Nat} {table : Signatures} {Γ : TEnv}
    {regions : List Region} {sigs : List RegionSig} {call : CallSem} {env : Env}
    (checked : regionsCheck ib table Γ regions sigs = true) (he : EnvTyped ib Γ env)
    (typed : CallTyped ib table call) (M : FloatModel) (right : List (Ctx → RegionSem))
    (agree : RegionsAgree (Family M ib) ib regions (evalRegions opSem call env regions) right) :
    ∃ pairs : List BodyPair,
      pairs.map Prod.fst = evalRegions opSem call env regions ∧ pairs.map Prod.snd = right ∧
      List.Forall₂ (RegionRelated M ib) pairs sigs := by
  induction regions generalizing sigs right with
  | nil =>
    cases sigs <;> cases right <;> simp_all [regionsCheck, RegionsAgree, evalRegions]
  | cons region regions ih =>
    cases region with
    | mk blocks =>
      cases sigs with
      | nil => cases checked
      | cons sig sigs =>
        cases right with
        | nil => cases agree
        | cons r rs =>
          simp only [regionsCheck, Bool.and_eq_true] at checked
          simp only [evalRegions, RegionsAgree] at agree
          obtain ⟨pairs, hs, ht, hp⟩ := ih checked.2 rs agree.2
          refine ⟨((fun ctx => evalRegion opSem ctx call env (.mk blocks)), r) :: pairs,
            by simp [evalRegions, hs], by simp [ht], .cons ?_ hp⟩
          intro ctx hm hib hd args ha
          have inputs := (region_arguments checked.1 args).mpr ha
          exact ⟨agree.1 ctx ⟨hm, hib⟩ args inputs,
            regionCheck_preserves ib table Γ (.mk blocks) sig call env checked.1 he typed ctx hib args ha⟩

theorem primitive_checked {ib : Nat} {table : Signatures} {Γ : TEnv} {op : Op}
    {ats rts : List VTy} {sigs : List RegionSig} {d : Nat} {ctx : Ctx}
    {call : CallSem} {env : Env} {args : List RVal} {M : FloatModel} {D : CallDomain}
    (hrule : typingRule ib table d op ats rts = some sigs)
    (hregions : regionsCheck ib table Γ op.regions sigs = true)
    (he : EnvTyped ib Γ env) (typed : CallTyped ib table call) (admit : CallAdmits ib table D)
    (family : Family M ib ctx) (ha : HasTys ib args ats) :
    PrimitiveTyping D (Family M ib) ib ctx op args call (evalRegions opSem call env op.regions) := by
  intro right rt calls agree
  obtain ⟨pairs, hs, ht, hp⟩ := related_callbacks hregions he typed M rt agree
  have hp' : List.Forall₂ (RegionRelated ctx.M ib) pairs sigs := by rw [family.1]; exact hp
  have result := typingRule_sound hrule ctx call right args pairs family.2 ha hp' D admit calls typed
  simpa only [hs, ht] using result.1

theorem operation_preserves {ib : Nat} {table : Signatures} {Γ : TEnv} {op : Op}
    {ats rts : List VTy} {sigs : List RegionSig} {d : Nat} {ctx : Ctx}
    {call : CallSem} {env : Env} {args : List RVal}
    (hrule : typingRule ib table d op ats rts = some sigs)
    (hregions : regionsCheck ib table Γ op.regions sigs = true)
    (he : EnvTyped ib Γ env) (typed : CallTyped ib table call)
    (hib : ctx.indexBits = ib) (ha : HasTys ib args ats) :
    ∀ values, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values →
      HasTys ib values rts := by
  have hb := regionsCheck_preserves ib table Γ op.regions sigs call env hregions he typed
  have result := typingRule_sound hrule ctx call call args
    ((evalRegions opSem call env op.regions).map fun r => (r, r)) hib ha
    (preserving_pairs hb ctx.M) (fun _ _ => True) (fun _ _ _ _ _ => trivial)
    (fun _ _ _ => rfl) typed
  simpa only [List.map_map, Function.comp_def, List.map_id_fun', id_eq] using result.2

mutual
  theorem opsCheck_ready (ib : Nat) (table : Signatures) (D : CallDomain) (M : FloatModel)
      (admit : CallAdmits ib table D) (call : CallSem) (typed : CallTyped ib table call) (d : Nat) :
      ∀ Γ ops yields scope path index env ctx,
        opsCheck ib table d Γ ops yields = true → EnvTyped ib Γ env →
        Invariant ib scope env → Family M ib ctx →
        ReadyOps D (Family M ib) ib ctx call scope path index env ops
    | _, [], _, _, _, _, _, _, h, _, _, _ => by cases h
    | Γ, (.mk n results operands attrs props regions successors) :: rest, yields,
        scope, path, index, env, ctx, h, he, valid, family => by
      rw [ReadyOps.eq_2]
      split
      · trivial
      · next ht =>
        simp only [opsCheck, ht, Bool.false_eq_true, ↓reduceIte] at h
        split at h
        · cases h
        · next ats hats =>
          split at h
          · cases h
          · next rts hrts =>
            split at h
            · cases h
            · next sigs hrule =>
              simp only [Bool.and_eq_true] at h
              refine ⟨regionsCheck_ready ib table D M admit call typed Γ regions sigs scope
                (path ++ [index]) 0 env h.1 he valid, ?_⟩
              intro args fetched
              obtain ⟨actual, got, ha⟩ := he.getAll operands ats hats
              have eq : args = actual := Outcome.ok.inj (fetched.symm.trans got)
              subst args
              refine ⟨fun _ => primitive_checked hrule h.1 he typed admit family ha, ?_⟩
              intro values bound executed boundEq
              have hv := operation_preserves hrule h.1 he typed family.2 ha values executed
              have facts := result_facts hrts hv executed
              refine ⟨facts, ?_⟩
              have hlen : results.length = values.length := by
                rw [← optionMapM_length _ _ _ hrts]; exact (HasTys.length hv).symm
              have bound' : bound = (results.map (·.id)).zip values ++ env := by
                have hbind := bindAll_ok (env := env) hlen
                exact (Outcome.ok.inj (boundEq.symm.trans hbind))
              subst bound
              exact opsCheck_ready ib table D M admit call typed d _ rest yields
                _ path (index + 1) _ ctx h.2 (he.extend _ values rts hv)
                (results_bind valid facts boundEq) family

  theorem regionsCheck_ready (ib : Nat) (table : Signatures) (D : CallDomain) (M : FloatModel)
      (admit : CallAdmits ib table D) (call : CallSem) (typed : CallTyped ib table call) :
      ∀ Γ regions sigs scope path index env,
        regionsCheck ib table Γ regions sigs = true → EnvTyped ib Γ env →
        Invariant ib scope env → ReadyRegions D (Family M ib) ib call scope path index env regions
    | _, [], [], _, _, _, _, _, _, _ => by simp only [ReadyRegions]
    | Γ, .mk blocks :: rs, sig :: sigs, scope, path, index, env, h, he, valid => by
      simp only [regionsCheck, Bool.and_eq_true] at h
      simp only [ReadyRegions]
      exact ⟨regionCheck_ready ib table D M admit call typed Γ (.mk blocks) sig scope
        (path ++ [index]) 0 env h.1 he valid,
        regionsCheck_ready ib table D M admit call typed Γ rs sigs scope path (index + 1) env h.2 he valid⟩
    | _, [], _ :: _, _, _, _, _, h, _, _ => by cases h
    | _, _ :: _, [], _, _, _, _, h, _, _ => by cases h

  theorem regionCheck_ready (ib : Nat) (table : Signatures) (D : CallDomain) (M : FloatModel)
      (admit : CallAdmits ib table D) (call : CallSem) (typed : CallTyped ib table call) :
      ∀ Γ region sig scope path index env,
        regionCheck ib table Γ region sig = true → EnvTyped ib Γ env →
        Invariant ib scope env → ReadyBlocks D (Family M ib) ib call scope path index env region.blocks
    | Γ, .mk [.mk args ops], sig, scope, path, index, env, h, he, valid => by
      simp only [regionCheck, Bool.and_eq_true, beq_iff_eq] at h
      simp only [Region.blocks, ReadyBlocks]
      intro ctx family values ha bound boundEq
      have hv := (arguments_types args sig.args h.1 values).mp ha
      have hlen : args.length = values.length := by
        rw [← optionMapM_length _ _ _ h.1]; exact (HasTys.length hv).symm
      have bound' : bound = (args.map (·.id)).zip values ++ env :=
        Outcome.ok.inj (boundEq.symm.trans (bindAll_ok (env := env) hlen))
      subst bound
      exact opsCheck_ready ib table D M admit call typed sig.depth _ ops sig.yields
        _ (path ++ [index]) 0 _ ctx h.2 (he.extend _ values sig.args hv)
        (arguments_bind valid ha boundEq) family
    | _, .mk [], _, _, _, _, _, h, _, _ => by cases h
    | _, .mk ((.mk _ _) :: _ :: _), _, _, _, _, _, h, _, _ => by cases h
end

/-- The previously assumed SourceTyping is now a consequence of the checker and explicit
external result contracts. No numerical assumption or policy approval occurs here. -/
theorem source_typing {ib : Nat} {source : Module} (checked : programCheck ib source = true)
    (M : FloatModel) (external : ∀ ctx, Family M ib ctx → ExternalResults ib source ctx) :
    SourceTyping source (Family M ib) ib := by
  intro ctx family fuel name f found
  have hf := function_checked checked found
  unfold functionCheck at hf
  cases hs : (signatures ib source).lookup f.name with
  | none => simp [hs] at hf
  | some sig =>
    by_cases declaration : f.isDecl = true
    · have emptyBody : f.body.blocks = [] := by
        have view : f.isDecl = f.body.blocks.isEmpty := by
          have member := List.mem_of_find?_eq_some found
          obtain ⟨op, _, hop⟩ := List.mem_filterMap.1 member
          unfold Op.asFunc? at hop
          split at hop <;> try cases hop
          split at hop <;> simp_all
          all_goals cases hop; exact List.isEmpty_iff.1 declaration
        rw [view] at declaration
        exact List.isEmpty_iff.1 declaration
      rw [emptyBody]
      simp only [ReadyBlocks]
    · have body : regionCheck ib (signatures ib source) [] f.body ⟨sig.1, sig.2, 0⟩ = true := by
        simpa [hs, declaration] using hf.2
      exact regionCheck_ready ib (signatures ib source) (Inputs source ib) M (calls_admitted checked)
        _ (calls_preserve checked ctx family.2 (external ctx family) fuel) [] f.body _ [] [] 0 [] body
        (fun _ _ impossible => by cases impossible) (empty ib)

#print axioms opsCheck_ready
#print axioms primitive_checked
#print axioms source_typing
end MerlinVerify.Rules.ScalarSquareTyping

/- Finite loop execution, bound and initialization proofs for the additive candidate. -/
import MerlinVerify.Rules.FrontendMaskProgram

namespace MerlinVerify.Rules.FrontendMaskExecution
open MerlinVerify FrontendMaskRuntime FrontendMaskBounds FrontendMaskProgram

/-- A mask value at the cut consists of initialized i1 elements. -/
def MaskMatches (mask : Tensor) (bits : Nat → Bool) (n : Nat) : Prop :=
  mask.shape = [n] ∧ ∀ i, i < n → mask.get [i] = Sem.Ops.Arith.boolVal (bits i)

/-- The mathematical gather invariant is connected to execution of the actual MLIR evaluator,
including scf.if, tensor.extract/insert, arith.addi and both scf.yield terminators. -/
theorem gather_loop (limit : Nat) (ctx : Ctx) (call : CallSem)
    (mask values : Tensor) (bits : Nat → Bool) (total cap : Nat)
    (hm : MaskMatches mask bits total) (hv : values.shape = [total])
    (ht : total < 2^63) (hc : cap < 2^63) :
    ∀ n start buf pos,
      start + n ≤ total → buf.shape = [cap] → pos + countFrom bits n start ≤ cap →
      loopN 64 (evalRegion (sem limit) ctx call (capturedEnv mask values) (body false total))
        1 n start [.tensor buf, index pos] =
      let final := gather bits (fun i => values.get [i]) n start buf pos
      .ok [.tensor final.1, index final.2]
  | 0, _, _, _, _, _, _ => rfl
  | n + 1, start, buf, pos, hn, hs, hp => by
    have hi : start < total := by omega
    have hw : bits start = true → pos < cap := by
      intro hb; simp [countFrom, hb] at hp; omega
    have hbody := gather_body limit ctx call mask values buf total cap start pos
      (bits start) hm.1 hv hs ht hc hi (hm.2 start hi) hw
    cases hb : bits start
    · have hrest : pos + countFrom bits n (start+1) ≤ cap := by simpa [countFrom, hb] using hp
      simp only [loopN, hbody, hb, Bool.false_eq_true, ↓reduceIte, Outcome.bind_ok,
        List.length_cons, List.length_nil, gather]
      exact gather_loop limit ctx call mask values bits total cap hm hv ht hc n (start+1)
        buf pos (by omega) hs hrest
    · have hrest : pos+1 + countFrom bits n (start+1) ≤ cap := by
        simp [countFrom, hb] at hp; omega
      simp only [loopN, hbody, hb, ↓reduceIte, Outcome.bind_ok,
        List.length_cons, List.length_nil, gather]
      exact gather_loop limit ctx call mask values bits total cap hm hv ht hc n (start+1)
        ((buf.set [pos] (values.get [start])).canonize) (pos+1) (by omega) hs hrest

theorem scatter_loop (limit : Nat) (ctx : Ctx) (call : CallSem)
    (mask compact : Tensor) (bits : Nat → Bool) (total cap : Nat)
    (hm : MaskMatches mask bits total) (hv : compact.shape = [cap])
    (ht : total < 2^63) (hc : cap < 2^63) :
    ∀ n start seed pos,
      start + n ≤ total → seed.shape = [total] → pos + countFrom bits n start ≤ cap →
      loopN 64 (evalRegion (sem limit) ctx call (capturedEnv mask compact) (body true total))
        1 n start [.tensor seed, index pos] =
      let final := scatter bits compact n start seed pos
      .ok [.tensor final.1, index final.2]
  | 0, _, _, _, _, _, _ => rfl
  | n + 1, start, seed, pos, hn, hs, hp => by
    have hi : start < total := by omega
    have hw : bits start = true → pos < cap := by
      intro hb; simp [countFrom, hb] at hp; omega
    have hbody := scatter_body limit ctx call mask compact seed total cap start pos
      (bits start) hm.1 hv hs ht hc hi (hm.2 start hi) hw
    cases hb : bits start
    · have hrest : pos + countFrom bits n (start+1) ≤ cap := by simpa [countFrom, hb] using hp
      simp only [loopN, hbody, hb, Bool.false_eq_true, ↓reduceIte, Outcome.bind_ok,
        List.length_cons, List.length_nil, scatter]
      exact scatter_loop limit ctx call mask compact bits total cap hm hv ht hc n (start+1)
        seed pos (by omega) hs hrest
    · have hrest : pos+1 + countFrom bits n (start+1) ≤ cap := by
        simp [countFrom, hb] at hp; omega
      simp only [loopN, hbody, hb, ↓reduceIte, Outcome.bind_ok,
        List.length_cons, List.length_nil, scatter]
      exact scatter_loop limit ctx call mask compact bits total cap hm hv ht hc n (start+1)
        ((seed.set [start] (compact.get [pos])).canonize) (pos+1) (by omega) hs hrest

/-- Runtime loop controls are initialized; the finite budget and final signed advancement are
checked before executing the body. The theorem allows an empty range. -/
theorem for_carried (limit : Nat) (ctx : Ctx) (n : Nat) (carried : List RVal)
    (region : Ctx → RegionSem) (hib : ctx.indexBits = 64) (hn : n < 2^63) (hl : n ≤ limit) :
    forSem limit ctx ([index 0, index n, index 1] ++ carried) [region] =
      loopN 64 (region ctx) 1 n 0 carried := by
  simp [forSem, hib, index, indexNat_encode (by decide : 0 < 64) hn,
    indexNat_encode (by decide : 0 < 64) (by decide : 0 < 2^(64-1)),
    indexNat_encode (by decide : 0 < 64) (by decide : 1 < 2^(64-1)), hn, hl]

def emptyOp : Op := .mk "tensor.empty"
  [{id := 0, ty := .tensor [none] (.int 64) none}] [] [] [] [] []

theorem empty_alloc (limit : Nat) (ctx : Ctx) (call : CallSem) (cap : Nat)
    (hib : ctx.indexBits = 64) (hc : cap < 2^63) :
    sem limit ctx call emptyOp [index cap] [] =
      .ok [.tensor (Tensor.full [cap] (.int 64) .poison)] := by
  have hd : Sem.Ops.dialectOf "tensor.empty" = "tensor" := by cbv
  simp [sem, emptyOp, hd, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Sem.Ops.Tensor.resultTensorTy,
    Ty.staticTensor?, empty, resolveShape, hib, indexNat_encode (by decide : 0 < 64) hc]

/-- Initialized integer cut inputs suffice for both loops. The compact buffer starts poison;
its exact mask count is essential, and the returned compact and scatter tensors are initialized.
This theorem exposes that count premise, rather than assuming a Python-computed safety flag. -/
theorem selection_execution (limit : Nat) (ctx : Ctx) (call : CallSem)
    (mask values seed : Tensor) (bits : Nat → Bool) (n : Nat)
    (hm : MaskMatches mask bits n) (hv : values.shape = [n]) (hs : seed.shape = [n])
    (hdv : RVal.Defined (.tensor values)) (hds : RVal.Defined (.tensor seed))
    (hib : ctx.indexBits = 64) (hn : n < 2^63) (hl : n ≤ limit) :
    let cap := countFrom bits n 0
    let buf := Tensor.full [cap] (.int 64) .poison
    let compact := (gather bits (fun i => values.get [i]) n 0 buf 0).1
    let final := (scatter bits compact n 0 seed 0).1
    sem limit ctx call emptyOp [index cap] [] = .ok [.tensor buf] ∧
    forSem limit ctx [index 0, index n, index 1, .tensor buf, index 0]
      [fun inner => evalRegion (sem limit) inner call (capturedEnv mask values) (body false n)] =
        .ok [.tensor compact, index cap] ∧
    forSem limit ctx [index 0, index n, index 1, .tensor seed, index 0]
      [fun inner => evalRegion (sem limit) inner call (capturedEnv mask compact) (body true n)] =
        .ok [.tensor final, index cap] ∧
    RVal.Defined (.tensor compact) ∧ RVal.Defined (.tensor final) := by
  dsimp only
  let cap := countFrom bits n 0
  have hcap : cap ≤ n := countFrom_le bits n 0
  have hc : cap < 2^63 := by omega
  let buf := Tensor.full [cap] (.int 64) .poison
  let g := gather bits (fun i => values.get [i]) n 0 buf 0
  have hgi := gather_invariant bits (fun i => values.get [i]) n 0 buf 0 cap
    rfl (by intro i hi; omega) (by omega)
    (fun i _ hi => hdv [i] (by rw [hv]; exact one_inBounds.mpr (by simpa using hi)))
  have hgc : g.2 = cap := by simpa [g, cap] using hgi.2.1
  have hgp : PrefixInitialized g.1 cap := by simpa only [← hgc] using hgi.2.2
  have hgd := prefix_full hgi.1 hgp
  have hsi := scatter_invariant bits g.1 n 0 seed 0 cap n hgi.1 hgp hs hds (by omega) (by omega)
  have hsc : (scatter bits g.1 n 0 seed 0).2 = cap := by simpa [cap] using hsi.2.1
  refine ⟨empty_alloc limit ctx call cap hib hc, ?_, ?_, hgd, hsi.2.2⟩
  · rw [show [index 0, index n, index 1, RVal.tensor buf, index 0] =
      [index 0, index n, index 1] ++ [.tensor buf, index 0] from rfl,
      for_carried limit ctx n _ _ hib hn hl]
    simpa only [g, hgc] using gather_loop limit ctx call mask values bits n cap hm hv hn hc
      n 0 buf 0 (by omega) rfl (by omega)
  · rw [show [index 0, index n, index 1, RVal.tensor seed, index 0] =
      [index 0, index n, index 1] ++ [.tensor seed, index 0] from rfl,
      for_carried limit ctx n _ _ hib hn hl]
    simpa only [hsc] using scatter_loop limit ctx call mask g.1 bits n cap hm hgi.1 hn hc
      n 0 seed 0 (by omega) hs (by omega)

/-- Actual-region acceptance transports the complete finite gather execution. No region model
or Python flag is used in place of executing its decoded operations. -/
theorem accepted_gather_loop {actual : Region} {maskId valuesId oneId : Nat}
    (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env)
    (mask values : Tensor) (bits : Nat → Bool) (n cap : Nat)
    (hcheck : checkBody false n maskId valuesId oneId actual = true)
    (hu : FrontendBoolFptosi.UneCorrect ctx.M)
    (he : Check.Corr [(0, maskId), (1, valuesId), (2, oneId)] (capturedEnv mask values) env)
    (hm : MaskMatches mask bits n) (hv : values.shape = [n])
    (hn : n < 2^63) (hc : cap < 2^63) (buf : Tensor) (hs : buf.shape = [cap])
    (hp : countFrom bits n 0 ≤ cap) :
    loopN 64 (evalRegion (sem limit) ctx call env actual) 1 n 0 [.tensor buf, index 0] =
      let final := gather bits (fun i => values.get [i]) n 0 buf 0
      .ok [.tensor final.1, index final.2] := by
  rw [← checkBody_sound hcheck limit ctx hu call (capturedEnv mask values) env he]
  exact gather_loop limit ctx call mask values bits n cap hm hv hn hc n 0 buf 0
    (by omega) hs (by simpa using hp)

theorem accepted_scatter_loop {actual : Region} {maskId valuesId oneId : Nat}
    (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env)
    (mask compact : Tensor) (bits : Nat → Bool) (n cap : Nat)
    (hcheck : checkBody true n maskId valuesId oneId actual = true)
    (hu : FrontendBoolFptosi.UneCorrect ctx.M)
    (he : Check.Corr [(0, maskId), (1, valuesId), (2, oneId)] (capturedEnv mask compact) env)
    (hm : MaskMatches mask bits n) (hv : compact.shape = [cap])
    (hn : n < 2^63) (hc : cap < 2^63) (seed : Tensor) (hs : seed.shape = [n])
    (hp : countFrom bits n 0 ≤ cap) :
    loopN 64 (evalRegion (sem limit) ctx call env actual) 1 n 0 [.tensor seed, index 0] =
      let final := scatter bits compact n 0 seed 0
      .ok [.tensor final.1, index final.2] := by
  rw [← checkBody_sound hcheck limit ctx hu call (capturedEnv mask compact) env he]
  exact scatter_loop limit ctx call mask compact bits n cap hm hv hn hc n 0 seed 0
    (by omega) hs (by simpa using hp)

#print axioms accepted_gather_loop
#print axioms accepted_scatter_loop
#print axioms selection_execution
#print axioms gather_loop
#print axioms scatter_loop
end MerlinVerify.Rules.FrontendMaskExecution

/- Actual-IR mask-body checking and its connection to the additive evaluator. -/
import MerlinVerify.Rules.FrontendMaskBounds
import MerlinVerify.Rules.FrontendBoolSimulationWith

set_option maxRecDepth 10000
set_option maxHeartbeats 1500000

namespace MerlinVerify.Rules.FrontendMaskProgram
open MerlinVerify FrontendMaskRuntime FrontendMaskBounds
open Sem.Ops

abbrev index (n : Nat) : RVal := .scalar (.some (.bits 64 (BitVec.ofNat 64 n)))
def tensorTy (scatter : Bool) (n : Nat) : Ty :=
  .tensor [if scatter then some n else none] (.int 64) none

def extractOp : Op := .mk "tensor.extract" [{id := 9, ty := .int 64}] [] [] [] [] []
def insertOp : Op := .mk "tensor.insert" [{id := 10, ty := .tensor [none] (.int 64) none}] [] [] [] [] []
def addOp : Op := .mk "arith.addi" [{id := 11, ty := .index}] []
  [("overflowFlags", .enum "arith" "overflow" "none")] [] [] []

/-- The second loop scatters compact values at selected source positions; false positions
retain the seed tensor. It is not a second gather. -/
def body (scatter : Bool) (n : Nat) : Region :=
  .mk [.mk [{id := 3, ty := .index}, {id := 4, ty := tensorTy scatter n}, {id := 5, ty := .index}] [
    .mk "tensor.extract" [{id := 6, ty := .int 1}] [0, 3] [] [] [] [],
    .mk "scf.if" [{id := 7, ty := tensorTy scatter n}, {id := 8, ty := .index}] [6] [] [] [
      .mk [.mk [] [
        .mk "tensor.extract" [{id := 9, ty := .int 64}] [1, if scatter then 5 else 3] [] [] [] [],
        .mk "tensor.insert" [{id := 10, ty := tensorTy scatter n}]
          [9, 4, if scatter then 3 else 5] [] [] [] [],
        .mk "arith.addi" [{id := 11, ty := .index}] [5, 2]
          [("overflowFlags", .enum "arith" "overflow" "none")] [] [] [],
        .mk "scf.yield" [] [10, 11] [] [] [] []]],
      .mk [.mk [] [.mk "scf.yield" [] [4, 5] [] [] [] []]]] [],
    .mk "scf.yield" [] [7, 8] [] [] [] []]]

/-- Recomputes types, attributes, SSA uses, branch yields and freshness from decoded IR. -/
def checkBody (scatter : Bool) (n mask values one : Nat) (actual : Region) : Bool :=
  FrontendBoolSimulation.region [] [(0, mask), (1, values), (2, one)] (body scatter n) actual

theorem index_toInt {n : Nat} (hn : n < 2^63) :
    (BitVec.ofNat 64 n).toInt = (n : Int) := by
  change (BitVec.ofInt 64 (n : Int)).toInt = (n : Int)
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

theorem extract_sem (limit : Nat) (ctx : Ctx) (call : CallSem) (t : MerlinVerify.Tensor) {cap i : Nat}
    (hs : t.shape = [cap]) (hi : i < cap) (hcap : cap < 2^63) :
    sem limit ctx call extractOp [.tensor t, index i] [] = .ok [.scalar (t.get [i])] := by
  have hto := index_toInt (n := i) (by omega)
  have hd : dialectOf "tensor.extract" = "tensor" := by cbv
  simp [sem, extractOp, opSem, dispatch, hd, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Tensor.sem,
    Tensor.indexVal?, hs, index, hto, hi, Int.natCast_nonneg, List.mapM_cons, List.mapM_nil, Outcome.bind]

theorem insert_sem (limit : Nat) (ctx : Ctx) (call : CallSem) (t : MerlinVerify.Tensor)
    (value : Val) {cap i : Nat} (hs : t.shape = [cap]) (hi : i < cap) (hcap : cap < 2^63) :
    sem limit ctx call insertOp [.scalar value, .tensor t, index i] [] =
      .ok [.tensor (t.set [i] value).canonize] := by
  have hto := index_toInt (n := i) (by omega)
  have hd : dialectOf "tensor.insert" = "tensor" := by cbv
  simp [sem, insertOp, opSem, dispatch, hd, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Tensor.sem,
    Tensor.indexVal?, hs, index, hto, hi, Int.natCast_nonneg, List.mapM_cons, List.mapM_nil, Outcome.bind]

theorem add_sem (limit : Nat) (ctx : Ctx) (call : CallSem) (n : Nat) :
    sem limit ctx call addOp [index n, index 1] [] = .ok [index (n+1)] := by
  have hd : dialectOf "arith.addi" = "arith" := by cbv
  have hf : flagList "none" = ["none"] := by cbv
  simp [sem, addOp, opSem, dispatch, hd, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
    Arith.sem, Arith.intBin, Arith.arithWithFlags, Arith.bv, overflowFlags,
    Arith.resultElt, index, Arith.binOp, Arith.pureN, liftEager, List.mapM_cons,
    List.mapM_nil, Outcome.bind, attrEnum?, hf, BitVec.ofNat_add]

def capturedEnv (mask values : MerlinVerify.Tensor) : Env :=
  [(0, .tensor mask), (1, .tensor values), (2, index 1)]

/-- Successful evaluation of the actual gather body is derived from initialized mask bits and
mathematical bounds, not asserted by a certificate. -/
theorem gather_body (limit : Nat) (ctx : Ctx) (call : CallSem)
    (mask values buf : MerlinVerify.Tensor) (n cap iv pos : Nat) (selected : Bool)
    (hmask : mask.shape = [n]) (hvalues : values.shape = [n]) (hbuf : buf.shape = [cap])
    (hn : n < 2^63) (hc : cap < 2^63) (hi : iv < n)
    (hbit : mask.get [iv] = Arith.boolVal selected) (hp : selected = true → pos < cap) :
    evalRegion (sem limit) ctx call (capturedEnv mask values) (body false n)
      [index iv, .tensor buf, index pos] =
      .ok (if selected then [.tensor (buf.set [pos] (values.get [iv])).canonize, index (pos+1)]
        else [.tensor buf, index pos]) := by
  have hdi : dialectOf "tensor.insert" = "tensor" := by cbv
  have hde : dialectOf "tensor.extract" = "tensor" := by cbv
  have hda : dialectOf "arith.addi" = "arith" := by cbv
  have hds : dialectOf "scf.if" = "scf" := by cbv
  have hf : flagList "none" = ["none"] := by cbv
  have hiv := index_toInt (n := iv) (by omega)
  cases selected
  all_goals try have hpos : pos < cap := hp rfl
  all_goals try have hpo : (BitVec.ofNat 64 pos).toInt = (pos : Int) := index_toInt (by omega)
  all_goals
    simp (maxSteps := 100000) [*, evalRegion, evalBlocks, evalBlock, evalOps, body, tensorTy, capturedEnv,
      Op.isTerminator, terminatorNames, Env.bindAll, Env.getAll, Env.get?, List.lookup,
      Outcome.mapM, Outcome.ofOption, sem, ifSem, evalRegions, opSem, dispatch,
      unmodelledAttr?, interpretedAttrs, hdi, hde, hda, hds, hf,
      Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
      Tensor.sem, Tensor.indexVal?, Arith.sem, Arith.boolVal, Arith.binOp, Arith.pureN,
      Arith.resultElt, liftEager, Arith.intBin, overflowFlags, attrEnum?, Arith.arithWithFlags,
      Arith.bv, List.mapM_cons, List.mapM_nil, Outcome.bind, index, hmask, hvalues, hbuf,
      hiv, hbit, hi, BitVec.ofNat_add, show (0 : Int) ≤ (iv : Int) by omega]

/-- The checked scatter body reads the next compact value and writes the original source index. -/
theorem scatter_body (limit : Nat) (ctx : Ctx) (call : CallSem)
    (mask compact seed : MerlinVerify.Tensor) (n cap iv pos : Nat) (selected : Bool)
    (hmask : mask.shape = [n]) (hcompact : compact.shape = [cap]) (hseed : seed.shape = [n])
    (hn : n < 2^63) (hc : cap < 2^63) (hi : iv < n)
    (hbit : mask.get [iv] = Arith.boolVal selected) (hp : selected = true → pos < cap) :
    evalRegion (sem limit) ctx call (capturedEnv mask compact) (body true n)
      [index iv, .tensor seed, index pos] =
      .ok (if selected then [.tensor (seed.set [iv] (compact.get [pos])).canonize, index (pos+1)]
        else [.tensor seed, index pos]) := by
  have hdi : dialectOf "tensor.insert" = "tensor" := by cbv
  have hde : dialectOf "tensor.extract" = "tensor" := by cbv
  have hda : dialectOf "arith.addi" = "arith" := by cbv
  have hds : dialectOf "scf.if" = "scf" := by cbv
  have hf : flagList "none" = ["none"] := by cbv
  have hiv := index_toInt (n := iv) (by omega)
  cases selected
  all_goals try have hpos : pos < cap := hp rfl
  all_goals try have hpo : (BitVec.ofNat 64 pos).toInt = (pos : Int) := index_toInt (by omega)
  all_goals
    simp (maxSteps := 100000) [*, evalRegion, evalBlocks, evalBlock, evalOps, body, tensorTy, capturedEnv,
      Op.isTerminator, terminatorNames, Env.bindAll, Env.getAll, Env.get?, List.lookup,
      Outcome.mapM, Outcome.ofOption, sem, ifSem, evalRegions, opSem, dispatch,
      unmodelledAttr?, interpretedAttrs, hdi, hde, hda, hds, hf,
      Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
      Tensor.sem, Tensor.indexVal?, Arith.sem, Arith.boolVal, Arith.binOp, Arith.pureN,
      Arith.resultElt, liftEager, Arith.intBin, overflowFlags, attrEnum?, Arith.arithWithFlags,
      Arith.bv, List.mapM_cons, List.mapM_nil, Outcome.bind, index, hmask, hcompact, hseed,
      hiv, hbit, hi, BitVec.ofNat_add, show (0 : Int) ≤ (iv : Int) by omega]


/-- Acceptance transports execution of the canonical body to the actual captured region.
The environment correspondence identifies all captured operands; the existing evaluator owns
block arguments, SSA binding, selected branch execution and terminators. -/
theorem checkBody_sound {scatter : Bool} {n mask values one : Nat} {actual : Region}
    (h : checkBody scatter n mask values one actual = true)
    (limit : Nat) (ctx : Ctx) (hu : FrontendBoolFptosi.UneCorrect ctx.M) (call : CallSem)
    (es et : Env) (hc : Check.Corr [(0, mask), (1, values), (2, one)] es et) :
    evalRegion (sem limit) ctx call es (body scatter n) =
      evalRegion (sem limit) ctx call et actual := by
  have he := FrontendBoolSimulationWith.region_sound (sem limit)
    (FrontendBoolSimulationWith.candidateLaws limit) [] call _ _ _ ctx es et h hu hc
  simpa only [FrontendBoolSimulationWith.policy_nil] using he

/-- Hints are extracted from the actual region and then checked by the proved AST checker. -/
def bodyInputs? (actual : Region) : Option (Nat × Nat × Nat) := do
  let .mk [.mk _ (first :: branch :: _)] := actual | none
  let [mask, _] := first.operands | none
  let [yes, _] := branch.regions | none
  let .mk [.mk _ (read :: _ :: increment :: _)] := yes | none
  let [values, _] := read.operands | none
  let [_, one] := increment.operands | none
  pure (mask, values, one)

def checkedBody (scatter : Bool) (n : Nat) (actual : Region) : Bool :=
  match bodyInputs? actual with
  | some (mask, values, one) => checkBody scatter n mask values one actual
  | none => false

theorem checkedBody_sound {scatter : Bool} {n : Nat} {actual : Region}
    (h : checkedBody scatter n actual = true) :
    ∃ mask values one, bodyInputs? actual = some (mask, values, one) ∧
      ∀ (limit : Nat) (ctx : Ctx), FrontendBoolFptosi.UneCorrect ctx.M →
      ∀ (call : CallSem) (es et : Env),
      Check.Corr [(0, mask), (1, values), (2, one)] es et →
      evalRegion (sem limit) ctx call es (body scatter n) =
        evalRegion (sem limit) ctx call et actual := by
  unfold checkedBody at h
  split at h
  · next mask values one he =>
    exact ⟨mask, values, one, he, fun limit ctx hu call es et hc =>
      checkBody_sound h limit ctx hu call es et hc⟩
  · simp at h

#print axioms gather_body
#print axioms scatter_body
#print axioms checkBody_sound

end MerlinVerify.Rules.FrontendMaskProgram

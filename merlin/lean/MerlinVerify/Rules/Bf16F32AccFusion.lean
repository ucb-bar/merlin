/-
Fusion of a `contract`-licensed multiply-add is an instance of the parametric float model.

MLIR permits the lowering to evaluate `acc + ext(a) * ext(b)`, with `contract` on both operations,
as one fused multiply-add. The frozen semantics instead gives each flagged operation its own
uninterpreted function (`Bf16F32AccContract.contract_distinct`). `fused_step_on_image` shows that
for any base model `M`, any fused operation `fma`, and any operand-format conversion, there is
a model that agrees with `M` on every other operation key and whose contract step is exactly
`fma(ext a, ext b, acc)`. Recovering the extended operands needs no conversion injectivity.
The historical pinned `fused_step` retains its stronger, unnecessary injectivity premise.

Premises a claim citing this must discharge: (1) the program's only `contract`-flagged f32
`arith.mulf`/`arith.addf` operations are contract steps in one operand format `k`; (2) the target
fuses either every such step or none. These are program/target correspondence premises, not
conclusions of the local step theorem. No bit-shift implementation or non-NaN domain is assumed.
-/
import MerlinVerify.Rules.Bf16F32AccContract

namespace MerlinVerify.Rules.Bf16F32AccFusion

open MerlinVerify Bf16F32AccContract

/-- Two 16-bit operands in one binary32 bit pattern; `unpack` recovers them exactly. -/
def pack (a b : BitVec 16) : BitVec 32 := BitVec.ofNat 32 (a.toNat * 65536 + b.toNat)

def unpack (code : BitVec 32) : BitVec 16 × BitVec 16 :=
  (BitVec.ofNat 16 (code.toNat / 65536), BitVec.ofNat 16 (code.toNat % 65536))

theorem unpack_pack (a b : BitVec 16) : unpack (pack a b) = (a, b) := by
  have ha := a.isLt
  have hb := b.isLt
  have hlt : a.toNat * 65536 + b.toNat < 2 ^ 32 := by omega
  simp only [unpack, pack, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hlt, Prod.mk.injEq]
  constructor
  · apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_ofNat]
    omega
  · apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_ofNat]
    omega

/-- The operand whose extension is `x` (any choice off the image; never consulted there). -/
noncomputable def source (M : FloatModel) (k : FKind) (x : FBits .f32) : FBits k :=
  Function.invFun (ext M k) x

/-- `M`, except that the contract multiply records its operands and the contract add fuses. -/
noncomputable def fused (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32) : FloatModel :=
  { M with op := fun key kind args =>
      match kind, args with
      | .f32, [x, y] =>
        if key = mulKey .contract then
          pack (BitVec.cast hk (source M k x)) (BitVec.cast hk (source M k y))
        else if key = addKey .contract then
          fma (ext M k (BitVec.cast hk.symm (unpack y).1)) (ext M k (BitVec.cast hk.symm (unpack y).2)) x
        else M.op key .f32 [x, y]
      | kind, args => M.op key kind args }

theorem add_ne_mul : addKey .contract ≠ mulKey .contract := by
  rw [addKey_literal, mulKey_literal]
  decide

/-- Every operation other than the two contract keys is interpreted exactly as by `M`. -/
theorem fused_agrees (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32) (key : FloatOpKey) (kind : FKind)
    (args : List (FBits kind)) (hm : key ≠ mulKey .contract) (ha : key ≠ addKey .contract) :
    (fused M k hk fma).op key kind args = M.op key kind args := by
  unfold fused
  dsimp only
  split <;> simp [hm, ha]

theorem fused_conversions (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32) :
    (fused M k hk fma).cmp = M.cmp ∧ (fused M k hk fma).convert = M.convert ∧
      (fused M k hk fma).toInt = M.toInt ∧ (fused M k hk fma).ofInt = M.ofInt :=
  ⟨rfl, rfl, rfl, rfl⟩

private theorem fused_ext (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32) (k' : FKind) (x : FBits k') :
    ext (fused M k hk fma) k' x = ext M k' x := rfl

private theorem fused_binary (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32) (key : FloatOpKey) (x y : FBits .f32) :
    (fused M k hk fma).op key .f32 [x, y] =
      if key = mulKey .contract then
        pack (BitVec.cast hk (source M k x)) (BitVec.cast hk (source M k y))
      else if key = addKey .contract then
        fma (ext M k (BitVec.cast hk.symm (unpack y).1)) (ext M k (BitVec.cast hk.symm (unpack y).2)) x
      else M.op key .f32 [x, y] := rfl

/-- The contract step under the fused model is one fused multiply-add of the extended operands. -/
theorem fused_step (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32)
    (injective : Function.Injective (ext M k)) (a b : FBits k) (acc : FBits .f32) :
    step (fused M k hk fma) ⟨.contract, .contract⟩ a b acc = fma (ext M k a) (ext M k b) acc := by
  have recover (x : FBits k) : source M k (ext M k x) = x := Function.leftInverse_invFun injective x
  simp only [step, fused_ext]
  rw [fused_binary, fused_binary]
  simp only [add_ne_mul, ↓reduceIte, recover, unpack_pack, BitVec.cast_cast, BitVec.cast_eq]

/-- Only the converted values are observed by FMA. An arbitrary selected preimage
recreates those values even when conversion merges NaNs or other bit patterns.
This additive theorem preserves the historical pinned statement above. -/
theorem fused_step_on_image (M : FloatModel) (k : FKind) (hk : k.width = 16)
    (fma : FBits .f32 → FBits .f32 → FBits .f32 → FBits .f32)
    (a b : FBits k) (acc : FBits .f32) :
    step (fused M k hk fma) ⟨.contract, .contract⟩ a b acc = fma (ext M k a) (ext M k b) acc := by
  have recover (x : FBits k) : ext M k (source M k (ext M k x)) = ext M k x :=
    Function.invFun_eq ⟨x, rfl⟩
  simp only [step, fused_ext]
  rw [fused_binary, fused_binary]
  simp only [add_ne_mul, ↓reduceIte, unpack_pack, BitVec.cast_cast, BitVec.cast_eq, recover]

#print axioms unpack_pack
#print axioms fused_agrees
#print axioms fused_step
#print axioms fused_step_on_image

end MerlinVerify.Rules.Bf16F32AccFusion

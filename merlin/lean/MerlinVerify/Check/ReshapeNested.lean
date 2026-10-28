/- Per-block reshape provenance lifted through the shared exact structured-body traversal.
Captured tensors obtain their runtime types from the checked body scope; no outer reshape fact
is assumed at a region boundary. Opposite-side syntax supplies only candidate headers. -/
import MerlinVerify.Check.ReshapeNormalize
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizeC1

namespace MerlinVerify.Check.ReshapeNested

open MerlinVerify MerlinVerify.Sem.Ops

mutual
  /-- Separate the current block's headers from descendants in one structural traversal. -/
  def hintParts : List Op → List Op × List Op
    | [] => ([], [])
    | (.mk name results operands attrs props regions successors) :: rest =>
      let op := Op.mk name results operands attrs props regions successors
      let (localHeaders, nestedHeaders) := hintParts rest
      (if Rules.ReshapeCompose.Names op then op :: localHeaders else localHeaders,
        hintRegions regions ++ nestedHeaders)

  def hintRegions : List Region → List Op
    | [] => []
    | .mk blocks :: rest => hintBlocks blocks ++ hintRegions rest

  def hintBlocks : List Block → List Op
    | [] => []
    | .mk _ ops :: rest =>
      let (localHeaders, nestedHeaders) := hintParts ops
      localHeaders ++ nestedHeaders ++ hintBlocks rest
end

/-- Current-block candidates retain priority over descendants, including earlier enclosing ops.
The ordering affects conservative alignment choices only; every candidate remains rechecked. -/
def hintOps (ops : List Op) : List Op :=
  let (localHeaders, nestedHeaders) := hintParts ops
  localHeaders ++ nestedHeaders

/-- Shape facts start empty at each block. The generic lifter supplies the actual typed scope,
including captures and shadowing block arguments, while uniform facts belong to other families. -/
def base (ib : Nat) (headers : List Op) : SplatNormalizer.Normalizer ib where
  run _ Γ _ ops _ := ReshapeNormalize.normalize ib Γ [] headers ops
  sound d Γ _ ops normalized yields ctx call env hc hn he _ hi hd :=
    ReshapeNormalize.normalize_sound d Γ [] headers ops normalized yields ctx call env hc hn he
      (by intro f hf; cases hf) hi hd

def normalizer : TypedNormalizer where
  run ib d Γ ops hints yields :=
    (BodyNormalize.recursive (base ib (hintOps hints))
      (max (RegionDepth.ops ops) (RegionDepth.ops hints))).run d Γ [] ops yields
  sound ib d Γ ops hints normalized yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (base ib (hintOps hints))
      (max (RegionDepth.ops ops) (RegionDepth.ops hints))).sound
      d Γ [] ops normalized yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  TypedNormalizeC1.check normalizer ib src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := TypedNormalizeC1.check_sound normalizer h

end MerlinVerify.Check.ReshapeNested

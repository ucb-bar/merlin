/- Full byte-vector certificate, in checked ABI order. No digest substitutes for tensor equality.
Archive authenticity, byte decoding, and the eventual runtime descriptor delivery stay explicit. -/
import MerlinVerify.Check.Qinner
namespace MerlinVerify.Check.QinnerRuntime
open MerlinVerify Rules.QinnerBinding
structure Blob where
  key : String
  ty : Ty
  bytes : List UInt8
  deriving DecidableEq

def width? : Ty → Option Nat
  | .int 1 => some 1
  | .int n => if [8,16,32,64].contains n then some (n/8) else none
  | .float .f16 => some 2
  | .float .f32 => some 4
  | .float .f64 => some 8
  | _ => none

def sized (b : Blob) : Bool :=
  match b.ty.staticTensor? with
  | none => false
  | some (shape,elt) => match width? elt with
    | none => false
    | some width => b.bytes.length == shapeSize shape * width

def Matches (archive : List Blob) (slot : Slot) (delivered : Blob) : Prop :=
  archive.find? (fun b => b.key == slot.key) = some delivered ∧
    delivered.ty = slot.arg.ty ∧ sized delivered = true

def rows (archive : List Blob) : List Slot → List Blob → Bool
  | [],[] => true
  | slot::slots,delivered::rest =>
    archive.find? (fun b => b.key == slot.key) == some delivered &&
    delivered.ty == slot.arg.ty && sized delivered && rows archive slots rest
  | _,_ => false

def check (archive : List Blob) (slots : List Slot) (delivered : List Blob) : Bool :=
  decide (archive.map (·.key)).Nodup && decide (slots.map (·.key)).Nodup && rows archive slots delivered

theorem rows_sound {archive : List Blob} : ∀ {slots : List Slot} {delivered : List Blob},
    rows archive slots delivered = true → List.Forall₂ (Matches archive) slots delivered
  | [],[],_ => .nil
  | slot::slots,delivered::rest,h => by
    simp only [rows,Bool.and_eq_true,beq_iff_eq] at h
    exact .cons ⟨h.1.1.1,h.1.1.2,h.1.2⟩ (rows_sound h.2)
  | [],_::_,h | _::_,[],h => by simp [rows] at h

theorem check_sound {archive : List Blob} {slots : List Slot} {delivered : List Blob}
    (h : check archive slots delivered = true) : List.Forall₂ (Matches archive) slots delivered := by
  simp only [check,Bool.and_eq_true] at h
  exact rows_sound h.2

def binding (archive : List Blob) (decode : Blob → RVal) (key : String) : RVal :=
  match archive.find? (fun b => b.key == key) with
  | some b => decode b
  | none => .scalar .poison

theorem values_equal {archive : List Blob} {slots : List Slot} {delivered : List Blob}
    (h : check archive slots delivered = true) (decode : Blob → RVal) :
    delivered.map decode = Rules.QinnerABI.values slots (binding archive decode) := by
  have matched := check_sound h
  clear h
  induction matched with
  | nil => rfl
  | @cons slot delivered slots rest first tail ih =>
    simp only [List.map_cons,Rules.QinnerABI.values,List.map_cons,binding,first.1]
    exact congrArg (List.cons (decode delivered)) ih
theorem program_equal {s t : Module} {archive delivered : List Blob}
    (transformed : Qinner.check s t = true) (changed : (Qinner.sitesOps s.ops).isEmpty = false)
    (runtime : check archive (Qinner.slots s t) delivered = true)
    (decode : Blob → RVal) (ctx : Ctx) {a : Func} (source : s.func? "forward" = some a)
    (args : List RVal) (arity : a.body.blocks.head!.args.length = args.length) :
    denote (sem (binding archive decode)) ctx s "forward" args =
      denote Sem.Ops.opSem ctx t "forward" (args ++ delivered.map decode) := by
  rw [values_equal runtime decode]
  exact Qinner.check_sound transformed changed _ ctx source args arity
#print axioms program_equal
#print axioms check_sound
#print axioms values_equal
end MerlinVerify.Check.QinnerRuntime

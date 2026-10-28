/- Concrete-decoder specialization of the complete Qinner correspondence claim. Accepted payloads
are typed, initialized and bit-exact. Archive authenticity and actual delivery remain explicit. -/
import MerlinVerify.Rules.QinnerDecoder
namespace MerlinVerify.Check.QinnerDecoded
open MerlinVerify Rules.QinnerBinding Rules.QinnerDecoder
open QinnerRuntime (Blob)

def check (order layout : String) (archive : List Blob) (slots : List Slot) (delivered : List Blob) : Bool :=
  order == "little" && layout == "row_major" && QinnerRuntime.check archive slots delivered &&
  archive.all (fun b => payloadCheck order layout b.ty b.bytes) &&
  delivered.all (fun b => payloadCheck order layout b.ty b.bytes)

/-- Standalone payload checking includes the caller's expected tensor type and shape. -/
def expectedCheck (order layout : String) (expected : Ty) (b : Blob) : Bool :=
  decide (b.ty = expected) && payloadCheck order layout b.ty b.bytes

theorem expectedCheck_sound {order layout : String} {expected : Ty} {b : Blob}
    (h : expectedCheck order layout expected b = true) (ib : Nat) :
    ∃ value vt, decode? order layout b.ty b.bytes = some value ∧ vtyOf ib expected = some vt ∧
      RVal.HasTy ib value vt ∧ value.Defined := by
  simp only [expectedCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  obtain ⟨value,vt,decoded,typed,hasTy,defined⟩ := decode_typed_initialized h.2 ib
  exact ⟨value,vt,decoded,by simpa [h.1] using typed,hasTy,defined⟩

/-- Failure is kept explicit; only a payload that passes the checker can use the success lemmas. -/
def interpret (b : Blob) : RVal :=
  (decode? "little" "row_major" b.ty b.bytes).getD (.scalar .poison)

theorem premises {order layout : String} {archive delivered : List Blob} {slots : List Slot}
    (h : check order layout archive slots delivered = true) :
    order = "little" ∧ layout = "row_major" ∧ QinnerRuntime.check archive slots delivered = true ∧
      (∀ b ∈ archive, payloadCheck "little" "row_major" b.ty b.bytes = true) ∧
      (∀ b ∈ delivered, payloadCheck "little" "row_major" b.ty b.bytes = true) := by
  simp only [check,Bool.and_eq_true,beq_iff_eq,List.all_eq_true] at h
  obtain ⟨⟨⟨⟨orderEq,layoutEq⟩,runtime⟩,left⟩,right⟩ := h
  subst order
  subst layout
  exact ⟨rfl,rfl,runtime,left,right⟩

theorem payload_typed_initialized {b : Blob}
    (h : payloadCheck "little" "row_major" b.ty b.bytes = true) (ib : Nat) :
    ∃ vt, vtyOf ib b.ty = some vt ∧ RVal.HasTy ib (interpret b) vt ∧ (interpret b).Defined := by
  obtain ⟨value,vt,decoded,typed,hasTy,defined⟩ := decode_typed_initialized h ib
  exact ⟨vt,typed,by simpa [interpret,decoded] using hasTy,by simpa [interpret,decoded] using defined⟩

theorem delivered_sound {order layout : String} {archive delivered : List Blob} {slots : List Slot}
    (h : check order layout archive slots delivered = true) (ib : Nat) :
    ∀ b ∈ delivered, ∃ vt, vtyOf ib b.ty = some vt ∧
      RVal.HasTy ib (interpret b) vt ∧ (interpret b).Defined := by
  intro b member
  exact payload_typed_initialized ((premises h).2.2.2.2 b member) ib

theorem slot_witness {archive : List Blob} {slots : List Slot} {delivered : List Blob}
    (matched : List.Forall₂ (QinnerRuntime.Matches archive) slots delivered) {slot : Slot}
    (member : slot ∈ slots) : ∃ b ∈ delivered, QinnerRuntime.Matches archive slot b := by
  induction matched with
  | nil => cases member
  | @cons a b slots rest first tail ih =>
    rcases List.mem_cons.mp member with rfl | member
    · exact ⟨b,by simp,first⟩
    · obtain ⟨value,mem,rel⟩ := ih member
      exact ⟨value,List.mem_cons_of_mem _ mem,rel⟩

theorem binding_typed_initialized {order layout : String} {archive delivered : List Blob} {slots : List Slot}
    (h : check order layout archive slots delivered = true) (ib : Nat) :
    BindingTyped ib slots (QinnerRuntime.binding archive interpret) ∧
      BindingClean slots (QinnerRuntime.binding archive interpret) := by
  have matched := QinnerRuntime.check_sound (premises h).2.2.1
  constructor
  · intro slot member
    obtain ⟨b,mem,found,ty,_⟩ := slot_witness matched member
    obtain ⟨vt,typed,hasTy,_⟩ := delivered_sound h ib b mem
    refine ⟨vt,by simpa [ty] using typed,?_⟩
    simpa [QinnerRuntime.binding,found] using hasTy
  · intro slot member
    obtain ⟨b,mem,found,_,_⟩ := slot_witness matched member
    obtain ⟨_,_,_,defined⟩ := delivered_sound h ib b mem
    simpa [QinnerRuntime.binding,found] using defined

/-- Exact whole-program equality now uses the concrete bit decoder, not a decoder premise.
This is a correspondence theorem; unchanged model computations may still be UB or stuck. -/
theorem program_equal {order layout : String} {s t : Module} {archive delivered : List Blob}
    (transformed : Qinner.check s t = true) (changed : (Qinner.sitesOps s.ops).isEmpty = false)
    (runtime : check order layout archive (Qinner.slots s t) delivered = true)
    (ctx : Ctx) {a : Func} (source : s.func? "forward" = some a)
    (args : List RVal) (arity : a.body.blocks.head!.args.length = args.length) :
    denote (sem (QinnerRuntime.binding archive interpret)) ctx s "forward" args =
      denote Sem.Ops.opSem ctx t "forward" (args ++ delivered.map interpret) :=
  QinnerRuntime.program_equal transformed changed (premises runtime).2.2.1 interpret ctx source args arity

#print axioms expectedCheck_sound
#print axioms delivered_sound
#print axioms binding_typed_initialized
#print axioms program_equal
end MerlinVerify.Check.QinnerDecoded

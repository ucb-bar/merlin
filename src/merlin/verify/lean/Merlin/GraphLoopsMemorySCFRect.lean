/- Rank1/rank2 complete initializer coverage and captured IV discipline. -/
import Merlin.GraphLoopsMemorySCFCoverage

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

def Initializer.rows (program : Initializer) : Nat := program.shape.headD 0
def Initializer.columns (program : Initializer) : Nat := program.shape[1]?.getD 0
def Initializer.outerIV (program : Initializer) : Nat := program.ivs.headD 0
def Initializer.innerIV (program : Initializer) : Nat := program.ivs[1]?.getD 0

def Initializer.One (program : Initializer) : Prop :=
  program.shape = [program.rows] ∧ program.ivs = [program.outerIV] ∧ 0 < program.rows ∧ program.outerPrefix = []

def Initializer.Two (program : Initializer) : Prop :=
  program.shape = [program.rows, program.columns] ∧ program.ivs = [program.outerIV, program.innerIV] ∧
  0 < program.rows ∧ 0 < program.columns ∧ program.outerIV ≠ program.innerIV

def Initializer.WellFormed (program : Initializer) : Prop :=
  (program.One ∨ program.Two) ∧ program.leaf.indices = program.ivs ∧
  (∀ name ∈ program.ivs, name ∉ definitions program.leaf.ops) ∧
  ∀ name ∈ program.ivs, name ∉ definitions program.outerPrefix

instance (program : Initializer) : Decidable program.One := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))
instance (program : Initializer) : Decidable program.Two := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))
instance (program : Initializer) : Decidable program.WellFormed := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

theorem rank_one_covers {Value : Type} {meaning : PureMeaning Value} {program : Initializer}
    {extent iv tile : Nat} {captures : Env Value} {writes : Log Value}
    (shape : program.shape = [extent]) (ivs : program.ivs = [iv])
    (indices : program.leaf.indices = [iv])
    (noShadow : ∀ name ∈ program.leaf.indices, name ∉ definitions program.leaf.ops)
    (run : Run meaning program tile false captures writes) :
    ∀ index, index < extent → ∃ value, (index, value) ∈ writes := by
  obtain ⟨trace, run⟩ := run
  have source : GraphLoopsSegment.Execution (LeafBody meaning program.leaf [extent] iv captures) extent 1 0 [] trace writes := by
    simpa only [Run, Body, shape, ivs, List.headD_cons, Bool.false_eq_true, ↓reduceIte] using run
  exact SCF_covers id (fun _ _ _ event => LeafBody.monotone event)
    (fun _ _ _ event => LeafBody.emits_one event indices noShadow) source

theorem rank_two_covers {Value : Type} {meaning : PureMeaning Value} {program : Initializer}
    {rows columns outerIV innerIV tile : Nat} {captures : Env Value} {writes : Log Value}
    (shape : program.shape = [rows, columns]) (ivs : program.ivs = [outerIV, innerIV])
    (positiveColumns : 0 < columns) (different : outerIV ≠ innerIV)
    (indices : program.leaf.indices = [outerIV, innerIV])
    (outerFresh : outerIV ∉ definitions program.outerPrefix)
    (leafFresh : ∀ name ∈ program.leaf.indices, name ∉ definitions program.leaf.ops)
    (run : Run meaning program tile false captures writes) :
    ∀ index, index < rows * columns → ∃ value, (index, value) ∈ writes := by
  obtain ⟨trace, run⟩ := run
  have source : GraphLoopsSegment.Execution
      (NestedBody meaning program.outerPrefix program.leaf rows columns outerIV innerIV captures) rows 1 0 [] trace writes := by
    simpa only [Run, Body, shape, ivs, List.headD_cons, Bool.false_eq_true, ↓reduceIte] using run
  intro index inside
  have rowInside : index / columns < rows := (Nat.div_lt_iff_lt_mul positiveColumns).mpr inside
  have columnInside : index % columns < columns := Nat.mod_lt _ positiveColumns
  obtain ⟨value, member⟩ := SCF_covers (fun row => row * columns + index % columns)
    (fun _ _ _ event => NestedBody.monotone event)
    (fun _ _ _ event => NestedBody.emits_row event indices different outerFresh leafFresh _ columnInside) source _ rowInside
  refine ⟨value, ?_⟩
  have equal : index / columns * columns + index % columns = index := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod index columns
  simpa only [equal] using member

/-- Every successful ordered initializer log writes EVERY valid cell. Pure
operations may be partial/nondeterministic; correlated row-prefix choices stay
inside the same nested execution. No per-coordinate totality is postulated. -/
theorem Run.covers {Value : Type} {meaning : PureMeaning Value} {program : Initializer} {tile : Nat}
    (valid : program.WellFormed) {captures : Env Value} {writes : Log Value}
    (run : Run meaning program tile false captures writes) :
    ∀ index, index < GraphLoopsReuse.Geometry.volume program.shape → ∃ value, (index, value) ∈ writes := by
  obtain ⟨rank, indices, leafFresh, outerFresh⟩ := valid
  have leaves : ∀ name ∈ program.leaf.indices, name ∉ definitions program.leaf.ops := by
    simpa only [indices] using leafFresh
  rcases rank with one | two
  · obtain ⟨shape, ivs, _positive, _outerEmpty⟩ := one
    have bound := rank_one_covers shape ivs (indices.trans ivs) leaves run
    simpa only [shape, GraphLoopsReuse.Geometry.volume, Nat.mul_one] using bound
  · obtain ⟨shape, ivs, _positiveRows, positiveColumns, different⟩ := two
    have fresh : program.outerIV ∉ definitions program.outerPrefix :=
      outerFresh _ (by rw [ivs]; exact List.mem_cons_self)
    have bound := rank_two_covers shape ivs positiveColumns different (indices.trans ivs) fresh leaves run
    simpa only [shape, GraphLoopsReuse.Geometry.volume, Nat.mul_one] using bound

theorem Initializer.positive_volume {program : Initializer} (valid : program.WellFormed) :
    0 < GraphLoopsReuse.Geometry.volume program.shape := by
  rcases valid.1 with one | two
  · obtain ⟨shape, _ivs, positive, _outer⟩ := one
    simpa only [shape, GraphLoopsReuse.Geometry.volume, Nat.mul_one] using positive
  · obtain ⟨shape, _ivs, rowsPositive, columnsPositive, _distinct⟩ := two
    simpa only [shape, GraphLoopsReuse.Geometry.volume, Nat.mul_one] using Nat.mul_pos rowsPositive columnsPositive

end Merlin.GraphLoopsMemorySCF

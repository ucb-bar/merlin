/- Graham D3 independent witness: the total/private/unobservable allocation premise
of GraphLoopsReuse / GraphLoopsMemorySCF is necessary for any finite-capacity claim.

Shape of the retained `resource_gap` case (reuse-mode-v1/README.md:52-55; its MLIR is
not retained locally): first and last `memref<2xi32>` (8 bytes) share slot 0; the
intervening `memref<1x2xi32>` (8 bytes) uses slot 1. The source frees the first object
before allocating the middle one; the reuse candidate keeps slot 0 reserved across the
gap. This file (1) shows the frozen protocol checker ACCEPTS that epoch program, so the
sealed theorem equates all complete outcomes in the total-allocation model, and (2)
shows a capacity-8 allocator succeeds on the source but exhausts on the candidate. -/
import Merlin.GraphLoopsReuseCountermodels

set_option autoImplicit false

namespace GrahamD3.AllocationGap
open Merlin.GraphLoopsReuse

def objects : Objects := [⟨0, [2], 32, 16⟩, ⟨1, [1, 2], 32, 16⟩, ⟨0, [2], 32, 16⟩]

def program : List Instruction := [
  .allocate 0 true, .initialize 0 "a", .load 0 "read", .release 0 false,
  .allocate 1 true, .initialize 1 "b", .load 1 "read", .release 1 true,
  .allocate 2 false, .initialize 2 "c", .load 2 "read", .release 2 true]

/-- The sealed epoch/slot protocol accepts this reuse plan. -/
theorem accepted : checkProgram objects emptyStatus program = true := by decide

/-- Hence the sealed theorem equates every complete outcome in the TOTAL model. -/
theorem model_equates {Value Public : Type} (primitives : Primitives Value Public)
    (source target : Memory Value Public) (same : source.observed = target.observed) (outcome : Public) :
    (∃ final, Execution objects primitives false program source final ∧ final.observed = outcome) ↔
    (∃ final, Execution objects primitives true program target final ∧ final.observed = outcome) :=
  checked_observations objects program accepted primitives source target same outcome

def bytes (object : Nat) : Nat := (objectAt objects object).extent * (objectAt objects object).elementBits / 8

/-- Finite allocator for the SOURCE: every allocation requests its bytes; every release returns them. -/
def sourceRun (capacity : Nat) : List Instruction → Nat → Option Nat
  | [], used => some used
  | .allocate object _ :: rest, used =>
      (Countermodels.attemptAllocate capacity used (bytes object)).bind (sourceRun capacity rest)
  | .release object _ :: rest, used => sourceRun capacity rest (used - bytes object)
  | _ :: rest, used => sourceRun capacity rest used

/-- Finite allocator for the CANDIDATE: only slot-reserving (retained) allocations request
bytes and only retained releases return them, exactly the `retained` flags the sealed
`Step` uses for physical clearing. -/
def candidateRun (capacity : Nat) : List Instruction → Nat → Option Nat
  | [], used => some used
  | .allocate object retained :: rest, used =>
      if retained then (Countermodels.attemptAllocate capacity used (bytes object)).bind (candidateRun capacity rest)
      else candidateRun capacity rest used
  | .release object retained :: rest, used =>
      candidateRun capacity rest (if retained then used - bytes object else used)
  | _ :: rest, used => candidateRun capacity rest used

def sourcePeak : List Instruction → Nat → Nat
  | [], _ => 0
  | .allocate object _ :: rest, used => max (used + bytes object) (sourcePeak rest (used + bytes object))
  | .release object _ :: rest, used => sourcePeak rest (used - bytes object)
  | _ :: rest, used => sourcePeak rest used

def candidatePeak : List Instruction → Nat → Nat
  | [], _ => 0
  | .allocate object retained :: rest, used =>
      let next := if retained then used + bytes object else used
      max next (candidatePeak rest next)
  | .release object retained :: rest, used => candidatePeak rest (if retained then used - bytes object else used)
  | _ :: rest, used => candidatePeak rest used

theorem peaks : sourcePeak program 0 = 8 ∧ candidatePeak program 0 = 16 := by decide

theorem source_fits_capacity_eight : sourceRun 8 program 0 = some 0 := by decide

theorem candidate_exhausts_capacity_eight : candidateRun 8 program 0 = none := by decide

theorem both_fit_capacity_sixteen : sourceRun 16 program 0 = some 0 ∧ candidateRun 16 program 0 = some 0 := by decide

/-- Nonzero total-model witness: the source has a complete outcome [7, 9, 11]. -/
def primitives : Primitives Nat (List Nat) where
  initialization key _ values := values = fun _ => if key = "a" then 7 else if key = "b" then 9 else 11
  address _ _ index := index = 0
  load _ value before after := after = before ++ [value]
  store _ _ _ _ := False
  dimension _ _ _ _ := False
  retained _ _ _ := False

def start : Memory Nat (List Nat) := ⟨[], fun _ _ => none⟩
def stale : Memory Nat (List Nat) := ⟨[], fun _ _ => some 91⟩

def m1 : Memory Nat (List Nat) := clearMemory objects false 0 true start
def m2 : Memory Nat (List Nat) := initializeMemory objects false 0 (fun _ => 7) m1
def m3 : Memory Nat (List Nat) := ⟨[7], m2.heap⟩
def m4 : Memory Nat (List Nat) := clearMemory objects false 0 false m3
def m5 : Memory Nat (List Nat) := clearMemory objects false 1 true m4
def m6 : Memory Nat (List Nat) := initializeMemory objects false 1 (fun _ => 9) m5
def m7 : Memory Nat (List Nat) := ⟨[7, 9], m6.heap⟩
def m8 : Memory Nat (List Nat) := clearMemory objects false 1 true m7
def m9 : Memory Nat (List Nat) := clearMemory objects false 2 false m8
def m10 : Memory Nat (List Nat) := initializeMemory objects false 2 (fun _ => 11) m9
def m11 : Memory Nat (List Nat) := ⟨[7, 9, 11], m10.heap⟩
def m12 : Memory Nat (List Nat) := clearMemory objects false 2 true m11

theorem source_run : Execution objects primitives false program start m12 :=
  .more (.allocate 0 true start)
  (.more (.initialized 0 "a" m1 (fun _ => 7) rfl)
  (.more (.load 0 0 "read" m2 7 [7] rfl (by decide) rfl rfl)
  (.more (.release 0 false m3)
  (.more (.allocate 1 true m4)
  (.more (.initialized 1 "b" m5 (fun _ => 9) rfl)
  (.more (.load 1 0 "read" m6 9 [7, 9] rfl (by decide) rfl rfl)
  (.more (.release 1 true m7)
  (.more (.allocate 2 false m8)
  (.more (.initialized 2 "c" m9 (fun _ => 11) rfl)
  (.more (.load 2 0 "read" m10 11 [7, 9, 11] rfl (by decide) rfl rfl)
  (.more (.release 2 true m11) (.done m12))))))))))))

/-- Nonzero complete source outcome in the total model. -/
theorem source_complete : ∃ final, Execution objects primitives false program start final ∧ final.observed = [7, 9, 11] :=
  ⟨m12, source_run, rfl⟩

/-- The total model therefore also gives the candidate the same complete outcome,
from stale physical contents. -/
theorem candidate_complete : ∃ final, Execution objects primitives true program stale final ∧ final.observed = [7, 9, 11] :=
  (model_equates primitives start stale rfl [7, 9, 11]).mp source_complete

/-- Adding finite-capacity admission as an observation separates the two programs:
the source keeps its complete outcome while the candidate has none. So the sealed
relation cannot be read as a finite-allocator or peak-memory preservation claim. -/
theorem finite_capacity_distinguishes :
    ((∃ final, Execution objects primitives false program start final ∧ final.observed = [7, 9, 11]) ∧
      (sourceRun 8 program 0).isSome) ∧
    ¬ ((∃ final, Execution objects primitives true program stale final ∧ final.observed = [7, 9, 11]) ∧
      (candidateRun 8 program 0).isSome) := by
  refine ⟨⟨source_complete, by decide⟩, ?_⟩
  rintro ⟨_, fits⟩
  simp [candidate_exhausts_capacity_eight] at fits

end GrahamD3.AllocationGap

#print axioms GrahamD3.AllocationGap.accepted
#print axioms GrahamD3.AllocationGap.model_equates
#print axioms GrahamD3.AllocationGap.peaks
#print axioms GrahamD3.AllocationGap.source_fits_capacity_eight
#print axioms GrahamD3.AllocationGap.candidate_exhausts_capacity_eight
#print axioms GrahamD3.AllocationGap.both_fit_capacity_sixteen
#print axioms GrahamD3.AllocationGap.source_complete
#print axioms GrahamD3.AllocationGap.candidate_complete
#print axioms GrahamD3.AllocationGap.finite_capacity_distinguishes

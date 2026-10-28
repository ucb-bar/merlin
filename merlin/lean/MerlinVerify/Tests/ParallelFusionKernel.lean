/- Semantic-kernel controls, independent of the future fusion syntax matcher. -/
import MerlinVerify.Rules.ParallelFusion
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ParallelFusionKernel

open MerlinVerify MerlinVerify.Sem.Ops.Linalg
open MerlinVerify.Rules.ParallelFusion MerlinVerify.Tests.Semantics

def bits (n : Int) : Val := .some (.bits 13 (BitVec.ofInt 13 n))
def invert : Val → Val
  | .some (.bits w x) => .some (.bits w (~~~x))
  | _ => .poison

/-- The two original destinations are deliberately unrelated to the fresh fused one. -/
def oldProducer (shape : List Nat) := Tensor.full shape (.int 13) (bits 37)
def oldConsumer (shape : List Nat) := Tensor.full shape (.int 13) (bits (-15))
def fresh (shape : List Nat) := Tensor.full shape (.int 13) .poison
def payload (shape : List Nat) : Tensor :=
  { shape, elt := .int 13, get := fun pt => bits (linearIndex shape pt) }

def copy : Ctx → RegionSem := fun _ args =>
  match args with
  | [.scalar v, .scalar _] => .ok [.scalar v]
  | _ => .stuck "copy body arity/type"
def complement : Ctx → RegionSem := fun _ args =>
  match args with
  | [.scalar v, .scalar _] => .ok [.scalar (invert v)]
  | _ => .stuck "complement body arity/type"
def fail : Ctx → RegionSem := fun _ _ => .ub

def loop (shape : List Nat) (input init : Tensor) (body : Ctx → RegionSem) :=
  genericSem ctx [identityMap shape.length, identityMap shape.length]
    (List.replicate shape.length .parallel) [.tensor input] [.tensor init] body
def source (shape : List Nat) (input : Tensor) :=
  (loop shape input (oldProducer shape) copy).bind fun produced =>
    genericSem ctx [identityMap shape.length, identityMap shape.length]
      (List.replicate shape.length .parallel) produced [.tensor (oldConsumer shape)] complement
def target (shape : List Nat) (input : Tensor) := loop shape input (fresh shape) complement

private theorem actual_read (t : Tensor) (shape : List Nat) (ht : t.shape = shape)
    (pt : Idx) (hp : Idx.InBounds shape pt) :
    readAt (.tensor t) (identityMap shape.length) pt = .ok (t.get pt) := by
  subst shape
  exact readAt_identity hp

-- Ordinary, unit, rank-zero and empty iteration spaces; the expected values are independent.
#guard readInts (source [3] (payload [3])) == some [[some (-1), some (-2), some (-3)]]
#guard readInts (target [3] (payload [3])) == some [[some (-1), some (-2), some (-3)]]
#guard readInts (source [] (payload [])) == some [[some (-1)]]
#guard readInts (target [] (payload [])) == some [[some (-1)]]
#guard readInts (source [1, 1] (payload [1, 1])) == some [[some (-1)]]
#guard readInts (target [1, 1] (payload [1, 1])) == some [[some (-1)]]
#guard readInts (source [2, 3] (payload [2, 3])) ==
  some [[some (-1), some (-2), some (-3), some (-4), some (-5), some (-6)]]
#guard readInts (source [2, 3] (payload [2, 3])) == readInts (target [2, 3] (payload [2, 3]))
#guard readInts (source [2, 0, 3] (payload [2, 0, 3])) == some [[]]
#guard readInts (target [2, 0, 3] (payload [2, 0, 3])) == some [[]]

-- Payload poison survives; arbitrary poison in the old and fresh destinations is irrelevant.
#guard readInts (source [2] (Tensor.full [2] (.int 13) .poison)) == some [[none, none]]
#guard readInts (target [2] (Tensor.full [2] (.int 13) .poison)) == some [[none, none]]
#guard readInts (source [] (Tensor.full [] (.int 13) .poison)) == some [[none]]
#guard readInts (target [] (Tensor.full [] (.int 13) .poison)) == some [[none]]

-- The kernel does not turn a failed body into a successful one on a nonempty domain.
#guard isUb (loop [1] (payload [1]) (oldProducer [1]) fail)
#guard isUb (loop [] (payload []) (oldProducer []) fail)
#guard readInts (loop [0] (payload [0]) (oldProducer [0]) fail) == some [[]]
#guard readInts (loop [0, 2] (payload [0, 2]) (oldProducer [0, 2]) fail) == some [[]]
#guard isStuck (loop [2] (payload [3]) (oldProducer [2]) copy)

-- Actual output element metadata is retained rather than inferred from the body values.
#guard (match loop [1] (payload [1]) (Tensor.full [1] (.int 7) .poison) complement with
  | .ok [.tensor t] => t.elt == .int 7
  | _ => false)
#guard (match target [2] (payload [2]) with
  | .ok [.tensor t] => decide (t.get [2] = .poison)
  | _ => false)

-- An empty input list is still geometrically determined by its output.
#guard match loopRanges [identityMap 2] [[0, 4]] 2 with
  | .ok ds => ds == [0, 4]
  | _ => false
#guard match loopRanges [identityMap 0] [[]] 0 with
  | .ok ds => ds == []
  | _ => false

-- Captures are actual closures, not assumptions about the absence of free scalar values.
def capture (v : Val) : Ctx → RegionSem := fun _ _ => .ok [.scalar v]
#guard readInts (loop [2] (payload [2]) (fresh [2]) (capture (bits 19))) ==
  some [[some 19, some 19]]
#guard readInts (loop [2] (payload [2]) (fresh [2]) (capture .poison)) == some [[none, none]]

-- Reading the old output cannot justify arbitrary fresh-output replacement.
def readDestination : Ctx → RegionSem := fun _ args =>
  match args with
  | [.scalar _, .scalar v] => .ok [.scalar v]
  | _ => .stuck "destination body arity/type"
#guard readInts (loop [1] (payload [1]) (oldConsumer [1]) readDestination) == some [[some (-15)]]
#guard readInts (loop [1] (payload [1]) (fresh [1]) readDestination) == some [[none]]
#guard isStuck (loop [1] (payload [1]) (fresh [1]) (fun _ _ => .ok []))

-- Multiple input maps and a genuinely empty input list retain their exact operand arity.
#guard readInts (genericSem ctx (List.replicate 3 (identityMap 1)) [.parallel]
  [.tensor (payload [2]), .tensor (fresh [2])] [.tensor (oldConsumer [2])]
  (capture (bits 7))) == some [[some 7, some 7]]
#guard readInts (genericSem ctx [identityMap 1] [.parallel] [] [.tensor (fresh [2])]
  (capture (bits 23))) == some [[some 23, some 23]]
#guard readInts (genericSem ctx [identityMap 1] [.parallel] [] [.tensor (fresh [0])]
  fail) == some [[]]

/-- A symbolic use of the fusion theorem checks all three actual-body premises. -/
example (shape : List Nat) : source shape (payload shape) = target shape (payload shape) := by
  apply identity_overwrite_fusion ctx shape [payload shape]
    (oldProducer shape) (oldConsumer shape) (fresh shape) copy complement complement
    (fun pt => (payload shape).get pt) (fun pt => invert ((payload shape).get pt))
  · intro t ht
    simp only [List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at ht
    rcases ht with ht | ht | ht | ht <;> rw [ht] <;> rfl
  · rfl
  · intro pt hp
    simp [bodyAt, copy,
      actual_read (payload shape) shape rfl pt hp,
      actual_read (oldProducer shape) shape rfl pt hp]
  · intro pt hp
    simp only [bodyAt, List.zip_cons_cons, List.zip_nil_left, List.mapM_cons,
      List.mapM_nil]
    rw [actual_read (Tensor.canon shape (oldProducer shape).elt
      (fun pt => (payload shape).get pt)) shape rfl pt hp,
      actual_read (oldConsumer shape) shape rfl pt hp]
    simp [complement, Tensor.canon, hp]
  · intro pt hp
    simp [bodyAt, complement,
      actual_read (payload shape) shape rfl pt hp,
      actual_read (fresh shape) shape rfl pt hp]

end MerlinVerify.Tests.ParallelFusionKernel

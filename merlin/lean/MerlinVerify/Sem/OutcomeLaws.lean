/-
`Outcome` is a lawful monad, so the standard `List.mapM`/`foldlM` lemmas apply to it.
-/
import MerlinVerify.Sem.Outcome

namespace MerlinVerify

instance : LawfulMonad Outcome := LawfulMonad.mk'
  (id_map := fun x => by cases x <;> rfl)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun x _ _ => by cases x <;> rfl)

end MerlinVerify

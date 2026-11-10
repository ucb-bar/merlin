# Saturn support ownership

Target-specific vector backend and dialect implementation belong here, not in
shared Merlin. Provider identity is Saturn, separate from the sibling RVV support.
Shared generic emitters stay in Merlin; do not duplicate them here.

Preserve historical source receipts. Changes after migration require their own
explicit evidence and must not rewrite the original source hashes. Pure import
and emission tests are not native execution or hardware certification.

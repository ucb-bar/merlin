---
section: implementation_focus
merge: append
---
Treat this as a post-global-optimization recovery task. Prefer Merlin and IREE seams that recover accelerator semantics after global optimization and before dispatch creation, and keep the design aligned with dialect registration, pass plumbing, and structured lowering.

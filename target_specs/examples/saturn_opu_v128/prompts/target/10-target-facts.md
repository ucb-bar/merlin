---
section: target_facts
merge: append
---
This target models Saturn OPU with an LLVM-exposed vendor extension path. Keep the implicit machine-state assumptions explicit, preserve VLEN-sensitive tiling, and call out any packing or unpacking decisions that follow from the matrix engine geometry.

---
section: target_facts
merge: append
---
This target is the SpacemiT X60 path. Preserve the CPU-side LLVM and ukernel flow, keep `{{ isa.features | comma_list }}` visible in derived compile views, and treat vector-length-driven tile decisions as first-class target facts.

# Radiance Phase 1 inputs

Public target-specific inputs belong here, not in shared Merlin implementation.
The descriptor selects the contract and explicitly denies its example-kernel subtree
for the no-kernel treatment. Preserve that denial when moving or adding inputs.
The ISA definition imports its sibling patterns; keep them together. The assembly
file is a retained supplied reference, not a newly generated experiment result.
Kernel-library selection and materialization live in `kernel_library/`; generated
exports stay at an explicit artifact destination. Never copy private holdouts here.

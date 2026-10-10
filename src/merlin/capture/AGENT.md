# AGENT.md — src/merlin/capture

Owns shared capture bundle access, deterministic rewrites, and workload identities.
No dependency on experiment orchestration, baseline frameworks, or DSE analysis.
Preserve on-disk formats and numerical tolerances when moving code.

`integerization.py` validates complete integer/preserved-floating contraction
accounting without importing a framework. Preserved BF16/FP16 Q/DQ is source
floating arithmetic, never target admission or integer coverage; unresolved
rewrites and unmatched refusal inventories remain errors.

`safetensors.py` owns bounded header framing and JSON decoding for capture rewrites
and weight packers. It preserves metadata and entry order, never reads tensor data,
and rejects malformed, duplicate-key or non-finite JSON. Its default 16 MiB header
bound is a Merlin policy, not a format limit. Consumers still own tensor layout,
dtype, payload identity and execution-authority validation.

`contraction_formats.py` observes explicitly selected complete original call and
session rosters against actual parsed reduction bodies. The first positive domain
is flat rank-two mm/matmul with signed i8 operands and i32 accumulation. Other
original forms, incomplete lineage, dynamic geometry and multiplicity remain
UNKNOWN and block optional complete-integer registration. Counts and static MACs
retain plain/preserved floats and all unsupported original calls; these observations
do not establish source-producer authority, numerical equivalence, accuracy,
offload, runtime or timing. Saved census JSON is never an input authority.

The explicit original-program selection adapter reopens a pre-dispatch graph pin
roster and joins fresh emitted trace, MLIR and complete session membership. It
never extracts an original graph from post-capture metadata. These selections
bind source-format bytes only; original-producer qualification remains separate.

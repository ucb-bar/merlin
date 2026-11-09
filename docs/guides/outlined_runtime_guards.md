---
title: Retaining source guards in outlined programs
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [target_resolution]
code_refs: [src/merlin/llvmlower/outlined_guards.py, src/merlin/xdsl_dialects/lowering/dispatch_program.py, src/merlin/runtime/outlined_chain.py]
---

Outlined DAG pruning removes only views whose source operation has a proven pure
standard form. Unknown effects and assertions remain live even when they produce
no result. Keeping a node is not execution admission: the strict deployable
planner still refuses unsupported effects and contracts.

A supported `cf.assert` consumes a zero-index `tensor.extract` of a dense rank-zero
`tensor<i1>`. The command buffer retains the original IR and an ordered guard
node reading the underlying completed tensor, so its producer and storage remain
live. The scalar extract can disappear when nothing else consumes it. Other
assertion forms refuse rather than being interpreted as metadata.

All finite reductions and other inference arithmetic execute on assigned compute
workers. Their produced boolean bytes must be canonical zero or one. After the
preceding team is quiescent, the controller may inspect this returned flag as
control data. False or malformed flags return a failure before the next kernel
or provider preparation; a trailing guard fails before final output publication.
The existing staged runtime abort/join protocol remains responsible for worker
quiescence, and a failed context is poisoned against reuse.

The arity-checked `prov.argument_source_node_ids` array records original argument
identity on the original driver and plan. It grants no numerical permission and
is not redistributed across differently shaped outlined kernel interfaces.
Unknown function contracts continue to refuse.

Partial section slicing refuses a program containing guards or unproved driver
effects. Selecting every compute region retains the complete original ordered
DAG, including guards and their producers. No caller-supplied boundary flag is
invented to remove a source condition from a smaller signature.

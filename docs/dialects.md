# Dialects

Rule: create a dialect only when information must **survive passes, be verified, be transformed,
and eventually lower**. Reporting/search/bookkeeping data stays as schemas/YAML/JSON.

## Four core dialects

| Dialect            | Answers                                                            |
| ------------------ | ----------------------------------------------------------------- |
| `merlin.contract`  | What is true? What must the compiler prove? What does HW promise? |
| `merlin.schedule`  | Which compiler decisions are selected? (tiling, layout, placement, dispatch grouping, memory-state assignment) |
| `merlin.interface` | Which target-independent HW/SW abstraction is exposed? (resident tensors, accumulators, event tokens, command objects) |
| `merlin.runtime`   | How is it launched/synchronized/persisted/measured?               |

Then each real target has its own dialect (e.g. `toynpu`, gemmini, radiance).

## Not dialects (yet)

- **DSE search spaces** → schemas first (`dse_result`, `interface_candidate`). Create a
  `merlin.dse` dialect only if symbolic choices must live inside the pipeline.
- **Kernel-derived policies** → schemas first (`kernel_record`, `abstraction_candidate`,
  `policy_rule`). They later feed `merlin.schedule`.

## Where the question lands

| Question                                | Belongs in            |
| --------------------------------------- | --------------------- |
| Should I tile 64x64x128?                | merlin.schedule       |
| Should W be resident?                   | merlin.schedule / interface |
| How do I express resident W abstractly? | merlin.interface      |
| Which hardware op implements it?        | target dialect        |
| Which command buffer / queue runs it?   | merlin.runtime        |

Prototype all four in `merlin/python/merlin/xdsl_dialects/`; promote stable ones to
`merlin/compiler/include/merlin/Dialect/` + `lib/Dialect/`.

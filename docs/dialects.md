# Dialects

Rule: create a dialect only when information must **survive passes, be verified, be transformed,
and eventually lower**. Reporting/search/bookkeeping data stays as schemas/YAML/JSON.

Dialect namespaces are bare — no `m` prefix, no `merlin.` prefix.

## Five core dialects

| Dialect     | Answers                                                            |
| ----------- | ----------------------------------------------------------------- |
| `contract`  | What is true? What must the compiler prove? What does HW promise? |
| `schedule`  | Which compiler decisions are selected? (tiling, layout, placement, dispatch grouping, memory-state assignment) |
| `interface` | Which target-independent HW/SW abstraction is exposed? (resident tensors, accumulators, event tokens, command objects) |
| `runtime`   | How is it launched/synchronized/persisted/measured?               |
| `dse`       | Which interface candidates exist, and what did their variant runs measure? (minimal, descriptive — never lowers) |

Then each real target has its own dialect (e.g. `toynpu`, gemmini, saturn, radiance).

## Not dialects (yet)

- **Kernel-derived policies** → schemas first (`kernel_record`, `abstraction_candidate`,
  `policy_rule`). They later feed the `schedule` dialect.
- **Search machinery** → stays Python tooling; `dse` IR only *records* candidates and
  results (mirroring `interface_candidate`/`dse_result`/`exploitability_report` schemas).

## Where the question lands

| Question                                | Belongs in            |
| --------------------------------------- | --------------------- |
| Should I tile 64x64x128?                | schedule              |
| Should W be resident?                   | schedule / interface  |
| How do I express resident W abstractly? | interface             |
| Which hardware op implements it?        | target dialect        |
| Which command buffer / queue runs it?   | runtime               |
| Was software-visible residency worth it? | dse                  |

Prototype all five in `merlin/python/merlin/xdsl_dialects/`; promote stable ones to a future
MLIR/C++ plane (**not yet built** — see `docs/design/compiler_plane.md`).

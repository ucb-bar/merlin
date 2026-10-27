# TVM Gemmini comparison

Start at **[FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md)** for the
current deliverables, exact commands, readiness and parallel remote job plan.
The shared tuning chunks for **all three models** are published in
[the checksum-verified chunk delivery](../../../../out/artifacts/handoff/gemmini-shared-model-chunks/v1/latest/README.md).

- ResNet: committed baseline ELF and restorable parent/chunks/fixtures; tune
  existing contraction schedules and compare baseline versus tuned E2E.
- TinyLlama: 10 frozen projection/MLP chunks at 16 and 1 rows, with independent
  references. Complete attention/session preparation remains separate; final
  baseline/tuned E2E uses **32/32**, with cache capacity 64.
- SmolVLA: 8 ready flow/expert chunks and one blocked connected diagnostic.
  Chunk publication/tuning is authorized; full selected E2E quality is unfinished.
- Use stock TVM EvolutionarySearch/XGB through the existing Gemmini backend.
  Graph fusion stays at 2. Proposed host/packing knobs are not authorized.

| Reference | Purpose |
| --- | --- |
| [Numerical policy](../../numerics/CURRENT_RUN_POLICY.md) | Frozen W8A8 arithmetic and separate model-quality gates |
| [Adapter contract](ADAPTER_CONTRACT.md) | Real live active-cycle cutoff, moving incumbents and cancellation qualification |
| [Optimization ownership/knobs](OPTIMIZATION_KNOBS.md) | Existing decisions and research-only ideas |
| [Input audit](EL4_INPUT_AUDIT.md) | Dated source/shape/input findings; current readiness is in the handoff |
| [Generation](GENERATION.md) | Compiler flow and ownership |
| [Performance audit](PERFORMANCE_AUDIT.md) | Hardware/resource and historical cycle analysis |
| [Numerical diagnostics](NUMERICAL_DIAGNOSTICS.md) | Existing first-divergence tools |

The older overlapping setup/run pages now link to the single current handoff.
Historical artifacts and reports remain unchanged. The prior long setup/run
reference is retained in [Git history](https://github.com/ucb-bar/merlin/blob/1900c7ff8b4c4495479e20a73a27bdbd79413abf/examples/gemmini/comparisons/tvm/README.md).

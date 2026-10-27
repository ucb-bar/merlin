# TVM Gemmini comparison

Start at **[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md)** for the current approved
workloads, knobs, ELF build commands and full-session preparation instructions.
Integration is `82b321a89`; model protocol is exclusively `8ec99308` with the
approved TinyLlama 32+32 correction. Use only its quality policy.

The published chunk delivery contains 18 ResNet, 10 TinyLlama and 8 ready SmolVLA
chunks: 47 initial distinct contractions across 26 owners. TinyLlama tunes at
16/1 rows and runs E2E at 32/32. Begin with fusion=2/prepacking/bias preload and
then search existing graph options and TensorIR schedules, within 4h/1h/1h total
per-model tuning allowances. Final E2E is separate. The remote worker owns
slots, deployment, FireSim-only guest validation and tuning-time accounting.

| Reference | Purpose |
| --- | --- |
| [Selected protocol](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol) | Sole E2E recipe/input/quality authority; approved correction in handoff |
| [Adapter contract](ADAPTER_CONTRACT.md) | Real live active-cycle cutoff, moving incumbents and cancellation qualification |
| [Optimization ownership/knobs](OPTIMIZATION_KNOBS.md) | Existing decisions and research-only ideas |
| [Input audit](EL4_INPUT_AUDIT.md) | Dated source/shape/input findings; current readiness is in the handoff |
| [Generation](GENERATION.md) | Compiler flow and ownership |
| [Performance audit](PERFORMANCE_AUDIT.md) | Hardware/resource and historical cycle analysis |
| [Numerical diagnostics](NUMERICAL_DIAGNOSTICS.md) | Existing first-divergence tools |

The older overlapping setup/run pages now link to the single current handoff.
Historical artifacts and reports remain unchanged. The prior long setup/run
reference is retained in [Git history](https://github.com/ucb-bar/merlin/blob/1900c7ff8b4c4495479e20a73a27bdbd79413abf/examples/gemmini/comparisons/tvm/README.md).

At the per-model tuning limit, stop admitting new work and **finish active runs**.
Do not start queued jobs during this completion period. Record the time beyond
budget separately; preserve cycle-pruning and failure/hang safeguards. This adds
no new tuning allowance and does not renew a continuation's budget.

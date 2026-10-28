# TVM Gemmini comparison

Start at **[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md)** for the current approved
workloads, knobs, ELF build commands and full-session preparation instructions.
The frozen model delivery originated at `82b321a89`; the simplified compiler
and median-watchdog runner are supplied by the current handoff. Model protocol is exclusively `8ec99308` with the
approved TinyLlama 32+32 correction. Use only its quality policy.

The next campaign is **ResNet50 plus 16 kernel cases**, with
up to 25 ReplayTrace proposals total per kernel/chunk, without ML. Full TinyLlama/SmolVLA model runs are
backlogged; their isolated kernel cases remain in scope. ResNet's caps are
4h DSE and 30min per E2E run. See the handoff for trial accounting and the 24 slot-hour kernel ceiling
and the separate [kernel study](KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md).
The restored 36-chunk delivery remains available as historical input evidence;
its TinyLlama/SmolVLA model roster is not the new kernel roster.

| Reference | Purpose |
| --- | --- |
| [Selected protocol](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol) | Sole E2E recipe/input/quality authority; approved correction in handoff |
| [Adapter contract](ADAPTER_CONTRACT.md) | Approved median wall watchdog, cycle measurement and remote cancellation qualification |
| [Architecture and paper scope](ARCHITECTURE.md) | Current top-down path, stock/custom boundaries and limits on claims |
| [Operation support](TENSORIR_OP_AUDIT.md) | Exact spec families, primitive/composite mappings and all 51 fixture boundaries |
| [Optimization ownership/knobs](OPTIMIZATION_KNOBS.md) | Compiler simplification, scheduling/cost-model audit and approved scope |
| [Input audit](EL4_INPUT_AUDIT.md) | Dated source/shape/input findings; current readiness is in the handoff |
| [Generation](GENERATION.md) | Compiler flow and ownership |
| [Performance audit](PERFORMANCE_AUDIT.md) | Hardware/resource and historical cycle analysis |
| [Numerical diagnostics](NUMERICAL_DIAGNOSTICS.md) | Existing first-divergence tools |

The older overlapping setup/run pages now link to the single current handoff.
Historical artifacts and reports remain unchanged. The prior long setup/run
reference is retained in [Git history](https://github.com/ucb-bar/merlin/blob/1900c7ff8b4c4495479e20a73a27bdbd79413abf/examples/gemmini/comparisons/tvm/README.md).

At the per-model tuning limit, stop admitting new work and **finish active runs**.
Do not start queued jobs during this completion period. Record the time beyond
budget separately; preserve the median wall watchdog and failure safeguards. This adds
no new tuning allowance and does not renew a continuation's budget.

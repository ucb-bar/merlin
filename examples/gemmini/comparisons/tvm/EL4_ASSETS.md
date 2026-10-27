# EL4 prepared assets

Current approved preparation and ELF instructions are in
**[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md)**. All three models are enabled.
Use the shared chunk publication at Merlin `82b321a89`: 18 ResNet, 10 TinyLlama
and 8 ready SmolVLA chunks, with the failing normalization/update diagnostic
excluded. Tune TinyLlama at 16/1 and run full E2E at 32/32.

The only full-model protocol is `8ec99308`, including the approved run-local
numerics correction and its quality policy. Published chunk references do not
establish complete model-session admission. Full TinyLlama/SmolVLA selected
exports and references remain separate preparation work.

ResNet gets four hours of tuning, TinyLlama and SmolVLA one hour each; final E2E
is excluded. Published graph settings come first, then existing graph options
and TensorIR schedules within those budgets. Deployment belongs to the remote
worker. Earlier paused-model/shared-budget instructions are superseded.

At the per-model tuning limit, stop admitting new work and **finish active runs**.
Do not start queued jobs during this completion period. Record the time beyond
budget separately; preserve cycle-pruning and failure/hang safeguards. This adds
no new tuning allowance and does not renew a continuation's budget.

# AGENT.md — merlin/experiments/agent_bench

Reusable, target-agnostic **agent benchmark scaffold**: compare a baseline agent vs a
merlin-assisted agent on a held-out task. Tracked source only (task prompts
`TASK_baseline.md`/`TASK_merlin_assisted.md`, `setup_baseline_sandbox.sh`, `grade.sh`, `hidden/`
test cases, `README.md`). No generated output lives here — runs route to `runs/`. See `README.md`
for the protocol.

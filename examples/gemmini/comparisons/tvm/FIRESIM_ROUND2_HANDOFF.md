# Earlier FireSim handoff

The current instructions are in **[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md)**.
Use that page for approved workloads, graph/TensorIR knobs, ELF builds, the
8ec99308 protocol and remote responsibilities. ResNet has four hours of tuning;
TinyLlama and SmolVLA have one hour each. Stop new submissions at the limit and
finish active runs, recording the extra completion time. Final E2E is separate.
TinyLlama tunes at 16/1 rows and runs E2E at 32/32.

The previous instructions are retained in
[Git history](https://github.com/ucb-bar/merlin/blob/82b321a89a8a646877484f4bd09637040ce0fa26/examples/gemmini/comparisons/tvm/FIRESIM_ROUND2_HANDOFF.md).
Historical run artifacts remain unchanged; their old budgets, model pauses and
numerical policies do not select the current campaign.

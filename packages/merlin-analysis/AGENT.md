# AGENT.md — packages/merlin-analysis

Owns optional baseline comparisons, scientific measurement controllers, and plotting.
Stable imports remain merlin.baselines, merlin.compare, merlin.plotting, merlin.agentreport and
merlin.verify.plots, merlin.verify.replay, merlin.verify.replay_layers and merlin.perf.recovery through the
shared namespace. Core capture and ISA-audit implementations live in merlin.capture and
merlin.runtime; do not duplicate them. Preserve frozen study/evidence semantics.

# AGENT.md — packages/merlin-timing/include/merlin_timing

Keep state bounded by the declared scoreboard, resource capacities, queue
capacities and cache geometry. Resources receive explicit duration and demand;
result latency and occupancy remain separate. Preserve RAW/WAW/WAR hazards,
unknown propagation and refusal to recycle live state.

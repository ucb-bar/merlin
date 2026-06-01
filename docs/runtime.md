# Runtime

`merlin/runtime/` is the **target-independent** runtime substrate. Target-specific runtime code
lives in external target repos or under `merlin/targets/<target>/runtime/` for toy examples.

```
common/          shared runtime types
command_buffer/  enqueue / submit / wait
simulator/       event/cost simulator (drives DSE)
baremetal/       bare-metal backend (scaffold)
zephyr/          Zephyr RTOS backend (scaffold)
```

These map onto the `merlin.runtime` dialect concepts: command buffers, dispatches, queues,
persistent handles, waits, profiling regions. See `docs/dialects.md`.

# Which lever moves which signal

Generic levers for raising array utilization. Each names the signal it should move, where to find
candidate mechanisms and how to verify it. Mechanism ids and APIs for this target are in
`OPTIMIZATION_SUGGESTIONS.json` and in each entry's `suggested_mechanisms` (`pass-library` skill); the
Facets column names the catalogue `family` values those hints draw from. Verify every lever the same way: a
variant or knob, the same cases, numerics unchanged, cycles compared (`ab-library-pass` skill).

| Signal you see | Likely cause | Lever | Facets |
|---|---|---|---|
| `bound_by: compute`, low `array_utilization_pct`, `useful_fraction_of_padded_work` below 1 | tiles do not fill the array; edge tiles padded | Fill the array dimensions with the tiling: choose tile extents from the array and storage facts so each compute command uses full rows/columns; split tails so padding is small | scheduling, storage |
| `bound_by: compute`, gap well above 1, busy issue unit (`cpu.issue`, an accelerator `issue` lane) | many small commands, configuration per tile | Cut configuration and issue overhead: hoist or merge configuration, larger commands, fewer fences, loop-invariant setup outside loops | issue, synchronization, scheduling |
| `bound_by: read requests`, traffic ratio above 1 | operands fetched more than once | Keep operands resident: reuse loaded tiles across the reduction or output loop, pick the loop order that keeps the larger operand stationary, size tiles to the local storage | storage, movement |
| `bound_by: read/write requests`, traffic ratio near 1, low overlap between transfer and execute units | loads, compute and readout serialized | Overlap loads and readout with compute: double-buffer, prefetch the next tile, issue readout as soon as a reduction finishes | overlap, movement |
| busy `memory.*` or load/store units, many small transfers | transfer count, not bytes, dominates | Coalesce copies into fewer larger transfers | movement |
| busy `accelerator.storage.<unit>` (accumulator, scratchpad) | the unit's capacity or port is the bottleneck | Shorten lifetimes, separate operand and partial-sum storage, read out earlier; mechanisms naming that unit come first in the hints | storage, overlap |
| host `cpu.*` work between device commands; epilogue on the host | epilogue or layout work outside the accelerator | Fuse epilogues (bias, scale, activation, requantization) into the device readout when the target supports it; move packing out of the timed window when legal | fusion, packing, host_compute |
| connected model: one kernel holds most of `share_pct` | that kernel's schedule | Work that kernel's family first; check `device_mac_fraction` stays at its value | all |

Signals that do not mean "more work": a high `modeled_activity` is a modeled reservation, not a stall;
the request floors are ideal service at peak rate. Rank candidates by complete cycles of the same cases,
never by an activity or ratio alone.

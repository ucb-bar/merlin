# Resource floors in Phase 2 feedback

A lower bound answers how many target cycles are necessary. It does not predict
latency, establish achievable overlap, or qualify accelerator execution. The
original host owns source, executable, numerical, interface-crossing, timing-window
and hardware identities. Pure arithmetic never issues those authorities.

`merlin.perf.staged_roofline` emits V2. Saved V1 documents retain their original
bytes and interpretation. V2 uses useful MACs against the structural multiplier
capacity. Its padded MAC count and selected tiling describe a hypothetical
schedule; they are not unavoidable workload work. Store bank counts establish
neither overlap nor serialization. Requested bytes and hypothetical beat-rounded
segments remain distinct from physical traffic. Non-dense or unknown-granularity
requests have no hypothetical physical volume. A movement floor requires explicit
`compulsory_transfer_bytes={read, write, provenance}` stating that the named
interface must carry that payload. The old numeric host estimate is diagnostic,
not silently a compulsory serial lower bound.

`capsule_roofline.compose_service_floor(resources, events, *, completion_chains,
unresolved, scope)` composes independently derived service facts. Each resource
states its unit, structural `capacity_per_cycle_upper`, and provenance. Each event
states unique identity, demand by resource, compulsory status and provenance.
An unavailable rate or demand is `None`, never zero. Arithmetic uses exact
fractions before the final integer-cycle ceiling.

```python
resources = {
    "shared_port": {
        "unit": "bytes", "capacity_per_cycle_upper": 16,
        "provenance": "selected single read/write service path",
    }
}
events = [
    {"id": "load", "demand": {"shared_port": 64}, "compulsory": True,
     "provenance": "independent compulsory crossing proof"},
    {"id": "publish", "demand": {"shared_port": 32}, "compulsory": True,
     "provenance": "complete live-out and selected readout proof"},
]
# Six necessary service cycles on the shared path. No overlap assertion.
```

All demand on the same resource is summed. Different resource floors use the
maximum: every execution must satisfy each necessary term, whether or not their
actual services overlap. Addition across different services requires an explicit
`completion_chains=[{events: [...], provenance: ...}]` proof that each listed
service completes before the next begins. Command issue order, data availability,
queue capacity and a fence name alone are insufficient. In particular, a pending
preload descriptor can survive a fence. A selected input-row service fact prices
neither mesh propagation nor final completion.

`scope="workload"` accepts only compulsory work. Extra requests, padding, spills
and reloads from one candidate belong to `scope="emitted_schedule"`. They can
explain waste, but cannot raise the frozen workload target. These documents
remain partial if complete-call host work, packing, readout, physical ports,
resource sharing or completion effects are unresolved. They never independently
permit a roofline-convergence claim.

The ordinary pure join accepts `service_evidence` with schema
`merlin.phase2.service_effect_evidence.v1`. It requires exact `source_sha256` on
both member and compiled observation, the compiled `elf_sha256`, and the selected
`device_sha256` and `scope_sha256`. The envelope carries resources, events,
completion chains, unresolved terms and scope. The join recomputes the floor;
a supplied saved verdict or floor is ignored. Missing bindings remain UNKNOWN.
Only independently admitted, same-boundary measured target cycles can yield an
attainment or gap. Empirical or native event estimates never supply measured
cycles. A measured result below the floor refutes it without clamping. Whole-trace
resource busy times remain whole-trace diagnostics and cannot be summed or
attributed to a smaller counter window.

Target packages own instruction/effect and physical transaction derivations.
An interface width is a rate ceiling, not a packet trace. A maximum request size
is not a compulsory packet size; logical tensor bytes do not prove physical
traffic, residency, bank conflicts or DRAM misses. Missing selected sources or
measurement evidence stay explicit. Generic aggregation contains no target
names, opcodes, geometry or fitted latency prices.

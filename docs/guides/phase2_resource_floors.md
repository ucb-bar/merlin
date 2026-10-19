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
Only independently admitted, same-boundary measured target cycles can yield a
`partial_floor_ratio` or diagnostic gap. `attainment` remains null while complete
floor terms are unresolved. Empirical or native event estimates never supply measured
cycles. A measured result below the floor refutes it without clamping. Whole-trace
resource busy times remain whole-trace diagnostics and cannot be summed or
attributed to a smaller counter window.

Target packages own instruction/effect and physical transaction derivations.
An interface width is a rate ceiling, not a packet trace. A maximum request size
is not a compulsory packet size; logical tensor bytes do not prove physical
traffic, residency, bank conflicts or DRAM misses. Missing selected sources or
measurement evidence stay explicit. Generic aggregation contains no target
names, opcodes, geometry or fitted latency prices.

`merlin.perf.channel_service.compose_transfer_service(contract, transfers)`
connects an explicitly selected adapter's client-channel contract to the same
service-floor arithmetic. Contract schema `merlin.perf.channel_service_contract.v1`
provides resources, transfer kinds, unresolved terms and provenance. Each kind
states its data-beat width and the resource/demand law for request and response
channels: `units_per_packet` or `units_per_data_beat`. Transfers contain complete
adapter-derived packet inventories with payload bytes, envelope bytes and data
beats. Full-beat envelopes must match the selected width; the consumer does not
invent rounding, packet sizes, roles or resource independence.

For example, a read request may require one request-channel handshake while its
response requires four data beats. A write may use four request data beats and
one response acknowledgement. These laws belong to the selected adapter, not to
the generic consumer. Reusing a resource identity explicitly adds demands;
different resources use max and assert no achievable overlap. Every request
remains noncompulsory `emitted_schedule` demand. Returned envelope bytes are
neither observed physical traffic nor DRAM traffic. Completion latency, retries,
translation, downstream sharing and a fixed workload floor remain unresolved.
This additive path does not change an existing timing model or measured result.

Ordinary feedback also emits `cost_boundaries` with separate setup, verification,
declared execution-window and whole-trace context blocks. Original compiler
analysis walltime is reported as setup when present; compile, input staging and
verification times stay null until their original receipt costs are exposed.
No amount is inferred by subtracting target counts or simulator walltime.
An execution-window label does not prove complete-model coverage or cold/warm
state. Packing, layout conversion, transfers, dispatch, synchronization, feeding
host arithmetic and carried-state movement stay execution costs whenever they
occur inside that declared window.

The ordinary archive owner also reopens original public source bytes and gSIM
measurement context for historical diagnostics. Interface work uses the canonical
command-buffer work observer: resident handles retain their source geometry, and
an elementwise operation cannot become a contraction merely because it has a
left operand and output shape. Unsupported work remains unknown.

A queued arm can reuse historical cycles only when its actual compiled interface,
ELF, consumed caller-source producer chain, embedded input context, device and
counter coordinates match. The current frozen-capsule input hash and original
embedded-caller hash keep their distinct meanings; an explicit caller bridge
records both. Every original repeat remains visible, and a missing or failed
repeat leaves that execution unavailable. The worst repeat supplies the
historical diagnostic representative. These are retained measurements, with
`measured_now: false`; they never populate fresh `timing_cycles`, incumbent or
plateau values. Feedback V3 reports `historical_timing_cycles` separately. Changed
executables keep their qualified prediction and have no borrowed measurement.
Logical readout shape and requested store counts still do not prove physical
readout ownership, so this join does not invent a movement floor.

An ordinary model capsule has no timing family merely because its descriptor
says `op: model`. The archive owner can now recognize the complete current typed
source as one direct signed i8 by signed i8, zero-initialized rank-two dot with
complete raw i32 output. It reopens the exact original source and compiled tensor
plan, requires the existing source/emission join, and matches the entire
`(M, K, N, types, layout, initialization, readout)` tuple against public TRAIN
source evidence. Independent axis ranges and metadata labels do not admit a new
tuple. The original source-domain receipt is rederived when feedback is consumed.
This grants only source-family selection: complete functional results, actual
event counts, reviewed target/runtime semantics and raw-activity guards still
apply. Connected arithmetic, batched attention, new layouts/readouts and capacity
extrapolation remain UNKNOWN. Original archive held denominators and the frozen
native model do not change; this route creates no new qualification labels.

The original host may explicitly select a separately reviewed live capability
with mechanism `emitted_schedule_service_effects` through
`NativeArchiveScreenSelection.service_effects`. It must have the native timing
owner's exact five-field binding. The original bound `evidence` callback reopens
the current same-ELF annotated observer process and counter review; saved success
documents and foreign callbacks refuse. The archive owner writes private
per-arm `service_effects.json` and passes only checked effect inventory to the
existing pure feedback join. Missing current source/emission custody, an
unsupported window or an absent capability leaves the term UNKNOWN.

This optional route changes its own screen identity while the absent-selection
behavior keeps its previous interpretation. Its owner supplies no measured
latency, workload floor, qualification or promotion authority. In particular,
scalar retirement opportunities of an emitted host fill loop can explain that
loop's overhead without raising the frozen workload floor when a later compiler
removes it. Necessary emitted-schedule service remains separate from historical
measurements and native estimated latency.

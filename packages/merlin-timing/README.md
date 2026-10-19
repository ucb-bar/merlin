# Portable timing tools

`merlin-timing` provides a C++17 streaming timing engine and Python tooling with
no runtime dependencies on the compiler. ISA decoding and target policies are
supplied by separate observers. Every timing price is an explicit input.

## C++ engine

Include `merlin_timing/engine.hpp`. `Engine` consumes ephemeral `Event` views;
it retains fixed scoreboard entries, resource lanes and bounded waiting queues,
not an instruction trace. `ResourceSpec` declares capacity and waiting-request
capacity. `Demand.occupancy` reserves resource units; `Event.latency` controls
result readiness independently. The stream's issue frontier advances through
dependencies and queue backpressure, without automatically advancing to result
completion. Declare an issue resource to model issue throughput.

Tokens represent mutable storage. Reads wait for their latest writer; writes
wait for prior writers and readers. Set `hold_reads_until_ready` for asynchronous
reads whose input storage remains live through completion. Ordinary operands
are consumed at issue. `retained_reads` holds a separate operand set until
completion, allowing captured CPU registers and live memory to share an event. `reset` refuses live or unknown tokens. `barrier` drains
all modeled work without assuming a fence-instruction price. `publish` attaches
an already scheduled asynchronous result to a completed token without advancing
the issue frontier; it refuses live tokens and backdated results.

Unknown prices or dependencies produce incomplete accounting and propagate to
subsequent affected work. Reports contain a modeled makespan and descriptive
reservations/busy-unit counts; these are neither qualified hardware cycles nor
correctness evidence. Callers must supply complete resource/alias relationships.
The reservation model assumes greedy capacity admission and does not discover
hardware arbitration, speculative execution, coherence or memory consistency.

`SetAssociativeCache` classifies bounded LRU hits, misses and dirty evictions.
It assigns no hit, miss, writeback or coherence cost. The caller owns those
policies and must not assume absent observations are free. `probe` observes
residency/dirty state without filling, counting accesses or changing LRU;
`invalidate` reports whether a dirty line was removed. `clear` removes residency
without erasing the access census. Access results and hit probes expose the
bounded physical slot index (`set * ways + way`), allowing callers to attach
pending-fill dependencies without retaining every accessed address. These
mechanisms assign no coherence price.

## Build and export

```sh
cmake -S . -B /path/to/build -DBUILD_TESTING=ON
cmake --build /path/to/build
ctest --test-dir /path/to/build --output-on-failure
```

`merlin_timing.export.export_cpp_core(destination)` exports only the canonical
header, native test, CMake file, README and license into a fresh directory, with
per-file SHA-256 identities. The same source payload is included in the wheel,
so export works without a compiler checkout. `verify_cpp_core` detects changed,
missing or additional files. Build outside the exported source directory.

## Target provider bundles

`merlin_timing.provider_bundle.export_bundle(payloads, destination,
entrypoint="observer.cpp")` packages a caller-selected map of filenames to source
bytes. The manifest declares the translation unit and complete source roster;
the builder contains no accelerator name, instruction semantics or hardware
parameters. Exports are limited to one MiB and 64 files.

The export includes the canonical `build_provider.py`, which runs independently
of this package using Python's standard library:

```sh
python3 /path/to/provider-source/build_provider.py --verify-only
python3 /path/to/provider-source/build_provider.py \
  --spike-root /path/to/annotated-spike \
  --output-library /path/to/libtiming_provider.so
```

Compilation uses one C++17 compiler process, pins the source bundle and generic
runtime headers before and after building, and publishes a new library inode.
Target simulator sources and facts belong in the simulator repository's examples,
independently of the generated OOT compiler. The shared builder supplies no
kernel, schedule, timing prices or calibration evidence. Candidate grants expose
reviewed runtime endpoints through the broker; simulator sources remain operator
build inputs.
# Asynchronous reservation APIs

The additive `Engine::submit_detached` API schedules a controller effect without
advancing CPU admission to its future completion. `advance_issue` separately
applies explicit frontend backpressure. Original `submit` and `barrier` behavior
is unchanged. Adapters must establish actual dependency and completion effects;
these APIs supply no physical timing qualification.

`ReservationWindow` tracks bounded entries by generation and keeps each owned
until its explicit release stage. Allocation can wait for a known release;
unresolved releases cannot become zero cost or free entries. A selected adapter
may exclude an exact outstanding ticket from its drain predicate only when its
original hardware/software completion contract allows that exception. Waiting
resource queues and completion-owned reservations are different abstractions.

The portable source export now emits version 2, including the reservation header
and its executable checks. Version 1 exports retain their exact original roster
and remain verifiable without being rewritten. The simulator's source package
must retain the manifest of its exact canonical export; copied vendor bytes are
a versioned source snapshot, never another maintained implementation.

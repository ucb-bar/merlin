---
title: Reporting observed region profiles
kind: guide
status: current
owner: runtime
last_verified: 2026-11-08
related: [simulator_selection, reproducibility, verification]
code_refs: [src/merlin/runtime/profile_evidence.py, src/merlin/perf/debug_companion.py]
---

# Reporting observed region profiles

`merlin.runtime.profile_evidence` reduces existing, byte-bound observations into
JSON and a Chrome/Perfetto trace. It never executes an ELF, programs counters,
changes a simulator, estimates missing timing or fits a model. Numerical checking
and execution ownership remain separate caller obligations.

## Select the exact files and counter meaning

Bind each input to an absolute path and its lowercase SHA256 digest. A selected
numerical gate must contain an actual true pass field; its method and coverage
must describe what was checked. The caller must associate that gate with the
execution. A matching filename or digest does not establish that association.

```python
from merlin.runtime.profile_evidence import (
    BoundFile, CounterDomain, Gate, RegionSource, collect, write_report,
)

source = RegionSource(
    name="selected-execution",
    records=BoundFile(report_path, report_sha256),
    elf=BoundFile(executed_elf_path, executed_elf_sha256),
    gate=Gate(
        BoundFile(report_path, report_sha256),
        ("pass",),
        method="complete independent output comparison and guard checks",
        coverage="all declared outputs and guards",
    ),
    domain=CounterDomain(
        "functional_instructions", "instruction_count_not_modeled_cycles"
    ),
    expected_regions=expected_region_count,
    record_path=("regions",),
)
report, trace = collect([source], max_regions=expected_region_count)
write_report(fresh_output_directory, report, trace)
```

The `kind` is explicitly `functional_instructions`, `modeled_cycles` or
`hardware_cycles`. Functional counters never produce cycle values or elapsed
seconds. Cycle-to-time conversion requires independently qualified `clock_hz`;
a frequency alone does not align separate executions or harts. Counter width is
32 or 64 bits. Wrapped, reset or reversed endpoints require separate admission
and are refused by this reader.

## Supply a complete region roster

Each record requires `case_id`, `region`, `hart`, `start_cycle`, `end_cycle` and
`pass: true`. Field names do not determine counter meaning. Repeated trials need
unique case IDs. If supplied, `instret_delta` must be unsigned and agree with
both retirement endpoints. If an expected checksum is supplied, the observed
checksum must match. Other fields remain unchanged under `raw_record`, preserving
partial metrics and their existing qualification.

Set `record_path` to the JSON array location. Alternatively, select a single
caller-owned console token with `console_prefix`; this is mutually exclusive
with `record_path`. The reader requires exactly `expected_regions` records and
refuses duplicate identities or malformed selected records. Console completion,
exit and full-output checks remain the execution postgate's responsibility.

## Join source labels without inventing attribution

Optionally set `labels=BoundFile(labels_path, labels_sha256)`. Its JSON contains
`elf_sha256` and an `entries` array. Every entry selects the exact region identity
and may provide these source-declared labels:

```json
{
  "elf_sha256": "<executed-ELF SHA256>",
  "entries": [{
    "case_id": "case0",
    "region": "operator",
    "hart": 0,
    "model": null,
    "part": "projection",
    "stage": "prefill",
    "fqn": null,
    "function": null,
    "variant": "selected-variant",
    "route": "selected-route",
    "scope": "compiler-generated operator"
  }]
}
```

Unknown labels stay null. A source FQN is a provenance join, and a declared
function is a call-site label; neither measures function self time. Variants,
routes and stages form separate groups. The reader reports raw samples,
medians, inclusive sums and interval unions separately. Overlapping recorded
intervals may be nested software scopes; they do not prove physical engine
overlap. `inclusive_retired_instructions` sums region deltas, including nested
intervals; it is not a count of unique retired instructions. Physical busy and
traffic remain unavailable.

For an existing PC census, `attribute_pc_census` reuses the structural ELF and
debug-companion verifier plus LLVM symbolizer JSON reducer. Both sidecars bind
their exact ELF bytes. The result attributes observed PC counts to functions,
source lines and inline stacks, retaining unknown symbols. It does not convert
counts into function elapsed time or claim complete sampling coverage.

## Interpret the trace coordinates

Each execution and hart gets a separate track and origin. With no qualified
frequency, one raw counter tick uses one microsecond display coordinate. The
trace labels this convention explicitly: it is not elapsed time. Keep unlike
counter domains separate when comparing reports. Loading the trace into a viewer
does not establish shared epochs, unit utilization, memory bandwidth or timing
accuracy.

All inputs are verified again after reduction. `write_report` creates a fresh
output directory and refuses to overwrite existing results.

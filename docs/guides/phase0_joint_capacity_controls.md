---
title: Retained joint-store capacity controls
kind: guide
status: current
owner: experiments
last_verified: 2026-10-20
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase0/development_joint_capacity.py, packages/merlin-experiments/src/merlin_experiments/phase0/development_capacity_materialize.py]
---

# Retained joint-store capacity controls

`development_joint_capacity` accounts for simultaneous complete padded A, B and
raw i32 C buffers. Select the operand and accumulator **address-space views**
from pinned facts explicitly. The reader checks total bytes against bank count,
depth, element width and physical row width. Physical census aliases must not be
summed into the address-space capacity.

Select positive M/N/K padding granules and row-aligned reserved bytes explicitly.
They are experimental placement premises, not semantics inferred from array or
bank counts. Each A/B/C row receives its own physical-row-aligned stride. The
proof produces disjoint linear intervals, overflow amounts and exact maximum
aligned K for both one-copy and two-copy placements. Fixed reservations occur
once per store. Both operand copies and accumulator copies must fit for the
two-copy case; fitting either store alone is insufficient.

The selected 262144-byte operand and 65536-byte accumulator facts already have
four useful retained guards. With the original explicit 16-element granules and
zero reservations, they prove these **conditional full-buffer** boundaries:

| Existing ID | M / K / N | One-copy A+B / C bytes | Two-copy A+B / C bytes | Conditional regime |
| --- | --- | --- | --- | --- |
| decisive33:27 | 512 / 240 / 16 | 126720 / 32768 | 253440 / 65536 | double fit, last aligned K |
| decisive33:28 | 512 / 256 / 16 | 135168 / 32768 | 270336 / 65536 | single fit, first double spill |
| decisive33:25 | 1024 / 240 / 16 | 249600 / 65536 | 499200 / 131072 | single fit, last aligned K |
| decisive33:26 | 1024 / 256 / 16 | 266240 / 65536 | 532480 / 131072 | spill, first single spill |

These are the existing source IDs and their original functional roles. The tool
does not create benchmark identities, move guards into an objective, change
numerical policies, or remove refused/unselected members. Existing smaller and
larger capacity-family controls can be checked through the same API. A pair
whose K values skip an admissible granule is reported as a wider bracket, not
an exact last-fit/first-spill pair.

`development_capacity_materialize` reopens the complete frozen source inventory,
copies selected original source/input/reference bytes, and uses the ordinary
source stager. It also recomputes every output with the existing bounded blocked
reference owner and compares every value with the retained independent result.
It checks the entire signed input seed and bounds archive expansion before
loading arrays. Select an original reference supplement through the existing
intake when a member has no original complete NPZ; this tool preserves that
refusal rather than inventing reference authority.

An example operator command, under the existing shared compute namespace, is:

```sh
python -m merlin_experiments.phase0.development_capacity_materialize \
  --sources "$FROZEN_SOURCES" --manifest-sha256 "$MANIFEST_SHA256" \
  --facts "$FACTS" --facts-sha256 "$FACTS_SHA256" \
  --operand-memory "$OPERAND_ADDRESS_SPACE" \
  --accumulator-memory "$ACCUMULATOR_ADDRESS_SPACE" \
  --axis-granules '{"M":16,"N":16,"K":16}' \
  --reserved-bytes '{"operand":0,"accumulator":0}' \
  --ordinals 25 26 27 28 --budgets "$REFERENCE_BUDGETS" \
  --block '{"M":32,"N":64,"K":128}' --output "$FRESH_OUTPUT"
```

The budget document requires finite `max_member_work`, `max_total_work`,
`max_input_bytes`, `max_output_bytes`, `max_workspace_bytes` and
`member_timeout_s` (at most 120). Keep numerical preparation costs separate from
target latency. Original source and fact identities are reopened after the
operation; the resulting index retains the entire original member roster.

The proof and materialization do **not** establish that an emitted tiled program
holds whole source tensors in either store. Actual address ownership, live
intervals, copy counts, bank decoding, port arbitration, command dependencies,
completion and spill traffic still need the independent target effect owner.
Double-buffer fit does not prove DMA/compute overlap. No target cycle floor,
timing qualification, sealed capture, writer publication or formal performance
cohort admission follows from these artifacts.

---
title: Representative capsules per archetype
kind: design
status: current
owner: core
last_verified: 2026-09-01
related: [dialect_test_bar, compiler_plane, performance_levers_per_archetype]
code_refs: [merlin/python/merlin/targetgen/conformance.py, build_tools/scripts/check_conformance_coverage.py, merlin/contract/capsules/conformance, merlin/python/merlin/targetgen/boundary.py, merlin/python/merlin/targetgen/memory_regime.py, merlin/python/merlin/perf/falsifier.py]
---

# Representative capsules per archetype

## The rule

A corpus is not a hand-picked list. It is the **closure of a derived requirement**, and the
requirement is evidence rather than authorship — regenerate it, never hand-edit it:

```
build_tools/scripts/check_conformance_coverage.py --target <t> --write \
    merlin/contract/capsules/conformance/<t>.yaml
```

Three axes, each derived from the target's own declarations plus real captured models:

| axis | derivation |
|---|---|
| **cells** | `admitted ∩ observed` over `(semantic_family, dtype, tile_alignment)`. `admitted` = the capability manifest's `compute_units[].semantic_capabilities`; `observed` = `model_coverage.regions_from_module` over each capture. |
| **composition** | which shapes the captures actually assemble: `A`, `A→A`, `H→A→H`, `A→H→A`, `routing`. |
| **memory regime** | `fits_double` / `fits_single` / `fits_on_reuse` / `spills`, classified against the target's own operand-store capacity. |

Each axis has three states, never two: **required-and-covered**, **required-and-uncovered**, and
**undeterminable**. The third is the one that gets lost. A regime axis reporting `0/0` because the
operand-store fact is absent reads exactly like a satisfied axis, and a capsule written against it
proves nothing.

## Why the corpus cannot be cloned between targets

The axes are shared; the **falsifier is not**, because it is a property of the archetype
(`dispatch × datapath_kind`), not of the workload.

| | gemmini · mx_gemmini | atlas | radiance · muon |
|---|---|---|---|
| dispatch × datapath | host_instruction × systolic | **device_native** × systolic | host_instruction × **SIMT** |
| hazards resolved by | hardware reservation station | **nothing** — the program orders | warp scheduler |
| tile edge | 16, hardware | 32, hardware | 16, **software default** |
| L3 oracle | verilator (`sim_via: chipyard`) | arc cosim (`sim_via: ""`) | cyclotron |
| **falsifier** | **η**: reordering is bit-exact, so correctness proves *nothing* about a schedule | **bit-exactness**: separation is real, a wrong order yields wrong data | **warp occupancy / divergence**: occupancy is lane-width, not a busy bit |

Consequence: an A/B on an interlocked target must be gated on η. Measured 2026-09-01 over four
schedules of provably identical work (`chk` equal across all four): `hoist_all` −0.0489 against
baseline, while `pipeline_1` (−0.0016) and `batch_2` (+0.0013) sat inside the noise. A
correctness-gated capsule would have passed all four and distinguished nothing.

## Measured state, 2026-09-01

Derived against the same 20 captures for every target.

| target | required cells | covered | capsules | notes |
|---|---|---|---|---|
| gemmini | 8 | **8/8** | 48 | composition 6/6, regimes 3/3 |
| atlas | 12 | **7/12** | 43 | `fits_single` unreached by any capsule though 9.5% of real regions land there |
| mx_gemmini | 10 | **5/10** | 17 | all five uncovered are contraction, including `mxfp4/6/8` at partial alignment |
| radiance | **56** | **19/56** | 44 | composition **1/5**, regimes **0/0 (vacuous)** — fans out because a SIMT target admits far more (family × dtype) pairs |

## Radiance is not a lower priority; it is a different shape

Measured: **19 of 56 cells covered, 37 uncovered; composition 1 of 5; memory regime 0/0 and vacuous**
(`operand store None rows`). 56 required cells against 44 capsules means the corpus cannot close the
requirement by construction.
That is not a reason to write 56 capsules. It is the signal that the **declared** capability surface
is wider than the **evidenced** one — radiance declares 8 semantic families and evidences 2. So the
order of work is:

1. **Decide which declared families are genuinely reachable** on this target's own datapath. A family
   that no compute unit can serve should not be admitted, and a requirement derived over it is noise.
2. **Check the observation set.** All four targets currently derive `observed` from the same 20
   captures (gemma2 / llama / spectformer) — systolic-friendly workloads. For a SIMT target that is
   probably the wrong observation set and will mis-shape the requirement in both directions.
3. **Only then** author capsules, against the cells that survive 1 and 2.

Radiance also has **0 `layers` capsules** (gemmini has 11), which is the direct cause of composition
1/5: without layer-scale capsules there is nothing to assemble the `A→A` / `H→A→H` / `A→H→A` shapes out
of, whatever the cell coverage.

Two derivations worth preserving rather than "fixing", both emitted by the checker itself:

* **attention and softmax appear in NO capture as a region of their own**, because the importer
  decomposes them. They are required on the evidence that every primitive they decompose into is
  observed. Taking the region census literally would drop attention from a transformer corpus.
* **the alignment axis uses tile edge 16, a SOFTWARE tiling default for this target, not a hardware
  boundary** — so `aligned` / `partial` here is a statement about the compiler's chosen tiling, not
  about a mesh. It must not be read as the same kind of fact gemmini's 16 is.

## Facts that must exist before capsules are worth writing

A requirement derived over an absent fact is vacuous, and vacuous axes are worse than missing ones
because they report as satisfied.

- `mx_gemmini`: `block_scale_group: null` — the MX target does not declare the block-scale fact the
  MX axis depends on. Radiance derives `32` from its own `mx_mmio.group`; mx_gemmini derives nothing.
- `atlas`, `mx_gemmini`: `operand_store_bytes: null` in the written spec, even though the live check
  derived `49152 rows` for atlas. The spec therefore cannot reproduce its own regime verdict.
- `radiance`, `mx_gemmini`: empty RTL facts blocks, so the `admitted` half of `admitted ∩ observed`
  is unfounded.

## Per-target gates that generalise

Both defects found on gemmini on 2026-09-01 apply to every target and should be checked per target:

- **`depends_on` adoption.** 0 of 48 gemmini capsules declare it, so every agent edit invalidates
  every certificate. The machinery is complete; only adoption is missing. Declare it *before* a cohort
  freeze — never mid-round, which changes what the agent is graded against.
- **Carrier consistency.** A capsule may not declare an epilogue the command-buffer schema cannot
  express. `GP0/GP1/GP2` declare `epilogue: ['maxpool']` against a schema carrying only
  `{bias_add, bias, requant, acc_scale, relu}`; `additionalProperties: True` then let an invented
  carrier validate silently. Ratcheted in
  `merlin/tests/targetgen/test_capsule_carrier_consistency.py`.

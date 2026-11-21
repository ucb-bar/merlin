# Recovered instruction observations in CCA

CCA retains the selected ISA observer's recovered operand fields in its
`instructions` record. Machine and emitted-IR comparisons consume this same typed
record. It carries class/role counts, selected decoder hashes, field samples,
population/sequence fingerprints, missingness, evidence identities and explicit
unknowns. JSON and CCA MLIR preserve the entire record.

These are static observations alongside semantic facets. They do not populate
optimization levers, count as additional semantic comparison axes, or establish
complete schedules. Repeated static sites are not dynamic invocation counts.
Equal opcode counts can conceal different geometry, addresses or scales; changed
field values can also reflect different mathematical inputs. Matching all expert
values is therefore not an optimization objective.

## Native author interface

The existing selected reference-analysis channel returns
`report.observation_comparison` for supported machine and emitted instruction
views. Its bounded response preserves actual scalar values and immutable detail
pointers. Value differences are prioritized over differences in site counts.
When the field population budget is exhausted, localized sequence differences
remain available with explicit partial recovery.

Use the original selected ISA/decoded field roster to request specific fields:

```sh
python agent_selfcheck.py --reference-analysis compare \
  --reference-path reference-kernels/expert.elf --reference-region REGION \
  --candidate-path artifacts/kernel.dis --candidate-symbol kernel_entry \
  --reference-observation-field INSTRUCTION.field
```

The field option can repeat up to eight times. An unobserved field returns an
explicit unknown; unsupported representation domains refuse field comparison.
The source, raw command buffer and actual emitted LLVM can also be pinned with
`--write-reference-views` and compared as `artifact_views`. Decoder input pins and
count bases must agree. Static machine sites and static IR sites cannot be
silently treated as the same population.

The copied client stays standard-library-only. Decoding/comparison remains in
the host analysis owner; this feature adds no source or filesystem grants and
does not bypass the profile's reference-analysis service gate. Raw-only profiles
continue to withhold CCA. Current compiler binding still requires the original
native producer record; copying a file to `artifacts/` establishes no such proof.

## Corpus and study boundary

The v1 corpus and selected banks remain frozen. Regenerated products use
`expert-kernel-database-v2` and `expert-kernel-subsets-v2` aliases under
`out/artifacts/kernel-mining/gemmini/`. Their original files, deduplicated body
identities and nested subset order are unchanged; the CCA records now retain the
decoded instruction evidence formerly available only in sidecars.

All six study conditions retain identical raw references. The CCA arms additionally
receive only their selected derived bank. The whole corpus is operator material:
do not mount it into any author workspace. Offline bank preparation is separate
from native subset enforcement and qualification of the actual common seed.

Reconstruction remains conditional on the pinned ISA/RTL definitions. Matching
upstream source/build records are unavailable for Jack's stripped ELFs. Neither
semantic equivalence, dynamic storage lifetimes nor physical performance follows
from these new observations. Hypotheses, scheduling choices and correctness and
performance checks still belong to the author and native evaluators.

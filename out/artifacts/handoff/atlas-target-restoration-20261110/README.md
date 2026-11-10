# Atlas target restoration on main

The restored Merlin `feat/atlas-target` starts at main commit
`d56133b168bbee3d73cb8b4304d3622d4191fc6a`. Its worktree is
`/home/jeremy/Projects/merlin-atlas-target`. This is a port of the earlier work,
not a merge of the old branch's unrelated parent history.

The previous branch names were deleted locally and remotely, but their commits
survived. The recovered Merlin changes were `cdb99f2931a5d88d7a40c9fb012c148f236c8996`
(numerics), `22d0fc0c8c0a20fb0bf515ba95b714d7122a35a4` (ordered floating dense),
`d31366d6fac639c1c463a0940349f97fe03d5308` (timing decisions), and
`e44289aea1ae953780d6b504c7debe4be0d28e80` (investigation).

The [original investigation and MX precision plan](../atlas-target-investigation-20261109/README.md)
is retained byte for byte as historical evidence. Its original baseline and
test totals describe the November 9 checkout. This note describes the restoration.
One correction to that report: Atlas's native Merlin binding was **not retired**.
Its import failure was an incompatible Merlin baseline; the paired
`feat/atlas-model-compilation` branches contain that integration. Those separate
`feat/atlas-shared-passes` worktrees and branches remain available.

## Scope and prerequisite ports

Current main lacked the compact dense planner/checker used by the old work.
The restoration reuses their self-contained implementation from
`525506143`, with its blocking and independent execution/mutation tests and
typed C index emission. It adds checked-kernel/index binding before emission.
It does not import the later selected-contract, bias/readout, pipeline-state,
or shared-library-export subsystems. Tests of those unported APIs are excluded
from the prerequisite test copy; all tests for the restored functionality run.

The restored ordered-floating reader and logical tile codec reuse main's
`Load`, `Preload`, `Compute`, `Store`, and counted kernel IR. Gathered loads are
explicitly refused by the codec. The timing checker uses a small shared
`memory_effects.EncodedAccess` contract instead of pulling in the unrelated
final-program segmentation implementation. `schedule.time_annotation` and
`schedule.overlap` retain their original issue/completion and resource checks.

FP4 E2M1 and FP6 E3M2 were already in main's scalar format registry. The
restoration adds FP6 E2M3 and the exact binary scaling fix, with exhaustive
code, signed-zero, round-to-nearest-even tie, saturation and sub-byte-storage
refusal tests across all three formats. These are scalar numerical codecs.
MX block scales, FP6 LUT indices, physical packing, and MX target compilation
remain implementation steps in the restored plan. Atlas execution here remains
the explicit FP8 E4M3 / ordered BF16 policy; this does not add FP4/FP6 hardware
execution to Atlas.

## Companion and verification

The Atlas companion `feat/atlas-target` is restored exactly at
`cf81623ce65c1a06f47834d765f404325ba3373b` in
`/home/jeremy/Projects/atlas-mlir-atlas-target`. Atlas has no `main` branch;
its original compiler baseline is `c9f50c3310a68d9eb8fa318e68492896a5a45737`.
The compatible execution model remains
`47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` in
`/home/jeremy/Projects/npu-model-atlas-target`. The selected RTL identity is
`0079c0541111197741a231c002e3843fa6f545b2`; no RTL execution was performed.

The restored branch passed:

- 166 focused Merlin tests covering numerical codecs, ordered source recognition,
  compact scheduling and its mutations, C execution, timing/dialect roundtrips,
  existing kernel emission, FP8 constraints and source-layout gates.
- 13 Atlas adapter tests, with Merlin imports pinned to this restored worktree.
- All 78 model cases: four matrix shapes, seven numerical witnesses, source-order
  and optimized compilation, direct Atlas references, ReLU, and a two-stage PACK
  chain retaining both outputs. Every case checks decoded/re-encoded words,
  numerical results and guard bytes.
- Repository structure checks, including module size and generated package docs.

The 32x32x32 shared programs are byte-identical to their direct Atlas references:
37 words / 3271 model cycles in source order, and 42 words / 3183 cycles optimized.
Full local results are in `out/build/atlas-restore/model/results.json`.
Other restored shapes reproduce the previous counts:

| M,K,N | Source-order words / cycles | Optimized words / cycles |
| --- | --- | --- |
| 32,64,32 | 53 / 4921 | 66 / 4712 |
| 64,32,64 | 125 / 11362 | 146 / 10122 |
| 64,64,32 | 90 / 8163 | 108 / 7344 |

The broader access-registry check has one bubblewrap subprocess failure,
`test_installed_grader_is_unreadable_in_real_bwrap`. It fails identically on
an isolated unmodified checkout of the main baseline; its log is
`out/build/atlas-restore/main-access-check.log`. No access-policy code was changed.

To repeat model checks, run from the Atlas worktree, using a fresh output directory:

```bash
PYTHONPATH=/home/jeremy/Projects/merlin-atlas-target/src:python:. \
  /home/jeremy/Projects/merlin/.venv/bin/python tools/check_shared_dense.py \
  --output-dir /home/jeremy/Projects/merlin-atlas-target/out/build/atlas-restore/model-rerun \
  --atlas-bin-dir /home/jeremy/Projects/atlas-mlir/build/bin \
  --model-root /home/jeremy/Projects/npu-model-atlas-target \
  --model-python /home/jeremy/Projects/npu_model/.venv/bin/python
```

This is the existing delegated Atlas lowering path. No OOT compiler generation
or optimization agent loop was run. Physical Atlas timing integration and the
OOT fusion contract remain follow-up work in the investigation.

# S1 div.Scalar port report

Branch: `s1`; worktree: `/scratch2/loren/aws/m27-fix/wt-s1`.
Base: `fda3ce0` — S1: seeded Codex m27 run final submission (56/57).
Commit: `ff6f921d59d068b7719e609ce2aa99b2e26a3ab7` — Port exact device scalar division onto S1.
Author and committer: Loren Hung <lorenhung@berkeley.edu>; no trailers.
The `s1` worktree is clean after committing.

## Implementation

Copied `div_scalar.py` byte for byte from `fa90f7baeadc9a3f52888e7d7a66dd1ef4da776f`.
Reapplied its two backend hooks and frontend recognition branch against S1.
The complete change is three files, 253 insertions and one deletion. No other
compiler module changes, including `elementwise.py`, `host.py`, and `device.py`.

The frontend recognizes a standalone parallel f32 input/literal division before
S1's generic floating host-island fallback. S1's `elementwise.py` has no competing
f32 div matcher, so it requires no edit. Integer map/reduction selection remains
earlier in the frontend and keeps the existing lowering. Folded divisor storage
avoids an unused allocation, and the new device command dispatches to the exact
quantized byte-predicate/MAC implementation. Recognition depends on typed SSA,
indexing maps, and arithmetic semantics, never capsule names or provenance.

## div.Scalar before and after

| Field | Before (S1) | After (s1-h1) |
| --- | --- | --- |
| result | wrong | lowered-on-device |
| ok | FAIL | PASS |
| numerical | fail | pass |
| host_compute | fail | pass |
| host_ops_per_element | 7.5 | 0.0 |
| host_value_ops_per_element | 0.5 | 0.0 |
| host_payload_ratio | 1.5 | 0.0 |
| trace_check | pass | pass |
| meets_full | NO | YES |

After: bit exact, maximum absolute error 0.0; offloaded; 3,338 accelerator
instructions retired; all full-definition checks pass. Quantized arithmetic uses
scale 2^-6, entry round-half-even, exact rational division and one saturated byte
readout, as in the donor implementation.

Baseline evidence:
- `/scratch2/loren/backend-verify-author/RESULT-193-m27-S1.md`
- `/scratch2/loren/backend-verify-author/result-193-m27-S1.csv`, row `064`
- `/scratch2/loren/backend-verify-author/runs193/m27-S1/f_064/{result,full}.json`

After evidence:
- `/scratch2/loren/aws/m27-fix/queue/done/s1-h1/RESULT-193-m27fix-s1-h1.md`
- `/scratch2/loren/aws/m27-fix/queue/done/s1-h1/result-193-m27fix-s1-h1.csv`, row `064`
- `/scratch2/loren/aws/m27-fix/queue/done/s1-h1/runs193/f_064/{result,full,cb}.json`

## Full h193 comparison

Queue request `s1-h1`, `KIND=h193`, completed `rc=0`.

| Group | Rows | Unchanged | Changed | Full before | Full after |
| --- | --- | --- | --- | --- | --- |
| DEVICE57 | 57 | 56 | 1 (div.Scalar) | 56 | 57 |
| HOST136 | 136 | 136 | 0 | n/a | n/a |
| All | 193 | 192 | 1 | | |

Compared every CSV column except `elf`, whose path identifies the run. All other
192 rows match exactly, including device instruction counts, float-capture
results, audit metrics and reasons, trace results, and full-definition status.
HOST136 retain all prior outcomes, including pre-existing declines/crashes.
Machine-readable comparison: `notes-s1/h193-comparison.json`.

## Corpus regrades

Both requests completed `rc=0`: `s1-div1` is 3/3 pass and `s1-reg1` is
15/15 pass. All 18 capsule statuses remain `pass`, all numerical reports have
zero mismatches, and L2 passes for every capsule. The three div capsules also
pass L0, L1, and trace. Numeric reports, complete trace reports, and tier statuses
match the original S1 capsule result JSONs exactly for all 18 cases.

| Capsule | Before | After | Numeric | Trace |
| --- | --- | --- | --- | --- |
| SY_reduction_scalar_div_i8_aligned | pass | pass | pass | pass |
| SY_reduction_scalar_div_i8_partial | pass | pass | pass | pass |
| SY_reduction_scalar_div_i8_sub_tile | pass | pass | pass | pass |
| SY_elementwise_map_i8_aligned | pass | pass | pass | pass |
| SY_elementwise_map_cast_widen_i8_sub_tile | pass | pass | pass | pass |
| SY_elementwise_map_relu_i8_partial | pass | pass | pass | pass |
| SY_elementwise_map_scalar_add_i8_sub_tile | pass | pass | pass | pass |
| SY_elementwise_map_scalar_mul_i8_partial | pass | pass | pass | pass |
| SY_elementwise_map_scalar_neg_i8_aligned | pass | pass | pass | pass |
| SY_elementwise_map_scalar_sub_i8_partial | pass | pass | pass | pass |
| SY_elementwise_map_tensor_sub_i8_sub_tile | pass | pass | pass | pass |
| SY_composition_scaled_operand_sum_q_f32_per_operand_sum_aligned | pass | pass | pass | fail |
| SY_composition_scaled_operand_sum_q_f32_per_operand_sum_partial | pass | pass | pass | fail |
| SY_composition_scaled_operand_sum_q_f32_tensor_k_tail | pass | pass | pass | fail |
| SY_composition_scaled_operand_sum_q_f32_tensor_multi_block | pass | pass | pass | fail |
| SY_reduction_sum_i8_partial | pass | pass | pass | pass |
| SY_reduction_cumsum_i8_sub_tile | pass | pass | pass | pass |
| SY_reduction_mean_all_small_i8_aligned | pass | pass | pass | pass |

Evidence roots:
- `/scratch2/loren/aws/m27-fix/queue/done/s1-div1/out/results/`
- `/scratch2/loren/aws/m27-fix/queue/done/s1-reg1/out/results/`
- Baseline: `/scratch/loren/eval/results/m27-S1-20261105T073453Z/results/`

Within each root, results are `runs-*/runs/*/<capsule>/capsule_result.json`.
`notes-s1/regrade-results.json` records every exact before/after JSON path and
comparison; `notes-s1/regrade-names.txt` records the selection.

The four composition capsules report overall `pass` but retain the baseline's
`trace_check=fail` diagnostic, `composition audit: 'composition_evidence'`, with
`evidence_status=UNKNOWN`. Their numeric reports pass exactly; L0/L1 are skipped
as in the baseline. Several other capsules likewise retain skipped L0/L1.
These are unchanged baseline details, not new failures. The capsule JSONs do not
publish a `host_compute` field; the separate h193 full audit supplies the host-work
verdicts above.

The first comparison script incorrectly required every capsule tier and trace
to pass, beyond the grader's overall status. Inspection of original S1 JSONs
confirmed the existing skips and four trace diagnostics; the script now checks
unchanged baseline detail as well as passing status, numeric, and L2. The final
comparison exits successfully; see `notes-s1/comparison.log`.

## Local validation and snapshot integrity

- Python compilation of all three changed modules passed.
- `check-lowering.log`: 14 shape/divisor forms, including rank 0, rank 1,
  sub-tile, partial-tile and multi-tile; every form emits the device division
  command and no host load/store/alloca/fdiv/fmul.
- `check-div.log`: 30,776 binary32 inputs across nine divisors, all entry
  midpoints and adjacent words, saturation, infinities, NaNs, bounded byte weights
  and exact sums, checked against an independent rational reference.
- `check-device.log`: simulated byte-clamping DMA and exact int32 MAC execution,
  774 boundary/nonfinite inputs across three divisors.
- `snapshot-sha256.json`: all 24 compiler package files match each of the three
  queue snapshots byte for byte. Sources have not changed since pickup.
- `compare_results.py`: reproducible baseline/result comparisons.

System Python lacked xDSL; local lowering checks used the existing
`/scratch/loren/venvs/gca/bin/python` environment. The external queue ID regex
initially excluded digits; the user fixed the runner before pickup. No request
was resubmitted and no grader or runner file was modified by this fixer.

## Unresolved

No requested validation remains pending and no compiler regression was found.
The four pre-existing composition trace diagnostics above remain unchanged; no
fix to those unrelated diagnostics was attempted.
The matcher retains the donor's scope: standalone f32 source-input division by a
finite nonzero, nonidentity literal. Compound float islands retain S1 behavior.

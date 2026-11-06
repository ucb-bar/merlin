# RESULT-10k int1-verilator

Engine: verilator. Package: `/scratch/loren/eval/cands/m27-int-5cb502c`.
Package SHA256: `1deabc7d687236a53bfa37917173a32572ee28d04d084c205331c6c4f963153d`.
Scope: `int`. Filters: `{"dtypes": null, "idx_file": null, "overloads": null}`.
Sample: None; canonical-only: False.

GATING: **466/466 pass (100.00%)**.
Selected/executed: 466/466; not run: 0 (no pass credit).
Failure classes: {'pass': 466}.
Elapsed: 2752.3s; cores 24-31; workers 4.

All selected cases remain in the denominator, including unavailable references.

| overload | pass/total | declined | crashed | wrong | no-device | reference_unavailable | timeout | not run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| _to_copy.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| add.Scalar | 15/15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| add.Tensor | 15/15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| addmm.default | 13/13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| alias.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| as_strided.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bmm.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cat.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clone.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| constant_pad_nd.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| convolution.default | 11/11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| copy.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cumsum.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagonal.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| div.Scalar | 11/11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| expand.default | 8/8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fill.Scalar | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| flip.default | 11/11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| full.default | 12/12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| full_like.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mm.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mul.Scalar | 14/14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| neg.default | 15/15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| permute.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reflection_pad1d.default | 8/8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reflection_pad2d.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reflection_pad3d.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| relu.default | 6/6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| repeat.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replication_pad2d.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replication_pad3d.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scalar_tensor.default | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| select.int | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| select_scatter.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slice.Tensor | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slice_scatter.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| split_with_sizes.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| squeeze.dim | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| squeeze.dims | 11/11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sub.Scalar | 13/13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sub.Tensor | 15/15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sum.dim_IntList | 16/16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unsqueeze.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| view.default | 10/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Full selected spec_int8 reference audit: {"available": 860}.
Historical Merlin output-byte agreement: {"total_with_recorded_result_digest": 860, "checked": 0, "agree": 0, "disagree": 0, "uncheckable": 860}.
The recorded result_sha256 hashes runner JSON; missing historical shards prevent a byte-agreement claim. See reference-agreement.json.

Failure details per overload (decline/crash reasons grouped by recorded cause).

ULP is the maximum distance between representable storage values; complex errors use components. ULP includes finite pairs only; NaN/infinity mismatches are counted separately. Integer ULP is not applicable. Dtype/shape/count failures have no numeric distance.

| overload | class | count | grouped reason / maximum error |
|---|---|---:|---|

Report-only counts (never executed or gated): {"DEVICE57/host/bounded": 225, "HOST136/device/bounded": 453, "HOST136/host/bounded": 6788, "HOST136/host/corpus_193": 136}

Limits: numeric outputs and RoCC retirement only. Mutation, aliasing, full metadata, host-compute veto, and full grader trace semantics are not certified.

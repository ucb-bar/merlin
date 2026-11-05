# Frozen 193-overload evaluation harness

This is the 2026-11-04 grading instrument for the frozen Merlin branch
`mlsys27/test-generation-coverage`, base `69aa427c525c58351cbee5dd16cca677d4507f40`.
It contains the scripts and every case input needed to evaluate a candidate without
reading either live `backend-verify-author` directory. External compiler tools and
the prepared Merlin release environment are still required; they are described below.
This instrument is evaluator material, never an input to the candidate author.

A **57/57 supported** claim means all 57 DEVICE57 rows have `meets_full=YES`
in `result-193-<label>.csv` with `FULL=1`. FULL is the conjunction of:

- numerical: correct output on Spike, with retired accelerator instructions;
- host_compute: the grader's host-compute audit passes;
- trace: the grader's instruction-trace check passes, with its requirements available.

The other 136 overloads (HOST136) must execute correctly with zero accelerator
opcodes, both static and retired. Float host outputs use `host_f32` tolerance
(`1e-5 + 1e-4 * abs(reference)`); integer/bool outputs are exact. Uninitialized
`empty`/`empty_strided` cases use the declared metadata comparison. Host-compute
and trace are n/a for HOST136. The canonical f32 capture is also evaluated for each
device overload, but its result is a secondary column, not the device headline.
This snapshot does not assert that an arbitrary candidate achieves 57/57; no full
grade was run while preparing these commits.

## Lanes and the 2026-11-04 decisions

`canon193.json` fixes the roster, classification, selected slug and lane. Its 57
selected device forms comprise:

| lane | overloads | graded input and reference |
|---|---:|---|
| DEVICE_FORMS | 32 | Registry `corpus_spec.build` integer form and exact integer reference. |
| window_sum | 11 | Registry reduce/pool/window form and exact integer reference. |
| widened | 4 | Registered widened integer linalg form and exact reference. |
| SPEC_FORM | 9 | Spec form-table builder at the aligned cell (M=16, K=32, N=16), canonical leaves and exact integer golden. |
| SPEC_INT8 | 1 | `div.Scalar` f32 capture, exact comparison against the declared quantized model. |

The ten former capture-lane ops are graded on their spec-declared forms: cumsum,
diagonal, select, split_with_sizes, convolution_backward, upsample_nearest2d,
full, full_like, scalar_tensor and div.Scalar. The first nine use SPEC_FORM;
div.Scalar uses SPEC_INT8. Every selected device form is invoked with
`tools/run_case.py --policy exact`. In particular, div.Scalar does **not** use
run_case.py's looser policy named `spec_int8`.

A program whose persisted command buffer has census outcome `silent` fails FULL,
even if the host-compute veto does not fire. Strict FULL requires host_compute
`pass`; the separately reported `meets_full_declared_host_ok` is a diagnostic
lenient count and does not establish 57/57. The trace check decodes the emitted
LLVM MLIR, not Spike's instruction log; `capsule_meta.json` provides the capsule's
`expected` requirements.

## Scope against the canonical PyTorch calls

The following table is copied from the 2026-11-04 decision note
`h193-spec-forms.md` (original location:
`/tmp/claude-2629/-scratch2-loren-aws/ff436844-5087-49c7-b6b0-e2ab69f3dcac/scratchpad/m27/`).
These are claims about the selected forms, not all shapes, axes, layouts, dtypes,
or options of each canonical PyTorch overload.

| op | spec row / form (builder op) | graded case | narrows the claim vs the canonical call |
|---|---|---|---|
| cumsum.default | `reduction_prefix_sum` / `cumsum` (prefix_sum, component_program) | inclusive scan down axis 0 of i8 [16,16], **raw i32 exit** | **Yes, two ways.** (1) The axis is the outer one only; the canonical call is `cumsum(x, dim=1)`, the last axis. (2) The exit width is i32, but torch returns **i64** for an int8 input. An inner-axis scan or a widened exit is not graded. |
| diagonal.default | `movement` / `diagonal` (as_strided, sizes [16], strides [17]) | main diagonal of a square i8 [16,16] | Minor. The canonical call is offset 0 on a non-square [3,4]. The graded case is square with offset 0. Other offsets and other dim pairs are not graded. |
| select.int | `movement` / `select` (movement_chain: slice rows [8:9], then a reshape that drops the dim) | rank-2 [16,16], selecting **dim 0**, index 8 | **Yes.** The canonical call is `select(x[2,3,4], dim=1, index=1)`, which is rank 3 on an inner dim. The graded case is an outer-dim select on rank 2. |
| split_with_sizes.default | `movement` / `split_sizes` (movement_chain of slices) | [16,16] split into [3,5,8] **along dim 0**, three results | **Yes.** The canonical call is `split_with_sizes(x[2,5], [2,3], dim=1)`, an inner-dim split with two results. The graded case is an outer-dim split. |
| convolution_backward.default | contraction / `convolution_backward_spatial` (conv_backward_spatial; the fixed canonical 3x3, stride 1, pad 1 case) | inputs Xcol [16,36] (**already im2col'd**), dY [16,16], W [36,16]. Outputs grad_weight i32 [36,16], grad_input i32 [1,4,4,4], grad_bias i32 [1,16] | **Yes.** (1) The input arrives as im2col columns, not NCHW. (2) The grad_input columns pass through a declared accumulator readout scale (2^-11 to i8) before the exact fold. The canonical call is an integer composite with exact i32 outputs. (3) Only the spatial form is graded; the pointwise form `convolution_backward` (conv_backward) is also declared in the spec but not graded. |
| upsample_nearest2d.vec | `movement` / `upsample_nearest` (movement, scales [2,2,1]) | **channels-last** i8 [16,2,16] -> [32,4,16] | **Yes.** The canonical call is f32 NCHW [1,1,3,3] -> [1,1,6,6], which resamples the contiguous axis. The spec row explicitly does not claim that case ("an NCHW tensor ... needs a layout permutation ... which this row does not claim"). Only the integer channels-last form is graded. |
| full.default | `movement` / `fill` (fill, code 5) | i8 [16,16], no input | Minor. The canonical call is `full([2,3], 3, int8)`. Same semantics; a different value and shape. |
| full_like.default | `movement` / `fill` | the same fill program as full | **Yes, slightly.** There is no "like" tensor operand: the shape is static. The graded case is identical to full.default. |
| scalar_tensor.default | `movement` / `fill_scalar` (fill, rank 0, code 5) | rank-0 i8 | None of substance. The canonical call is `scalar_tensor(3, int8)`. |
| div.Scalar | `reduction_scalar_div` + `quantized_native_fallback` / spec_int8 (`tools/gen_spec_int8_form.py`) | **unchanged program**: the f32 capture `divf x, 2.0` on [2,3]. The reference is the declared quantized model (s = 2^-6, round-half-even entry, exact rational body, one round-half-even byte readout), compared exactly | **Yes.** The result is the spec's quantized model, not IEEE f32: e.g. -0.171875 where torch gives -0.175. Only this divisor (2.0) is graded. |

The other 47 device ops are unchanged: same case, lane and reference. The cumsum, convolution_backward and
upsample_nearest2d rows are the ones where the spec claim is materially narrower than the canonical call.


## Included files and dependency boundary

- `run_193.sh` stages the instrument and candidate, runs numerical and FULL checks,
  pulls results, and aggregates them. The case runner is **`tools/run_case.py`**
  (the live source has no top-level `run_case.py`).
- `tools/drive193.sh`, `bva_lib.py`, `fullcheck_drive.sh`, `ns_full.sh`,
  `fullcheck.py` and `aggregate193.py` implement that path. The five files in
  `tools/rt/` are the C/assembly/linker runtime inputs read during each build.
- `canon193/` contains four files for each of 193 canonical captures:
  `capsule.linalg.mlir`, `inputs.json`, `golden.json`, `request.json`.
- `forms93/` contains those four inputs plus `capsule_meta.json` for each of the
  57 selected forms, including all nine `spec_*` directories and `div_Scalar`.
  `forms.json` retains the original 93-row catalogue; only the 57 slugs selected
  by `canon193.json` are required for this evaluation. `spec_forms.json` describes
  the nine spec-builder forms.
- `controls/` contains positive movement and deliberately corrupted-reference
  controls, four inputs each. `cases.json` and `overloads93.json` preserve the
  source catalogues; normal grading does not read them.
- Four form generators and `select_193.py` are retained as construction provenance.
  Normal grading consumes the committed inputs and never invokes a generator.

Excluded: `*.bak*`, `.pre-*` versions, `runs193/`, `runs/`, `runs-forms/`,
`idx-*.jsonl`, result CSV/Markdown and logs; unused `cases/` and `variants/`
(including `variants.json`); the 36 unselected form rows' case directories,
superseded capture forms, and `forms93/_cap/` capture intermediates.
`drive193.sh` reads none of these. The original `cases/` numbering is a separate
catalogue from `canon193/`; its required canonical inputs are already materialized
in `canon193/`. The variants belong to earlier evaluation lanes and are never
selected by this driver. `capture-work/meta.json` is not read by run_case.py.
Unrelated 93-form/variant drivers, aggregate tools and unused runtime sources are
also omitted. No ELF, object, or generated run `kernel.ll` is an input, and none
is committed. All required form inputs fit comfortably below 40 MB, so grading
requires no generated-input download or regeneration step.

## Paths and prerequisites

Defaults preserve the garden setup:

| setting | default | purpose |
|---|---|---|
| `H193_H`, falling back to `H` | `/scratch2/loren/backend-verify-author` | Instrument inputs, pulled run data and aggregate outputs. |
| `R` | `/scratch/loren/backend-verify-author` | Runtime staging, candidate copies, build/run outputs and FULL namespace bind. |
| `H193_TAG` | `m27` | Prepared release under `$P12/stage/<tag>/{root,merlin}`. |
| `H193_SSH` | `ssh -F /scratch2/loren/aws/ssh/config -o BatchMode=yes merlin-compute-1` | Transport; use `bash -c` locally. |
| `CPUS` | `0-23` | Runtime affinity; choose available cores. |
| `H193_PY` | `/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python3` | Case-driver Python (NumPy, PyYAML, candidate dependencies). |
| `P12` | `/scratch/loren/p12-unsealed` | Prepared namespace and its `env.sh`. |
| `P12_SEAL_ROOT` | `/scratch/loren/megaphase-core-aten/seal4` | Same-path mount expected by the prepared release. |

`H193_H` takes precedence because the external eval wrapper uses `H` for its own
tooling directory. `R` and `H193_TAG` are explicitly passed to the remote FULL
stage. Other toolchain overrides should be configured in the runtime host's
environment (local mode inherits them): `MERLIN_CLANG_INSTALL`, `RV_GCC`, `SPIKE`,
`GEMMINI_EXTLIB`, `F2B_RT`, `F2B_PKG_PYBIN`, `F2B_PKG_PYTHONPATH` (see
`tools/bva_lib.py`). `M2M_PY` and `M2M_DIR` configure the optional capture generator.
The FULL stage sources the prepared release's `env.sh`, which supplies `$PY`
and the frozen Merlin grader imports. It requires bubblewrap and taskset.
Use the grader at the base revision above to preserve the grading definition.

Use absolute paths without whitespace, a simple unique label, and distinct source
and runtime roots. Keep `R` outside `/tmp`: the FULL namespace mounts a fresh
tmpfs at `/tmp`. Never point a reproduction run at either active live harness
root: the driver replaces `runs193/<label>` and stages tools in `R`.

Some environment-specific paths remain: `ns_full.sh` binds facts at the recorded
`/scratch/loren/s4facts` path and the user's `$HOME/.codex` directory; its source
roots have `P12_C_ROOT` and `P12_FACTS` overrides. `bva_lib.py` retains garden
compiler/Spike/extlib defaults with the overrides above. SSH config and agent
socket defaults remain configurable via `H193_SSH` and `SSH_AUTH_SOCK`.
`select_193.py` retains the historical `/scratch2/loren/aws/tooling/exercised`,
`/scratch2/loren/backend-verify/results-all.csv`, SSH config, and shard-host paths.
That archival selector needs excluded shard indexes and is unnecessary with the
committed `canon193.json`; do not delete that index and invoke its fallback.
Absolute paths inside copied JSON are provenance strings and are not opened by
the grading path. The CSV `elf` locator retains its historical `compute-1:`
prefix even in local mode; its directory now follows `R`.

## Run with the existing evaluation wrapper

From a checkout of this branch, set the candidate to a frozen package directory
containing `manifest.yaml` and `compiler.py`. This local, harness-only example uses
`eval_candidate.sh` from `/scratch2/loren/aws/tooling/eval`:

```bash
export REPO=/scratch2/loren/m27-h193
export BV="$REPO/experiments/mlsys27/h193"
export H193_H="$BV"
export R=/scratch/loren/h193-repro-$USER
export H193_TAG=m27
export CANDIDATE=/absolute/path/to/frozen-candidate

SSHC="bash -c" H193_SSH="bash -c" \
  NOREGRADE=1 CPUS_REG=0 CPUS_H=0-7 FULLW=4 P193=4 \
  nice -n 10 bash /scratch2/loren/aws/tooling/eval/eval_candidate.sh \
  "$CANDIDATE" h193-repro-20261104 fin6yn
```

Set `REPO` to the actual worktree location. `NOREGRADE=1` skips native corpus
regrading, and explicit `CPUS_REG`/`CPUS_H` skip idle-core sampling. The wrapper
always checks for an existing `/scratch/loren/eval/templates/fin6yn/COPIED`, even
with NOREGRADE; use an already registered local template. Its bootstrap may inspect
run-private inputs, so use the direct command below if the template is absent.
The wrapper also writes candidate staging and results under `/scratch/loren/eval`
and `/scratch2/loren/aws/tooling/eval/results/<label>`. For native corpus regrading
as well, omit NOREGRADE, give the regrade and harness disjoint CPU lists, and use
the wrapper's documented corpus/template setup. `SSHC` controls that wrapper;
`H193_SSH` independently controls this harness's transport.

For this 193-overload harness alone, with no wrapper/template dependency:

```bash
H193_SSH="bash -c" FULL=1 CPUS=0-7 FULLW=4 \
  nice -n 10 bash "$BV/run_193.sh" "$CANDIDATE" h193-repro-direct 4
```

Both commands use the exported `H193_H`, `R` and `H193_TAG`. In remote mode leave
`H193_SSH` unset (and leave SSHC unset for the wrapper); ensure the runtime host
has the same prepared release, dependencies and source-path mounts.

Inspect `result-193-<label>.csv` and `RESULT-193-<label>.md` under `BV`, especially
`meets_full`, `numerical`, `host_compute`, `trace_check`, and both control outcomes.
`runs193/<label>/PACKAGE_ID` identifies the candidate file set. The runtime ELF
files remain under `R/runs193/<label>`; source-side pulls omit ELF, object and
`kernel.ll` products. Reuse of a label overwrites that label's run directory.

## Construction provenance and integrity

The initial snapshot commit contains all 1,088 mirrored files byte-for-byte from
`/scratch2/loren/backend-verify-author`, whose symlink target is
`/scratch2/loren/prob-merlin-stuff/backend-verify-author`. All 1,085 of those files
present in `/scratch/loren/backend-verify-author` also matched byte-for-byte.
The runtime copy lacks only `run_193.sh`, `cases.json` and `overloads93.json`.
The subsequent commit changes configuration/path plumbing, adds this README and
output ignores, and refreshes the manifest. `tools/run_case.py`, `bva_lib.py`,
`fullcheck.py`, and all grading inputs remain byte-identical to the source.

`MANIFEST.sha256` contains three commented sections: hashes of every current
committed harness file; original live-author hashes for all mirrored files; and
live-runtime hashes where that file was present. The manifest excludes itself,
since a file cannot contain its own stable hash. Verify locally without the live
sources by selecting the first section, or verify all sections on garden:

```bash
cd "$BV"
awk '/^# Live author source/ {exit} /^[0-9a-f]/ {print}' MANIFEST.sha256 | sha256sum -c -
sha256sum --quiet -c MANIFEST.sha256  # also checks the two live, read-only snapshots
```

Configuration edits are deliberately distinguishable from original-source hashes;
compare the two commits to audit them. The manifest records a snapshot: later
changes to a running live harness may cause its absolute-path checks to fail.

The committed inputs are authoritative. The retained generators show construction;
regeneration is optional and should target a separate writable copy. With that
copy as `H193_H` and the frozen Merlin root as `REPO`, the spec forms are built by:

```bash
export MERLIN_REPO_ROOT="$REPO"
export PYTHONPATH="$REPO/src:$REPO/packages/merlin-experiments/src${PYTHONPATH:+:$PYTHONPATH}"
nice -n 10 "$H193_PY" "$H193_H/tools/gen_spec_forms.py" "$REPO"
nice -n 10 "$H193_PY" "$H193_H/tools/gen_spec_int8_form.py" "$REPO" div_Scalar
```

Set `H193_PY` explicitly to the environment's Python before these commands; add
other `packages/*/src` directories if required by that environment. The nine spec
forms use `corpus_form_table.FORM_TABLE`, canonical leaves, and
`capsule_golden._recompute_golden`; component programs also bind the spec's
`numerical_semantics` and Phase 0 integer bound. div.Scalar uses the exact rational
quantized model at scale 2^-6 with round-half-even entry and a single byte readout.
Its regenerated request embeds the selected REPO path as provenance.

Historical base generation was `gen_forms.py OPCAPS_ENV_ROOT OUTDIR OVERLOADS.json`
followed by `gen_expected.py FORMS_DIR` in the opcaps environment. Rebuilding that
historical catalogue additionally requires model2MLIR and capture intermediates
not needed for grading. Do not run gen_expected.py across the final SPEC_FORM or
SPEC_INT8 rows: they have no registry capsule. Do not overwrite the final
`forms.json` or `canon193.json` with gen_forms.py/select_193.py output: restore the
committed indexes to retain the selected spec lanes. The two spec generators write
the case data/metadata but do not patch those indexes.

Preparation validation: all four shell scripts passed `bash -n`; all nine Python
scripts passed `python3 -m py_compile` with its cache redirected outside the
checkout. Every JSON input parsed, all roster-selected files were present, and
source/runtime SHA-256 checks passed. The aggregation AST matched the original
apart from root configuration and the ELF display path. Synthetic aggregation
exercised silent census, declared-host, unavailable trace requirements, wrong
numerical output and host-compute vetoes. No candidate compilation, Spike run or
full grade was performed.

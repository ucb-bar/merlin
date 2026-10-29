# MACC fix and remaining pass integration — 2026-10-29

Continue on **`feat/pass-verif-integration`**. The MACC freshness fix is
`ae144d959` (`fix(verify): require current audit-only proof objects`), based on
the teammate's Q/DQ integration through `b0f3f8f1d`. This handoff supersedes
earlier statements that Q/DQ has no compiler selector. It does not approve any
numerical policy or source specification.

## What was implemented

The MACC gate could audit stale proof objects while binding newer proof sources.
`CheckerTools` checked freshness of `merlin-lean` and `merlin-lean-audit`, but
an audit-only module such as `MerlinVerify.Rules.Bf16F32AccFusion` belongs to
neither executable's dependency tree. Missing objects caused late `unavailable`
failures; existing stale objects could produce apparently current evidence.

The fix in [pass_validation.py](../../../../src/merlin/verify/pass_validation.py)
requires **`MerlinVerify` plus both executables** to be current. It retains
`--no-build`: validation refuses rather than silently rebuilding anything.
The [new regression](../../../../merlin/tests/ir/test_pass_gate_audit_freshness.py)
builds real minimal Lean projects and checks current, stale-source and
missing-object cases. The stale/missing cases fail on the old implementation
and pass with the fix. The [gate guide](../../../../docs/guides/lean_pass_gate.md)
documents the complete-library prerequisite.

The soundness check reproduced acceptance of an actual f16 boundary after adding
an unsafe helper to an audit-only source without rebuilding its object. A fresh
private compilation made the auditor reject that helper; the repaired admission
refused before a MACC report was issued. This demonstrates an evidence-binding
defect, **not a disproof of the MACC theorem or an invalid rewrite accepted by
the intended mathematical checker**.

No production transformation, Lean theorem, frozen semantics, `SPEC_LOCK`,
admission record or approval flag was changed by this fix.

## Verification and evidence cuts

- Before the teammate's latest Q/DQ commits: **117 MACC-related tests passed,
  zero skips; two retained SmolVLA cases explicitly deselected**. Four actual
  bf16/f16 matmul/batched-matmul boundaries passed, followed by four fresh
  separate-process replays. These results bind the earlier `14cd21711` plus fix.
- Combined branch at `ae144d959`: **119 MACC-related tests passed, zero skips,
  two retained SmolVLA cases deselected; 22 Q/DQ integration controls passed,
  zero skips**. The two extra MACC cases cover canonical dispatch refusal for
  the newly registered Q/DQ worker. Four fresh MACC boundaries and four separate
  process replays passed on this combined source cut; older receipts are not
  relabeled. The handoff-only documentation commit changes no tested gate source.
- The full Lean/lock audit passed: **567 source modules; 550 shared modules /
  15,304 shared theorems; zero violations**, plus separate driver audits.
  Neither this fix nor the incoming Q/DQ commits changes its Lean, lock or guard
  inputs. Driver/import counts overlap and must not be summed as unique coverage.
- The original freshness regression run gives **two expected failures on old
  code, one positive pass**; all three cases pass with the fix.

Raw logs, original failed attempts, the unsafe-helper counterexample, private
rejecting control and initial source inventories remain local under
`out/artifacts/handoff/soundness-review-20261029/`. Postmerge logs and receipts
remain local alongside this README. Only this handoff Markdown is published
from these evidence directories; do not assume ignored SDKs/logs/receipts exist
on another host. Reproduction commands are below.

## Current integration inventory

The authoritative registry is
[`pass_validation.CHECKS` / `DEFAULT_CHECKS`](../../../../src/merlin/verify/pass_validation.py).
Transformation selection and validation selection remain separate.

| Pass/family | Compiler integration now | Established relation and remaining work |
| --- | --- | --- |
| Tensor DCE: `prune_dead_pure_tensor_ops` | In `all`; actual xDSL preprocessing boundary | Typed tensor refinement with source support. This is DCE, not general dead-store elimination. |
| Leading tensor `cse` | In `all`; actual native runner boundary | Same tensor refinement. This does not cover every downstream CSE invocation. |
| Ordered loop unrolling | In `all`; optional source-SCF transformation | All defined entries preserve modeled returns/errors/terminal world. Current compiler profile accepts 0–16 inputs, 1–16 word carries and multiple changed definitions. Broader loop modes need proof bridges. |
| Ordered loop segmentation | In `all`; source-SCF and private-buffer routes | Scalar route has return/error/world equality; memory route has the complete chain's successful-observation relation. These are different claims. |
| Private buffer reuse | In `all`; normalization → reuse → segmentation chain | Equal possible successful public observations under `total_private_unobservable` allocation. Identity normalization and one-to-one reuse prefixes are supported. Untiled memory and broader allocator semantics remain separate. |
| MACC: `bf16_f32acc_draft_contract` | Explicit-only; actual pass and optional compiler-entry invocation | Output equals a source-derived draft f32-accumulation reference, plus strict nonempty site qualification. Independent numerical-policy review remains pending. |
| Q/DQ: `lower_quant_ext` | **Explicit-only, now integrated** on xDSL and textual routes | Typed refinement under the proposed Q/DQ model; optional fresh original-source binding. Source approval, framework equivalence and native fidelity remain separate. |
| Configuration-write elision | Direct `prepare/check/deliver/replay` API; no registered compiler selector | Conditional complete-entry state/error equality. Needs a production caller that captures actual instruction-MLIR boundaries and consumes the live completion. |
| Scoped configuration hoist | Direct API; no registered compiler selector | Conditional complete Kernel state/error equality for one supported positive counted-loop hoist. Needs production Kernel boundary wiring. |
| Exact integer simplification, scalar redundancy elimination, scalar invariant motion | Larger source profiles exist; **public activation is held** | `lean_graph_pipeline_gate_profiles.public_check` refuses and `compiler_eligible` is false. Need reviewed provider/operational bridges and compiler wiring, not just selector registration. |
| Other tensor/local rulebook rewrites | Rules/checkers and composition are not blanket pass registration | Identify a real production pass, prove its complete actual endpoints in the supported fragment, then wire a canonical owned worker. |
| W8A8 numerical policy; upstream MLIR/LLVM/hardware | Separate work | No approval or end-to-end correctness follows from these compiler gates. |

The five checks in `all` remain DCE, CSE, unrolling, segmentation and buffer
reuse. Both MACC and Q/DQ require explicit selection. Merely enabling a check
does not enable a transformation; unchanged stages submit no positive proof job.

MACC cites `MerlinVerify.Check.Bf16F32AccBoundary.check_sound`: output/reference
denotation equality under `ParametricAlpha.Laws`, nonempty source-site coverage
and strict local qualification. Local qualification does not establish an
initialized successful whole-program execution. Its numerical verdict remains
`abstracted`, with `policy_reviewed=false` and no original-source equivalence.

## What the next owner should do

### 1. Wire the two configuration APIs into their real compiler callers

Use the existing APIs rather than replacing their proven model:

- [Elision guide](../../../../docs/guides/lean_configuration_elision_gate.md),
  `configuration_elision_{intake,gate}.py`: capture complete actual instruction
  MLIR before/after redundant-write removal. Bind the original ISA/effect facts
  and software selection. The reference remains
  `4499eebb959c0f92cefd92cc6d80646ba66474ed`.
- [Hoist guide](../../../../docs/guides/lean_configuration_hoist_gate.md),
  `configuration_hoist_{intake,gate}.py`: capture complete actual Kernel text
  before/after the hoist. Preserve the detached
  `e0484f4886ae0780d0325cdfa998842b61bb647c` reference and its required source
  manifest; use `MERLIN_HOIST_REFERENCE` for the selected read-only checkout.

Add explicit validation selection at the layer that owns these programs, and
hold returned products until `check` and live `deliver` succeed. A status string
or saved receipt is insufficient. Keep redundant setter purity/totality,
nonaliasing configuration fields, body-frame and complete observer-state
premises visible. Test retained failing calls as well as successful execution.
Hoist zero-trip/nested/IV-dependent/completion-token/asynchronous cases and body
writes to setup fields remain outside the supported domain.

If current compiler output differs from the historical producer profile, add a
scoped current-compiler intake. Preserve the old standalone profile and pins;
do not relabel historical captures as current compiler evidence.

### 2. Activate scalar/integer profiles only after the missing bridge is reviewed

Start at
[`lean_graph_pipeline_gate_profiles.py`](../../../../src/merlin/verify/lean_graph_pipeline_gate_profiles.py),
`lean_graph_pipeline_gate_profile_prepare.py` and
`lean_graph_pipeline_gate_profile_requests.py`. The four-transform scalar profile
and integer-prefix profile are descriptive source preparations with held public
activation; they do not issue compiler completion authority.

Implement a scoped operational provider that checks the generated complete
program, audits exact roots/types/imports and binds the selected tools, sources,
objects and actual endpoints through live delivery. Respect canonical word-entry
requirements in the integer profile and its narrower historical geometry.
Then extend `compiler_transforms` and canonical dispatch for the exact supported
production transformations. `compiler_loop_intake` currently admits unrolling
and segmentation; it is not a generic proof handler for folding/CSE/LICM.
Keep compiler validation at index64: a source-only index32 preparation does not
resolve the retained native/model mismatch in ledger S6.

### 3. Extend existing loop/buffer domains separately from wiring

Unroll tails, empty/fully inlined loops, arbitrary nesting, internal calls and
untiled memory initializers require corresponding proof/intake extensions.
Standalone shape normalization is not currently an independently registered
compiler selector. Do not count an identity prefix as positive rewrite coverage,
manufacture tile-one evidence, or reinterpret buffer success observations as
full error/world or finite-capacity allocator equality.

### 4. Keep Q/DQ follow-up aligned with the newly merged implementation

Read the [Q/DQ gate guide](../../../../docs/guides/lean_qdq_gate.md) and
[qualification handoff](../quantization-lean-qualification-20261029/README.md).
The actual xDSL pass is observed; textual validation captures the complete outer
preprocessing result, including wrapper edits. A dedicated label handles textual
per-channel dequantization. Do not add duplicate nested observations.

The optional `SourceCapture` path reopens original PT2, complete graph arguments
and returns, producer bytes, Torch wheel identity and exact boundary input both
before and after the Lean check. Its original-program source check is distinct
from native numerical replay and from source-spec approval. Source intake still
rejects lossy scalar materialization, unsupported overloads/TorchAO paths and
unbound historical captures. Freshly qualify captures after bound sources change.
Provision exact pinned environments on a new machine using the qualification
handoff; machine-local `out/build` paths are not distributed by Git.

### 5. Finish review and qualification without broadening claims

MACC independent contract/admission review, Q/DQ source-model review and quantizer
definedness remain open. Keep them distinct from compiler wiring. Tensor
refinement, modeled loop equality, buffer successful observations and numerical
draft conformance must retain their different report claims.

The broader legacy regression is incomplete. The latest interrupted broad run
retained three tensor-gate source-support failures; older full-suite counts
belong to their own source cuts. Repository structure/dependency gates also have
failures in unchanged files. Preserve these results; neither the targeted MACC
pass nor this handoff claims an all-green repository or full-model campaign.

Publication checks pass for docs, formatting, artifact layout, local paths and
diff whitespace. The manual pre-commit run stops at two unchanged overfit-register
findings (`phase1/task_staging.py`, `phase2/owned_model_observation.py`). The
provenance scan reports two older retained reports without provenance and notes
its 20,000-entry scan cap. These flagged files match the remote baseline; no
ratchets, approval flags or gate expectations were changed to publish this fix.

## Acceptance criteria for any newly integrated pass

1. Capture the real complete before/after program at the owning compiler layer,
   including enclosing edits, all functions, declarations and exact stage joins.
2. Choose the theorem, input domain and relation explicitly. Require typing,
   source support or definedness only as actually established; report missing
   strengthenings and assumptions separately.
3. Use canonical owned workers and fresh kernel/axiom/type/import audits; refuse
   stale/missing objects, tool/source mutations, forged/copied completions and
   custom handler rerouting. Saved JSON grants no live authority.
4. Hold compiler products until selected checks accept; on failure retain evidence
   and verify no final LLVM/object/shared-library product is returned.
5. Exercise CLI, environment and typed API selection, identity/mixed stages,
   mutations in another function, bad real producer outputs, and scope refusals.
   Compare gated/ungated outputs and run meaningful independent native references.
6. Replay in a separate process from the exact tested sources. Preserve failed
   attempts and explicit skipped/unavailable/deselected checks. Do not alter
   specifications, locks or approval flags to obtain green results.

## Reproduce the relevant checks

Select a real Lean 4.35.0 SDK `bin` directory (not an Elan wrapper relying on a
global default), the project's pinned Mathlib dependencies and the compiler
Python containing `torch_mlir`. For example, if those local installs exist:

```sh
export MERLIN_LEAN_SDK_BIN="$PWD/out/build/lean/toolchain/lean-4.35.0-linux/bin"
export MERLIN_COMPILER_PYTHON="$PWD/out/build/lean/mlir-venv/bin/python"
export MERLIN_CLANG="$MERLIN_LEAN_SDK_BIN/clang"
export MERLIN_LEAN_REQUIRED=1
export PATH="$MERLIN_LEAN_SDK_BIN:$PATH"
(
  cd merlin/lean
  lake build MerlinVerify merlin-lean merlin-lean-audit
)
.venv/bin/python -m pytest -q -ra \
  merlin/tests/ir/test_pass_gate_audit_freshness.py \
  merlin/tests/ir/test_macc_root_audit.py \
  merlin/tests/ir/test_compiler_macc.py \
  merlin/tests/ir/test_compiler_macc_wiring.py \
  merlin/tests/ir/test_bf16_installed_owner.py \
  merlin/tests/ir/test_lean_bf16_policy_gate.py \
  merlin/tests/ir/test_lean_bf16_f32acc_native.py \
  merlin/tests/ir/test_compiler_qdq.py \
  merlin/tests/ir/test_compiler_qdq_source.py \
  -k 'not retained_smolvla'
.venv/bin/python build_tools/scripts/check_lean_guard.py
```

The two excluded SmolVLA native cases depend on retained historical artifacts;
run and report them separately when those artifacts are available. The five
Q/DQ source-reconstruction host controls substitute frontends; they are not a
new pinned-Torch capture campaign. Run the capture qualification driver for that
separate guarantee. Do not silently skip missing Lean/MLIR/PyTorch tools.

For fresh MACC receipts, use `compiler_macc.check_boundary(before, after,
destination, sdk_bin=..., compiler_python=...)`, then `validate_completion` and
`deliver`. In a **new process**, use `compiler_macc.replay(report, fresh_destination,
sdk_bin=..., compiler_python=...)` followed by live delivery. Generate a named
matmul and a generic batched contraction for each of bf16/f16 from the actual
production pass. Keep source/target byte hashes, the entire source/tool inventory,
audit artifacts and every premise in the resulting reports. Avoid reusing test
receipts whose inputs were deliberately mutated by negative controls.

## Paper wording that fits this result

> For supported bf16/f16 contraction sites, Merlin validates actual MACC compiler
> boundaries against a source-derived draft f32-accumulation reference. A
> Lean-checked theorem establishes complete output/reference execution equality
> under explicit SSA/call laws and strict local site qualification. Native
> references and mutation controls exercise the implementation and evidence
> binding. Independent numerical-policy approval, original half-precision source
> equivalence, whole-model definedness and downstream machine-code fidelity remain
> separate obligations.

See the [assumption ledger](../../../../merlin/lean/LEDGER.md),
[compiler stack](../../../../docs/guides/lean_compiler_stack.md) and
[prior integration handoff](../pass-verif-integration-20261028/README.md).

# Merlin pass verification: plan to submission

Proposed 2026-10-19 by Jeremy. For submission work, this replaces the resume checklist in the
[2026-10-19 pause report](../../shared-passes/checkpoints/20261019T041154Z/PROGRESS.md).

**Calendar.** D1 = Tue 2026-10-20, D14 = Mon 2026-11-02. Confirm the real submission date and shift
the table if it differs. Agents run every day, weekends included.

**Files in this directory**

| File | For |
|---|---|
| `README.md` | This plan: claim, scope, agents, schedule, gates, risks |
| [`RULES.md`](RULES.md) | Rules every agent follows |
| [`A0-coordinator.md`](A0-coordinator.md) | Coordinator and integrator |
| [`A1-inventory.md`](A1-inventory.md) | Real-model inventory, then paper evidence |
| [`A2-specs.md`](A2-specs.md) | Blind spec author (TorchAO contract, frontend intent, W8A8 policy) |
| [`A3-w8a8.md`](A3-w8a8.md) | W8A8 policy checker |
| [`A4-quant-lowerings.md`](A4-quant-lowerings.md) | TorchAO lowering, quant-inner lift, `quant_ext` variants |
| [`A5-frontend.md`](A5-frontend.md) | Frontend repairs, over-rank collapse, scalar squares, bf16 |
| [`A6-integration.md`](A6-integration.md) | Int8 recording and admission, composed claim, trust table, nightly run |

Start each agent with:

> Read `out/artifacts/verification/merlin-passes/plan/RULES.md` and
> `out/artifacts/verification/merlin-passes/plan/<your brief>.md` in full, write your first status
> file, then begin.

---

## 1. Why the plan changed

- **The previous campaign built a strong foundation**: the Lean semantics, the rule checker and
  its general theorem, typed checking, source definedness, receipts and chain replay. It also built
  47 rewrite families. About 40% of the Lean code targets upstream MLIR passes (canonicalize, CSE,
  elementwise fusion, named-op generalization). Those passes aren't Merlin's code, and the design
  rates them Tier B.
- **The paper's claim rests on Merlin-authored passes.** Those are preparation (above all the
  W8A8 quantization), device routing, runtime artifacts and memory layout, and almost none of them
  are covered yet.
- **Axon's lesson** ([arXiv 2605.01660](https://arxiv.org/abs/2605.01660)): credible compilation is
  cheap when the passes you own emit certificates and a general checker validates them. It's
  expensive when the checker has to recognize every rewrite of a pass that emits nothing. So:
  Merlin's passes supply hints, and upstream MLIR is trusted for submission.
- **Wording.** Don't say "shared passes" in briefs or in the paper. Say *Merlin-authored passes*
  (rows 2–4, 7–9 below) or *upstream MLIR passes* (rows 5–6).

## 2. The claim

For **one** validation program built with the paper configuration (decision 1 in §7):

> For every typed, initialized input on which the captured program `S0` is defined, under every
> interpretation of floating-point operations satisfying the laws named in the assumption ledger,
> each preparation boundary either refines its input or refines its declared spec transformer
> applied to its input: the frontend intent, the W8A8 policy, or the argument map of the
> quant-inner lift. The composed statement names every intermediate program where a transformer
> applies, because a policy transformer is not monotone under refinement (design §5.5).

Every boundary after preparation is verified, assumed or trusted, as the table below shows.
Accelerator calls stay contract holes (assume-guarantee, design D8).

**Trust table (the paper's main figure, in the style of Axon Fig. 1)**

| Row | Boundary | Planned status | Who reports it |
|---|---|---|---|
| 1 | Capture: PyTorch → MLIR (model2MLIR) | Trusted; differential evidence if time allows | — |
| 2 | **Preparation** (`_prepare_model_mlir`, `src/merlin/runtime/backends/zephyr_model.py:466`) | **Verified (this plan)** | A2–A6 |
| 3 | Device routing and the declaration text patch | Assumed (accelerator contracts); tested by L3 grading | A1 |
| — | OOT accelerator compiler | Assumed contracts; post-submission work | — |
| 4 | `lower_model` xDSL preprocessing | Verified where the existing gate applies | A6 |
| 5 | Upstream canonicalize, CSE, fusion, generalization | Trusted; the existing families are reported as partial validation, with a measured percentage | A1 |
| 6 | Bufferization → loops → LLVM | Trusted | — |
| 7 | `ciface_inline` LLVM transform | Trusted, tested | A1 |
| 8 | Runtime artifacts (argument table, `weights.bin`, IO maps, device shim) | Tested | A1 |
| 9 | Link layout; clang/ld | Tested; toolchain trusted | A1 |

## 3. Row 2 work list

The stage names are the recorded boundary names in `src/merlin/verify/lean/plans.py:151`.

| Stage | Pass | Claim | Owner | Estimate |
|---|---|---|---|---|
| `prepare-torchao-affine` | `lower_torchao_affine_quant` (`llvmlower/torchao_affine.py`) | Refines an independent TorchAO source contract (`Spec/TorchAO.lean`, A2) | A4 | 2–4 d |
| `prepare-scalar-squares` | `canonicalize_scalar_squares` (`llvmlower/scalar_square.py`) | Refines under one named float law in `Assumptions/` (decision 4); otherwise assumed | A5 | 0.5–1 d |
| `prepare-collapse-overrank-matmul` | `collapse_overrank_matmul` (`llvmlower/passes_xdsl.py:54`) | Refines the declared frontend intent (`Spec/FrontendIntent.lean`, A2) | A5 | 1–2 d |
| `prepare-propagate-quant-inner` | `_propagate_quant_inner` (`runtime/dispatch_runtime.py:1210`) | No semantic change, if it only edits `prov.*` / semantics-free attributes; confirm | A4 | 0.5 d |
| `prepare-qinner-lift` | `qinner.lift` (`llvmlower/qinner.py`) | `equivArgs` / argument map; the meaning of tagged `tensor.empty` is declared (decision 6) | A4 | 2–3 d |
| **(new)** `prepare-quantize-int8` | `apply_quant` → `lower_contraction_int8` (`llvmlower/passes_quant_int.py:644`) | `policy "w8a8"`: after ⊑ P(before) (`Spec/Policy/W8A8.lean`, A2) | A3, with A6 recording it | 5–8 d |
| `prepare-lower-quant-ext` | `lower_quant_ext` (`llvmlower/passes_xdsl.py:133`) | Already proved for signed-i8/i32/f32; extend only to variants the model uses | A4 | 0–1 d |
| `prepare-bf16-matmul-f32acc` | `lower_bf16_matmul_f32acc` | Only if it fires on the chosen model; otherwise an unchanged boundary | A5 | 0 or 1–2 d |
| `prepare-fix-bool-sitofp`, `prepare-fix-bool-fptosi` | `fix_bool_sitofp`, `fix_bool_fptosi` | Refines the declared frontend intent | A5 | 1–2 d |
| `prepare-scalarize-splat-inputs` | `scalarize_splat_inputs` | Already verified | — | 0 |
| `prepare-finish` | composite tail | Existing roster; device routing is row 3 | A6 | — |

Facts that shape the work:

- **`apply_quant` is not recorded today.** `zephyr_model.py` calls it outside
  `_lean_stages.run`, so its changes sit invisibly between the `prepare-qinner-lift` *after* text and
  the `prepare-lower-quant-ext` *before* text. A6 records it.
- **`preparation._admit` refuses int8 quantization and `device=`** (`src/merlin/verify/lean/preparation.py:44`).
  The paper build uses both. A6 admits `contraction_int8` only, and ends the verified chain at the
  routing boundary.
- **The claim types already exist.** `Check/Types.lean` defines `policy name`, `equivArgs` and
  `contractSeam`, so no frozen-interface change is needed to state these claims.
- **Hints already exist.** `passes_quant_int` tags every op it emits with `prov.*` roles
  (`_carry_prov`), and `Spec/Annotations.lean` makes `prov.*` semantics-free. The checker may use
  those roles as untrusted site hints and must recompute everything else.
- **Conv isn't a gap here.** model2MLIR lowers convolutions to an im2col gather plus `linalg.matmul`,
  and `lower_conv_int8` never fires on these models (its docstring says so). Pooling may still
  appear; A1's census will show.
- **A boundary that changes nothing on the chosen model is verified as unchanged.** A1's census
  decides which rows actually need proofs.

## 4. Agents

| Agent | Workstream | Branch | Starts | Hard dependency |
|---|---|---|---|---|
| A0 | Coordinator and integrator (Jeremy with one Codex agent) | `feat/pass-verification` | D0 | — |
| A1 | Inventory, then evidence and paper numbers | `pv/inventory` | D1 | Model environment |
| A2 | Spec author, blind | `pv/specs` | D1 | None (sparse checkout, §9) |
| A3 | W8A8 policy checker | `pv/w8a8` | D1 (scaffolding) | A2's W8A8 statement by D5 |
| A4 | Quantization lowerings | `pv/quant-lowerings` | D1 | A2's TorchAO statement by D3; decision 6 |
| A5 | Frontend repairs and structural rewrites | `pv/frontend` | D1 | A2's intent statements by D3; decision 4 |
| A6 | Pipeline integration | `pv/integration` | D1 | — |

**With fewer agents:** merge A4 and A5, let A0 absorb A6, and let A1 continue into the evidence
work. The minimum is A0, A2, A3 and a merged A4/A5, plus A1 for the first two days.

## 5. Schedule and gates

| Day | Work | Gate |
|---|---|---|
| D0 (today) | Send Agustin the decisions (§7). Commit this plan. Create the worktrees (§9). | |
| D1–D2 | A1: environment, inventory and census on the candidate models. A2: starts the TorchAO and W8A8 specs. A3: policy-claim scaffolding and side theorems. A4: quant-inner tags and qinner lift. A5: intent-relative rules on fixtures. A6: records and admits the int8 boundary. | **Gate A (end of D2):** the model is chosen; the list of passes that fire is known; Agustin's decisions are in. Drop rows that don't fire. If `S0` hits ops the semantics doesn't model, choose another model or a single function (for example, one decode step). |
| D3 | A2 delivers the TorchAO contract and frontend-intent statements. Jeremy reviews them; A0 pins them in `SPEC_LOCK`. A6 proposes the composed-claim shape for review. | |
| D3–D5 | A3, A4 and A5 build checkers against the frozen statements. A6 composes chains across transformer boundaries. | **Gate B (end of D5):** the W8A8 statement is frozen and reviewed. If it isn't, W8A8 falls back to "assumed, with differential tests against PyTorch", and A3 moves to the leftover row-2 work. |
| D6–D9 | Every boundary is checked on the chosen model's actual output. A6 runs the end-to-end chain nightly. A1 measures row 5 and collects the row-3/7/8/9 evidence. | **Gate C (end of D9):** every row-2 boundary is verified or explicitly assumed, and one composed claim exists for the model. |
| D10–D11 | A0 runs one integrated qualification: mutation matrix, chain replay, runtime measurement, trust table. | |
| D12 | Evidence freeze: a manifest with commit IDs, file hashes and tool fingerprints. | **Gate D:** freeze |
| D13–D14 | Paper writing; fixes only. | |

## 6. Definition of done for a boundary

1. The theorem statement was written first, reviewed by Jeremy, and pinned in `SPEC_LOCK` by A0.
2. The checker accepts the **actual boundary from the chosen model's build**. Fixtures alone don't
   count.
3. At least five MLIR-valid corrupted outputs are refused, using whichever apply: swapped operands,
   a changed constant or scale, a changed indexing map, a dropped operation, the wrong rounding mode,
   wrong clamp bounds.
4. Source support and definedness hold for the boundary's input on the declared domain, or the
   domain limit is written down.
5. The receipt replays and the chain composes.
6. `build_tools/scripts/check_lean_guard.py` passes, and `#print axioms` shows only approved
   `Assumptions/` entries.
7. Tests live in `merlin/tests/ir/test_lean_<area>.py`, plus a `_native` companion when numbers
   change.
8. The ledger (`merlin/lean/LEDGER.md`) or findings (`merlin/lean/findings.md`) is updated if an
   assumption or a pass bug appears, and the status file is updated.

## 7. Decisions needed from Agustin (send on D0)

1. **Model and build configuration** for the demonstration. Recommendation: the smallest program on
   the private roster whose capture uses only modeled ops (A1's census confirms by end of D2).
2. **Upstream MLIR passes are trusted** for submission. The existing partial validation is reported
   as a bonus row.
3. **Optional passes are a stretch goal**, not a requirement for submission.
4. **One named float law**, `powf(x, 2.0) = x * x`, or else the scalar-squares boundary is marked
   assumed.
5. **Declared frontend intent** for the repairs: `sitofp` on `i1` means boolean to {0.0, 1.0};
   float to `i1` means `x != 0`; an over-rank matmul means `aten.linear`'s batched semantics.
6. **Meaning of the quant-inner tagged `tensor.empty`**: the runtime binds it from the quantized
   tensor subclass's inner tensor, which turns the lift into an argument map.
7. **Who reviews the specs, and when**: the TorchAO and intent specs by D3, W8A8 by D5 (Jeremy;
   Agustin for W8A8).
8. **Rows 3, 8 and 9 appear as assumed or tested** in the paper.

Draft message:

> Hi Agustin! Quick update on the Lean verification and a few decisions I need from you to hit the
> deadline. The agent's last two weeks went mostly into verifying upstream MLIR passes
> (canonicalize/CSE/fusion), which gave us a solid Lean foundation but isn't the part the paper
> claim needs. For the next two weeks I'd like to focus on Merlin's own preparation passes,
> especially W8A8 quantization, for one model end to end, and report everything else in an
> Axon-style trust table (verified / assumed / tested / trusted). Can you confirm:
> (1) which model and build config we demonstrate (I'll propose the smallest one with full op
> support by Wednesday);
> (2) that upstream MLIR passes can be listed as trusted (we'll report our partial validation of
> them as a bonus);
> (3) that optional passes are a stretch goal;
> (4) that we can state one named float law, `powf(x,2) = x*x`, as an explicit assumption;
> (5) the intended meaning of the bool-cast and over-rank-matmul frontend repairs;
> (6) that device routing, runtime ABI and memory layout can appear as assumed/tested for
> submission;
> and (7) whether you can review the W8A8 spec around Oct 25.

## 8. Stop list (until Gate D)

- No new upstream MLIR rewrite families. No work on groups 13/14 or constant combination from the
  pause report. No native-gate extensions.
- No optional passes: int-softmax/gelu/silu/rsqrt, weight-quantize hoisting, quant round fusion,
  abs-max vectorization, static arena, `quantize_before_gather`.
- No model2MLIR verification.
- No OOT, pass-catalog or register/storage work (post-submission, per Agustin).
- No per-batch full regression ceremony. One integrated qualification runs at D10–D11.

## 9. Setup (Jeremy, D0)

Commit this plan first. Worktrees cut from an uncommitted tree won't see it.

```sh
cd ~/projects/merlin
W=out/build/agent-workspaces/merlin-passes
mkdir -p "$W" out/artifacts/verification/merlin-passes/status
for a in A1:pv/inventory A3:pv/w8a8 A4:pv/quant-lowerings A5:pv/frontend A6:pv/integration; do
  git worktree add "$W/${a%%:*}" -b "${a#*:}" feat/pass-verification
done

# A2 is blind to the implementations it specifies (information barrier, design §1.2 rule 6).
git worktree add --no-checkout "$W/A2" -b pv/specs feat/pass-verification
cd "$W/A2"
git sparse-checkout set --no-cone '/*' \
  '!/src/merlin/llvmlower/passes_quant_int.py' '!/src/merlin/llvmlower/quant_passes.py' \
  '!/src/merlin/llvmlower/quant_scope.py' '!/src/merlin/llvmlower/torchao_affine.py' \
  '!/src/merlin/llvmlower/qinner.py' '!/src/merlin/llvmlower/quant_hoist.py' \
  '!/src/merlin/llvmlower/quant_round.py' '!/src/merlin/llvmlower/passes_xdsl.py' \
  '!/src/merlin/llvmlower/scalar_square.py' '!/src/merlin/runtime/dispatch_runtime.py' \
  '!/merlin/tests/ir/*quant*' '!/merlin/tests/ir/test_lean_dequant*'
git checkout pv/specs
ls src/merlin/llvmlower/passes_quant_int.py   # must fail: file absent
cd -
```

Give every agent the same environment:

```sh
export STATUS_DIR=$HOME/projects/merlin/out/artifacts/verification/merlin-passes/status
export PATH=$HOME/.elan/bin:$PATH
export MERLIN_M2M_PYTHON=/path/to/compiler-environment/bin/python
export MERLIN_TORCH_PYTHON=/path/to/pinned-torch-oracle/bin/python   # A2, A3, A4
```

## 10. Risks and fallbacks

| Risk | Early signal | Fallback |
|---|---|---|
| The model environment (model2MLIR, checkpoints, private W8A8 recipe) doesn't work | A1 blocked for more than 4 h on D1 | Use a public capsule and the public W8A8 stand-in (`--dtype int8 --scheme int8_dyn_act_int8_weight`) for development; use the private build only for final numbers |
| `S0` uses ops the semantics doesn't model | A1's census | Choose a model or function with full support, or extend `Sem/` for one or two ops (A0 assigns, as an interface change request) |
| The checker is too slow at model size | A1/A6 timing on D2–D3 | Check per function or per dispatch group, and report the scaling honestly |
| The W8A8 spec is late or disputed | Gate B | Mark it assumed with differential tests; ship everything else |
| The spec author sees an implementation | — | Sparse checkout. If the barrier is breached, Jeremy re-reviews the spec line by line against the TorchAO source |
| Agents spin on hard proofs (Axon: narrow scope, bailing out) | No commit for 6 h on one theorem | A0 escalates: strengthen the checker's side conditions or split the rule. Never weaken a statement silently |
| Work lost in local clones (happened in the previous campaign) | — | Push every branch at least daily |
| `SPEC_LOCK` merge conflicts | — | Only A0 edits `SPEC_LOCK` |

## 11. Paper evidence to collect (A1 owns; A0 freezes)

- The trust table for the chosen model, **generated, never hand-edited**.
- Per boundary: changed sites, verified sites, abstentions with reasons, and the assumptions used.
- Bugs and discrepancies found: F6 (dequantization order), S4 (bufferization of captured tensors),
  S6 (index32), plus anything new.
- Effort, the way Axon reports it: days, agent count, Lean lines by category (semantics, checker,
  rules, specs), commits.
- The human review burden: lines in `Spec/` and `Assumptions/`.
- Checker runtime on the model, compared with compile time.
- The row-5 partial-validation percentage.

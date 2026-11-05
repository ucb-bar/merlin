---
name: inspect-artifacts
description: Find and read what a self-check left behind for one case, and diff two runs or variants. Open it when you need the files behind a verdict row (your_artifacts), want to know which file is your compiler's output and which is executed evidence, are unsure which selfcheck_out copy is current, or want to compare submission against a variant or an older request byte by byte. Covers the selfcheck_out and .qa_channel layout, what each file proves, and a script that lists every copy of a case with its verdict and diffs two copies.
---

# Inspect a case's artifacts

`SKILL` below is this skill's directory (`.agents/skills/inspect-artifacts` or `.claude/skills/inspect-artifacts`).
Run commands from the workspace root. Layout details and code citations: `reference/layout.md`.

## 1. Know where a check writes

| Path | Written by | What it is |
|---|---|---|
| `.qa_channel/resp_RID.json` | broker | the redacted verdict of request RID: `per_capsule` rows, counts, `note`, optional `performance_brief`, `offload_census` |
| `selfcheck_out/requests/RID/CASE/` | broker | that request's copy of your three files; never overwritten |
| `selfcheck_out/CASE/` | older single-slot path | the latest copy from a check without a per-request root; may be stale |
| `selfcheck_out/[requests/RID/]model_layers/runs/SUITE/CASE/` | `--model-layers` only | a full run directory: `generated/`, `artifacts/`, `capsule_result.json`, `run_manifest.yaml` |

RID is the id `agent_selfcheck.py` printed (also `--list`). A failing row's `your_artifacts` gives the exact paths.

## 2. Know what each file proves

- `command_buffer.json` is **your compiler's output** from the command-buffer stage: what you asked for, not proof it ran.
- `lowered.llvm.mlir` is **your compiler's output** from the target-artifact stage. The graded object is built from these exact bytes.
- `instruction_trace.json` is a **harness decode** of `lowered.llvm.mlir`: decoded classes in text order and counted loops. These are static sites. Nothing was executed to produce it.
- **Executed evidence** sits only in the verdict row. `tiers` gives each tier's status, `numeric` gives the mismatch count, magnitudes and first mismatch (expected values withheld), `barrier_cycles` is the tier's cycle count, and `sim_console_tail` is the console tail for a failing row. The model-layer run trees also hold `generated/numeric_report.yaml` and `artifacts/*_console.log`.
- A row with `declined` means your backend emitted no program for that case. It is a coverage gap, not wrong arithmetic.

## 3. List every copy of a case

```
python SKILL/scripts/case_artifacts.py list CASE            # newest first, joined to each request's verdict row
python SKILL/scripts/case_artifacts.py list CASE --limit 0 --json
```
Pick the copy whose `verdict.submission` is the tree you mean (`submission` or `variants/NAME`) and whose time follows your edit. Do not read `selfcheck_out/CASE/` as current without checking its time.

## 4. Diff two copies

```
python SKILL/scripts/case_artifacts.py diff selfcheck_out/requests/RID_A/CASE selfcheck_out/requests/RID_B/CASE
```
The output reports, in order:
- each file as identical or differing;
- tensor changes, per-opcode command counts and the first differing command;
- the change in decoded classes, plus loop headers with their trip counts;
- the op histogram and function names of the lowered IR.

Read it top to bottom:
- **Same command buffer, different trace:** the change is in emission (schedule, unrolling, configuration).
- **Different command buffer:** the change is in planning or lowering upstream of emission. Dump the stages with the `inspect-ir` skill to find which one.

To compare a variant, run one check per tree with the same `--capsules`, adding `--submission variants/NAME` for the variant. Then diff the two request directories.

## 5. Go deeper

- Per-command hardware state and cycles of a command buffer: the `read-command-buffer` skill (`isa_tools.py debug`).
- Rebuild the object, check the stack frame and symbols, and map loops to tasks: the `inspect-kernel` skill.
- Every intermediate stage of your compiler for one case: the `inspect-ir` skill.

The files under `selfcheck_out/` and `.qa_channel/` belong to the broker. Read them and copy them elsewhere, but do not edit them in place: the next check overwrites shared paths and per-request copies are evidence.

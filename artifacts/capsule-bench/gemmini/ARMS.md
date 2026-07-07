# The three arms — what each is, and why

A controlled A/B/C: **the same task, the same model/flags, and identical grading** — the *only* thing
that varies is the **authoring aid** the agent is given. That isolation is what makes any measured
difference attributable to the aid rather than to luck or task framing.

**Shared task (all arms):** author a Gemmini MLIR out-of-tree backend package; iterate against a
**redacted** QA verdict on the 4 pilot capsules (A0/A2/A4/B0) until all pass (or the round cap is hit);
the frozen package is then graded **only through its 4 CLI entrypoints** and must be integrity-clean
(no `import merlin`, no oracle routes — Merlin may *author*, never be a runtime dependency).

**Shared flags:** `--model claude-opus-4-8 --effort high --max-rounds 8 --max-rate-limit-waits 3`.

| Arm | run-id (batch `abc1`) | Driver / bundle | What the agent is given | The question it answers |
|---|---|---|---|---|
| **baseline** | `rb_abc1` | `run_baseline_qa_loop.py --arm raw_baseline` · `raw_baseline_public_v0` | the spec, the Gemmini headers, and the bench contract — **nothing else**. Writes the backend from scratch. | the control: how well does an un-tooled agent do? |
| **merlin** | `merlin_abc1` | `run_baseline_qa_loop.py --arm merlin_assisted` · `merlin_assisted_public_v0` | baseline **+ all Merlin authoring aids**, working in **xDSL**: plan synthesis (`targetgen/synthesize/`), scaffold generators (`targetgen/generate/` ∖ `runtime_adapter.py`), xDSL dialect patterns (`xdsl_dialects/` ∖ `lowering/`), interface grammar (`interface_emit.py`). **No CIRCT checks.** | do Merlin's authoring tools help over the baseline? |
| **merlin + CIRCT** | `merlincirct_abc1` | `run_rtlchecks_qa_loop.py` · `merlin_assisted_rtlchecks_public_v0` | **identical** authoring aids to `merlin`, **plus the CIRCT-compiled-from-RTL checks** delivered as advisory feedback each round: FileCheck over the agent's emitted MLIR + the decoded RoCC trace, with legal bounds extracted (via CIRCT) from the real Gemmini RTL (`facts.json`). The fast-iteration aid we built, hoped to **significantly improve** convergence and correctness. | does grounding feedback in the *actual RTL* help **beyond** the authoring tools? |

The arms are **nested in capability**: `merlin+CIRCT ⊃ merlin ⊃ baseline`. Because each adds exactly one
thing, the deltas are clean: (merlin − baseline) = the value of authoring tooling; (merlin+CIRCT − merlin)
= the value of the CIRCT-grounded checks specifically.

## Two dimensions we compare (both first-class)
1. **Authoring effort** — cost ($), tokens, tool-calls, rounds-to-converge, cumulative spend, and the
   activity trajectory through the transcript. *Does the aid make authoring cheaper / faster / fewer rounds?*
2. **Dialect completeness / correctness** — each frozen dialect is re-graded on the **entire 25-capsule
   corpus** (20 public + 5 hidden) across the oracle ladder **L2 (spike, functional) + L3 (verilator,
   cycle-accurate RTL)** via `full_suite_audit.py`: per-capsule pass/fail, failure plane, L3 cycles, and
   per-workload-class coverage. The QA loop only iterates on the 4 pilot capsules, so this audit is what
   reveals *how far each dialect generalizes beyond what it was tuned on.* *Does the aid yield a more
   complete / more correct dialect?*

## Provenance / how to tell the arms apart downstream
`environment.yaml::bundle_id` is the discriminator (`raw_baseline_public_v0` / `merlin_assisted_public_v0`
/ `merlin_assisted_rtlchecks_public_v0`). Both merlin arms write to `runs/merlin_assisted/`; the CIRCT one
is the `…rtlchecks…` bundle. This is the same identity `agg_agentic_results.py` keys on.

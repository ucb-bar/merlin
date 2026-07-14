# merlin similarity / leakage audit — pilot_merlin_0001

Operator-side, post-freeze. Compares the generated artifact against the forbidden prior backends (which the agent never saw). Inputs are NOT exposed to the agent.

- submission: `/path/to/merlin/experiments/gemmini_capsule_bench_v0/runs/merlin_assisted/pilot_merlin_0001/submission`
- prior backends compared: agent_spec_v0_mlir_oot, agent_spec_v1_mlir_oot, hand_smoke_oot, merlin_native_v0
- source files in submission: 10
- high-similarity threshold: 0.85 (normalized difflib ratio)

## Verdict: CLEAN

No exact matches, no high-similarity files, no manifest-structure copy.

## Exact content matches

_none_

## High-similarity files

_none_

## Manifest-structure overlap

_none above threshold_

## Comparability impact

No leakage signal; the artifact appears independently authored. Comparable.


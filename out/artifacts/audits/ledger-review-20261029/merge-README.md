# Integration with the concurrent memory-gate changes

This source cut combines ledger-review commit `8c8c1e5dd` with the published
`feat/pass-verif-integration` tip `df8799c42`. The merge retains both independent
canonical gate handlers and both worker-result validators. The status table
retains host fusion and the newer memory DCE/CSE qualifications. No proof,
specification, lock or approval flag is changed by the conflict resolutions.

The original [ledger review](README.md), [results](results.json) and prior
counterexamples remain evidence for their original source cut. This merge
changes the ledger and compiler source bindings, so those old receipts are not
current-tree qualification. Fresh checks and independent-process replays are
retained under `merge-df8799c42/`; their hashes, test results, commands and exact
merged source hashes are in [merge-results.json](merge-results.json).

The merge checks exercise the host-fusion gate, memory DCE gate, memory
origin protocol, actual default-all untiled memory compilation, tile-size policy,
changed formal-source interface assertions and existing host graph behavior. Q/DQ generation,
replay and mutation checks are repeated on both routes. Separate durable MACC
generation/replay covers bf16 and f16, and both actual host-fusion compiler
boundary pairs are independently replayed. Failed-evidence mutation tests stay
separate from durable replayable boundaries.

The initial broad invocation is retained, including a Python-version-dependent
compatibility-test failure. Python 3.14's default AST dump omits empty fields;
requesting those fields restores the original Python 3.12 fingerprint without
changing the pinned value. The corrected 46-case memory-origin suite passes,
and the original fingerprint also matches under the installed Python 3.12.3.
The separate interface/tile/graph compatibility suite passes 97 cases.
Across the completed cohorts, 179 distinct cases pass after that correction.
The untiled native case records two accepted workers and 12 matching kernel
observations. This count is not a claim that the entire broad suite finished.

The broad invocation is stopped after the untiled default-all native case.
The tiled default-all case and the full legacy loop/native corpus are not part
of this merged-source rerun. Their earlier qualification remains on its own
source cut. Completed cases, the retained failed attempt, interruption status
and corrected follow-up are reported separately in the manifest.
The intentional interrupt also exposed a pytest temporary-directory teardown
`KeyError`; its traceback is retained and is not counted as a clean suite exit.

Whole-tree structure/dependency checks retain the same 16 structure findings
and two optional-package imports, in unchanged files. The staged formatter check
against the local parent also reports pre-existing import ordering in
`src/merlin/llvmlower/pipeline.py`, whose bytes equal the published remote.
That file is outside this push's diff from the remote. Push-range checks are
recorded separately; no gate or allowlist is relaxed.

The push-range formatter required three touched graph files to be formatted.
Their parsed Python ASTs are exactly unchanged, with before/after hashes in the
manifest. All 73 existing graph and native compiler cases pass on the formatted
sources. Both host-fusion boundaries were then independently replayed again in
`host-fusion-format-0/` and `host-fusion-format-1/`. The earlier host receipts are
explicitly superseded; Q/DQ and MACC bindings are unaffected and remain current.

Use the tool environment in the original review and a new output directory.

```sh
mkdir -p out/build/push-merge-replay
.venv/bin/python -m pytest -q -ra --tb=short \
  -o tmp_path_retention_policy=all \
  merlin/tests/ir/test_compiler_host_graph_fusion.py \
  merlin/tests/ir/test_compiler_memory_dce.py \
  merlin/tests/ir/test_compiler_memory_origin.py \
  merlin/tests/ir/test_compiler_default_memory.py \
  merlin/tests/ir/test_private_buffer_tile_policy.py \
  merlin/tests/ir/test_compiler_formal_transforms.py \
  merlin/tests/ir/test_host_graph_optimizations.py \
  merlin/tests/ir/test_host_graph_policy.py \
  --basetemp=out/build/push-merge-replay/integration
```

The original review supplies the Q/DQ and durable MACC reproduction commands.
Host-fusion replay uses `compiler_host_graph_fusion.replay` on either retained
`host-fusion-*/REPORT.json` with a new destination and the selected Lean binary.
None of these checks approves a numerical policy or establishes downstream
native fidelity. Historical capture/tool skips remain explicit in the original
results and are not converted to passes by this merge.

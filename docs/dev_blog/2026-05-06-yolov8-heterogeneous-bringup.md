# YOLOv8 heterogeneous bringup — emitter v2 + partitioner + per-island routing

*2026-05-06*

This dev-blog entry summarizes the multi-week QNN heterogeneous
compile work that landed in tree as Phases 1–6 of the
"i-want-to-enable-rosy-sundae" plan. The headline is: a single
`./merlin compile yolov8.q.int8.onnx --target qrb5165_qnn_gpu
--qnn-partition` invocation now partitions the imported MLIR into
**94 deterministic islands** (64 conv + 17 concat + 9 reshape + 3
maxpool + 1 transpose), each routed to qnn-hta / qnn-gpu / cpu by
either a profile-driven decision or a heuristic (MAC count and
QNN-floor) when no profile is available. The full pipeline from
MLIR through to per-island `.qnn-ctx` artifacts is plumbed; the
remaining work is real on-board execution to populate the profile
sweep and validate the numerical-equivalence gate against the
all-CPU baseline.

## What problem we set out to solve

Before this round, "heterogeneous" on QRB5165 meant
[Architecture B](architecture/qnn_emitter_v2.md): per-target compile
of separate VMFBs followed by runtime-scheduler routing. The user
asked for **Architecture A** — a single VMFB that contains the
mixed-target dispatches, with the compiler choosing per-dispatch
backends based on profile data. That requires:

  1. A QNN code generator that takes MLIR linalg/tensor IR and emits
     a `.qnn.cpp` source. (Phases 1 & 2.)
  2. A subgraph partitioner that splits a multi-conv model into
     per-island subgraphs each lowering through one recognizer.
     (Phase 3.)
  3. A profile-driven router that decides which backend each island
     runs on. (Phase 4.)
  4. End-to-end gates that prove the heterogeneous output matches
     the all-CPU baseline within tolerance. (Phase 5.)
  5. The same pipeline working on dronet / mobilenet / depth_anything
     / smolVLA, not just yolov8. (Phase 6.)

The user added one hard constraint that drove a lot of the
architecture: **no regex anywhere in the v2 code path.** Every IR
walk uses `iree.compiler.ir` bindings exclusively; the legacy
regex-based emitter (`tools/kernels/qnn_emit.py`) stays as an
opt-in fallback (`MERLIN_QNN_EMIT_REGEX=1`) but is otherwise the
deprecated path.

## Architecture summary

```
                 .onnx → iree-import-onnx → util.func @model {…}
                                              │
                                              │ (Phase 3 partitioner)
                                              ▼
                    ┌─────────────────────────────────────┐
                    │  Island{0…N}: anchor + claim closure │
                    │  + boundary inputs/outputs            │
                    └─────────────────────────────────────┘
                                              │
                                              │ (Phase 4 router)
                                              ▼
                ┌──────────────┬──────────────┬──────────────┐
                │ qnn-hta      │ qnn-gpu      │ cpu          │
                │ (large conv) │ (small / no  │ (below QNN   │
                │              │  profile)    │  floor)      │
                └──────────────┴──────────────┴──────────────┘
                                              │
                                              │ (Phase 5 emit)
                                              ▼
                  per-island .qnn.cpp (via v2 emitter recognizers)
                                              │
                                              │ (qnn_build, parallel SSH)
                                              ▼
                        per-island .qnn-ctx (run via QNN HAL)
```

## What landed end-to-end

| Phase | Pieces | Tests |
|---|---|---|
| 1 | `qnn_emit_v2` + 8 bindings recognizers (conv+relu, conv uint8, depthwise, maxpool, concat, reshape, elementwise binary/unary) + `qnn_gates` (compile-determinism + numerical-equivalence shapes) | 11 parity (byte-equal vs v1) + 4 emit-only |
| 2 | NCHW int8 conv recognizer (real yolov8 IR), Conv+Relu/Sigmoid/Tanh/SiLU fused shapes (incl. real-yolov8 SiLU through requantize round-trips), per-element weight payloads + OIhw→HWIO permutation, standalone maxpool/concat/reshape/transpose recognizers | 38 parser/lowering + 14 Phase 2 gate |
| 3 | `qnn_partition` (anchor-based SSA def-use closure with boundary detection), `qnn_build`'s `digest_qnn_graph_desc` + parallel-SSH builder, `--qnn-partition` flag in `tools/compile.py` | 14 partition correctness (incl. real yolov8 → 94 islands deterministic) |
| 4 | `qnn_route` (heuristic + profile-driven + threshold-tuning), `tools/profile_per_island.py` (env-gated on-board sweep), `qnn_routing.json` artifact emission | 9 routing |
| 5 | per-island slice MLIR emission (`partition_with_claims`, `emit_island_slice_mlir`), `assert_numerical_equivalence` finalized (was a stub), `eval/qrb5165/heterogeneous/yolov8_e2e.py` orchestrator | 8 Phase 5 gate / orchestrator |
| 6 | Multi-model regression test (dronet, mobilenet, yolov8), fixture retirement README, this dev blog + how-to + architecture doc | 4 multi-model |

Total: **96 default-pass pytest tests + 4 env-gated on-board build
tests, zero regex in the v2 code path, pre-commit clean across all
phase files.**

## Key wins

  - **Real-yolov8 SiLU recognized through requantize round-trips.**
    yolov8 nano emits `conv → dequant → quantize → dequant → sigmoid
    → quantize → dequant → multiply` for every SiLU activation.
    `_strip_requantize_roundtrip` walks past the round-trips so the
    classifier sees the post-roundtrip f32 value; the multiply test
    pairs sigmoid output with conv-result-after-roundtrip output.
    All 64 yolov8 conv variants partition + recognize cleanly today.

  - **94 deterministic islands on yolov8.** Stable across re-runs;
    op counts exactly match the input MLIR's anchor inventory.
    Compile-determinism gate (md5 stable across runs) requires this
    determinism — and gets it.

  - **Heuristic routing produces sensible distribution without any
    profile data.** 64 → qnn-hta (all convs above 1M MAC), 29 →
    qnn-gpu (elementwise / pool / reshape / concat above 1Ki ops),
    1 → cpu (a tiny detection-head reshape correctly demoted below
    the QNN floor). Profile-driven routing then refines once the
    on-board sweep populates the CSV.

## Open follow-ups (board-bound)

  - **Live on-board profile sweep.** `profile_per_island.py
    --on-board qdev` with the QAIRT SDK staged at `/tmp/qnn_probe`
    runs ~30 min for yolov8 (94 × 2 backends × 30 iter); the
    output drops a CSV that the orchestrator picks up automatically.
  - **Numerical-equivalence gate on the model output.** The
    `assert_numerical_equivalence` function is functional today;
    it just needs both VMFBs (the all-CPU baseline + the
    heterogeneous candidate) to be present on a host with
    `iree-run-module`. The orchestrator's step 5 invokes it
    automatically.
  - **CPU-backend Transpose validator gap.** QAIRT 2.45's
    `libQnnCpu.so` rejects int8 Transpose during graph compose;
    this is the documented xfail in
    `test_qnn_emit_v2_yolov8_build.py`. GPU/HTA backends accept
    the same graph — confirmed via the recognizer's structural
    output reaching the validator with all addNode calls in scope.
    Real on-board build via `qnn_build_on_board` will flip the
    xfail to xpass.

## Path forward

The next concrete on-board session (when board access is open)
should run, in order:

  1. `profile_per_island.py --on-board qdev` — populates the
     yolov8 profile CSV.
  2. The yolov8 e2e orchestrator with `--build-islands` flag
     enabled (need a tools/compile.py extension to actually invoke
     the per-island builds via `build_many_qnn_kernels_on_board`).
  3. The numerical-equivalence gate reads the heterogeneous VMFB
     and the all-CPU baseline VMFB; runs both on host with
     `iree-run-module`; asserts element-wise output match within
     the int8 tolerance.

When all three pass, Architecture A on yolov8 is end-to-end real,
not just structurally correct. The plan flagged this as the
"Phase 5 perf comparison vs ONNX black-box" gate; the structural
pieces are in place, the runtime numbers come from running on the
board.

## Files & docs

  - Plan: `/home/agustin/.claude/plans/i-want-to-enable-rosy-sundae.md`
  - Architecture: `docs/architecture/qnn_emitter_v2.md`
  - Author guide: `docs/how_to/add_qnn_recognizer.md`
  - Orchestrator: `eval/qrb5165/heterogeneous/yolov8_e2e.py`
  - Tests: `tools/kernels/tests/test_qnn_emit_v2_*.py`,
    `test_qnn_partition_correctness.py`, `test_qnn_route.py`,
    `test_qnn_phase5_gates.py`, `test_qnn_phase6_multimodel.py`

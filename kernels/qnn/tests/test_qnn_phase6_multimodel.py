"""Phase 6 — multi-model regression test.

Drives the partitioner + heuristic router through every available
phase-1 IR and asserts deterministic, recognizer-coverage-correct
results. Each model is a parametric case so failures stay attributed
to the specific shape that broke.

Models exercised:
  - yolov8 nano (the canonical Phase 5 target)
  - dronet (q.int8)
  - mobilenet_v2 (q.int8 — partial coverage; depthwise convs are a
    Phase-6+ recognizer extension)
  - depth_anything_v2 / smolVLA — when phase-1 IR is built (skip
    when missing).

The assertions are structural: partition count, per-recognizer
counts, target distribution. Numerical-equivalence and on-board
runtime gates are exercised by `eval/qrb5165/heterogeneous/yolov8_e2e.py`
on a per-model basis when board access is available.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


# Each entry: (model_label, phase1_path_relative_to_repo, expected_counts).
# When a path doesn't exist, the test skips with a clear message.
# Counts are exact for stable models; "ge" comparisons would mask
# regressions — we want a count change to fail the test loudly.
_MODELS: tuple[tuple[str, str, dict[str, int]], ...] = (
    (
        "yolov8_nano",
        "build/compiled_models/yolov8_nano/qrb5165/phases/yolov8n_q_int8.1.input.mlir",
        {
            "nchw_int8_conv": 64,
            "nchw_int8_concat": 17,
            "nchw_int8_reshape": 9,
            "nchw_int8_pool": 3,
            "nchw_int8_transpose": 1,
        },
    ),
    (
        "dronet",
        "build/compiled_models/dronet/qrb5165_aarch64_a77_dronet.q.int8/phases/dronet_q_int8.1.input.mlir",
        {
            "nchw_int8_conv": 10,
            "nchw_int8_pool": 1,
            "nchw_int8_reshape": 6,
            "nchw_int8_transpose": 2,
        },
    ),
    (
        "mobilenet_v2",
        "build/compiled_models/mobilenet_v2/qrb5165/phases/mobilenet_v2.1.input.mlir",
        # mobilenet uses depthwise-int8 convs which the partitioner
        # doesn't yet anchor; only reshape + transpose surface today.
        # This expectation INVERTS once the depthwise int8 recognizer
        # lands (planned: depthwise + standalone-pool + concat anchors
        # for a fuller mobilenet partition).
        {
            "nchw_int8_reshape": 1,
            "nchw_int8_transpose": 1,
        },
    ),
)


@pytest.mark.parametrize("model_label,phase1_rel,expected_counts", _MODELS, ids=[m[0] for m in _MODELS])
def test_model_partition_recognizer_counts(
    model_label: str,
    phase1_rel: str,
    expected_counts: dict[str, int],
) -> None:
    """For each available model, assert the partitioner's per-
    recognizer island counts exactly match the expected distribution."""
    from qnn_partition import parse_and_partition

    phase1 = REPO_ROOT / phase1_rel
    if not phase1.exists():
        pytest.skip(
            f"{model_label} phase-1 IR not built; run "
            f"`./merlin compile models/{model_label}/...{model_label.replace('_', '-')}.onnx "
            f"--target qrb5165_aarch64 --dump-phases` to populate"
        )

    islands = parse_and_partition(phase1.read_text())
    counts = dict(Counter(i.recognizer_name for i in islands))
    assert counts == expected_counts, (
        f"{model_label}: partitioner counts diverged.\n"
        f"  expected: {expected_counts}\n"
        f"  observed: {counts}\n"
        f"  (a count change is either a recognizer regression or "
        f"a new model variant — update the expected map after "
        f"reviewing the diff)"
    )


@pytest.mark.parametrize(
    "model_label,phase1_rel,_unused",
    [(label, path, expected) for label, path, expected in _MODELS],
    ids=[m[0] for m in _MODELS],
)
def test_model_partition_routing_distribution(model_label: str, phase1_rel: str, _unused) -> None:
    """For each model, the heuristic-only routing produces a
    non-empty target distribution and every island gets a routing
    decision (no orphans)."""
    from qnn_partition import parse_and_partition
    from qnn_route import route_islands, summarize

    phase1 = REPO_ROOT / phase1_rel
    if not phase1.exists():
        pytest.skip(f"{model_label} phase-1 IR not built")

    islands = parse_and_partition(phase1.read_text())
    decisions = route_islands(islands)
    assert len(decisions) == len(islands)
    counts = summarize(decisions)
    # Total across all targets matches island count (no orphans).
    assert sum(counts.values()) == len(islands)
    # Every island is routed to one of the known targets.
    assert all(t in {"cpu", "qnn-gpu", "qnn-hta"} for t in counts)


def test_model_partition_is_deterministic() -> None:
    """Run the partitioner twice on each available model. Outputs
    must match exactly (compile-determinism gate prerequisite for
    multi-model release)."""
    from qnn_partition import parse_and_partition

    for label, phase1_rel, _ in _MODELS:
        phase1 = REPO_ROOT / phase1_rel
        if not phase1.exists():
            continue
        text = phase1.read_text()
        a = parse_and_partition(text)
        b = parse_and_partition(text)
        assert len(a) == len(b), label
        for ia, ib in zip(a, b):
            assert ia.name == ib.name, label
            assert ia.recognizer_name == ib.recognizer_name, label
            assert ia.target == ib.target, label
            assert ia.op_names == ib.op_names, label

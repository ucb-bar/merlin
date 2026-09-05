"""Synchronization points must be counted from the emitted program, and never guessed."""
from __future__ import annotations

import json
import pathlib

import pytest

from merlin.perf import barrier_arms as B


def _buf(*counts):
    """A command stream: (n_matmul, n_commit)."""
    n_mm, n_commit = counts
    return {"commands": [{"opcode": "RES_PACK"},
                         *[{"opcode": "MATMUL_RESIDENT"} for _ in range(n_mm)],
                         *[{"opcode": "COMMIT"} for _ in range(n_commit)],
                         {"opcode": "EVICT"}]}


def test_it_counts_completion_points_in_a_real_emitted_buffer():
    """Read a buffer the harness actually produced, not a stand-in, when one is available."""
    root = pathlib.Path("/scratch/agustin/tmp/gsim_recert_20260904/captures49")
    found = sorted(root.glob("PQ*/command_buffer.json")) if root.is_dir() else []
    if not found:
        pytest.skip("no captured synchronization buffer on this machine")
    counted = 0
    for path in found:
        out = B.count_barriers(json.loads(path.read_text()))
        assert out["status"] == "counted", path.name
        # the family varies its job count, so the completion points must vary with it
        assert out["barriers"] >= 1
        counted += 1
    assert counted >= 1


def test_a_buffer_with_no_completion_opcode_is_unknown_not_zero():
    out = B.count_barriers({"commands": [{"opcode": "MATMUL_RESIDENT"}]})
    assert out["status"] == B.UNKNOWN
    assert "cannot be counted" in out["reason"]
    assert "barriers" not in out, "an uncountable program must not report a count"


@pytest.mark.parametrize("bad", [None, 42, "commands", {}, {"commands": []},
                                 {"commands": [{"operands": {}}]}])
def test_an_unreadable_buffer_is_unknown_with_a_reason(bad):
    out = B.count_barriers(bad)
    assert out["status"] == B.UNKNOWN and out["reason"]


def test_paired_removal_reports_what_the_candidate_dropped():
    out = B.paired_removal(_buf(16, 16), _buf(16, 1))
    assert out["removed"] == 15
    assert out["baseline_barriers"] == 16 and out["candidate_barriers"] == 1


def test_paired_removal_is_unknown_when_either_side_cannot_be_read():
    out = B.paired_removal(_buf(16, 16), {"commands": [{"opcode": "MATMUL_RESIDENT"}]})
    assert out["status"] == B.UNKNOWN and "candidate" in out["reason"]


def test_the_claim_needs_a_varying_removed_count():
    same = [{"removed": 3, "cycles_saved": 300}, {"removed": 3, "cycles_saved": 310}]
    out = B.analyze_barrier_claim(same)
    assert out["verdict"] == B.REFUSED
    assert "GROWS" in out["reason"]


def test_a_saving_that_grows_with_removed_barriers_is_established():
    out = B.analyze_barrier_claim([{"removed": 1, "cycles_saved": 100},
                                   {"removed": 7, "cycles_saved": 700},
                                   {"removed": 15, "cycles_saved": 1500}])
    assert out["verdict"] == B.ESTABLISHED
    assert out["measured"]["cycles_per_removed_barrier"] == pytest.approx(100.0)


def test_barriers_that_cost_nothing_refute_the_claim_rather_than_refusing_it():
    out = B.analyze_barrier_claim([{"removed": 1, "cycles_saved": 5},
                                   {"removed": 15, "cycles_saved": 3}])
    assert out["verdict"] == B.REFUTED
    assert out["verdict"] != B.REFUSED, "a measured null result is evidence, not missing evidence"


@pytest.mark.parametrize("rows", [[], "nope", [{"removed": 1}], [{"cycles_saved": 1}],
                                  [{"removed": True, "cycles_saved": 1}]])
def test_incomplete_pairs_are_refused(rows):
    assert B.analyze_barrier_claim(rows)["verdict"] == B.REFUSED

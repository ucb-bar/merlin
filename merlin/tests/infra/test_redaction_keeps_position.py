"""A POSITION is not a golden value: the divergence index must survive the numeric scrub."""

from __future__ import annotations

import sys

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))
import qa_check  # noqa: E402


def test_the_divergence_index_survives():
    txt = "your command buffer does not compute the declared operation (first divergence at index=2)"
    out = qa_check._scrub_numbers(txt)
    assert "index=2" in out, out


def test_expected_and_observed_values_still_scrub():
    out = qa_check._scrub_numbers("expected 0.546875, observed 24183284")
    assert "0.546875" not in out and "24183284" not in out


def test_a_shape_still_survives():
    out = qa_check._scrub_numbers('tensor<16x16xi8>')
    assert "16x16xi8" in out

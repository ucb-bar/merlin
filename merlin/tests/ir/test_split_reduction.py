"""An exact arithmetic obligation for a candidate split reduction."""

from __future__ import annotations

import pytest

from merlin.verification.split_reduction import maximum_safe_chunk, verify_split_reduction

pytest.importorskip("z3")


def test_split_reduction_proves_all_inputs_at_one_shape() -> None:
    import pickle

    from merlin.verification import split_reduction as owner
    from merlin.verify import split_reduction as legacy

    assert legacy is owner
    # A 4-bit signed input has maximum absolute product 64. Two terms need
    # 9 signed bits; the source's 10-bit accumulation handles all three.
    assert maximum_safe_chunk(4, 9) == 3
    verdict = verify_split_reduction(k=3, operand_width=4, partial_width=9, output_width=10, chunks=(2, 1))
    assert verdict.status == "verified"
    assert verdict.counterexample is None
    assert verdict.__class__.__module__ == "merlin.verify.split_reduction"
    assert pickle.loads(pickle.dumps(verdict)) == verdict


def test_narrow_partial_accumulator_is_refuted() -> None:
    verdict = verify_split_reduction(k=3, operand_width=4, partial_width=8, output_width=10, chunks=(3,))
    assert verdict.status == "refuted"
    assert verdict.counterexample is not None


def test_missing_reduction_term_is_not_a_theorem() -> None:
    with pytest.raises(ValueError, match="cover every reduction term"):
        verify_split_reduction(k=3, operand_width=4, partial_width=9, output_width=10, chunks=(2,))


def test_widths_are_parameters_not_target_defaults() -> None:
    assert maximum_safe_chunk(8, 20) == 31
    assert maximum_safe_chunk(8, 32) == 131071


def test_cli_exit_codes_distinguish_proof_counterexample_and_invalid_partition(capsys) -> None:
    import json

    from merlin.verify.cli import main

    widths = ["--operand-width", "4", "--output-width", "10"]
    assert main(["split-reduction", "--k", "3", "--partial-width", "9", "--chunks", "2,1", *widths]) == 0
    proved = json.loads(capsys.readouterr().out)
    assert proved["status"] == "verified"
    assert proved["maximum_safe_chunk"] == 3

    assert main(["split-reduction", "--k", "3", "--partial-width", "8", "--chunks", "3", *widths]) == 1
    refuted = json.loads(capsys.readouterr().out)
    assert refuted["status"] == "refuted" and refuted["counterexample"]

    assert main(["split-reduction", "--k", "3", "--partial-width", "9", "--chunks", "2", *widths]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"

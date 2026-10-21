"""Actual native wide-command execution, including a physically wrong codec."""

import json

import pytest
from test_compact_pe_native import native_pair


@pytest.mark.parametrize(
    "dtypes,widths,unified,tile",
    [
        (("i8", "i8", "i32"), (1, 1, 4), False, (2, 3, 4)),
        (("i16", "i8", "i32"), (2, 1, 4), True, (3, 2, 5)),
        (("i16", "i16", "i32"), (2, 2, 4), False, (1, 4, 3)),
    ],
)
def test_actual_wide_commands_execute_blocked_mapping_and_full_carried_outputs(tmp_path, dtypes, widths, unified, tile):
    native_pair(tmp_path, dtypes, widths, unified, transfer_groups=True, tile=tile)
    report = json.loads((tmp_path / "pair.json").read_text())
    for original, selected in zip(report["arms"][0]["actual_native"], report["arms"][1]["actual_native"], strict=True):
        assert selected["dma_commands"] < original["dma_commands"]
        assert selected["group_commands"] > 0 == original["group_commands"]
        assert selected["dram_bytes"] == original["dram_bytes"]
        assert selected["computes"] == original["computes"]


def test_independent_native_reference_rejects_wrong_actual_blocked_destination(tmp_path):
    # The deliberately false native implementation writes every component to
    # the first block. Logical plan/projection acceptance cannot qualify it.
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        native_pair(tmp_path, ("i8", "i8", "i32"), (1, 1, 4), False, transfer_groups=True, bad_mapping=True)
    assert (tmp_path / "program.c").is_file() and (tmp_path / "program.so").is_file()

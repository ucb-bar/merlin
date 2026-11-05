"""Which harts carry a matrix unit: declared per board, enforced before a routed image is built.

A vector unit without the matrix unit (a DMA-style core beside matrix-equipped ones) traps in a routed
image's tile loop, which the OpenMP split puts on every worker. A hart count alone cannot say which harts
those are, so the board declares them and a matrix image that would reach any other hart is refused.
"""

from __future__ import annotations

import inspect

import pytest

from merlin.runtime import boards

_ENTRY = (
    "    dram_bytes: 256 MiB\n    dram_base: 0x80000000\n    harts: 3\n    vector_harts: 2\n"
    "    console: htif\n    flow: baremetal\n    code_reserve: 1 MiB\n"
)


def _board(**over):
    facts = {
        "name": "b",
        "dram_bytes": 1 << 28,
        "dram_base": 0x80000000,
        "harts": 3,
        "console": boards.CONSOLE_HTIF,
        "flow": boards.FLOW_BAREMETAL,
        "code_reserve": 1 << 20,
    }
    facts.update(over)
    return boards.Board(**facts)


def test_matrix_harts_default_to_the_vector_harts():
    plain = _board(vector_harts=2)
    assert plain.matrix_harts == (0, 1)
    plain.check_matrix_image(2)  # (3 harts is the vector-hart rule's to refuse, in build_app)


def test_matrix_harts_can_be_narrower_than_the_vector_harts():
    narrow = _board(vector_harts=3, matrix_hart_ids=(0, 1))
    assert narrow.hart_ids_for("rvv") == (0, 1, 2) and narrow.matrix_harts == (0, 1)
    narrow.check_matrix_image(1)
    narrow.check_matrix_image(2)
    with pytest.raises(boards.BoardRegistryError, match="carry no matrix unit"):
        narrow.check_matrix_image(3)
    odd = _board(vector_hart_ids=(0, 2), matrix_hart_ids=(2,))
    with pytest.raises(boards.BoardRegistryError):
        odd.check_matrix_image(1)  # the first vector hart (0) has no matrix unit


def test_the_catalog_reads_and_validates_matrix_hart_ids(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(f"schema_version: 1\nboards:\n  b:\n{_ENTRY}    matrix_hart_ids: [1]\n", encoding="utf-8")
    assert boards.load_boards(good)["b"].matrix_harts == (1,)


@pytest.mark.parametrize("ids", ["[2]", "[0, 0]", "[]"])
def test_matrix_hart_ids_must_be_distinct_vector_harts(tmp_path, ids):
    reg = tmp_path / "boards.yaml"
    reg.write_text(f"schema_version: 1\nboards:\n  b:\n{_ENTRY}    matrix_hart_ids: {ids}\n", encoding="utf-8")
    with pytest.raises(boards.BoardRegistryError, match="matrix_hart_ids"):
        boards.load_boards(reg)


def test_the_pair_builder_refuses_before_building():
    from merlin.runtime.backends import matrix_image

    src = inspect.getsource(matrix_image.build_matrix_pair)
    assert src.index("check_matrix_image(harts)") < src.index("def build(")

"""Committed scalar widths come from instructions, never padded XLEN spans."""

import pytest

from merlin.targetgen.riscv_memory_effects import rv64_scalar_store_width


@pytest.mark.parametrize(
    "word,width",
    [
        (0x23, 1), (0x1023, 2), (0x2023, 4), (0x3023, 8),
        (0x2027, 4), (0x3027, 8), (0x202F, 4), (0x302F, 8),
        (0x1800202F, 4), (0xE000302F, 8),
        (0xA000, 8), (0xC000, 4), (0xE000, 8),
        (0xA002, 8), (0xC002, 4), (0xE002, 8),
    ],
)
def test_recognized_store_effect_width(word, width):
    assert rv64_scalar_store_width(word) == width


@pytest.mark.parametrize(
    "word",
    [True, -1, 1 << 32, 0x13, 0x3, 0x4023, 0x1027, 0x102F,
     0x1000202F, 0x2800202F, 0x10000 | 0xC000, 0xC001],
)
def test_nonstore_reserved_unknown_or_foreign_width_refuses(word):
    with pytest.raises(ValueError):
        rv64_scalar_store_width(word)

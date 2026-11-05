"""Check the complete transpose address bijection with an independent builder."""

import pytest

from merlin.llvmlower.dense_layout_loop import emit_dense_transpose


class Concrete:
    def __init__(self):
        self.reads, self.writes = [], {}

    def op(self, opcode, left, right):
        return left + right if opcode == "add" else left * right

    def gep(self, base, offset):
        return base + offset

    def load(self, address, dtype):
        assert dtype == "i8"
        self.reads.append(address)
        return address

    def store(self, value, address, dtype):
        assert dtype == "i8" and address not in self.writes
        self.writes[address] = value

    def loop(self, start, end, step, callback):
        for index in range(start, end, step):
            callback(index)


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 17), (19, 1), (7, 31), (16, 16)])
def test_complete_transpose_reads_once_and_writes_exact_bijection(rows, cols):
    b = Concrete()
    emit_dense_transpose(b, 0x100000, 0x200000, rows, cols, max_elements=rows * cols)
    assert sorted(b.reads) == list(range(0x100000, 0x100000 + rows * cols))
    assert sorted(b.writes) == list(range(0x200000, 0x200000 + rows * cols))
    for r in range(rows):
        for c in range(cols):
            assert b.writes[0x200000 + c * rows + r] == 0x100000 + r * cols + c


@pytest.mark.parametrize("rows,cols,bound,same", [(True, 3, 10, False), (0, 3, 10, False),
    (3, 4, 11, False), (3, 4, 12, True), (3, 4, True, False)])
def test_invalid_layout_refuses_before_emission(rows, cols, bound, same):
    b = Concrete()
    with pytest.raises(ValueError):
        emit_dense_transpose(b, 0x100000, 0x100000 if same else 0x200000,
                             rows, cols, max_elements=bound)
    assert b.reads == [] and b.writes == {}

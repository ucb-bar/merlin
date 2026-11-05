"""Compare emitted window rectangles with independently padded image coordinates."""

import numpy as np
import pytest

from merlin.compile.scheduling.window_transfer import Window2D


class Builder:
    def op(self, name, a, b, ty="i64"):
        value = {"add": lambda: a+b, "sub": lambda: a-b, "mul": lambda: a*b,
                 "udiv": lambda: a//b, "urem": lambda: a%b, "and": lambda: a & b}[name]()
        return value & ((1 << (1 if ty == "i1" else 64))-1)

    def cmp(self, pred, a, b):
        a, b = a & (2**64-1), b & (2**64-1)
        return int(a < b if pred == "ult" else a <= b)


@pytest.mark.parametrize("geometry", [
    Window2D((1, 5, 9, 3), (3, 3), padding=(1, 1, 1, 1)),
    Window2D((2, 7, 11, 5), (3, 2), stride=(2, 1), dilation=(1, 2), padding=(2, 1, 1, 2)),
    Window2D((2, 4, 17, 2), (2, 3), stride=(1, 2), padding=(1, 2, 0, 1)),
])
def test_every_accepted_rectangle_matches_the_padded_window(geometry):
    shape = geometry.input_shape
    values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    top, left, bottom, right = geometry.padding
    padded = np.pad(values, ((0, 0), (top, bottom), (left, right), (0, 0)), constant_values=-1)
    oh, ow = geometry.output_spatial
    accepted, refused = 0, 0
    for row in range(shape[0]*oh*ow):
        for rows in (1, 2, 4, 16):
            for ky in range(geometry.kernel[0]):
                for kx in range(geometry.kernel[1]):
                    run = geometry.emit_row_run(Builder(), row, rows, ky, kx, 0, shape[-1])
                    if not run["predicate"]:
                        refused += 1
                        continue
                    accepted += 1
                    assert row//ow == (row+rows-1)//ow
                    for i in range(rows):
                        image, spatial = divmod(row+i, oh*ow)
                        y, x = divmod(spatial, ow)
                        expected = padded[image, y*geometry.stride[0]+ky*geometry.dilation[0],
                                          x*geometry.stride[1]+kx*geometry.dilation[1], :]
                        first = run["first_element"] + i*run["row_stride_elements"]
                        actual = values.ravel()[first:first+shape[-1]]
                        np.testing.assert_array_equal(actual, expected)
    assert accepted and refused


def test_foreign_negative_or_overflowed_coordinates_cannot_enable_a_load():
    window = Window2D((1, 3, 8, 4), (1, 1))
    for row, ky, kx, channel in ((-1, 0, 0, 0), (24, 0, 0, 0), (0, -1, 0, 0),
                                 (0, 0, 1, 0), (0, 0, 0, 4)):
        assert not window.emit_row_run(Builder(), row, 2, ky, kx, channel, 1)["predicate"]
    with pytest.raises(ValueError, match="64-bit"):
        Window2D((2**62, 2, 1, 1), (1, 1))

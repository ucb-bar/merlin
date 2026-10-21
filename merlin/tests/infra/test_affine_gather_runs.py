"""All logical samples, including row/image tails, survive DMA run coalescing."""

import pytest

from merlin.compile.scheduling.affine_gather_runs import ConvolutionGatherRuns, emit_convolution_gather_runs


class IntegerBuilder:
    def op(self, operation, left, right):
        return {"add": lambda: left + right, "sub": lambda: left - right, "mul": lambda: left * right,
                "udiv": lambda: left // right, "urem": lambda: left % right}[operation]()

    def cmp(self, operation, left, right):
        assert operation == "ult"
        return left < right

    def select(self, condition, left, right):
        return left if condition else right


@pytest.mark.parametrize("ow", [1, 2, 3, 7, 14, 16, 17, 28, 56])
@pytest.mark.parametrize("stride,dilation", [((1, 1), (1, 1)), ((2, 3), (2, 1))])
def test_exhaustive_rectangles_preserve_every_bounded_source_and_destination_sample(ow, stride, dilation):
    b, oh, ci, kh, kw = 2, 3, 17, 2, 3
    sh, sw = stride
    dh, dw = dilation
    h, w = (oh - 1) * sh + (kh - 1) * dh + 1, (ow - 1) * sw + (kw - 1) * dw + 1
    plan = ConvolutionGatherRuns((b, h, w, ci), (b, oh, ow), stride, dilation, (kh, kw), 16)
    for first in range(b * oh * ow):
        for count in range(1, min(16, b * oh * ow - first) + 1):
            expected = plan.fragments(first, count)
            emitted = tuple(fragment for fragment in emit_convolution_gather_runs(plan, IntegerBuilder(), first, count)
                            if fragment[2])
            assert emitted == expected and len(emitted) <= plan.max_fragments(count)
            for ky in range(kh):
                for kx in range(kw):
                    for channel, channels in ((0, 16), (16, 1)):
                        observed = []
                        for base, destination, length in emitted:
                            for row in range(length):
                                for c in range(channels):
                                    source = base + ky * dh * w * ci + kx * dw * ci
                                    source += row * plan.source_row_stride + channel + c
                                    assert 0 <= source < b * h * w * ci
                                    observed.append((destination + row, channel + c, source))
                        exact = []
                        for row in range(count):
                            image, spatial = divmod(first + row, oh * ow)
                            y, x = divmod(spatial, ow)
                            for c in range(channels):
                                address = ((image * h + y * sh + ky * dh) * w + x * sw + kx * dw) * ci
                                exact.append((row, channel + c, address + channel + c))
                        assert observed == exact


def test_geometry_capacity_and_out_of_bounds_refuse():
    with pytest.raises(ValueError, match="complete exact valid window"):
        ConvolutionGatherRuns((1, 5, 5, 3), (1, 2, 2), (1, 1), (1, 1), (3, 3), 16)
    plan = ConvolutionGatherRuns((1, 5, 5, 3), (1, 3, 3), (1, 1), (1, 1), (3, 3), 16)
    for first, count in ((-1, 1), (9, 1), (0, 17), (8, 2), (0, 0)):
        with pytest.raises(ValueError):
            plan.fragments(first, count)
    with pytest.raises(ValueError, match="byte pitches"):
        ConvolutionGatherRuns((1, 1, 1, 17), (1, 1, 1), (1, 1 << 62), (1, 1), (1, 1), 16)

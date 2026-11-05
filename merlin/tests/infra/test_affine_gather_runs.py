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


class DeferredBuilder(IntegerBuilder):
    @staticmethod
    def evaluate(value, count):
        return value(count) if callable(value) else value

    def op(self, operation, left, right):
        return lambda count: super(DeferredBuilder, self).op(
            operation, self.evaluate(left, count), self.evaluate(right, count))

    def cmp(self, operation, left, right):
        return lambda count: super(DeferredBuilder, self).cmp(
            operation, self.evaluate(left, count), self.evaluate(right, count))

    def select(self, condition, left, right):
        return lambda count: self.evaluate(left if self.evaluate(condition, count) else right, count)


@pytest.mark.parametrize("ow", [1, 7, 14, 28])
def test_dynamic_row_tail_preserves_every_sample_across_source_rows_and_images(ow):
    from merlin.compile.scheduling.affine_gather_runs import emit_convolution_matrix_gather

    plan = ConvolutionGatherRuns((2, 5, 2 * ow - 1, 12), (2, 3, ow),
                                (2, 2), (1, 1), (1, 1), 16)
    builder = DeferredBuilder()
    for first in range(6 * ow):
        bound = min(16, 6 * ow - first)
        dynamic_count = lambda count: count
        emitted = emit_convolution_matrix_gather(
            plan, builder, first, dynamic_count, max_count=bound,
            reduction_block=2, reduction_tile=4)
        for count in range(bound + 1):
            actual = tuple(tuple(builder.evaluate(v, count) for v in fragment) for fragment in emitted)
            actual = tuple(fragment for fragment in actual if fragment[2])
            expected = tuple((base + 8, local, length) for base, local, length in
                             plan.fragments(first, count)) if count else ()
            assert actual == expected


def test_dynamic_gather_requires_a_valid_explicit_bound_and_keeps_static_count_checks():
    plan = ConvolutionGatherRuns((1, 5, 13, 12), (1, 3, 7),
                                (2, 2), (1, 1), (1, 1), 16)
    for bound in (None, 0, 17, True):
        with pytest.raises(ValueError, match="row count"):
            emit_convolution_gather_runs(plan, DeferredBuilder(), 0, lambda count: count,
                                         max_count=bound)
    with pytest.raises(ValueError, match="explicit bounded domain"):
        emit_convolution_gather_runs(plan, IntegerBuilder(), 0, 4, max_count=3)


@pytest.mark.parametrize('stride,dilation', [((1, 1), (1, 1)), ((2, 3), (2, 1))])
@pytest.mark.parametrize('channels,tile', [(16, 16), (32, 16), (6, 3)])
def test_implicit_matrix_tiles_equal_original_ordered_convolution(stride, dilation, channels, tile):
    from merlin.compile.scheduling.affine_gather_runs import emit_convolution_matrix_gather

    b, oh, ow, kh, kw = 2, 3, 7, 2, 3
    sh, sw = stride
    dh, dw = dilation
    h, w = (oh-1)*sh+(kh-1)*dh+1, (ow-1)*sw+(kw-1)*dw+1
    plan = ConvolutionGatherRuns((b, h, w, channels), (b, oh, ow), stride, dilation, (kh, kw), 4)
    for first in range(0, b*oh*ow, 4):
        count = min(4, b*oh*ow-first)
        for block in range(kh*kw*channels//tile):
            fragments = emit_convolution_matrix_gather(plan, IntegerBuilder(), first, count,
                                                       reduction_block=block, reduction_tile=tile)
            actual = []
            for offset, local, rows in fragments:
                for row in range(rows):
                    for c in range(tile):
                        actual.append((local+row, block*tile+c, offset+row*sw*channels+c))
            expected = []
            for row in range(count):
                image, spatial = divmod(first+row, oh*ow)
                y, x = divmod(spatial, ow)
                for c in range(tile):
                    tap, channel = divmod(block*tile+c, channels)
                    ky, kx = divmod(tap, kw)
                    offset = ((image*h+y*sh+ky*dh)*w+x*sw+kx*dw)*channels+channel
                    expected.append((row, block*tile+c, offset))
            assert actual == expected
    bad = ConvolutionGatherRuns((1, 5, 5, 17), (1, 3, 3), (1, 1), (1, 1), (3, 3), 4)
    with pytest.raises(ValueError, match='divide original input channels'):
        emit_convolution_matrix_gather(bad, IntegerBuilder(), 0, 4, reduction_block=0, reduction_tile=16)

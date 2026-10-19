"""Neutral dense provider for host-prepared convolution panels; no device gather."""

from dataclasses import replace

import numpy as np
from test_dense_source_provider import DenseProvider


class ConvolutionProvider(DenseProvider):
    def __init__(self, work, model, geometry, epochs, *, budget=None, route="auto", mutation=None):
        super().__init__(work, model, [], mutation=mutation)
        self.geometry, self.epochs, self.budget, self.route = geometry, tuple(epochs), budget, route

    def selection(self):
        from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects

        return replace(
            super().selection(),
            max_convolution_packing_bytes=self.budget,
            convolution_route=self.route,
            fuse_output_readout=False,
            borrowed_effects=BorrowedBufferEffects(
                "neutral-reviewed-synchronous-convolution-panel-ABI-v1", True, True, True
            ),
        )

    def columns(self, image):
        """Exact source K order ci,kh,kw and position order batch,oh,ow."""
        g = self.geometry
        batch, channels = image.shape[:2]
        kernel_h, kernel_w = g["kernel"]
        out_h, out_w = g["output"]
        step_h, step_w = g["stride"]
        dil_h, dil_w = g["dilation"]
        top, left, _, _ = g["padding"]
        output = np.empty((channels * kernel_h * kernel_w, batch * out_h * out_w), dtype=image.dtype)
        for ci in range(channels):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    k = (ci * kernel_h + kh) * kernel_w + kw
                    for n in range(batch):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                h, w = oh * step_h + kh * dil_h - top, ow * step_w + kw * dil_w - left
                                value = image[n, ci, h, w] if 0 <= h < image.shape[2] and 0 <= w < image.shape[3] else 0
                                output[k, (n * out_h + oh) * out_w + ow] = value
        return output

    def prepare(self, bindings, private_directory):
        widths = [b.compilation.compiled.kernel.args[1].shape[1] for b in bindings]
        batch = self.epochs[0][0].shape[0]
        oh, ow = self.geometry["output"]
        full_size = batch * oh * ow
        trace = []
        for image, weights, expected in self.epochs:
            packed = self.columns(image)
            flat_weights = weights.reshape(weights.shape[0], -1)
            if widths == [full_size]:
                trace.append((0, flat_weights.copy(), packed.copy(), expected.copy()))
                continue
            assert 1 <= len(widths) <= 2
            width = widths[0]
            full, tail = divmod(ow, width)
            assert widths == [width] + ([tail] if tail else [])
            assert width < full_size and full > 0
            # The source rewrite emits every complete row panel, followed by
            # one true-width tail per batch/output row. No independent K tiles.
            for n in range(batch):
                for h in range(oh):
                    for panel in range(full):
                        start = (n * oh + h) * ow + panel * width
                        trace.append(
                            (
                                0,
                                flat_weights.copy(),
                                packed[:, start : start + width].copy(),
                                expected[:, start : start + width].copy(),
                            )
                        )
            if tail:
                for n in range(batch):
                    for h in range(oh):
                        start = (n * oh + h) * ow + full * width
                        trace.append(
                            (
                                1,
                                flat_weights.copy(),
                                packed[:, start : start + tail].copy(),
                                expected[:, start : start + tail].copy(),
                            )
                        )
        self.trace = trace
        super().prepare(bindings, private_directory)

    @staticmethod
    def wrapper(index, symbol, bridge, problem, *, output_width=4):
        assert output_width == 4
        code = DenseProvider.wrapper(index, symbol, bridge, problem, typed_scratch=True)
        entry = f"void {bridge}(const descriptor *a,const descriptor *b,descriptor *out){{"
        assert code.count(entry) == 1
        observed = f"""
#ifdef DENSE_TARGET_CONSOLE
 htif_puts("MERLIN_CONV_PANEL {index}");
 htif_puts(" ");htif_putd((long)((uintptr_t)b->aligned+b->offset*{problem.rhs_bytes}));
 htif_puts(" ");htif_putd((long)b->sizes[0]);htif_puts(" ");htif_putd((long)b->sizes[1]);
 htif_puts(" ");htif_putd((long)b->strides[0]);htif_puts(" ");htif_putd((long)b->strides[1]);
 htif_puts("\\n");
#endif
"""
        return code.replace(entry, entry + observed)

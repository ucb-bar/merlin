"""Portable scalar transpose, bias and epilogue construction for explicit storage.

The caller binds pointers, padded row alignment, output formats and stage order.
No target instruction, physical buffer ownership or numerical policy is inferred.
"""

from xdsl.ir import SSAValue

from .host_tensor_codegen import LoweringDeclined, row_pitch


def _int_range(dtype):
    bits = int(dtype[1:])
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def pool_out_dims(h, w, size, stride, padding):
    ph, pw = size
    sh, sw = stride
    pt, pl, pb, pr = padding
    return (h + pt + pb - ph) // sh + 1, (w + pl + pr - pw) // sw + 1


class HostEpilogueEmitter:
    def __init__(self, builder, pointer_of, fp, *, row_alignment, operand_dtype, pool_floor):
        row_pitch(1, alignment=row_alignment)
        self.fb = builder
        self.ptr = pointer_of
        self.fp = fp
        self.row_alignment = row_alignment
        self.operand_dtype = operand_dtype
        self.pool_floor = int(pool_floor)

    # -- compiler-generated host lane --------------------------------------------------------
    def emit_transpose(self, ins) -> None:
        src, dst = ins.bufs
        rows, cols = ins.attrs["rows"], ins.attrs["cols"]
        dtype = ins.attrs["dtype"]
        sp, dp = self.ptr(src), self.ptr(dst)
        src_pitch = self.fb.const(row_pitch(cols, alignment=self.row_alignment))
        dst_pitch = self.fb.const(row_pitch(rows, alignment=self.row_alignment))

        def outer(i: SSAValue) -> None:
            def inner(j: SSAValue) -> None:
                si = self.fb.add_i(self.fb.mul_i(i, src_pitch), j)
                di = self.fb.add_i(self.fb.mul_i(j, dst_pitch), i)
                self.fb.store_i64(self.fb.load_i64(sp, si, dtype), dp, di, dtype)

            self.fb.for_range(cols, inner)

        self.fb.for_range(rows, outer)

    def emit_bias_add(self, ins) -> None:
        src, bias, dst = ins.bufs
        rows, cols = ins.attrs["rows"], ins.attrs["cols"]
        dtype = ins.attrs["dtype"]
        sp, bp, dp = self.ptr(src), self.ptr(bias), self.ptr(dst)
        pitch = self.fb.const(row_pitch(cols, alignment=self.row_alignment))

        def outer(i: SSAValue) -> None:
            def inner(j: SSAValue) -> None:
                idx = self.fb.add_i(self.fb.mul_i(i, pitch), j)
                v = self.fb.load_i64(sp, idx, dtype)
                b = self.fb.load_i64(bp, j, dtype)
                self.fb.store_i64(self.fb.add_i(v, b), dp, idx, dtype)

            self.fb.for_range(cols, inner)

        self.fb.for_range(rows, outer)

    def _acc_scale(self, value: SSAValue, a: dict) -> SSAValue:
        """Emit the caller-selected f32 scale, ties-even round and integer clamp."""
        scale = a.get("acc_scale")
        if scale is None:
            raise LoweringDeclined("the acc_scale epilogue stage declares no acc_scale value", op="acc_scale")
        return self.fp.f32_readout(value, scale, self.operand_dtype)

    def _apply_scalar_stages(self, value: SSAValue, stages: list[str], a: dict) -> SSAValue:
        """The elementwise part of an epilogue, in the order the ABI declares it."""
        for stage in stages:
            if stage == "relu":
                value = self.fb.max0(value)
            elif stage == "acc_scale":
                value = self._acc_scale(value, a)
            elif stage == "requant":
                shift = int(a["requant_shift"])
                if shift > 0:
                    value = self.fb.ashr_i(self.fb.add_i(value, self.fb.const(1 << (shift - 1))), shift)
                elif shift < 0:
                    value = self.fb.shl_i(value, -shift)
            else:
                raise LoweringDeclined(
                    f"epilogue stage {stage!r} has no generated CPU-lane readout "
                    f"(the caller must supply an implemented scalar stage)",
                    op=stage,
                )
        return value

    def emit_epilogue(self, ins) -> None:
        a = ins.attrs
        src, dst = ins.bufs[0], ins.bufs[1]
        rows, cols = a["rows"], a["cols"]
        stages = list(a["stages"])
        elementwise = [s for s in stages if s != "maxpool"]
        out_dtype = a["out_dtype"]
        stage_dtype = a["stage_dtype"]
        src_row_elems = int(a["src_row_elems"])
        src_row_off = int(a["src_row_offset"])
        sp = self.ptr(src)
        src_pitch = self.fb.const(src_row_elems)
        src_off = self.fb.const(src_row_off * src_row_elems)

        def src_index(i: SSAValue, j: SSAValue) -> SSAValue:
            return self.fb.add_i(self.fb.add_i(self.fb.mul_i(i, src_pitch), j), src_off)

        def gather(i: SSAValue, j: SSAValue) -> SSAValue:
            """One staged accumulator element with the elementwise stages already applied.

            The stages are folded into the gather rather than written back into the staging
            buffer: they are elementwise, so applying them per element is the ABI order either
            way, and the staging buffer stays write-once for its producer.
            """
            return self._apply_scalar_stages(self.fb.load_i64(sp, src_index(i, j), stage_dtype), elementwise, a)

        if "maxpool" in stages:
            self._emit_maxpool(ins, gather)
            return

        dp = self.ptr(dst)
        dst_row_elems = int(a["dst_row_elems"])
        dst_pitch = self.fb.const(dst_row_elems)
        dst_off = self.fb.const(int(a["dst_row_offset"]) * dst_row_elems)
        lo, hi = _int_range(out_dtype)

        def outer2(i: SSAValue) -> None:
            def inner2(j: SSAValue) -> None:
                v = self.fb.clamp(gather(i, j), lo, hi)
                di = self.fb.add_i(self.fb.add_i(self.fb.mul_i(i, dst_pitch), j), dst_off)
                self.fb.store_i64(v, dp, di, out_dtype)

            self.fb.for_range(cols, inner2)

        self.fb.for_range(rows, outer2)

    def _emit_maxpool(self, ins, gather) -> None:
        a = ins.attrs
        rows, cols = a["rows"], a["cols"]
        ih, iw = (int(v) for v in a["pool_in_dims"])
        ph, pw = (int(v) for v in a["pool_size"])
        sh, sw = (int(v) for v in a["pool_stride"])
        pt, pl, pb, pr = (int(v) for v in a["pool_padding"])
        pad_value = a["pool_pad_value"]
        plane = ih * iw
        if plane <= 0 or rows % plane:
            raise LoweringDeclined(
                f"maxpool geometry {ih}x{iw} does not divide the {rows} accumulator rows", op="maxpool"
            )
        batch = rows // plane
        ho, wo = pool_out_dims(ih, iw, (ph, pw), (sh, sw), (pt, pl, pb, pr))
        if ho < 1 or wo < 1:
            raise LoweringDeclined("maxpool window leaves no output position", op="maxpool")
        padded = any((pt, pl, pb, pr))
        if padded and pad_value is None:
            raise LoweringDeclined("maxpool padding declares no pool_pad_value", op="maxpool")
        out_dtype = a["out_dtype"]
        lo, hi = _int_range(out_dtype)
        dp = self.ptr(ins.bufs[1])
        dst_row_elems = int(a["dst_row_elems"])
        dst_pitch = self.fb.const(dst_row_elems)
        dst_off = self.fb.const(int(a["dst_row_offset"]) * dst_row_elems)
        iw_c = self.fb.const(iw)
        plane_c = self.fb.const(plane)
        wo_c = self.fb.const(wo)
        ho_c = self.fb.const(ho)
        pad_c = self.fb.const(int(pad_value) if pad_value is not None else 0)
        floor_c = self.fb.const(self.pool_floor)

        def loop_n(n: SSAValue) -> None:
            def loop_oy(oy: SSAValue) -> None:
                def loop_ox(ox: SSAValue) -> None:
                    y0 = self.fb.sub_i(self.fb.mul_i(oy, self.fb.const(sh)), self.fb.const(pt))
                    x0 = self.fb.sub_i(self.fb.mul_i(ox, self.fb.const(sw)), self.fb.const(pl))

                    def loop_c(cc: SSAValue) -> None:
                        best = floor_c
                        for ky in range(ph):
                            y = self.fb.add_i(y0, self.fb.const(ky))
                            for kx in range(pw):
                                x = self.fb.add_i(x0, self.fb.const(kx))
                                mask = None
                                gy, gx = y, x
                                if padded:
                                    mask = self.fb.and_i(
                                        self.fb.and_i(
                                            self.fb.mask_nonneg(y),
                                            self.fb.mask_neg(self.fb.sub_i(y, self.fb.const(ih))),
                                        ),
                                        self.fb.and_i(
                                            self.fb.mask_nonneg(x),
                                            self.fb.mask_neg(self.fb.sub_i(x, self.fb.const(iw))),
                                        ),
                                    )
                                    # keep the gather in bounds; the blend below substitutes the
                                    # declared pad value wherever this clamp invented a cell.
                                    gy = self.fb.clamp(y, 0, ih - 1)
                                    gx = self.fb.clamp(x, 0, iw - 1)
                                row = self.fb.add_i(
                                    self.fb.mul_i(n, plane_c), self.fb.add_i(self.fb.mul_i(gy, iw_c), gx)
                                )
                                v = gather(row, cc)
                                if mask is not None:
                                    v = self.fb.blend(mask, v, pad_c)
                                best = self.fb.smax(best, v)
                        best = self.fb.clamp(best, lo, hi)
                        out_row = self.fb.add_i(self.fb.mul_i(self.fb.add_i(self.fb.mul_i(n, ho_c), oy), wo_c), ox)
                        di = self.fb.add_i(self.fb.add_i(self.fb.mul_i(out_row, dst_pitch), cc), dst_off)
                        self.fb.store_i64(best, dp, di, out_dtype)

                    self.fb.for_range(cols, loop_c)

                self.fb.for_range(wo, loop_ox)

            self.fb.for_range(ho, loop_oy)

        self.fb.for_range(batch, loop_n)

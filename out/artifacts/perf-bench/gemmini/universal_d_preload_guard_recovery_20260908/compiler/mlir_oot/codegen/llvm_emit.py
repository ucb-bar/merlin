"""Emit the target artifact: an LLVM-dialect module defining `@gemmini_kernel`.

Every accelerator command becomes one `llvm.inline_asm` wrapping a raw `.insn` directive whose
opcode / funct3 / funct7 come from the RTL-derived decode table, and whose two operands are
always SSA values defined earlier in the function.  DRAM addresses are `llvm.ptrtoint` of the
kernel's own pointer arguments (plus a constant tile offset) — never a literal address.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i8, i32, i64
from xdsl.ir import SSAValue

from ..lowering.plan import (DTYPE_BYTES, LoweringDeclined, Plan, pool_out_dims,
                             row_pitch)
from ..lowering.schedule import Instr
from ..tables import isa
from ..tables import rtl_facts as F
from .builder import INT_TYPES, PTR, FnBuilder
from .fpbuilder import FpBuilder
from .host_linalg import HostLinalg
from .loop_host_linalg import LoopHostLinalg

INT32_MIN = -(1 << 31)


class Emitter:
    def __init__(self, plan: Plan, instrs: list[Instr], staging: dict[str, Any]):
        self.plan = plan
        self.instrs = instrs
        self.staging = staging
        self.args = list(plan.kernel_args)
        self.fb = FnBuilder([PTR] * len(self.args))
        self.fb.unroll_all = True
        self.globals: list[llvm.GlobalOp] = []
        self.helpers: list[llvm.FuncOp] = []
        self._radix_helpers: dict[tuple[int, int, bool, bool, bool], str] = {}
        self._ptr: dict[str, SSAValue] = {}
        self._base: dict[str, SSAValue] = {}
        self._addr: dict[tuple[str, int], SSAValue] = {}
        self._host_scratch_slots: list[dict[str, Any]] = []
        self._host_scratch_cursor = 0
        for idx, name in enumerate(self.args):
            self._ptr[name] = self.fb.entry.args[idx]

    # -- buffers -----------------------------------------------------------------------------
    def _prologue(self, op):
        """Definitions every later command may use live at the top of the single entry block."""
        self.fb.prologue(op)
        return op.results[0]

    def ptr(self, name: str) -> SSAValue:
        hit = self._ptr.get(name)
        if hit is not None:
            return hit
        buf = self.staging.get(name) or self.plan.buffers.get(name)
        if buf is None or buf.role != "scratch":
            raise LoweringDeclined(
                f"buffer {name!r} is neither a kernel argument nor a compiler staging buffer, so "
                f"the kernel has no address for it")
        sym = f"__gemmini_stage_{len(self.globals)}"
        self.globals.append(
            llvm.GlobalOp(llvm.LLVMArrayType(buf.nbytes, i8), sym, "internal", alignment=64))
        value = self._prologue(llvm.AddressOfOp(sym, PTR))
        self._ptr[name] = value
        return value

    def base_int(self, name: str) -> SSAValue:
        hit = self._base.get(name)
        if hit is not None:
            return hit
        value = self._prologue(llvm.PtrToIntOp(self.ptr(name), i64))
        self._base[name] = value
        return value

    def dram_addr(self, name: str, offset: int) -> SSAValue:
        key = (name, int(offset))
        hit = self._addr.get(key)
        if hit is not None:
            return hit
        base = self.base_int(name)
        value = base if offset == 0 else self._prologue(
            llvm.AddOp(base, self.fb.const(offset)))
        self._addr[key] = value
        return value

    def host_scratch(self, nbytes: int) -> SSAValue:
        """Return one reusable aligned DRAM slot for a host-segment temporary."""
        slot_id = self._host_scratch_cursor
        self._host_scratch_cursor += 1
        if slot_id == len(self._host_scratch_slots):
            sym = f"__gemmini_host_stage_{slot_id}"
            global_index = len(self.globals)
            self.globals.append(
                llvm.GlobalOp(llvm.LLVMArrayType(nbytes, i8), sym, "internal", alignment=64))
            ptr = self._prologue(llvm.AddressOfOp(sym, PTR))
            self._host_scratch_slots.append(
                {"sym": sym, "bytes": nbytes, "global_index": global_index, "ptr": ptr})
            return ptr
        slot = self._host_scratch_slots[slot_id]
        if nbytes > slot["bytes"]:
            slot["bytes"] = nbytes
            self.globals[slot["global_index"]] = llvm.GlobalOp(
                llvm.LLVMArrayType(nbytes, i8), slot["sym"], "internal", alignment=64)
        return slot["ptr"]

    # -- instructions ------------------------------------------------------------------------
    def rocc(self, funct: int, rs1: SSAValue, rs2: SSAValue) -> None:
        self.fb.add(llvm.InlineAsmOp(isa.asm_string(funct), "r,r", [rs1, rs2], [],
                                     has_side_effects=True))

    def emit_command(self, ins: Instr) -> None:
        a = ins.attrs
        if ins.kind == "flush":
            f, rs1, rs2 = isa.flush(a.get("skip", 0))
        elif ins.kind == "config_ex":
            f, rs1, rs2 = isa.config_ex(dataflow=a["dataflow"], sys_act=a["act"],
                                        acc_scale=a["acc_scale"], a_stride=a["a_stride"],
                                        c_stride=a["c_stride"])
        elif ins.kind == "config_ld":
            f, rs1, rs2 = isa.config_ld(stride=a["stride"], scale=a.get("scale", 1.0),
                                        shrunk=a.get("shrunk", False),
                                        load_id=a.get("load_id", 0))
        elif ins.kind == "config_st":
            f, rs1, rs2 = isa.config_st(stride=a["stride"], acc_act=a["acc_act"],
                                        acc_scale=a["acc_scale"],
                                        pool_stride=a.get("pool_stride", 0),
                                        pool_size=a.get("pool_size", 0),
                                        pool_out_dim=a.get("pool_out_dim", 0),
                                        porows=a.get("porows", 0), pocols=a.get("pocols", 0),
                                        orows=a.get("orows", 0), ocols=a.get("ocols", 0),
                                        upad=a.get("upad", 0), lpad=a.get("lpad", 0))
        elif ins.kind == "mvin":
            f, _, rs2 = isa.mvin(local_addr=a["local"], cols=a["cols"], rows=a["rows"],
                                 load_id=a.get("load_id", 0))
            self.rocc(f, self.dram_addr(ins.bufs[0], a["offset"]), self.fb.const(rs2))
            return
        elif ins.kind == "mvout":
            f, _, rs2 = isa.mvout(local_addr=a["local"], cols=a["cols"], rows=a["rows"])
            self.rocc(f, self.dram_addr(ins.bufs[0], a["offset"]), self.fb.const(rs2))
            return
        elif ins.kind == "preload":
            f, rs1, rs2 = isa.preload(bd_addr=a["bd"], c_addr=a["c"], bd_cols=a["bd_cols"],
                                      bd_rows=a["bd_rows"], c_cols=a["c_cols"],
                                      c_rows=a["c_rows"])
        elif ins.kind == "compute":
            f, rs1, rs2 = isa.compute(a_addr=a["a"], bd_addr=a["bd"], a_cols=a["a_cols"],
                                      a_rows=a["a_rows"], accumulate=a["accumulate"])
        elif ins.kind == "fence":
            self.fb.add(llvm.InlineAsmOp("fence", "~{memory}", [], [], has_side_effects=True))
            return
        else:                                                     # pragma: no cover - guarded
            raise LoweringDeclined(f"no encoding for scheduled command {ins.kind!r}")
        self.rocc(f, self.fb.const(rs1), self.fb.const(rs2))

    # -- compiler-generated host lane --------------------------------------------------------
    def emit_transpose(self, ins: Instr) -> None:
        src, dst = ins.bufs
        rows, cols = ins.attrs["rows"], ins.attrs["cols"]
        dtype = ins.attrs["dtype"]
        sp, dp = self.ptr(src), self.ptr(dst)
        src_pitch = self.fb.const(row_pitch(cols))
        dst_pitch = self.fb.const(row_pitch(rows))

        def outer(i: SSAValue) -> None:
            def inner(j: SSAValue) -> None:
                si = self.fb.add_i(self.fb.mul_i(i, src_pitch), j)
                di = self.fb.add_i(self.fb.mul_i(j, dst_pitch), i)
                self.fb.store_i64(self.fb.load_i64(sp, si, dtype), dp, di, dtype)

            self.fb.for_range(cols, inner)

        self.fb.for_range(rows, outer)

    def emit_bias_add(self, ins: Instr) -> None:
        src, bias, dst = ins.bufs
        rows, cols = ins.attrs["rows"], ins.attrs["cols"]
        dtype = ins.attrs["dtype"]
        sp, bp, dp = self.ptr(src), self.ptr(bias), self.ptr(dst)
        pitch = self.fb.const(row_pitch(cols))

        def outer(i: SSAValue) -> None:
            def inner(j: SSAValue) -> None:
                idx = self.fb.add_i(self.fb.mul_i(i, pitch), j)
                v = self.fb.load_i64(sp, idx, dtype)
                b = self.fb.load_i64(bp, j, dtype)
                self.fb.store_i64(self.fb.add_i(v, b), dp, idx, dtype)

            self.fb.for_range(cols, inner)

        self.fb.for_range(rows, outer)

    def emit_im2col_row(self, ins: Instr) -> None:
        """Lower one target-dialect im2col row to bounded CPU gather loops."""
        a = ins.attrs
        src, dst = (self.ptr(name) for name in ins.bufs)
        ci, hi, wi = int(a["ci"]), int(a["hi"]), int(a["wi"])
        kh, kw, wo = int(a["kh"]), int(a["kw"]), int(a["wo"])
        sh, sw = int(a["stride_h"]), int(a["stride_w"])
        dh, dw = int(a["dilation_h"]), int(a["dilation_w"])
        pt, pl = int(a["pad_top"]), int(a["pad_left"])
        batch, oy = int(a["batch"]), int(a["out_y"])
        src_pitch, dst_pitch = row_pitch(wi), row_pitch(wo)
        kernel_area = kh * kw

        def loop_k(k) -> None:
            channel = self.fb.udiv_i(k, self.fb.const(kernel_area))
            rem = self.fb.urem_i(k, self.fb.const(kernel_area))
            ky = self.fb.udiv_i(rem, self.fb.const(kw))
            kx = self.fb.urem_i(rem, self.fb.const(kw))
            iy = self.fb.sub_i(
                self.fb.add_i(self.fb.const(oy * sh), self.fb.mul_i(ky, self.fb.const(dh))),
                self.fb.const(pt))

            def loop_x(ox) -> None:
                ix = self.fb.sub_i(
                    self.fb.add_i(self.fb.mul_i(ox, self.fb.const(sw)),
                                  self.fb.mul_i(kx, self.fb.const(dw))), self.fb.const(pl))
                valid_y = self.fb.and_i(self.fb.mask_nonneg(iy),
                    self.fb.mask_neg(self.fb.sub_i(iy, self.fb.const(hi))))
                valid_x = self.fb.and_i(self.fb.mask_nonneg(ix),
                    self.fb.mask_neg(self.fb.sub_i(ix, self.fb.const(wi))))
                valid = self.fb.and_i(valid_y, valid_x)
                gy, gx = self.fb.clamp(iy, 0, hi - 1), self.fb.clamp(ix, 0, wi - 1)
                src_index = self.fb.add_i(
                    self.fb.mul_i(
                        self.fb.add_i(self.fb.mul_i(self.fb.const(batch * ci), self.fb.const(hi)),
                                      self.fb.add_i(self.fb.mul_i(channel, self.fb.const(hi)), gy)),
                        self.fb.const(src_pitch)), gx)
                value = self.fb.load_i64(src, src_index, "i8")
                value = self.fb.blend(valid, value, self.fb.const(0))
                dst_index = self.fb.add_i(self.fb.mul_i(k, self.fb.const(dst_pitch)), ox)
                self.fb.store_i64(value, dst, dst_index, "i8")

            self.fb.for_range(wo, loop_x)

        self.fb.for_range(ci * kh * kw, loop_k)

    def emit_loop_ws_block(self, ins: Instr) -> None:
        """Lower a capacity-bounded matrix block through Gemmini's LOOP_WS sequencer."""
        a = ins.attrs
        if len(ins.bufs) == 4:
            a_name, b_name, d_name, c_name = ins.bufs
        else:
            a_name, b_name, c_name = ins.bufs
            d_name = c_name
        c_dtype = str(a.get("c_dtype", "i32"))
        d_stride = int(a.get("d_stride", a["c_stride"]))
        c_byte_stride = int(a["c_stride"]) * DTYPE_BYTES[c_dtype]
        d_byte_stride = d_stride * DTYPE_BYTES["i32"]

        # These are the same ordinary Gemmini configurations used by tiled_matmul_auto.  The
        # LOOP_WS engine consumes three independent load streams: A, B and the optional i32 D.
        configs = (
            isa.config_ex(dataflow=isa.WEIGHT_STATIONARY, sys_act=isa.NO_ACTIVATION),
            isa.config_st(stride=c_byte_stride, acc_act=isa.NO_ACTIVATION,
                          acc_scale=float(a.get("acc_scale", 1.0))),
            isa.config_ld(stride=int(a["a_stride"]), load_id=0),
            isa.config_ld(stride=int(a["b_stride"]), load_id=1),
            isa.config_ld(stride=d_byte_stride, shrunk=False, load_id=2),
        )
        for funct, rs1, rs2 in configs:
            self.rocc(funct, self.fb.const(rs1), self.fb.const(rs2))

        bounds, strides_ab, strides_dc, launch = isa.loop_ws_static(
            rows=int(a["rows"]), cols=int(a["cols"]), depth=int(a["depth"]),
            row_stride_a=int(a["a_stride"]), row_stride_b=int(a["b_stride"]),
            row_stride_c=int(a["c_stride"]), row_stride_d=d_stride,
            full_c=bool(a["full_c"]),
            accumulate=bool(a["accumulate"]))
        self.rocc(*[bounds[0], self.fb.const(bounds[1]), self.fb.const(bounds[2])])
        self.rocc(isa.K_LOOP_WS_CONFIG_ADDRS_AB,
                  self.dram_addr(a_name, int(a["a_offset"])),
                  self.dram_addr(b_name, int(a["b_offset"])))
        d_addr = (self.dram_addr(d_name, int(a["d_offset"]))
                  if bool(a["accumulate"]) else self.fb.const(0))
        self.rocc(isa.K_LOOP_WS_CONFIG_ADDRS_DC, d_addr,
                  self.dram_addr(c_name, int(a["c_offset"])))
        for record in (strides_ab, strides_dc, launch):
            self.rocc(record[0], self.fb.const(record[1]), self.fb.const(record[2]))

    def emit_radix128_step(self, ins: Instr) -> None:
        """Call a shape-specialized helper for one signed base-128 recovery digit.

        Thousands of tiles reuse a small set of shapes. Outlining their two loops prevents the
        LLVM CFG and xDSL verifier cost from growing by six blocks per digit store.
        """
        digit_name, correction_name, partial_name, dst_name = ins.bufs
        rows, cols = int(ins.attrs["rows"]), int(ins.attrs["cols"])
        reset = bool(ins.attrs["reset_partial"])
        commit = bool(ins.attrs["commit"])
        accumulate_dst = bool(ins.attrs["accumulate_dst"])
        key = (rows, cols, reset, commit, accumulate_dst)
        helper_name = self._radix_helpers.get(key)
        if helper_name is None:
            helper_name = f"__gemmini_radix128_{len(self._radix_helpers)}"
            self._radix_helpers[key] = helper_name
            helper = FnBuilder([PTR, PTR, PTR, PTR, i64, i64, i64, i64])
            digit, correction, partial, dst, pitch, unit, dst_pitch, dst_base = helper.entry.args

            def outer(i: SSAValue) -> None:
                def inner(j: SSAValue) -> None:
                    compact = helper.add_i(helper.mul_i(i, pitch), j)
                    recovered = (helper.const(0) if reset
                                 else helper.load_i64(partial, compact, "i32"))
                    digit_value = helper.load_i64(digit, compact, "i8")
                    recovered = helper.add_i(recovered, helper.mul_i(digit_value, unit))
                    helper.store_i64(recovered, partial, compact, "i32")
                    helper.store_i64(helper.sub_i(helper.const(0), recovered),
                                     correction, compact, "i32")
                    if commit:
                        target = helper.add_i(
                            dst_base, helper.add_i(helper.mul_i(i, dst_pitch), j))
                        value = (helper.add_i(helper.load_i64(dst, target, "i32"), recovered)
                                 if accumulate_dst else recovered)
                        helper.store_i64(value, dst, target, "i32")

                helper.for_range(cols, inner)

            helper.for_range(rows, outer)
            region = helper.finish()
            self.helpers.append(llvm.FuncOp(
                helper_name,
                llvm.LLVMFunctionType([PTR, PTR, PTR, PTR, i64, i64, i64, i64]),
                linkage=llvm.LinkageAttr("internal"), body=region))
        self.fb.add(llvm.CallOp(
            helper_name,
            self.ptr(digit_name), self.ptr(correction_name), self.ptr(partial_name),
            self.ptr(dst_name), self.fb.const(int(ins.attrs["row_elems"])),
            self.fb.const(int(ins.attrs["unit"])),
            self.fb.const(int(ins.attrs["dst_row_elems"])),
            self.fb.const(int(ins.attrs["dst_offset_elems"]))))

    @property
    def fp(self) -> FpBuilder:
        """One `FpBuilder` per emitted function, so its constants are shared."""
        hit = getattr(self, "_fp", None)
        if hit is None:
            hit = FpBuilder(self.fb)
            self._fp = hit
        return hit

    def _acc_scale(self, value: SSAValue, a: dict) -> SSAValue:
        """The `acc_scale` stage on the CPU lane, as the ISA header defines it.

        `gemmini_params.h` states the stage outright --
        `ACC_SCALE(x, scale) = clamp(ROUND_NEAR_EVEN((float)x * scale))` -- and the ABI's own
        line says the same thing (`clamp_i8(round_near_even(acc * scale))`).  Both are followed
        here rather than approximated: the product is taken in f32 (which is the width the
        hardware's scale unit multiplies in, so a value past 2**24 loses the same bits on both
        paths), the rounding is the header's round-half-to-even, and the clamp is to the
        target's OWN operand dtype read from the RTL facts, never a hardcoded width.

        This is the readout the store path cannot take: it applies its scale only on a
        NARROWING readout with a positive scale, so a full-width (or otherwise-shaped) commit
        used to have nowhere to put the stage and was refused.
        """
        scale = a.get("acc_scale")
        if scale is None:
            raise LoweringDeclined(
                "the acc_scale epilogue stage declares no acc_scale value", op="acc_scale")
        fp = self.fp
        scaled = fp.round_near_even(fp.fmul(fp.sitofp(value), fp.fconst(float(scale))))
        lo, hi = _int_range(F.OPERAND_DTYPE)
        # Clamp in the FLOAT domain first: a scale large enough to leave the integer range
        # would otherwise be converted before it is bounded.
        return self.fb.clamp(fp.fptosi(fp.fclamp(scaled, float(lo), float(hi))), lo, hi)

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
                    value = self.fb.ashr_i(
                        self.fb.add_i(value, self.fb.const(1 << (shift - 1))), shift)
                elif shift < 0:
                    value = self.fb.shl_i(value, -shift)
            else:
                raise LoweringDeclined(
                    f"epilogue stage {stage!r} has no generated CPU-lane readout "
                    f"(the accelerator store path covers it only for a narrowing readout)",
                    op=stage)
        return value

    def emit_epilogue(self, ins: Instr) -> None:
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
            way, and the staging buffer stays write-once (the accelerator's store DMA owns it).
            """
            return self._apply_scalar_stages(
                self.fb.load_i64(sp, src_index(i, j), stage_dtype), elementwise, a)

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
                v = gather(i, j)
                if a.get("integer_output_policy", "saturate") != "modular":
                    v = self.fb.clamp(v, lo, hi)
                di = self.fb.add_i(self.fb.add_i(self.fb.mul_i(i, dst_pitch), j), dst_off)
                self.fb.store_i64(v, dp, di, out_dtype)

            self.fb.for_range(cols, inner2)

        self.fb.for_range(rows, outer2)

    def _emit_maxpool(self, ins: Instr, gather) -> None:
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
                f"maxpool geometry {ih}x{iw} does not divide the {rows} accumulator rows",
                op="maxpool")
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
        floor_c = self.fb.const(INT32_MIN)

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
                                        self.fb.and_i(self.fb.mask_nonneg(y),
                                                      self.fb.mask_neg(
                                                          self.fb.sub_i(y, self.fb.const(ih)))),
                                        self.fb.and_i(self.fb.mask_nonneg(x),
                                                      self.fb.mask_neg(
                                                          self.fb.sub_i(x, self.fb.const(iw)))))
                                    # keep the gather in bounds; the blend below substitutes the
                                    # declared pad value wherever this clamp invented a cell.
                                    gy = self.fb.clamp(y, 0, ih - 1)
                                    gx = self.fb.clamp(x, 0, iw - 1)
                                row = self.fb.add_i(
                                    self.fb.mul_i(n, plane_c),
                                    self.fb.add_i(self.fb.mul_i(gy, iw_c), gx))
                                v = gather(row, cc)
                                if mask is not None:
                                    v = self.fb.blend(mask, v, pad_c)
                                best = self.fb.smax(best, v)
                        best = self.fb.clamp(best, lo, hi)
                        out_row = self.fb.add_i(
                            self.fb.mul_i(self.fb.add_i(self.fb.mul_i(n, ho_c), oy), wo_c), ox)
                        di = self.fb.add_i(
                            self.fb.add_i(self.fb.mul_i(out_row, dst_pitch), cc), dst_off)
                        self.fb.store_i64(best, dp, di, out_dtype)

                    self.fb.for_range(cols, loop_c)

                self.fb.for_range(wo, loop_ox)

            self.fb.for_range(ho, loop_oy)

        self.fb.for_range(batch, loop_n)

    # -- host lane ---------------------------------------------------------------------------
    def host_lowering(self) -> HostLinalg:
        """The one CPU-lane lowering this kernel uses, shared by every host segment.

        The scalar primitive/constant cache is shared, while each segment rebinds its live tensor
        values at the explicit memory seam.
        """
        hit = getattr(self, "_host", None)
        if hit is None:
            hit = LoopHostLinalg(FpBuilder(self.fb), [], [], self.host_scratch)
            self._host = hit
        return hit

    def emit_host_segment(self, ins: Instr) -> None:
        segment = ins.attrs["segment"]
        self._host_scratch_cursor = 0
        lowering = self.host_lowering()
        lowering.run_segment(
            segment.ops,
            [(value, self.ptr(name)) for value, name in segment.inputs],
            [(value, self.ptr(name)) for value, name in segment.outputs])

    def emit_host_linalg(self, ins: Instr) -> None:
        """Generate the CPU-lane program for a module placed entirely on the host lane."""
        fp = FpBuilder(self.fb)
        self._host_scratch_cursor = 0
        lowering = LoopHostLinalg(fp,
                              [self.ptr(n) for n in ins.attrs["arg_buffers"]],
                              [self.ptr(n) for n in ins.attrs["out_buffers"]],
                              self.host_scratch,
                              input_prologue=ins.attrs.get("input_prologue"))
        lowering.run(ins.attrs["func"])

    # -- driver ------------------------------------------------------------------------------
    def build(self) -> ModuleOp:
        for ins in self.instrs:
            self.fb.emission_attributes = ({"merlin.global_task": IntegerAttr(
                ins.attrs["global_task_index"], i64)}
                if "global_task_index" in ins.attrs else {})
            if ins.kind == "host_transpose":
                self.emit_transpose(ins)
            elif ins.kind == "host_bias_add":
                self.emit_bias_add(ins)
            elif ins.kind == "im2col_row":
                self.emit_im2col_row(ins)
            elif ins.kind == "loop_ws_block":
                self.emit_loop_ws_block(ins)
            elif ins.kind == "host_radix128_step":
                self.emit_radix128_step(ins)
            elif ins.kind == "host_epilogue":
                self.emit_epilogue(ins)
            elif ins.kind == "host_linalg":
                self.emit_host_linalg(ins)
            elif ins.kind == "host_segment":
                self.emit_host_segment(ins)
            else:
                self.emit_command(ins)
        self.fb.emission_attributes = {}
        region = self.fb.finish()
        fn = llvm.FuncOp("gemmini_kernel", llvm.LLVMFunctionType([PTR] * len(self.args)),
                         linkage=llvm.LinkageAttr("external"), body=region)
        module = ModuleOp([*self.globals, *self.helpers, fn])
        module.verify()
        return module


def _int_range(dtype: str) -> tuple[int, int]:
    bits = int(dtype[1:])
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def emit(plan: Plan, instrs: list[Instr], staging: dict[str, Any]) -> ModuleOp:
    return Emitter(plan, instrs, staging).build()


def declined_artifact(reason: str) -> ModuleOp:
    """The target ARTIFACT for a declined capsule: a kernel with no command in it.

    The refusal itself is carried by the command buffer's `declined` entry; this exists so that
    `emit_target_artifact` answers with a module rather than with nothing, which is what keeps the
    four entrypoints consistent about the same capsule.
    """
    fb = FnBuilder([])
    region = fb.finish()
    fn = llvm.FuncOp("gemmini_kernel", llvm.LLVMFunctionType([]),
                     linkage=llvm.LinkageAttr("external"), body=region)
    module = ModuleOp([fn])
    module.attributes["gemmini.declined"] = StringAttr(reason)
    module.verify()
    return module

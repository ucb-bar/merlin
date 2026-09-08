"""Emit the target artifact: an LLVM-dialect module defining `@gemmini_kernel`.

Every accelerator command becomes one `llvm.inline_asm` wrapping a raw `.insn` directive whose
opcode / funct3 / funct7 come from the RTL-derived decode table, and whose two operands are
always SSA values defined earlier in the function.  DRAM addresses are `llvm.ptrtoint` of the
kernel's own pointer arguments (plus a constant tile offset) — never a literal address.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, ModuleOp, StringAttr, TensorType, i8, i32, i64
from xdsl.ir import SSAValue

from ..lowering.plan import (DTYPE_BYTES, LoweringDeclined, Plan, pool_out_dims,
                             row_pitch)
from ..lowering.schedule import Instr
from ..tables import isa
from ..tables import rtl_facts as F
from .builder import INT_TYPES, PTR, FnBuilder
from .fpbuilder import FpBuilder
from ..tables import loop_ws
from ..tables import loop_conv
from .host_linalg import HostLinalg
from .loop_host_linalg import LoopHostLinalg

INT32_MIN = -(1 << 31)


def _universal_valid_x_span(*, wi: int, kw: int, wo: int, stride_w: int,
                            dilation_w: int, pad_left: int) -> tuple[int, int]:
    """Return the output-column interval valid for every kernel column.

    For ``ix = ox * stride_w + kx * dilation_w - pad_left``, validity is
    monotone in ``kx``.  The smallest input coordinate is therefore at
    ``kx == 0`` and the largest at ``kx == kw - 1``.  Classifying their common
    interval once lets the generated packer omit bounds masks and clamping for
    the interior without specializing on a model or shape.
    """
    if min(wi, kw, wo, stride_w, dilation_w) <= 0 or pad_left < 0:
        raise LoweringDeclined(
            "im2col span classification requires positive static extents/strides and "
            "nonnegative padding", op="conv2d")
    # ceil(pad_left / stride_w), followed by the first ox whose largest kx is
    # outside the input.  Clip because valid convolution geometry may have no
    # column common to every kernel position.
    lo = (pad_left + stride_w - 1) // stride_w
    hi = (wi - 1 + pad_left - (kw - 1) * dilation_w) // stride_w + 1
    lo = max(0, min(wo, lo))
    hi = max(lo, min(wo, hi))
    return lo, hi


class Emitter:
    def __init__(self, plan: Plan, instrs: list[Instr], staging: dict[str, Any]):
        self.plan = plan
        self.instrs = instrs
        self.staging = staging
        self.args = list(plan.kernel_args)
        self.fb = FnBuilder([PTR] * len(self.args))
        self.fb.unroll_all = True
        self.globals: list[llvm.GlobalOp] = []
        self._ptr: dict[str, SSAValue] = {}
        self._base: dict[str, SSAValue] = {}
        self._addr: dict[tuple[str, int], SSAValue] = {}
        self._host_lowerings: list[LoopHostLinalg] = []
        self._host_workspace_symbol = "__merlin_host_workspace"
        self._host_workspace_base = None
        self._host_workspace_offsets: dict[int, SSAValue] = {}
        for idx, name in enumerate(self.args):
            self._ptr[name] = self.fb.entry.args[idx]

    # -- buffers -----------------------------------------------------------------------------
    def _prologue(self, op):
        """Definitions every later command may use live at the top of the single entry block."""
        owner = getattr(self, "_source_conv_owner", None)
        if owner is not None:
            op.attributes["merlin.global_task"] = IntegerAttr(owner, i64)
            # This straight-line task block dominates every later use of its cached
            # staging address. Do not hoist new global address definitions across
            # earlier task work without a shared-global prologue contract.
            self.fb.add(op)
            return op.results[0]
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

    def host_workspace_pointer(self, offset: int) -> SSAValue:
        """A reusable byte-addressed host workspace slice, defined in the kernel prologue."""
        offset = int(offset)
        hit = self._host_workspace_offsets.get(offset)
        if hit is not None:
            return hit
        if self._host_workspace_base is None:
            self._host_workspace_base = self._prologue(
                llvm.AddressOfOp(self._host_workspace_symbol, PTR))
        value = self._host_workspace_base
        if offset:
            gep = llvm.GEPOp(
                value, [llvm.GEP_USE_SSA_VAL], i8,
                ssa_indices=[self.fb.const(offset)])
            value = self._prologue(gep)
        self._host_workspace_offsets[offset] = value
        return value

    def _new_host_lowering(self, arg_ptrs, out_ptrs) -> LoopHostLinalg:
        lowering = LoopHostLinalg(FpBuilder(self.fb), arg_ptrs, out_ptrs)
        lowering.constant_tensor = self.host_constant_tensor
        lowering.workspace_pointer = self.host_workspace_pointer
        self._host_lowerings.append(lowering)
        return lowering

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
                                        c_stride=a["c_stride"],
                                        a_transpose=a.get("a_transpose", False),
                                        b_transpose=a.get("b_transpose", False))
        elif ins.kind == "config_ld":
            f, rs1, rs2 = isa.config_ld(stride=a["stride"], scale=a.get("scale", 1.0),
                                        shrunk=a.get("shrunk", False),
                                        load_id=a.get("load_id", 0),
                                        block_stride=a.get("block_stride", isa.DIM),
                                        pixel_repeats=a.get("pixel_repeats", 1))
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

    def emit_zero_i32(self, ins: Instr) -> None:
        """Initialize the compiler-owned LoopConv LOAD3 carrier; staging globals are undef."""
        ptr = self.ptr(ins.bufs[0])
        self.fb.for_range(
            int(ins.attrs["elements"]),
            lambda index: self.fb.store_i64(self.fb.const(0), ptr, index, "i32"))

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
        """Lower one im2col row with affine interior spans and guarded borders.

        All geometry is a verified target-op attribute.  The old lowering
        rebuilt the flattened-k quotient/remainder, both bounds predicates,
        clamps, and the complete NCHW address inside every output-column
        iteration.  Here the k-derived row bases are explicit loop invariants,
        while the statically common x interior omits x bounds/clamps entirely.
        Border columns retain the exact guarded gather.  Rows for which every
        kernel y is in range similarly omit the y guard; padded top/bottom rows
        keep it.  This changes only host address generation, not the staging
        encoding, synchronization, or accelerator schedule.
        """
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
        interior_lo, interior_hi = _universal_valid_x_span(
            wi=wi, kw=kw, wo=wo, stride_w=sw,
            dilation_w=dw, pad_left=pl)
        min_iy = oy * sh - pt
        max_iy = oy * sh + (kh - 1) * dh - pt
        all_y_valid = 0 <= min_iy and max_iy < hi

        def loop_k(k) -> None:
            channel = self.fb.udiv_i(k, self.fb.const(kernel_area))
            rem = self.fb.urem_i(k, self.fb.const(kernel_area))
            ky = self.fb.udiv_i(rem, self.fb.const(kw))
            kx = self.fb.urem_i(rem, self.fb.const(kw))
            iy = self.fb.sub_i(
                self.fb.add_i(self.fb.const(oy * sh), self.fb.mul_i(ky, self.fb.const(dh))),
                self.fb.const(pt))
            if all_y_valid:
                gy = iy
                valid_y = None
            else:
                valid_y = self.fb.and_i(
                    self.fb.mask_nonneg(iy),
                    self.fb.mask_neg(self.fb.sub_i(iy, self.fb.const(hi))))
                gy = self.fb.clamp(iy, 0, hi - 1)
            src_row = self.fb.mul_i(
                self.fb.add_i(self.fb.const(batch * ci), channel),
                self.fb.const(hi))
            src_row = self.fb.mul_i(self.fb.add_i(src_row, gy), self.fb.const(src_pitch))
            dst_row = self.fb.mul_i(k, self.fb.const(dst_pitch))
            x_origin = self.fb.sub_i(
                self.fb.mul_i(kx, self.fb.const(dw)), self.fb.const(pl))

            def guarded_span(start: int, count: int) -> None:
                if count <= 0:
                    return

                def loop_x(local_x) -> None:
                    ox = (local_x if start == 0 else
                          self.fb.add_i(local_x, self.fb.const(start)))
                    ix = self.fb.add_i(
                        self.fb.mul_i(ox, self.fb.const(sw)), x_origin)
                    valid_x = self.fb.and_i(
                        self.fb.mask_nonneg(ix),
                        self.fb.mask_neg(self.fb.sub_i(ix, self.fb.const(wi))))
                    valid = valid_x if valid_y is None else self.fb.and_i(valid_y, valid_x)
                    gx = self.fb.clamp(ix, 0, wi - 1)
                    value = self.fb.load_i64(src, self.fb.add_i(src_row, gx), "i8")
                    value = self.fb.blend(valid, value, self.fb.const(0))
                    self.fb.store_i64(
                        value, dst, self.fb.add_i(dst_row, ox), "i8")

                self.fb.for_range(count, loop_x)

            guarded_span(0, interior_lo)

            interior_count = interior_hi - interior_lo
            if interior_count:
                # At this span every kx is in range.  Build segment bases once;
                # the hot loop advances by one output and by the static stride.
                src_segment = self.fb.add_i(
                    src_row,
                    self.fb.add_i(
                        x_origin, self.fb.const(interior_lo * sw)))
                dst_segment = self.fb.add_i(dst_row, self.fb.const(interior_lo))

                def loop_interior(local_x) -> None:
                    src_delta = (local_x if sw == 1 else
                                 self.fb.mul_i(local_x, self.fb.const(sw)))
                    value = self.fb.load_i64(
                        src, self.fb.add_i(src_segment, src_delta), "i8")
                    if valid_y is not None:
                        value = self.fb.blend(valid_y, value, self.fb.const(0))
                    self.fb.store_i64(
                        value, dst, self.fb.add_i(dst_segment, local_x), "i8")

                self.fb.for_range(interior_count, loop_interior)

            guarded_span(interior_hi, wo - interior_hi)

        self.fb.for_range(ci * kh * kw, loop_k)

    def emit_loop_ws_block(self, ins: Instr) -> None:
        """Lower a capacity-bounded matrix block through Gemmini's LOOP_WS sequencer."""
        a = ins.attrs
        a_name, b_name, c_name = ins.bufs
        c_dtype = str(a.get("c_dtype", "i32"))
        c_byte_stride = int(a["c_stride"]) * DTYPE_BYTES[c_dtype]

        # These are the same ordinary Gemmini configurations used by tiled_matmul_auto.  The
        # LOOP_WS engine consumes three independent load streams: A, B and the optional i32 D.
        configs = (
            isa.config_ex(dataflow=isa.WEIGHT_STATIONARY, sys_act=isa.NO_ACTIVATION),
            isa.config_st(stride=c_byte_stride,
                          acc_act=int(a.get("acc_act", isa.NO_ACTIVATION)),
                          acc_scale=float(a.get("acc_scale", 1.0))),
            isa.config_ld(stride=int(a["a_stride"]), load_id=0),
            isa.config_ld(stride=int(a["b_stride"]), load_id=1),
            isa.config_ld(stride=c_byte_stride, shrunk=False, load_id=2),
        )
        for funct, rs1, rs2 in configs:
            self.rocc(funct, self.fb.const(rs1), self.fb.const(rs2))

        bounds, strides_ab, strides_dc, launch = loop_ws.loop_ws_static(
            rows=int(a["rows"]), cols=int(a["cols"]), depth=int(a["depth"]),
            row_stride_a=int(a["a_stride"]), row_stride_b=int(a["b_stride"]),
            row_stride_c=int(a["c_stride"]), full_c=bool(a["full_c"]),
            accumulate=bool(a["accumulate"]))
        self.rocc(*[bounds[0], self.fb.const(bounds[1]), self.fb.const(bounds[2])])
        self.rocc(loop_ws.opcode("k_LOOP_WS_CONFIG_ADDRS_AB"),
                  self.dram_addr(a_name, int(a["a_offset"])),
                  self.dram_addr(b_name, int(a["b_offset"])))
        d_addr = (self.dram_addr(c_name, int(a["d_offset"]))
                  if bool(a["accumulate"]) else self.fb.const(0))
        self.rocc(loop_ws.opcode("k_LOOP_WS_CONFIG_ADDRS_DC"), d_addr,
                  self.dram_addr(c_name, int(a["c_offset"])))
        for record in (strides_ab, strides_dc, launch):
            self.rocc(record[0], self.fb.const(record[1]), self.fb.const(record[2]))

    def _emit_loop_ws_block_cached(self, ins: Instr, state: dict[str, tuple[int, int, int]]) -> None:
        """Emit one slot-complete LOOP_WS descriptor with shared config reuse.

        The ordinary Gemmini CONFIG_EX/LD/ST registers are global and may be
        reused while their values are unchanged.  LOOP_WS descriptor registers
        are different: the RTL owns one descriptor register file per concurrent
        loop slot.  Omitting an unchanged-looking bounds or stride record leaves
        the alternate slot stale when successive launches overlap.  Therefore
        every descriptor writes bounds, both stride records, both address
        records, and the launch, even when its static geometry matches the
        preceding descriptor.
        """
        a = ins.attrs
        a_name, b_name, c_name = ins.bufs
        c_dtype = str(a.get("c_dtype", "i32"))
        c_byte_stride = int(a["c_stride"]) * DTYPE_BYTES[c_dtype]
        configs = {
            "ex": isa.config_ex(dataflow=isa.WEIGHT_STATIONARY, sys_act=isa.NO_ACTIVATION),
            "st": isa.config_st(
                stride=c_byte_stride,
                acc_act=int(a.get("acc_act", isa.NO_ACTIVATION)),
                acc_scale=float(a.get("acc_scale", 1.0))),
            "ld0": isa.config_ld(stride=int(a["a_stride"]), load_id=0),
            "ld1": isa.config_ld(stride=int(a["b_stride"]), load_id=1),
            "ld2": isa.config_ld(stride=c_byte_stride, shrunk=False, load_id=2),
        }
        for key, config in configs.items():
            if state.get(key) != config:
                self.rocc(config[0], self.fb.const(config[1]), self.fb.const(config[2]))
                state[key] = config

        bounds, strides_ab, strides_dc, launch = loop_ws.loop_ws_static(
            rows=int(a["rows"]), cols=int(a["cols"]), depth=int(a["depth"]),
            row_stride_a=int(a["a_stride"]), row_stride_b=int(a["b_stride"]),
            row_stride_c=int(a["c_stride"]), full_c=bool(a["full_c"]),
            accumulate=bool(a["accumulate"]))
        self.rocc(bounds[0], self.fb.const(bounds[1]), self.fb.const(bounds[2]))
        self.rocc(loop_ws.opcode("k_LOOP_WS_CONFIG_ADDRS_AB"),
                  self.dram_addr(a_name, int(a["a_offset"])),
                  self.dram_addr(b_name, int(a["b_offset"])))
        d_addr = (self.dram_addr(c_name, int(a["d_offset"]))
                  if bool(a["accumulate"]) else self.fb.const(0))
        self.rocc(loop_ws.opcode("k_LOOP_WS_CONFIG_ADDRS_DC"), d_addr,
                  self.dram_addr(c_name, int(a["c_offset"])))
        for record in (strides_ab, strides_dc):
            self.rocc(record[0], self.fb.const(record[1]), self.fb.const(record[2]))
        self.rocc(launch[0], self.fb.const(launch[1]), self.fb.const(launch[2]))

    def emit_loop_conv_ws(self, ins: Instr) -> None:
        """Emit a slot-complete native convolution descriptor from the pinned public macro."""
        a = ins.attrs
        static_keys = (
            "batch_size", "in_row_dim", "in_col_dim", "in_channels", "out_channels",
            "out_row_dim", "out_col_dim", "pool_out_row_dim", "pool_out_col_dim",
            "stride", "padding", "kernel_dim", "kernel_dilation", "pool_size",
            "pool_stride", "pool_padding", "batches", "porows", "pocols", "pochs",
            "krows", "kcols", "kchs", "lpad", "rpad", "upad", "dpad", "plpad",
            "prpad", "pupad", "pdpad", "orows", "ocols", "in_stride",
            "weight_stride", "out_stride", "no_bias", "no_pool", "downsample",
            "wrot180", "input_dilated", "activation", "trans_output_1203",
            "trans_weight_1203", "trans_weight_0132", "trans_input_3120",
            "max_pixels_per_row", "dw", "a_spad_id", "b_spad_id")
        config1, config2, config3, config4, launch = loop_conv.static_descriptor(
            **{key: a[key] for key in static_keys})
        for funct, rs1, rs2 in (config1, config2, config3, config4):
            self.rocc(funct, self.fb.const(rs1), self.fb.const(rs2))
        input_name, weight_name, output_name = ins.bufs[:3]
        bias_name = ins.bufs[3] if len(ins.bufs) == 4 else None
        self.rocc(loop_conv.K_CONFIG_5,
                  self.dram_addr(weight_name, int(a["weight_offset"])),
                  (self.dram_addr(output_name, int(a["output_offset"]))
                   if bool(a["write_output"]) else self.fb.const(0)))
        self.rocc(loop_conv.K_CONFIG_6,
                  (self.dram_addr(bias_name, int(a["bias_offset"]))
                   if bias_name is not None and not bool(a["no_bias"])
                   else self.fb.const(0)),
                  self.dram_addr(input_name, int(a["input_offset"])))
        self.rocc(launch[0], self.fb.const(launch[1]), self.fb.const(launch[2]))

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
    def host_constant_tensor(self, value):
        """Immutable source tensor data, retaining exact encoded element bytes."""
        element = value.get_element_type()
        symbol = f"__host_constant_{len(self.globals)}"
        flattened = DenseIntOrFPElementsAttr(TensorType(element, [len(value)]), value.data)
        self.globals.append(llvm.GlobalOp(
            llvm.LLVMArrayType(len(value), element), symbol, "internal",
            constant=True, value=flattened, alignment=16))
        return self._prologue(llvm.AddressOfOp(symbol, PTR))

    def host_lowering(self) -> HostLinalg:
        """The one CPU-lane lowering this kernel uses, shared by every host segment.

        A mixed-lane program hands values from one host run to the next through SSA, not through
        memory, so the segments have to be lowered by the same instance -- a second one would
        have no value for anything the first computed.
        """
        hit = getattr(self, "_host", None)
        if hit is None:
            hit = self._new_host_lowering([], [])
            self._host = hit
        return hit

    def emit_host_segment(self, ins: Instr) -> None:
        segment = ins.attrs["segment"]
        lowering = self.host_lowering()
        lowering.run_segment(
            segment.ops,
            [(value, self.ptr(name), self.plan.buffers[name].storage_encoding)
             for value, name in segment.inputs],
            [(value, self.ptr(name), self.plan.buffers[name].storage_encoding)
             for value, name in segment.outputs],
            getattr(segment, "source_op_indices", ()))

    def emit_host_linalg(self, ins: Instr) -> None:
        """Generate the CPU-lane program for a module placed entirely on the host lane."""
        lowering = self._new_host_lowering(
            [self.ptr(n) for n in ins.attrs["arg_buffers"]],
            [self.ptr(n) for n in ins.attrs["out_buffers"]])
        lowering.run(ins.attrs["func"])

    # -- driver ------------------------------------------------------------------------------
    def build(self) -> ModuleOp:
        loop_ws_state: dict[str, tuple[int, int, int]] = {}
        for ins in self.instrs:
            self._source_conv_owner = (ins.attrs.get("global_task_index")
                if ins.kind in {"im2col_row", "loop_ws_block", "loop_conv_ws"} else None)
            previous_last = {block: block.last_op for block in self.fb.region.blocks}
            if ins.kind == "host_transpose":
                self.emit_transpose(ins)
            elif ins.kind == "host_bias_add":
                self.emit_bias_add(ins)
            elif ins.kind == "im2col_row":
                self.emit_im2col_row(ins)
            elif ins.kind == "loop_ws_block":
                # LOOP_WS load stream 2 supplies the optional D accumulator.  A fresh
                # (non-accumulating) block encodes a null D address, so that stream is idle and
                # its CONFIG_LD value is unobserved.  Temporarily mark precisely that value as
                # cached to let the shared emitter omit the dead configuration.  Do not retain a
                # value which was never written: a later accumulating block must still configure
                # stream 2 unless an identical value was genuinely emitted earlier.
                saved_ld2 = None
                had_ld2 = "ld2" in loop_ws_state
                if not bool(ins.attrs["accumulate"]):
                    c_byte_stride = int(ins.attrs["c_stride"]) * DTYPE_BYTES["i32"]
                    desired_ld2 = isa.config_ld(
                        stride=c_byte_stride, shrunk=False, load_id=2)
                    saved_ld2 = loop_ws_state.get("ld2")
                    loop_ws_state["ld2"] = desired_ld2
                self._emit_loop_ws_block_cached(ins, loop_ws_state)
                if not bool(ins.attrs["accumulate"]):
                    if had_ld2:
                        loop_ws_state["ld2"] = saved_ld2
                    else:
                        del loop_ws_state["ld2"]
            elif ins.kind == "loop_conv_ws":
                self.emit_loop_conv_ws(ins)
                loop_ws_state.clear()
            elif ins.kind == "host_zero_i32":
                self.emit_zero_i32(ins)
            elif ins.kind == "host_epilogue":
                self.emit_epilogue(ins)
            elif ins.kind == "host_linalg":
                self.emit_host_linalg(ins)
            elif ins.kind == "host_segment":
                self.emit_host_segment(ins)
            else:
                self.emit_command(ins)
                if ins.kind not in {"fence", "flush"}:
                    loop_ws_state.clear()
            if "global_task_index" in ins.attrs:
                for block in self.fb.region.blocks:
                    last = previous_last.get(block)
                    emitted = last.next_op if last is not None else block.first_op
                    while emitted is not None:
                        emitted.attributes["merlin.global_task"] = IntegerAttr(
                            ins.attrs["global_task_index"], i64)
                        emitted = emitted.next_op
        region = self.fb.finish()
        storage_rows = [lowering.storage_receipt() for lowering in self._host_lowerings]
        workspace_bytes = max(
            (row["workspace_global_bytes"] for row in storage_rows), default=0)
        if workspace_bytes:
            self.globals.append(llvm.GlobalOp(
                llvm.LLVMArrayType(workspace_bytes, i8), self._host_workspace_symbol,
                "internal", alignment=64))
        allocations = [owner for row in storage_rows for owner in row["largest_owners"]]
        largest = sorted(
            allocations,
            key=lambda row: (-row["bytes"], row["segment"], row["operation"]))[:16]
        global_plan = self.plan.command_buffer.get("params", {}).get("global_program_plan", {})
        spills = list(global_plan.get("host_tensor_spills") or [])
        storage_receipt = {
            "schema": "whole_program_host_storage_v1",
            "lifetime_unit": "ordered_host_segment",
            "workspace_global_bytes": workspace_bytes,
            "workspace_requested_bytes": sum(
                row["workspace_requested_bytes"] for row in storage_rows),
            "workspace_reuse_bytes": max(0, sum(
                row["workspace_requested_bytes"] for row in storage_rows) - workspace_bytes),
            "bounded_stack_frame_upper_bound_bytes": sum(
                row["bounded_stack_frame_upper_bound_bytes"] for row in storage_rows),
            "segment_workspace_peaks": [peak for row in storage_rows
                                        for peak in row["segment_workspace_peaks"]],
            "host_tensor_spill_bytes": sum(int(row["bytes"]) for row in spills),
            "host_tensor_spills": spills,
            "allocation_count": sum(row["allocation_count"] for row in storage_rows),
            "largest_owners": largest,
            "kernel_abi_pointer_added": False,
            "workspace_linkage": "internal_global",
        }
        self.plan.command_buffer.setdefault("params", {})["host_storage"] = storage_receipt
        fn = llvm.FuncOp("gemmini_kernel", llvm.LLVMFunctionType([PTR] * len(self.args)),
                         linkage=llvm.LinkageAttr("external"), body=region)
        module = ModuleOp([*self.globals, fn])
        module.attributes["merlin.host_workspace_bytes"] = IntegerAttr(workspace_bytes, i64)
        module.attributes["merlin.host_stack_upper_bound_bytes"] = IntegerAttr(
            storage_receipt["bounded_stack_frame_upper_bound_bytes"], i64)
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

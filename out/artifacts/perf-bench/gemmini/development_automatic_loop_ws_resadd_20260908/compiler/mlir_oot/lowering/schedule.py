"""Tile scheduling: a kernel plan -> an ordered stream of gemmini commands.

The schedule is derived from the RTL facts (mesh `DIM`, scratchpad depth, accumulator depth) and
from the extents of the capsule that was handed in.  Nothing here is keyed on a capsule name, a
particular shape, or a particular epilogue value: the loop nest walks `ceil(extent / DIM)` tiles
in each of M, K and N, and every field it packs is read from the plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tables import isa
from ..tables import rtl_facts as F
from .model_lane import HostSegment
from .source_conv_plan import Convolution
from .native_conv_selector import (
    select_native_loop_conv,
    select_native_loop_conv_compute_only,
)
from ..tables import loop_ws
from ..tables import loop_conv

from .plan import (
    Buffer,
    row_pitch,
    Contraction,
    DTYPE_BYTES,
    Epilogue,
    HostBiasAdd,
    LoweringDeclined,
    Movement,
    Plan,
    ResAdd,
    Transpose,
    pool_out_dims,
)

DIM = F.DIM

#: The emitted kernel is straight-line code (the program oracle faults on a branch), so the CPU
#: lane's cost is paid in emitted instructions.  Past this budget the lowering DECLINES rather
#: than producing a multi-hundred-megabyte artifact nobody can assemble.
HOST_LANE_OP_BUDGET = 120_000
#: The same guard for the accelerator stream itself.
COMMAND_BUDGET = 1_000_000


@dataclass
class Instr:
    """One scheduled command (an accelerator instruction or a compiler-generated host fixup)."""

    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    #: names of the DRAM buffers this command touches, in operand order
    bufs: list[str] = field(default_factory=list)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class Readout:
    """How one commit's epilogue is split between the store path and the CPU lane."""

    mode: str                                   # "native" (straight to dst) | "scratch"
    stage_dtype: str                            # element type the MVOUT writes
    acc_act: int                                # activation encoded in CONFIG_ST
    acc_scale: float                            # scale encoded in CONFIG_ST
    native_stages: list[str] = field(default_factory=list)
    host_stages: list[str] = field(default_factory=list)


def readout_plan(e: Epilogue) -> Readout:
    """Split the declared epilogue between the accumulator store path and the CPU lane.

    The store path applies its activation and float scale only on the NARROWING readout
    (`AccumulatorScale` feeds the full-width port the raw accumulator), so a full-width i32
    readout carries no stage.  `maxpool` always lands on the CPU lane because it changes the
    extent, but it commutes with the monotone stages before it, so those still go to the store
    path and the pool then runs over the already-narrowed values.
    """
    stages = [s for s in e.stages if s not in ("bias_add", "bias")]
    pool = "maxpool" in stages
    pre = [s for s in stages if s != "maxpool"]
    # The store path has exactly two readout widths: the accumulator's own dtype (full width, raw)
    # and the mesh operand dtype (narrowed through the activation + scale unit).  Any OTHER output
    # width has to be produced by the CPU lane from the full-width readout.
    narrowing = e.output_dtype == F.OPERAND_DTYPE
    full_width = e.output_dtype == F.ACCUMULATOR_DTYPE
    if (narrowing and e.integer_output_policy != "modular"
            and set(pre) <= {"acc_scale", "relu"} and e.acc_scale > 0.0):
        native_stages, host_stages, stage_dtype = pre, [], e.output_dtype
    else:
        native_stages, host_stages, stage_dtype = [], pre, F.ACCUMULATOR_DTYPE
    mode = ("native"
            if (not pool and not host_stages and stage_dtype == e.output_dtype)
            else "scratch")
    return Readout(
        mode=mode,
        stage_dtype=stage_dtype,
        # CONFIG_ST states the DECLARED activation even when a full-width readout bypasses it,
        # so the decoded trace reflects the epilogue the capsule asked for.
        acc_act=isa.RELU if "relu" in stages else isa.NO_ACTIVATION,
        acc_scale=e.acc_scale if "acc_scale" in native_stages else 1.0,
        native_stages=native_stages,
        host_stages=host_stages)


class Scheduler:
    """Walks a plan and produces the command stream plus any staging buffers it needs."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.instrs: list[Instr] = []
        self.staging: dict[str, Buffer] = {}
        self._ld_stride: int | None = None
        self._ld_shrunk: bool | None = None
        self._st_key: tuple | None = None
        self._ex_key: tuple | None = None
        self._stage_id = 0
        self._current_task_index = -1
        self._convolution_selections: list[dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------------------------
    def emit(self, __kind: str, __bufs: list[str] | None = None, **attrs) -> None:
        self.instrs.append(Instr(__kind, attrs, list(__bufs or [])))

    def stage_buffer(self, shape: list[int], dtype: str, hint: str) -> Buffer:
        name = f"__stage_{self._stage_id}_{hint}"
        self._stage_id += 1
        buf = Buffer(name, list(shape), dtype, "scratch")
        self.staging[name] = buf
        self.plan.buffers[name] = buf
        return buf

    def buf(self, name: str) -> Buffer:
        return self.plan.buffers[name]

    def config_ld(self, stride: int, *, shrunk: bool = False) -> None:
        if (stride, shrunk) == (self._ld_stride, self._ld_shrunk):
            return
        self._ld_stride, self._ld_shrunk = stride, shrunk
        self.emit("config_ld", stride=stride, shrunk=shrunk, load_id=0, scale=1.0)

    def config_st(self, **kw) -> None:
        key = tuple(sorted(kw.items()))
        if key == self._st_key:
            return
        self._st_key = key
        self.emit("config_st", **kw)

    def config_ex(self, **kw) -> None:
        key = tuple(sorted(kw.items()))
        if key == self._ex_key:
            return
        self._ex_key = key
        self.emit("config_ex", **kw)

    # -- entry point -------------------------------------------------------------------------
    def run(self) -> list[Instr]:
        if not self.plan.tasks:
            # Nothing was placed on the mesh.  Emit no accelerator command at all: a capsule
            # whose family this datapath does not admit must leave NOTHING in the instruction
            # stream, not even a flush, or the placement is not actually on the host lane.
            return self.instrs
        # open with a fence so the kernel starts from a quiescent accelerator, then flush the
        # pipeline before configuring it (the trace gate reads the opening fence).
        has_device_work = any(not isinstance(task, HostSegment) for task in self.plan.tasks)
        if has_device_work:
            self.emit("fence")
            self.emit("flush", skip=0)
        prologue_end = len(self.instrs)
        for instruction in self.instrs:
            instruction.attrs["global_task_index"] = -1
        receipt = (self.plan.command_buffer.get("params") or {}).get("global_program_plan")
        device_reads: set[str] = set()
        device_writes: set[str] = set()
        host_writes: set[str] = set()
        for task_index, task in enumerate(self.plan.tasks):
            self._current_task_index = task_index
            instruction_start = len(self.instrs)
            if receipt is not None:
                task_record = receipt["tasks"][task_index]
                reads, writes = set(task_record["reads"]), set(task_record["writes"])
                if isinstance(task, HostSegment):
                    hazards = (reads & device_writes) | (writes & (device_reads | device_writes))
                    if hazards:
                        self.emit("fence")
                        task_record["boundary_sync"] = {
                            "direction": "device_to_host", "tensors": sorted(hazards)}
                        device_reads.clear()
                        device_writes.clear()
                    host_writes.update(writes)
                else:
                    hazards = (reads | writes) & host_writes
                    if hazards:
                        self.emit("fence")
                        task_record["boundary_sync"] = {
                            "direction": "host_to_device", "tensors": sorted(hazards)}
                        host_writes.clear()
                    device_reads.update(reads)
                    device_writes.update(writes)
            if isinstance(task, Contraction):
                self.contraction(task)
                # A non-native readout ends the accelerator part of the contraction with a
                # fence and then performs its narrowing/epilogue synchronously on the host.
                # Account for that completion here.  Otherwise the global hazard frontier still
                # says the just-completed device write is outstanding and emits a second fence
                # before an immediately following HostSegment, even though both sides of that
                # boundary are now host work.
                if readout_plan(task.epilogue).mode != "native":
                    device_reads.clear()
                    device_writes.clear()
                    host_writes.add(task.dst)
            elif isinstance(task, Convolution):
                self.convolution(task)
            elif isinstance(task, Movement):
                self.movement(task)
            elif isinstance(task, Transpose):
                self.emit("host_transpose", [task.src, task.dst],
                          rows=task.rows, cols=task.cols, dtype=task.dtype)
            elif isinstance(task, HostSegment):
                self.emit("host_segment",
                          [n for _, n in task.inputs] + [n for _, n in task.outputs],
                          segment=task, regions_placed=list(task.regions) or ["glue"])
            elif isinstance(task, HostBiasAdd):
                self.bias_add(task)
            elif isinstance(task, ResAdd):
                self.resadd(task)
            else:                                                   # pragma: no cover - guarded
                raise LoweringDeclined(f"unscheduled task {type(task).__name__}")
            if receipt is not None:
                receipt["tasks"][task_index].update(
                    instruction_start=instruction_start, instruction_end=len(self.instrs))
                for instruction in self.instrs[instruction_start:]:
                    instruction.attrs["global_task_index"] = task_index
        epilogue_start = len(self.instrs)
        # A non-native contraction readout fences the accelerator before running its
        # synchronous host epilogue and clears the device hazard frontier above.  If only
        # host work follows, there is consequently no outstanding endpoint access for a
        # second kernel-exit fence to retire.  Keep the conservative fence when there is no
        # verified global-program receipt, or whenever tracked device accesses remain live.
        needs_completion_fence = has_device_work and (
            receipt is None or bool(device_reads) or bool(device_writes)
        )
        if needs_completion_fence:
            self.emit("fence")
            self.instrs[-1].attrs["global_task_index"] = -2
        if receipt is not None:
            receipt.update(schedule_instruction_count=len(self.instrs),
                           prologue_instruction_range=[0, prologue_end],
                           epilogue_instruction_range=[epilogue_start, len(self.instrs)])
        conv_receipt = (self.plan.command_buffer.get("params") or {}).get(
            "convolution_lowering")
        if conv_receipt is not None:
            conv_receipt.update(
                native_loop_conv_count=sum(
                    item["selected"].startswith("gemmini_loop_conv_ws")
                    for item in self._convolution_selections),
                native_loop_conv_narrow_count=sum(
                    item["selected"] == "gemmini_loop_conv_ws"
                    for item in self._convolution_selections),
                native_loop_conv_compute_only_count=sum(
                    item["selected"] == "gemmini_loop_conv_ws_compute_only"
                    for item in self._convolution_selections),
                fallback_count=sum(
                    not item["selected"].startswith("gemmini_loop_conv_ws")
                    for item in self._convolution_selections),
                selections=list(self._convolution_selections))
        if len(self.instrs) > COMMAND_BUDGET:
            raise LoweringDeclined(
                f"the tile schedule needs {len(self.instrs)} accelerator commands, past this "
                f"backend's {COMMAND_BUDGET} budget for a fully unrolled program")
        return self.instrs

    # -- data movement -----------------------------------------------------------------------
    def movement(self, mv: Movement) -> None:
        src, dst = self.buf(mv.src), self.buf(mv.dst)
        operand, accum = F.OPERAND_DTYPE, F.ACCUMULATOR_DTYPE
        # The two on-chip stores this round trip can pass through are the operand scratchpad
        # (`operand` elements) and the accumulator (`accum` elements).  A container the RTL has
        # neither width for would be moved as whatever the hardware's element width happens to
        # be -- the data would come back reinterpreted, which is a WRONG ANSWER wearing the shape
        # of a correct one -- so it is refused with the widths stated.
        legal = {(operand, operand), (operand, accum), (accum, accum)}
        if (src.dtype, dst.dtype) not in legal:
            raise LoweringDeclined(
                f"movement {src.dtype} -> {dst.dtype} has no on-chip container on this target: "
                f"the scratchpad holds {operand} elements and the accumulator {accum}, so the "
                f"legal round trips are {sorted(legal)}",
                op="movement", shape=[mv.rows, mv.cols])
        in_bytes = DTYPE_BYTES[src.dtype]
        out_bytes = DTYPE_BYTES[dst.dtype]
        # `widen` means the round trip is staged in the ACCUMULATOR rather than the scratchpad:
        # either because the readout is wider than the operand (i8 -> i32) or because the data is
        # already accumulator-width and the operand store cannot hold it (i32 -> i32).  The load
        # is `shrunk` only in the first case -- that bit says the DRAM side is operand-width.
        widen = dst.dtype == accum
        shrunk = widen and src.dtype == operand
        # the mesh is untouched by a pure round trip, but the dataflow/stride state is still
        # part of the accelerator's configuration, so state it explicitly.
        self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                       acc_scale=1.0, a_stride=1, c_stride=1)
        src_pitch, dst_pitch = src.pitch, dst.pitch
        self.config_ld(src_pitch * in_bytes, shrunk=shrunk)
        self.config_st(stride=dst_pitch * out_bytes, acc_act=isa.NO_ACTIVATION, acc_scale=1.0)
        slots = 8
        slot = 0
        for i in range(_ceil_div(mv.rows, DIM)):
            rows = min(DIM, mv.rows - i * DIM)
            for j in range(_ceil_div(mv.cols, DIM)):
                cols = min(DIM, mv.cols - j * DIM)
                if widen:
                    row = (slot % slots) * DIM
                    if row + DIM > F.ACC_ROWS:
                        row = 0
                    local_in = isa.acc_addr(row)
                    local_out = isa.acc_addr(row, full_row=True)
                else:
                    row = (slot % slots) * DIM
                    local_in = local_out = row
                slot += 1
                self.emit("mvin", [mv.src], local=local_in, rows=rows, cols=cols, load_id=0,
                          offset=(i * DIM * src_pitch + j * DIM) * in_bytes)
                self.emit("mvout", [mv.dst], local=local_out, rows=rows, cols=cols,
                          offset=(i * DIM * dst_pitch + j * DIM) * out_bytes)

    # -- exact residual add ------------------------------------------------------------------
    def resadd(self, task: ResAdd) -> None:
        """Tile an exact i8 residual add within LOOP_WS's accumulator partition."""
        lhs, rhs, dst = self.buf(task.lhs), self.buf(task.rhs), self.buf(task.dst)
        if any(buf.dtype != F.OPERAND_DTYPE for buf in (lhs, rhs, dst)):
            raise LoweringDeclined(
                "LOOP_WS resadd requires two i8 operands and an i8 result",
                op="resadd", shape=[task.rows, task.cols])
        # The resadd controller writes one accumulator row per I-row/J-tile pair and this
        # target partitions ACC_ROWS/2 rows per overlappable descriptor.
        max_acc_rows = loop_ws.contract()["capacity"]["max_acc_rows"]["rows"]
        max_cols = min(task.cols, (max_acc_rows // DIM) * DIM)
        j_tiles = _ceil_div(max_cols, DIM)
        max_i_rows = ((max_acc_rows // j_tiles) // DIM) * DIM
        if max_i_rows < DIM:                                      # pragma: no cover - guarded
            raise LoweringDeclined("LOOP_WS resadd tile cannot fit the accumulator partition",
                                   op="resadd", shape=[task.rows, task.cols])
        max_i_rows = min(max_i_rows, task.rows)
        for i0 in range(0, task.rows, max_i_rows):
            rows = min(max_i_rows, task.rows - i0)
            for j0 in range(0, task.cols, max_cols):
                cols = min(max_cols, task.cols - j0)
                self.emit(
                    "loop_ws_resadd", [task.lhs, task.rhs, task.dst],
                    rows=rows, cols=cols, relu=task.relu,
                    a_stride=lhs.pitch, b_stride=rhs.pitch, c_stride=dst.pitch,
                    a_offset=i0 * lhs.pitch + j0,
                    b_offset=i0 * rhs.pitch + j0,
                    c_offset=i0 * dst.pitch + j0)

    # -- the standalone bias add -------------------------------------------------------------
    def bias_add(self, t: HostBiasAdd) -> None:
        """`dst[i, j] = src[i, j] + bias[j]`, FOLDED INTO THE ACCUMULATOR READ-OUT.

        This datapath has no vector-add class, and the ABI says so: a target without one is
        expected to fold the op into its accumulator read-out.  The shipped ISA header spells the
        idiom out in `sp_tiled_matmul_ws` (the `repeating_bias` move-in with a ZERO DRAM row
        stride into `D_sp_addr_start = 1 << (ADDR_LEN-1)`) and in `sp_tiled_resadd` (a second
        move-in into `3 << (ADDR_LEN-2)`, the same address with the accumulate bit set, then a
        move-out of the summed row).  So per output tile: move the bias row in with a zero row
        stride so it repeats down the tile, move the source in ON TOP with the accumulate bit,
        and read the accumulator row out.

        Both operands are in the ACCUMULATOR's dtype -- the ABI states that outright, because the
        bias lands on the accumulator before any requant.  A declaration in any other width is
        left to the CPU lane rather than fed to an accumulator port that does not carry it.
        """
        src, bias, dst = self.buf(t.src), self.buf(t.bias), self.buf(t.dst)
        acc_dt = F.ACCUMULATOR_DTYPE
        if not (src.dtype == bias.dtype == dst.dtype == acc_dt):
            self.emit("host_bias_add", [t.src, t.bias, t.dst],
                      rows=t.rows, cols=t.cols, dtype=t.dtype)
            return
        wide = DTYPE_BYTES[acc_dt]
        src_pitch, dst_pitch = src.pitch, dst.pitch
        self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                       acc_scale=1.0, a_stride=1, c_stride=1)
        self.config_st(stride=dst_pitch * wide, acc_act=isa.NO_ACTIVATION, acc_scale=1.0)
        slot = 0
        for i in range(_ceil_div(t.rows, DIM)):
            rows = min(DIM, t.rows - i * DIM)
            for j in range(_ceil_div(t.cols, DIM)):
                cols = min(DIM, t.cols - j * DIM)
                row = (slot % 8) * DIM
                if row + DIM > F.ACC_ROWS:
                    row = 0
                slot += 1
                # the bias vector repeats down the tile: a zero DRAM row stride is what the ISA
                # header's `repeating_bias` path uses, so one length-N row feeds every output row
                self.config_ld(0)
                self.emit("mvin", [t.bias], load_id=0, local=isa.acc_addr(row),
                          rows=rows, cols=cols, offset=j * DIM * wide)
                self.config_ld(src_pitch * wide)
                self.emit("mvin", [t.src], load_id=0,
                          local=isa.acc_addr(row, accumulate=True), rows=rows, cols=cols,
                          offset=(i * DIM * src_pitch + j * DIM) * wide)
                self.emit("mvout", [t.dst], local=isa.acc_addr(row, full_row=True),
                          rows=rows, cols=cols,
                          offset=(i * DIM * dst_pitch + j * DIM) * wide)

    # -- budget ------------------------------------------------------------------------------
    #: The epilogue stages the GENERATED CPU-lane readout implements.  `maxpool` is the windowed
    #: max; the rest are the elementwise stages `codegen.llvm_emit._apply_scalar_stages` builds.
    #: Kept here, next to the split that decides what lands on the CPU lane, so a stage with no
    #: readout is refused while the PLAN is being built -- every entrypoint then answers the same
    #: way about the capsule, instead of the command buffer claiming a program the artifact
    #: cannot emit (which reads as a crashed tool, not as a refusal).
    HOST_READOUT_STAGES = frozenset({"relu", "requant", "acc_scale", "maxpool"})

    def _check_host_stages(self, c: Contraction, stages: list[str]) -> None:
        """Refuse an epilogue stage the generated CPU-lane readout does not implement."""
        unsupported = [s for s in stages if s not in self.HOST_READOUT_STAGES]
        if unsupported:
            raise LoweringDeclined(
                f"epilogue stage {unsupported[0]!r} has no generated CPU-lane readout, and this "
                f"commit's readout does not go through the accelerator store path",
                op=unsupported[0], shape=[c.m, c.n])

    def _check_host_budget(self, c: Contraction, e: Epilogue, ro, stages: list[str]) -> None:
        """Refuse a CPU-lane readout whose straight-line form would not be assemblable."""
        per_element = 6 + 4 * len(ro.host_stages)
        if e.has_pool:
            ih, iw = e.pool_in_dims
            ph, pw = e.pool_size
            ho, wo = pool_out_dims(ih, iw, e.pool_size, e.pool_stride, e.pool_padding)
            out_rows = (c.m // (ih * iw)) * ho * wo
            cost = out_rows * c.n * (ph * pw * (per_element + 4) + 10)
        else:
            cost = c.m * c.n * per_element
        if cost > HOST_LANE_OP_BUDGET:
            raise LoweringDeclined(
                f"the epilogue {stages} over a [{c.m}, {c.n}] readout needs about {cost} "
                f"straight-line CPU-lane instructions, past this backend's {HOST_LANE_OP_BUDGET} "
                f"budget; the emitted kernel must be single-block on this target, so the work "
                f"cannot be rolled into a loop",
                op=stages[-1] if stages else "commit", shape=[c.m, c.n])

    # -- contraction -------------------------------------------------------------------------
    def block_shape(self, mt: int, nt: int, kt: int) -> tuple[int, int, int]:
        """Tile-block extents that respect the RTL accumulator and scratchpad depths."""
        acc_tiles = F.ACC_ROWS // DIM                 # C tiles resident at once
        spad_tiles = F.SPAD_ROWS // DIM               # A + B tiles resident at once
        bn = min(nt, 8)
        bm = min(mt, max(1, acc_tiles // bn))
        bn = min(nt, max(1, acc_tiles // bm))
        bk = min(kt, max(1, spad_tiles // (bm + bn)))
        while bk > 1 and (bm + bn) * bk > spad_tiles:
            bk -= 1
        return bm, bn, bk

    def convolution(self, c: Convolution) -> None:
        """Generate one looped target op for direct DMA or output-row-streamed im2col.

        The matrix orientation is ``weight[Co,K] @ activation_row[K,Wo]``. This is deliberate:
        each readout row is one output channel, so it lands directly in NCHW and needs no NHWC
        staging/transpose. Only one output row of im2col is live at once. The target op lowers to
        Gemmini's hardware ``LOOP_WS`` command family, avoiding millions of statically unrolled
        mvin/preload/compute instructions.
        """
        activation, weight, dst = self.buf(c.activation), self.buf(c.weight), self.buf(c.dst)
        native, _reason = self.native_convolution_eligibility(c)
        self._convolution_selections.append({
            "task_index": self._current_task_index,
            "geometry": {"n": c.batch, "ci": c.ci, "hi": c.hi, "wi": c.wi,
                         "co": c.co, "kh": c.kh, "kw": c.kw,
                         "ho": c.ho, "wo": c.wo,
                         "stride": [c.stride_h, c.stride_w],
                         "dilation": [c.dilation_h, c.dilation_w],
                         "padding": [c.pad_top, c.pad_left, c.pad_bottom, c.pad_right]},
            "semantic_output_dtype": c.output_dtype,
            "layouts": [c.input_layout, c.weight_layout, c.output_layout],
            "selected": (("gemmini_loop_conv_ws_compute_only"
                          if c.compute_only_native else "gemmini_loop_conv_ws")
                         if native else "streamed_row_im2col_loop_ws"),
            "reason": _reason,
        })
        if native:
            self.native_convolution(c)
            return
        if (activation.dtype != "i8" or weight.dtype != "i8"
                or dst.dtype != c.output_dtype or c.output_dtype not in ("i8", "i32")):
            raise LoweringDeclined(
                f"streamed conv requires i8 x i8 -> i8/i32; got {activation.dtype} x "
                f"{weight.dtype} -> {dst.dtype}", op="conv2d")
        kdim = c.ci * c.kh * c.kw
        if weight.shape != [c.co, kdim]:
            raise LoweringDeclined(
                f"streamed conv weight ABI is [Co,K]=[{c.co},{kdim}], got {weight.shape}",
                op="conv2d")
        packed = (None if c.direct_dma else
                  [self.stage_buffer([kdim, c.wo], "i8", "im2col_row_ping"),
                   self.stage_buffer([kdim, c.wo], "i8", "im2col_row_pong")])
        lhs_pitch = weight.pitch
        out_w_pitch = row_pitch(c.wo)
        out_plane_pitch = c.ho * out_w_pitch
        mt, nt, kt = _ceil_div(c.co, DIM), _ceil_div(c.wo, DIM), _ceil_div(kdim, DIM)
        if c.output_dtype == F.OPERAND_DTYPE:
            native_shape = loop_ws.reduction_resident_block_shape(mt, nt, kt)
            if native_shape is None:
                raise LoweringDeclined(
                    "native narrow convolution cannot spill a partial i8 reduction",
                    op="conv2d", shape=[c.co, kdim, c.wo])
            bm, bn, bk = native_shape
        else:
            bm, bn, bk = loop_ws.block_shape(mt, nt, kt)
        for batch in range(c.batch):
            for oy in range(c.ho):
                if packed is None:
                    rhs_name = c.activation
                    rhs_pitch = c.hi * row_pitch(c.wi)
                    rhs_base = (batch * c.ci * c.hi + oy) * row_pitch(c.wi)
                else:
                    # Ping-pong slabs let the CPU pack row N+1 while Gemmini consumes row N.
                    # The fence after packing both publishes those writes and proves that the
                    # slab from two iterations ago is no longer live before it is reused.
                    slab = packed[(batch * c.ho + oy) & 1]
                    rhs_name = slab.name
                    rhs_pitch = row_pitch(c.wo)
                    rhs_base = 0
                    self.emit(
                        "im2col_row", [c.activation, slab.name], batch=batch, out_y=oy,
                        ci=c.ci, hi=c.hi, wi=c.wi, kh=c.kh, kw=c.kw, wo=c.wo,
                        stride_h=c.stride_h, stride_w=c.stride_w,
                        dilation_h=c.dilation_h, dilation_w=c.dilation_w,
                        pad_top=c.pad_top, pad_left=c.pad_left)
                    self.emit("fence")
                out_base = (batch * c.co * c.ho + oy) * out_w_pitch
                for m0 in range(0, mt, bm):
                    for n0 in range(0, nt, bn):
                        for k0 in range(0, kt, bk):
                            # Full-K blocks above need no synchronization.  If even a
                            # 1x1 output block cannot hold the reduction, serialize the
                            # explicit DRAM partial-result dependency before D reads C.
                            if k0 != 0:
                                self.emit("fence")
                            rows = min(bm * DIM, c.co - m0 * DIM)
                            cols = min(bn * DIM, c.wo - n0 * DIM)
                            depth = min(bk * DIM, kdim - k0 * DIM)
                            out_bytes = DTYPE_BYTES[c.output_dtype]
                            c_offset = (out_base + m0 * DIM * out_plane_pitch
                                        + n0 * DIM) * out_bytes
                            self.emit(
                                "loop_ws_block", [c.weight, rhs_name, c.dst],
                                rows=rows, cols=cols, depth=depth,
                                a_stride=lhs_pitch, b_stride=rhs_pitch,
                                c_stride=out_plane_pitch,
                                a_offset=m0 * DIM * lhs_pitch + k0 * DIM,
                                b_offset=rhs_base + k0 * DIM * rhs_pitch + n0 * DIM,
                                c_offset=c_offset, d_offset=c_offset,
                                accumulate=k0 != 0,
                                full_c=c.output_dtype == F.ACCUMULATOR_DTYPE,
                                c_dtype=c.output_dtype,
                                acc_act=isa.RELU if c.relu else isa.NO_ACTIVATION,
                                acc_scale=c.acc_scale,
                                implementation="gemmini_loop_ws")

    def native_convolution_eligibility(self, c: Convolution) -> tuple[bool, str]:
        """Whether the source semantics and physical encodings fit LOOP_CONV_WS exactly."""
        activation, weight, dst = self.buf(c.activation), self.buf(c.weight), self.buf(c.dst)
        if c.compute_only_native:
            return select_native_loop_conv_compute_only(c, activation, weight, dst)
        stages = (["bias"] if c.bias else [])
        if c.relu:
            stages.append("relu")
        if c.acc_scale != 1.0:
            stages.append("acc_scale")
        return select_native_loop_conv(
            c, activation, weight, dst, epilogue_stages=stages, bias=c.bias,
            bias_buffer=self.buf(c.bias) if c.bias else None, acc_scale=c.acc_scale)

    def native_convolution(self, c: Convolution) -> None:
        """Tile an eligible NHWC/HWIO convolution into complete LOOP_CONV_WS descriptors."""
        activation, weight, dst = self.buf(c.activation), self.buf(c.weight), self.buf(c.dst)
        batches, porows, pocols, pochs, krows, kcols, kchs = loop_conv.auto_tile(
            batch=c.batch, ho=c.ho, wo=c.wo, co=c.co, kh=c.kh, kw=c.kw, ci=c.ci,
            stride=c.stride_h, kernel_dilation=c.dilation_h,
            trans_input_3120=c.compute_only_native)
        in_stride, weight_stride, out_stride = activation.pitch, weight.pitch, dst.pitch
        zero_bias = (self.stage_buffer([pochs], F.ACCUMULATOR_DTYPE, "loop_conv_zero_bias")
                     if c.compute_only_native else None)
        if zero_bias is not None:
            # Compiler staging globals lower to LLVM ``undef``, not a BSS zero initializer.
            # Materialise the exact LOAD3 value every invocation and publish it before launch.
            self.emit("host_zero_i32", [zero_bias.name], elements=pochs)
            self.emit("fence")
        st = {"stride": out_stride,
              "acc_act": isa.RELU if c.relu else isa.NO_ACTIVATION,
              "acc_scale": c.acc_scale}
        # CONFIG_ST is global asynchronous state.  Retire a prior readout before changing it.
        if self._st_key is not None and tuple(sorted(st.items())) != self._st_key:
            self.emit("fence")
        self.config_st(**st)
        self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                       acc_scale=1.0, a_stride=c.stride_h, c_stride=1,
                       # The descriptor flag changes LOOP_CONV's loader and execute
                       # address formulas, but CONFIG_EX independently controls the
                       # systolic array's A transpose.  The public Gemmini helper sets
                       # both for an NCHW/CHWN input; omitting this bit computes a
                       # different matrix product without trapping.
                       a_transpose=c.compute_only_native, b_transpose=False)
        for b in range(0, c.batch, batches):
            for orow in range(0, c.ho, porows):
                for ocol in range(0, c.wo, pocols):
                    for poch in range(0, c.co, pochs):
                        for krow in range(0, c.kh, krows):
                            for kcol in range(0, c.kw, kcols):
                                for kch in range(0, c.ci, kchs):
                                    tile_b = min(batches, c.batch - b)
                                    tile_or = min(porows, c.ho - orow)
                                    tile_oc = min(pocols, c.wo - ocol)
                                    tile_och = min(pochs, c.co - poch)
                                    tile_kr = min(krows, c.kh - krow)
                                    tile_kc = min(kcols, c.kw - kcol)
                                    tile_kch = min(kchs, c.ci - kch)
                                    irow = orow * c.stride_h + krow * c.dilation_h - c.pad_top
                                    icol = ocol * c.stride_w + kcol * c.dilation_w - c.pad_left
                                    dil_kr = tile_kr + (c.dilation_h - 1) * (tile_kr - 1)
                                    dil_kc = tile_kc + (c.dilation_w - 1) * (tile_kc - 1)
                                    irows = tile_or * c.stride_h + dil_kr - 1
                                    icols = tile_oc * c.stride_w + dil_kc - 1
                                    upad, lpad = max(0, -irow), max(0, -icol)
                                    dpad = max(0, irow + irows - c.hi)
                                    rpad = max(0, icol + icols - c.wi)
                                    final = (krow + tile_kr == c.kh
                                             and kcol + tile_kc == c.kw
                                             and kch + tile_kch == c.ci)
                                    first = krow == 0 and kcol == 0 and kch == 0
                                    bufs = [c.activation, c.weight, c.dst]
                                    if c.bias and zero_bias is None:
                                        bufs.append(c.bias)
                                    if zero_bias is not None and first:
                                        # With a null output pointer LoopConv keeps its private C
                                        # accumulator base at row zero.  Its ordinary bias-load
                                        # path does not provide the public, stable C-zeroing
                                        # contract compute-only needs.  Clear C0 explicitly through
                                        # LOAD3 and keep CONFIG_6 bias null.  One source zero row is
                                        # replayed with DRAM stride zero over the tile's C layout.
                                        volume = tile_b * tile_or * tile_oc
                                        self.emit(
                                            "config_ld", stride=0, scale=1.0,
                                            shrunk=False, load_id=2,
                                            block_stride=volume, pixel_repeats=1)
                                        for local_och in range(0, tile_och, DIM):
                                            cols = min(DIM, tile_och - local_och)
                                            c_block = local_och // DIM
                                            for local_b in range(tile_b):
                                                for local_orow in range(tile_or):
                                                    for local_ocol in range(0, tile_oc, DIM):
                                                        rows = min(DIM, tile_oc - local_ocol)
                                                        local_row = (
                                                            c_block * volume
                                                            + local_b * tile_or * tile_oc
                                                            + local_orow * tile_oc + local_ocol)
                                                        self.emit(
                                                            "mvin", [zero_bias.name], load_id=2,
                                                            local=isa.acc_addr(local_row),
                                                            rows=rows, cols=cols, offset=0)
                                    self.emit(
                                        "loop_conv_ws", bufs,
                                        batch_size=c.batch, in_row_dim=c.hi,
                                        in_col_dim=c.wi, in_channels=c.ci,
                                        out_channels=c.co, out_row_dim=c.ho,
                                        out_col_dim=c.wo, pool_out_row_dim=c.ho,
                                        pool_out_col_dim=c.wo, stride=c.stride_h,
                                        padding=c.pad_top, kernel_dim=c.kh,
                                        kernel_dilation=c.dilation_h, pool_size=1,
                                        pool_stride=1, pool_padding=0, batches=tile_b,
                                        porows=tile_or, pocols=tile_oc, pochs=tile_och,
                                        krows=tile_kr, kcols=tile_kc, kchs=tile_kch,
                                        lpad=lpad, rpad=rpad, upad=upad, dpad=dpad,
                                        plpad=0, prpad=0, pupad=0, pdpad=0,
                                        orows=tile_or, ocols=tile_oc,
                                        in_stride=in_stride, weight_stride=weight_stride,
                                        out_stride=out_stride,
                                        # A null bias pointer makes LoopConvLdBias skip rather
                                        # than clear C.  Compute-only descriptors deliberately
                                        # reuse accumulator base zero, so LOAD3 an all-zero
                                        # compiler-owned vector for the first K slice of every
                                        # output tile; later slices accumulate without clearing.
                                        no_bias=(True if zero_bias is not None
                                                 else not (c.bias and first)), no_pool=True,
                                        downsample=False, wrot180=False, input_dilated=False,
                                        activation=isa.NO_ACTIVATION,
                                        trans_output_1203=False, trans_weight_1203=False,
                                        trans_weight_0132=False,
                                        trans_input_3120=c.compute_only_native,
                                        # Gemmini's public helper forces one pixel whenever
                                        # either matrix is transposed.  With transposed A, a
                                        # larger value makes K=pixels*kchs while the temporary
                                        # A stride is irows*icols; compute then reads beyond the
                                        # rows loaded for the channel-major tile.
                                        max_pixels_per_row=(
                                            1 if c.compute_only_native else
                                            min(tile_kc, max(1, DIM // tile_kch))), dw=False,
                                        a_spad_id=0, b_spad_id=0,
                                        input_offset=((kch * c.hi * c.wi
                                                       + (irow + upad) * c.wi
                                                       + (icol + lpad))
                                                      if c.compute_only_native else
                                                      ((b * c.hi * c.wi
                                                        + (irow + upad) * c.wi
                                                        + (icol + lpad)) * in_stride + kch)),
                                        weight_offset=((krow * c.kw * c.ci
                                                        + kcol * c.ci + kch)
                                                       * weight_stride + poch),
                                        output_offset=((b * c.ho * c.wo
                                                        + orow * c.wo + ocol)
                                                       * out_stride + poch),
                                        bias_offset=(0 if zero_bias is not None
                                                     else poch * DTYPE_BYTES["i32"]),
                                        write_output=final and not c.compute_only_native,
                                        implementation=("gemmini_loop_conv_ws_compute_only"
                                                        if c.compute_only_native
                                                        else "gemmini_loop_conv_ws"))
                                    if c.compute_only_native and final:
                                        # The loop controller does not rotate accumulator banks
                                        # when output_dram_addr is null, so this descriptor's C
                                        # starts at row zero. The controller blocks this direct
                                        # MVOUT until internal completion; the reservation station
                                        # orders its read against the next descriptor's write.
                                        self.config_st(
                                            stride=out_stride * DTYPE_BYTES[F.ACCUMULATOR_DTYPE],
                                            acc_act=isa.NO_ACTIVATION, acc_scale=1.0)
                                        for local_och in range(0, tile_och, DIM):
                                            cols = min(DIM, tile_och - local_och)
                                            channel = poch + local_och
                                            c_block = local_och // DIM
                                            for local_b in range(tile_b):
                                                for local_orow in range(tile_or):
                                                    # MVOUT's row count is a DIM-sized hardware
                                                    # field.  The LOOP_CONV tile may be wider than
                                                    # DIM, so mirror the public helper's ocol
                                                    # splitting rather than encoding 32 as zero.
                                                    for local_ocol in range(0, tile_oc, DIM):
                                                        rows = min(DIM, tile_oc - local_ocol)
                                                        local_row = (
                                                            c_block * tile_b * tile_or * tile_oc
                                                            + local_b * tile_or * tile_oc
                                                            + local_orow * tile_oc + local_ocol)
                                                        global_row = (
                                                            (b + local_b) * c.ho * c.wo
                                                            + (orow + local_orow) * c.wo
                                                            + ocol + local_ocol)
                                                        self.emit(
                                                            "mvout", [c.dst],
                                                            local=isa.acc_addr(
                                                                local_row, full_row=True),
                                                            rows=rows, cols=cols,
                                                            offset=(global_row * out_stride + channel)
                                                            * DTYPE_BYTES[F.ACCUMULATOR_DTYPE])

    def contraction(self, c: Contraction) -> None:
        lhs, rhs, dst = self.buf(c.lhs), self.buf(c.rhs), self.buf(c.dst)
        if lhs.dtype != "i8" or rhs.dtype != "i8":
            raise LoweringDeclined(
                f"the gemmini mesh contracts i8 operands; got {lhs.dtype} x {rhs.dtype}",
                op="matmul", shape=[c.m, c.k, c.n])
        e = c.epilogue
        ro = readout_plan(e)
        mode = ro.mode
        lhs_pitch = lhs.pitch if lhs.storage_encoding is not None else row_pitch(c.lhs_row_elems)
        rhs_pitch = rhs.pitch if rhs.storage_encoding is not None else row_pitch(c.rhs_row_elems)
        lhs_base = getattr(c, "lhs_batch_rows", 0) * lhs_pitch
        rhs_base = getattr(c, "rhs_batch_rows", 0) * rhs_pitch

        if mode == "native":
            out_buf = dst
            out_row_elems = row_pitch(c.dst_row_elems or c.n)
            out_row_offset = c.dst_row_offset
        else:
            out_buf = (self.buf(c.accumulator_temporary) if c.accumulator_temporary
                       else self.stage_buffer([c.m, c.n], ro.stage_dtype, "acc"))
            out_row_elems = row_pitch(c.n)
            out_row_offset = 0

        out_dtype = ro.stage_dtype
        full_row = out_dtype == "i32"
        out_bytes = DTYPE_BYTES[out_dtype]
        acc_act, acc_scale = ro.acc_act, ro.acc_scale

        mt, nt, kt = _ceil_div(c.m, DIM), _ceil_div(c.n, DIM), _ceil_div(c.k, DIM)
        # Prefer the target's pinned LOOP_WS sequencer when its full-width contract is an
        # exact match.  Bias initialization and narrowed/native readout still use the explicit
        # schedule below: LOOP_WS has one D input and a raw i32 C output, so silently folding
        # either case into it would change the declared epilogue.  For the admitted case each
        # descriptor owns one capacity-bounded [M,N,K] block; later K blocks read the prior i32
        # partial sum through D, preserving the source contraction while moving the repetitive
        # issue loop from the host instruction stream to the endpoint sequencer.
        use_loop_ws = (not e.bias and (
            (out_dtype == F.ACCUMULATOR_DTYPE
             and acc_act == isa.NO_ACTIVATION and acc_scale == 1.0)
            or (mode == "native" and out_dtype == F.OPERAND_DTYPE)))
        resident_loop_shape = (
            loop_ws.reduction_resident_block_shape(mt, nt, kt) if use_loop_ws else None
        )
        if resident_loop_shape is not None:
            bm, bn, bk = resident_loop_shape
            for m0 in range(0, mt, bm):
                for n0 in range(0, nt, bn):
                    for k0 in range(0, kt, bk):
                        rows = min(bm * DIM, c.m - m0 * DIM)
                        cols = min(bn * DIM, c.n - n0 * DIM)
                        depth = min(bk * DIM, c.k - k0 * DIM)
                        c_offset = ((out_row_offset + m0 * DIM) * out_row_elems
                                    + n0 * DIM) * out_bytes
                        self.emit(
                            "loop_ws_block", [c.lhs, c.rhs, out_buf.name],
                            rows=rows, cols=cols, depth=depth,
                            a_stride=lhs_pitch, b_stride=rhs_pitch,
                            c_stride=out_row_elems,
                            a_offset=(lhs_base + m0 * DIM * lhs_pitch + k0 * DIM)
                            * DTYPE_BYTES[lhs.dtype],
                            b_offset=(rhs_base + k0 * DIM * rhs_pitch + n0 * DIM)
                            * DTYPE_BYTES[rhs.dtype],
                            c_offset=c_offset, d_offset=c_offset,
                            accumulate=k0 != 0, full_c=full_row,
                            c_dtype=out_dtype, acc_act=acc_act, acc_scale=acc_scale,
                            implementation="gemmini_loop_ws")
        else:
            bm, bn, bk = self.block_shape(mt, nt, kt)
            a_base_tile = 0
            b_base_tile = bm * bk

            self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                           acc_scale=1.0, a_stride=1, c_stride=1)

            for m0 in range(0, mt, bm):
                for n0 in range(0, nt, bn):
                    for k0 in range(0, kt, bk):
                        mi = list(range(m0, min(m0 + bm, mt)))
                        ni = list(range(n0, min(n0 + bn, nt)))
                        ki = list(range(k0, min(k0 + bk, kt)))
                        if e.bias and k0 == 0:
                            bias_buf = self.buf(e.bias)
                            self.config_ld(0)
                            for a, i in enumerate(mi):
                                for d, j in enumerate(ni):
                                    self.emit("mvin", [e.bias], load_id=0,
                                              local=isa.acc_addr((a * bn + d) * DIM),
                                              rows=min(DIM, c.m - i * DIM),
                                              cols=min(DIM, c.n - j * DIM),
                                              offset=j * DIM * DTYPE_BYTES[bias_buf.dtype])
                        self.config_ld(lhs_pitch * DTYPE_BYTES[lhs.dtype])
                        for a, i in enumerate(mi):
                            for b, k in enumerate(ki):
                                self.emit("mvin", [c.lhs], load_id=0,
                                          local=(a_base_tile + a * bk + b) * DIM,
                                          rows=min(DIM, c.m - i * DIM),
                                          cols=min(DIM, c.k - k * DIM),
                                          offset=(lhs_base + i * DIM * lhs_pitch + k * DIM)
                                          * DTYPE_BYTES[lhs.dtype])
                        self.config_ld(rhs_pitch * DTYPE_BYTES[rhs.dtype])
                        for b, k in enumerate(ki):
                            for d, j in enumerate(ni):
                                self.emit("mvin", [c.rhs], load_id=0,
                                          local=(b_base_tile + b * bn + d) * DIM,
                                          rows=min(DIM, c.k - k * DIM),
                                          cols=min(DIM, c.n - j * DIM),
                                          offset=(rhs_base + k * DIM * rhs_pitch + j * DIM)
                                          * DTYPE_BYTES[rhs.dtype])
                        for b, k in enumerate(ki):
                            for d, j in enumerate(ni):
                                for a, i in enumerate(mi):
                                    first_i = a == 0
                                    accumulate = bool(e.bias) or k != 0
                                    c_row = (a * bn + d) * DIM
                                    self.emit(
                                        "preload",
                                        bd=(b_base_tile + b * bn + d) * DIM if first_i
                                        else isa.GARBAGE_ADDR,
                                        c=isa.acc_addr(c_row, accumulate=accumulate),
                                        bd_cols=min(DIM, c.n - j * DIM),
                                        bd_rows=min(DIM, c.k - k * DIM),
                                        c_cols=min(DIM, c.n - j * DIM),
                                        c_rows=min(DIM, c.m - i * DIM))
                                    self.emit(
                                        "compute",
                                        a=(a_base_tile + a * bk + b) * DIM,
                                        bd=isa.GARBAGE_ADDR,
                                        a_cols=min(DIM, c.k - k * DIM),
                                        a_rows=min(DIM, c.m - i * DIM),
                                        accumulate=not first_i)
                    # readout of this [bm x bn] accumulator block
                    mi = list(range(m0, min(m0 + bm, mt)))
                    ni = list(range(n0, min(n0 + bn, nt)))
                    self.config_st(stride=out_row_elems * out_bytes, acc_act=acc_act,
                                   acc_scale=acc_scale)
                    for a, i in enumerate(mi):
                        for d, j in enumerate(ni):
                            self.emit("mvout", [out_buf.name],
                                      local=isa.acc_addr((a * bn + d) * DIM, full_row=full_row),
                                      rows=min(DIM, c.m - i * DIM),
                                      cols=min(DIM, c.n - j * DIM),
                                      offset=((out_row_offset + i * DIM) * out_row_elems + j * DIM)
                                      * out_bytes)

        if mode == "native":
            return
        stages = list(ro.host_stages) + (["maxpool"] if e.has_pool else [])
        self._check_host_stages(c, stages)
        self._check_host_budget(c, e, ro, stages)
        self.emit("fence")
        bufs = [out_buf.name, c.dst]
        self.emit("host_epilogue", bufs, stages=stages, rows=c.m, cols=c.n,
                  stage_dtype=ro.stage_dtype,
                  out_dtype=e.output_dtype, acc_scale=e.acc_scale,
                  integer_output_policy=e.integer_output_policy,
                  requant_shift=e.requant_shift,
                  pool_in_dims=e.pool_in_dims, pool_size=e.pool_size,
                  pool_stride=e.pool_stride, pool_padding=e.pool_padding,
                  pool_pad_value=e.pool_pad_value,
                  src_row_elems=out_row_elems,
                  src_row_offset=out_row_offset,
                  dst_row_elems=row_pitch(dst.shape[-1]),
                  dst_row_offset=c.dst_row_offset)


def schedule(plan: Plan) -> tuple[list[Instr], dict[str, Buffer]]:
    s = Scheduler(plan)
    return s.run(), s.staging


def epilogue_out_rows(rows: int, e: Epilogue) -> int:
    if not e.has_pool:
        return rows
    ih, iw = e.pool_in_dims
    ho, wo = pool_out_dims(ih, iw, e.pool_size, e.pool_stride, e.pool_padding)
    return (rows // (ih * iw)) * ho * wo

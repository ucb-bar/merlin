"""The `gemmini` target dialect: one op per RoCC command class this backend emits.

The ops are real xDSL IRDL operations with verifiers, so the lowered module is checked against
the RTL-derived machine limits (mesh `DIM`, scratchpad depth, accumulator depth, the legal funct
set) before anything is encoded.  `gemmini.host_*` ops are the compiler-generated CPU-lane
fixups the accelerator store path cannot express; they lower to ordinary LLVM control flow, not
to a library call.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_result_def, var_operand_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from ..tables import rtl_facts as F
from ..tables import isa

DIM = F.DIM


def _val(attr: Any) -> Any:
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, IntegerAttr):
        return int(attr.value.data)
    if isinstance(attr, FloatAttr):
        return float(attr.value.data)
    if isinstance(attr, ArrayAttr):
        return [_val(a) for a in attr.data]
    return attr


class _GemminiOp(IRDLOperation):
    """Shared concrete syntax: `gemmini.<op> %operands {attrs} : (types) -> (types)`."""

    operands_ = var_operand_def()
    res = opt_result_def()

    def print(self, printer: Printer) -> None:
        if self.operands_:
            printer.print_string(" ")
            printer.print_list(self.operands_, printer.print_ssa_value)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_function_type(
            [v.type for v in self.operands_],
            [self.res.type] if self.res is not None else [],
        )

    @classmethod
    def parse(cls, parser: Parser) -> "_GemminiOp":
        pos = parser.pos
        unresolved = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_unresolved_operand,
                parser.parse_unresolved_operand,
            )
            or []
        )
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        ftype = parser.parse_function_type()
        op = cls(operands=[parser.resolve_operands(unresolved, ftype.inputs.data, pos)],
                 result_types=[list(ftype.outputs.data)])
        op.attributes |= attrs
        return op

    def a(self, key: str, default: Any = None) -> Any:
        attr = self.attributes.get(key)
        return default if attr is None else _val(attr)

    def _extent(self, key: str) -> None:
        n = self.a(key)
        if not isinstance(n, int) or not (1 <= n <= DIM):
            raise VerifyException(f"{self.name}: `{key}` = {n!r} must be in 1..{DIM} (mesh DIM)")

    def _local(self, key: str) -> None:
        addr = self.a(key)
        if not isinstance(addr, int) or addr < 0:
            raise VerifyException(f"{self.name}: `{key}` must be a non-negative local address")
        if addr == isa.GARBAGE_ADDR:
            return
        if addr & isa.ACC_ADDR_BIT:
            row = addr & 0x3FFF
            if row >= F.ACC_ROWS:
                raise VerifyException(
                    f"{self.name}: accumulator row {row} exceeds the RTL depth {F.ACC_ROWS}")
        elif addr >= F.SPAD_ROWS:
            raise VerifyException(
                f"{self.name}: scratchpad row {addr} exceeds the RTL depth {F.SPAD_ROWS}")


@irdl_op_definition
class FlushOp(_GemminiOp):
    """`gemmini.flush` — drain the accelerator (k_FLUSH)."""

    name = "gemmini.flush"


@irdl_op_definition
class ConfigExOp(_GemminiOp):
    """`gemmini.config_ex` — dataflow / activation / strides (k_CONFIG, CONFIG_EX)."""

    name = "gemmini.config_ex"

    def verify_(self) -> None:
        if self.a("dataflow") not in (0, 1):
            raise VerifyException("gemmini.config_ex: `dataflow` must be 0 or 1")


@irdl_op_definition
class ConfigLdOp(_GemminiOp):
    """`gemmini.config_ld` — DMA load stride / scale / id (k_CONFIG, CONFIG_LD)."""

    name = "gemmini.config_ld"

    def verify_(self) -> None:
        if self.a("load_id") not in (0, 1, 2):
            raise VerifyException("gemmini.config_ld: `load_id` must be 0, 1 or 2")
        if int(self.a("stride", 0)) < 0:
            raise VerifyException("gemmini.config_ld: `stride` must be non-negative")


@irdl_op_definition
class ConfigStOp(_GemminiOp):
    """`gemmini.config_st` — DMA store stride, accumulator activation/scale, pooling geometry."""

    name = "gemmini.config_st"

    def verify_(self) -> None:
        if self.a("acc_act") not in (0, 1):
            raise VerifyException("gemmini.config_st: `acc_act` must be NO_ACTIVATION or RELU")


@irdl_op_definition
class MvinOp(_GemminiOp):
    """`gemmini.mvin` — DRAM -> scratchpad/accumulator DMA (k_MVIN / k_MVIN2)."""

    name = "gemmini.mvin"

    def verify_(self) -> None:
        if len(self.operands_) != 1:
            raise VerifyException("gemmini.mvin takes the source DRAM pointer as its operand")
        self._extent("rows")
        cols = self.a("cols")
        if not isinstance(cols, int) or not (1 <= cols <= DIM):
            raise VerifyException(f"gemmini.mvin: `cols` = {cols!r} must be in 1..{DIM}")
        self._local("local")


@irdl_op_definition
class MvoutOp(_GemminiOp):
    """`gemmini.mvout` — scratchpad/accumulator -> DRAM DMA (k_MVOUT)."""

    name = "gemmini.mvout"

    def verify_(self) -> None:
        if len(self.operands_) != 1:
            raise VerifyException("gemmini.mvout takes the destination DRAM pointer as its operand")
        self._extent("rows")
        self._extent("cols")
        self._local("local")


@irdl_op_definition
class PreloadOp(_GemminiOp):
    """`gemmini.preload` — load the stationary operand into the mesh (k_PRELOAD)."""

    name = "gemmini.preload"

    def verify_(self) -> None:
        for key in ("bd_cols", "bd_rows", "c_cols", "c_rows"):
            self._extent(key)
        self._local("bd")
        self._local("c")


@irdl_op_definition
class ComputeOp(_GemminiOp):
    """`gemmini.compute` — stream the moving operand through the mesh (k_COMPUTE_*)."""

    name = "gemmini.compute"

    def verify_(self) -> None:
        self._extent("a_cols")
        self._extent("a_rows")
        self._local("a")


@irdl_op_definition
class FenceOp(_GemminiOp):
    """`gemmini.fence` — wait for every outstanding accelerator command to retire."""

    name = "gemmini.fence"


@irdl_op_definition
class ScratchOp(_GemminiOp):
    """`gemmini.scratch` — a compiler-allocated DRAM staging buffer."""

    name = "gemmini.scratch"

    def verify_(self) -> None:
        if int(self.a("bytes", 0)) <= 0:
            raise VerifyException("gemmini.scratch: `bytes` must be positive")


@irdl_op_definition
class HostEpilogueOp(_GemminiOp):
    """`gemmini.host_epilogue` — compiler-generated CPU-lane readout the store path cannot fuse.

    Operands are (source i32 staging buffer, destination buffer[, bias buffer]).  The `stages`
    attribute is the ordered ABI epilogue; codegen materialises a real loop nest for it.
    """

    name = "gemmini.host_epilogue"

    def verify_(self) -> None:
        if len(self.operands_) < 2:
            raise VerifyException("gemmini.host_epilogue needs a source and a destination")
        if self.a("stages") is None:
            raise VerifyException("gemmini.host_epilogue: `stages` is required")


@irdl_op_definition
class HostTransposeOp(_GemminiOp):
    """`gemmini.host_transpose` — compiler-generated 2-D transpose into a staging buffer."""

    name = "gemmini.host_transpose"

    def verify_(self) -> None:
        if len(self.operands_) != 2:
            raise VerifyException("gemmini.host_transpose needs a source and a destination")


@irdl_op_definition
class Im2ColRowOp(_GemminiOp):
    """`gemmini.im2col_row` — target-generated packing of one convolution output row.

    This is intentionally a target-dialect operation rather than a frontend tensor: lowering may
    select it, direct DMA, or a future native-convolution address generator from the same canonical
    source convolution.
    """

    name = "gemmini.im2col_row"

    def verify_(self) -> None:
        if len(self.operands_) != 2:
            raise VerifyException("gemmini.im2col_row needs source and row-slab buffers")
        for key in ("batch", "out_y", "ci", "hi", "wi", "kh", "kw", "wo",
                    "stride_h", "stride_w", "dilation_h", "dilation_w",
                    "pad_top", "pad_left"):
            if self.a(key) is None:
                raise VerifyException(f"gemmini.im2col_row: `{key}` is required")


@irdl_op_definition
class LoopWsBlockOp(_GemminiOp):
    """`gemmini.loop_ws_block` — one capacity-bounded hardware LOOP_WS launch.

    Operands are A, B and C/D, or A, B, D and C when recovery uses distinct buffers. Offsets and
    row strides make this operation
    useful for ordinary matmul tiles as well as convolution rows; it contains no model-specific
    geometry.
    """

    name = "gemmini.loop_ws_block"

    def verify_(self) -> None:
        if len(self.operands_) not in (3, 4):
            raise VerifyException("gemmini.loop_ws_block needs A, B, [D,] and C buffers")
        for key in ("rows", "cols", "depth", "a_stride", "b_stride", "c_stride",
                    "a_offset", "b_offset", "c_offset", "d_offset", "accumulate",
                    "full_c", "implementation"):
            if self.a(key) is None:
                raise VerifyException(f"gemmini.loop_ws_block: `{key}` is required")
        rows, cols, depth = (int(self.a(k)) for k in ("rows", "cols", "depth"))
        if min(rows, cols, depth) < 1:
            raise VerifyException("gemmini.loop_ws_block extents must be positive")
        ti, tj, tk = ((x + DIM - 1) // DIM for x in (rows, cols, depth))
        # LOOP_WS is a CISC engine with mandatory double-buffering, so one launch may consume
        # at most half of either private memory (the other half is reserved for its alternate
        # buffer).  Primitive mvin/compute schedules have the full capacities; this op does not.
        if ti * tj * DIM > F.ACC_ROWS // 2:
            raise VerifyException(
                "gemmini.loop_ws_block exceeds the double-buffered accumulator capacity")
        if (ti + tj) * tk * DIM > F.SPAD_ROWS // 2:
            raise VerifyException(
                "gemmini.loop_ws_block exceeds the double-buffered scratchpad capacity")
        if self.a("accumulate") not in (0, 1, False, True):
            raise VerifyException("gemmini.loop_ws_block `accumulate` must be boolean")


@irdl_op_definition
class HostRadix128StepOp(_GemminiOp):
    """Recover one balanced radix-128 digit from a legal narrow accumulator store."""

    name = "gemmini.host_radix128_step"

    def verify_(self) -> None:
        if len(self.operands_) != 4:
            raise VerifyException(
                "gemmini.host_radix128_step needs digit, correction, partial and destination")
        for key in ("rows", "cols", "row_elems", "unit", "reset_partial",
                    "commit", "dst_row_elems", "dst_offset_elems", "accumulate_dst"):
            if self.a(key) is None:
                raise VerifyException(f"gemmini.host_radix128_step: `{key}` is required")


@irdl_op_definition
class HostLaneProgramOp(_GemminiOp):
    """`gemmini.host_lane_program` — a region this datapath admits no lowering for, placed on
    the scalar lane.

    It carries NO accelerator instruction by construction: the operands are the DRAM pointers
    the compiler-generated CPU-lane program reads and writes, and `regions` names the interface
    regions whose placement produced it.  Its presence in the target module is what makes the
    routing decision visible in the IR rather than only in the command buffer.
    """

    name = "gemmini.host_lane_program"

    def verify_(self) -> None:
        if not self.operands_:
            raise VerifyException("gemmini.host_lane_program needs at least one buffer")
        if self.a("regions_placed") is None:
            raise VerifyException("gemmini.host_lane_program: `regions_placed` is required")


GEMMINI_OPS = (
    FlushOp, ConfigExOp, ConfigLdOp, ConfigStOp, MvinOp, MvoutOp, PreloadOp,
    ComputeOp, FenceOp, ScratchOp, HostEpilogueOp, HostTransposeOp, Im2ColRowOp,
    LoopWsBlockOp, HostRadix128StepOp,
    HostLaneProgramOp,
)

GEMMINI = Dialect("gemmini", list(GEMMINI_OPS), [])

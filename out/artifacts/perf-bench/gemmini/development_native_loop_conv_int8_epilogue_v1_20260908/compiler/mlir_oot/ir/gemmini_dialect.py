"""The `gemmini` target dialect: one op per RoCC command class this backend emits.

The ops are real xDSL IRDL operations with verifiers, so the lowered module is checked against
the RTL-derived machine limits (mesh `DIM`, scratchpad depth, accumulator depth, the legal funct
set) before anything is encoded.  `gemmini.host_*` ops are the compiler-generated CPU-lane
fixups the accelerator store path cannot express; they lower to ordinary LLVM control flow, not
to a library call.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, IntegerAttr, NoneAttr, StringAttr
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_result_def, var_operand_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from ..tables import rtl_facts as F
from ..tables import isa
from ..tables import loop_ws
from ..tables import loop_conv

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


def _integer_fields(op: _GemminiOp, fields: tuple[str, ...], minimum: int) -> None:
    for key in fields:
        attr = op.attributes.get(key)
        if not isinstance(attr, IntegerAttr) or op.a(key) < minimum:
            raise VerifyException(f"{op.name}: `{key}` must be an integer >= {minimum}")


def _pointer_arguments(op: _GemminiOp, count: int) -> None:
    if len(op.operands_) != count or op.results:
        raise VerifyException(f"{op.name}: expected {count} pointers and no results")
    for operand in op.operands_:
        ty = operand.type
        if (not isinstance(ty, LLVMPointerType) or
                not (isinstance(ty.addr_space, NoneAttr) or
                     isinstance(ty.addr_space, IntAttr) and ty.addr_space.data == 0)):
            raise VerifyException(f"{op.name}: expected address-space-zero LLVM pointers")


@irdl_op_definition
class Im2colRowOp(_GemminiOp):
    """Bounded CPU gather into a padded row slab, not hardware im2col.

    LLVM emission clamps source addresses and blends zero for padding. Storage
    extent and lifetime proofs remain the source/global plan's responsibility.
    """

    name = "gemmini.im2col_row"

    def verify_(self) -> None:
        _pointer_arguments(self, 2)
        _integer_fields(self, ("ci", "hi", "wi", "kh", "kw", "wo", "stride_h",
                               "stride_w", "dilation_h", "dilation_w"), 1)
        _integer_fields(self, ("batch", "out_y", "pad_top", "pad_left"), 0)


@irdl_op_definition
class LoopWsBlockOp(_GemminiOp):
    """Matrix block expanded through the pinned LOOP_WS ABI, not LOOP_CONV_WS.

    Strides count elements and offsets count bytes, matching the LLVM emitter.
    """

    name = "gemmini.loop_ws_block"

    def verify_(self) -> None:
        _pointer_arguments(self, 3)
        _integer_fields(self, ("rows", "cols", "depth", "a_stride", "b_stride", "c_stride"), 1)
        _integer_fields(self, ("a_offset", "b_offset", "c_offset", "d_offset"), 0)
        _integer_fields(self, ("full_c", "accumulate"), 0)
        if self.a("full_c") != 1 or self.a("accumulate") not in (0, 1):
            raise VerifyException(f"{self.name}: full-width output and boolean accumulate required")
        if any(self.a(stride) < self.a(extent) for stride, extent in
               (("a_stride", "depth"), ("b_stride", "cols"), ("c_stride", "cols"))):
            raise VerifyException(f"{self.name}: row stride is smaller than its block extent")
        try:
            c_byte_stride = self.a("c_stride") * (F.ACC_ROW_BYTES // F.DIM)
            if (isa.config_st(stride=c_byte_stride)[2] ^ isa.config_st(stride=0)[2]) != c_byte_stride:
                raise ValueError("output byte stride is not representable by CONFIG_ST")
            capacity = loop_ws.contract()["capacity"]
            mt, nt, kt = ((self.a(key) + F.DIM - 1) // F.DIM
                          for key in ("rows", "cols", "depth"))
            if (mt * nt * F.DIM > capacity["max_acc_rows"]["rows"] or
                    (mt + nt) * kt * F.DIM > capacity["max_spad_rows"]["rows"]):
                raise ValueError("block exceeds pinned LOOP_WS capacity")
            loop_ws.loop_ws_static(rows=self.a("rows"), cols=self.a("cols"),
                depth=self.a("depth"), row_stride_a=self.a("a_stride"),
                row_stride_b=self.a("b_stride"), row_stride_c=self.a("c_stride"),
                full_c=True, accumulate=bool(self.a("accumulate")))
        except ValueError as exc:
            raise VerifyException(f"{self.name}: {exc}") from exc


@irdl_op_definition
class LoopConvWsOp(_GemminiOp):
    """One complete native convolution descriptor (functs 16--21 followed by 15)."""

    name = "gemmini.loop_conv_ws"

    def verify_(self) -> None:
        if len(self.operands_) not in (3, 4):
            raise VerifyException(
                f"{self.name}: expected input, weight, output[, i32 bias] pointers")
        _pointer_arguments(self, len(self.operands_))
        positive = (
            "batch_size", "in_row_dim", "in_col_dim", "in_channels", "out_channels",
            "out_row_dim", "out_col_dim", "pool_out_row_dim", "pool_out_col_dim",
            "stride", "kernel_dim", "kernel_dilation", "pool_size", "pool_stride",
            "batches", "porows", "pocols", "pochs", "krows", "kcols", "kchs",
            "orows", "ocols", "in_stride", "weight_stride", "out_stride",
            "max_pixels_per_row")
        nonnegative = (
            "padding", "pool_padding", "lpad", "rpad", "upad", "dpad", "plpad",
            "prpad", "pupad", "pdpad", "input_offset", "weight_offset",
            "output_offset", "activation", "a_spad_id", "b_spad_id")
        boolean = (
            "no_bias", "no_pool", "downsample", "wrot180", "input_dilated",
            "trans_output_1203", "trans_weight_1203", "trans_weight_0132",
            "trans_input_3120", "dw", "write_output")
        _integer_fields(self, positive, 1)
        _integer_fields(self, nonnegative, 0)
        _integer_fields(self, boolean, 0)
        if any(self.a(key) not in (0, 1) for key in boolean):
            raise VerifyException(f"{self.name}: native convolution flags must be boolean")
        if any(self.a(key) >= (1 << 16) for key in (
                "batch_size", "in_row_dim", "in_col_dim", "in_channels", "out_channels",
                "out_row_dim", "out_col_dim", "pool_out_row_dim", "pool_out_col_dim",
                "batches", "porows", "pocols", "pochs", "krows", "kcols", "kchs",
                "orows", "ocols", "in_stride", "weight_stride", "out_stride")):
            raise VerifyException(f"{self.name}: a 16-bit descriptor field overflows")
        if self.a("kernel_dilation") >= (1 << 10):
            raise VerifyException(f"{self.name}: kernel_dilation overflows its 10-bit field")
        common = dict(
            stride=self.a("stride"), kernel_dilation=self.a("kernel_dilation"),
            batches=self.a("batches"), porows=self.a("porows"),
            pocols=self.a("pocols"), pochs=self.a("pochs"),
            krows=self.a("krows"), kcols=self.a("kcols"), kchs=self.a("kchs"))
        if (loop_conv.total_rows(accumulator=False, **common) > F.SPAD_ROWS // 2
                or loop_conv.total_rows(accumulator=True, **common) > F.ACC_ROWS // 2):
            raise VerifyException(f"{self.name}: tile exceeds double-buffered native capacity")
        try:
            loop_conv.static_descriptor(**{
                key: self.a(key) for key in (
                    "batch_size", "in_row_dim", "in_col_dim", "in_channels", "out_channels",
                    "out_row_dim", "out_col_dim", "pool_out_row_dim", "pool_out_col_dim",
                    "stride", "padding", "kernel_dim", "kernel_dilation", "pool_size",
                    "pool_stride", "pool_padding", "batches", "porows", "pocols", "pochs",
                    "krows", "kcols", "kchs", "lpad", "rpad", "upad", "dpad", "plpad",
                    "prpad", "pupad", "pdpad", "orows", "ocols", "in_stride",
                    "weight_stride", "out_stride", "no_bias", "no_pool", "downsample",
                    "wrot180", "input_dilated", "activation", "trans_output_1203",
                    "trans_weight_1203", "trans_weight_0132", "trans_input_3120",
                    "max_pixels_per_row", "dw", "a_spad_id", "b_spad_id")})
        except (KeyError, ValueError) as exc:
            raise VerifyException(f"{self.name}: {exc}") from exc


GEMMINI_OPS = (
    FlushOp, ConfigExOp, ConfigLdOp, ConfigStOp, MvinOp, MvoutOp, PreloadOp,
    ComputeOp, FenceOp, ScratchOp, HostEpilogueOp, HostTransposeOp,
    HostLaneProgramOp, Im2colRowOp, LoopWsBlockOp,
    LoopConvWsOp,
)

GEMMINI = Dialect("gemmini", list(GEMMINI_OPS), [])

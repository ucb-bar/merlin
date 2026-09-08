"""The `merlin_iface` v0.1 input dialect, defined as real xDSL IRDL ops and types.

The interface grammar is parsed with xDSL's own MLIR parser against these definitions, so a
malformed input module is rejected by the IR verifier (wrong operand count, wrong result type,
an undefined mnemonic) rather than by a hand-rolled text scanner.  No regular expressions and no
bespoke lexer are involved anywhere in this file.

Every op follows the same concrete syntax the grammar fixes::

    %r = merlin_iface.<mnemonic> %a, %b {attrs...} : (<operand types>) -> <result type>
    %r = merlin_iface.tensor {attrs...} : <result type>
    merlin_iface.evict %h : (!merlin_iface.resident) -> ()

so one shared parser/printer pair serves the whole dialect.
"""
from __future__ import annotations

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

GRAMMAR_VERSION = "0.1"

#: The epilogue vocabulary of the frozen ABI (command_buffer_abi.yaml / interface_grammar.md).
EPILOGUE_STAGES = ("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool")


@irdl_attr_definition
class ResidentType(ParametrizedAttribute, TypeAttribute):
    """`!merlin_iface.resident` — an opaque handle to a packed, stationary weight."""

    name = "merlin_iface.resident"


@irdl_attr_definition
class AccType(ParametrizedAttribute, TypeAttribute):
    """`!merlin_iface.acc<i32>` — an opaque integer accumulator handle."""

    name = "merlin_iface.acc"

    element_type: Attribute


class _IfaceOp(IRDLOperation):
    """Shared concrete syntax for every op of the fixed interface grammar."""

    operands_ = var_operand_def()
    res = opt_result_def()

    def print(self, printer: Printer) -> None:
        if self.operands_:
            printer.print_string(" ")
            printer.print_list(self.operands_, printer.print_ssa_value)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        if self.operands_ or self.res is None:
            printer.print_function_type(
                [v.type for v in self.operands_],
                [self.res.type] if self.res is not None else [],
            )
        else:
            printer.print_attribute(self.res.type)

    @classmethod
    def parse(cls, parser: Parser) -> "_IfaceOp":
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
        if unresolved:
            ftype = parser.parse_function_type()
            operands = parser.resolve_operands(unresolved, ftype.inputs.data, pos)
            results = list(ftype.outputs.data)
        else:
            first = parser.parse_optional_type()
            if first is None:                    # `: () -> <t>` shaped, no operands
                ftype = parser.parse_function_type()
                operands, results = [], list(ftype.outputs.data)
            else:
                operands, results = [], [first]
        op = cls(operands=[operands], result_types=[results])
        op.attributes |= attrs
        return op

    # -- shared verification helpers --------------------------------------------------------
    def _string(self, key: str) -> str | None:
        attr = self.attributes.get(key)
        data = getattr(attr, "data", None)
        return data if isinstance(data, str) else None

    def _require_name(self) -> None:
        if self._string("name") is None:
            raise VerifyException(f"{self.name}: a `name` string attribute is required")

    def _check_epilogue(self) -> None:
        attr = self.attributes.get("epilogue")
        if attr is None:
            return
        for entry in getattr(attr, "data", ()):  # ArrayAttr of StringAttr
            stage = getattr(entry, "data", None)
            if stage not in EPILOGUE_STAGES:
                raise VerifyException(
                    f"{self.name}: epilogue stage {stage!r} is not in {list(EPILOGUE_STAGES)}"
                )
        stages = [getattr(e, "data", None) for e in getattr(attr, "data", ())]
        # `acc_scale` without an explicit value means the identity scale
        # (`ACC_SCALE_IDENTITY` in gemmini_params.h); the shipped corpus exercises that spelling.
        if any(s in ("bias_add", "bias") for s in stages) and self._string("bias") is None:
            raise VerifyException(
                f"{self.name}: a bias epilogue stage names no bias tensor"
            )
        if "maxpool" in stages:
            for key in ("pool_in_dims", "pool_size", "pool_stride"):
                if key not in self.attributes:
                    raise VerifyException(
                        f"{self.name}: maxpool epilogue requires `{key}` (no default exists)"
                    )


@irdl_op_definition
class TensorOp(_IfaceOp):
    """`merlin_iface.tensor` — declare a leaf input/weight/bias."""

    name = "merlin_iface.tensor"

    def verify_(self) -> None:
        self._require_name()
        role = self._string("role")
        if role not in ("weight", "input", "bias"):
            raise VerifyException(f"merlin_iface.tensor: bad role {role!r}")
        if self.operands_:
            raise VerifyException("merlin_iface.tensor takes no operands")


@irdl_op_definition
class ResidentPackOp(_IfaceOp):
    """`merlin_iface.resident_pack` — make a weight resident."""

    name = "merlin_iface.resident_pack"

    def verify_(self) -> None:
        if len(self.operands_) != 1:
            raise VerifyException("merlin_iface.resident_pack takes exactly one operand")
        if not isinstance(self.res.type, ResidentType):
            raise VerifyException("merlin_iface.resident_pack must produce !merlin_iface.resident")


@irdl_op_definition
class MatmulOp(_IfaceOp):
    """`merlin_iface.matmul` — matmul of a streaming lhs against a resident weight."""

    name = "merlin_iface.matmul"

    def verify_(self) -> None:
        if len(self.operands_) != 2:
            raise VerifyException("merlin_iface.matmul takes exactly two operands")
        if not isinstance(self.operands_[1].type, ResidentType):
            raise VerifyException("merlin_iface.matmul rhs must be !merlin_iface.resident")
        if not isinstance(self.res.type, AccType):
            raise VerifyException("merlin_iface.matmul must produce !merlin_iface.acc")


@irdl_op_definition
class MatmulBatchedOp(_IfaceOp):
    """`merlin_iface.matmul_batched` — a stack of independent matmuls."""

    name = "merlin_iface.matmul_batched"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 2:
            raise VerifyException("merlin_iface.matmul_batched takes exactly two operands")


@irdl_op_definition
class CommitOp(_IfaceOp):
    """`merlin_iface.commit` — apply the epilogue and commit an accumulator."""

    name = "merlin_iface.commit"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 1:
            raise VerifyException("merlin_iface.commit takes exactly one operand")
        if not isinstance(self.operands_[0].type, AccType):
            raise VerifyException("merlin_iface.commit consumes an !merlin_iface.acc")
        if self._string("output_dtype") is None:
            raise VerifyException("merlin_iface.commit requires `output_dtype`")
        self._check_epilogue()


@irdl_op_definition
class EvictOp(_IfaceOp):
    """`merlin_iface.evict` — release a resident weight."""

    name = "merlin_iface.evict"

    def verify_(self) -> None:
        if len(self.operands_) != 1:
            raise VerifyException("merlin_iface.evict takes exactly one operand")
        if not isinstance(self.operands_[0].type, ResidentType):
            raise VerifyException("merlin_iface.evict consumes a !merlin_iface.resident")


@irdl_op_definition
class MovementOp(_IfaceOp):
    """`merlin_iface.movement` — an identity load/store round trip through the accelerator."""

    name = "merlin_iface.movement"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 1:
            raise VerifyException("merlin_iface.movement takes exactly one operand")


@irdl_op_definition
class Conv2dOp(_IfaceOp):
    """`merlin_iface.conv2d` — NHWC convolution against a pre-im2col'd weight."""

    name = "merlin_iface.conv2d"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 2:
            raise VerifyException("merlin_iface.conv2d takes exactly two operands")
        if "kernel" not in self.attributes:
            raise VerifyException("merlin_iface.conv2d requires a `kernel` attribute")
        layout = self._string("layout")
        if layout not in (None, "nhwc"):
            raise VerifyException(f"merlin_iface.conv2d layout {layout!r} is rejected (nhwc only)")
        self._check_epilogue()


@irdl_op_definition
class BiasAddOp(_IfaceOp):
    """`merlin_iface.bias_add` — a length-N vector added to every row of a committed tensor."""

    name = "merlin_iface.bias_add"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 2:
            raise VerifyException("merlin_iface.bias_add takes exactly two operands")


@irdl_op_definition
class AttentionQkOp(_IfaceOp):
    """`merlin_iface.attention_qk` — q @ transpose(k)."""

    name = "merlin_iface.attention_qk"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 2:
            raise VerifyException("merlin_iface.attention_qk takes exactly two operands")
        self._check_epilogue()


@irdl_op_definition
class AttentionPvOp(_IfaceOp):
    """`merlin_iface.attention_pv` — p @ v."""

    name = "merlin_iface.attention_pv"

    def verify_(self) -> None:
        self._require_name()
        if len(self.operands_) != 2:
            raise VerifyException("merlin_iface.attention_pv takes exactly two operands")
        self._check_epilogue()


@irdl_op_definition
class SoftmaxOp(_IfaceOp):
    """`merlin_iface.softmax` — row-wise softmax."""

    name = "merlin_iface.softmax"

    def verify_(self) -> None:
        self._require_name()


@irdl_op_definition
class RmsNormOp(_IfaceOp):
    """`merlin_iface.rmsnorm` — row-wise RMS normalisation."""

    name = "merlin_iface.rmsnorm"

    def verify_(self) -> None:
        self._require_name()


@irdl_op_definition
class RopeOp(_IfaceOp):
    """`merlin_iface.rope` — rotary positional embedding."""

    name = "merlin_iface.rope"

    def verify_(self) -> None:
        self._require_name()


IFACE_OPS = (
    TensorOp, ResidentPackOp, MatmulOp, MatmulBatchedOp, CommitOp, EvictOp,
    MovementOp, Conv2dOp, BiasAddOp, AttentionQkOp, AttentionPvOp,
    SoftmaxOp, RmsNormOp, RopeOp,
)

MERLIN_IFACE = Dialect("merlin_iface", list(IFACE_OPS), [ResidentType, AccType])

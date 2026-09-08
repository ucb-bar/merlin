"""Register the two builtin-style FP8 spellings used by interface MLIR."""
from xdsl.dialects.builtin import _FloatType
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition


class _FP8(ParametrizedAttribute, _FloatType):
    @property
    def bitwidth(self):
        return 8

    @property
    def compile_time_size(self):
        return 1

    def iter_unpack(self, buffer, /):
        raise NotImplementedError

    def unpack(self, buffer, num, /):
        raise NotImplementedError

    def pack_into(self, buffer, offset, value):
        raise NotImplementedError

    def pack(self, values):
        raise NotImplementedError


@irdl_attr_definition
class Float8E4M3FNType(_FP8):
    name = "f8E4M3FN"


@irdl_attr_definition
class Float8E5M2Type(_FP8):
    name = "f8E5M2"


_TYPES = {"f8E4M3FN": Float8E4M3FNType(), "f8E5M2": Float8E5M2Type()}


def register_fp8_types():
    from xdsl.parser.attribute_parser import AttrParser
    from xdsl.utils.mlir_lexer import MLIRTokenKind

    if getattr(AttrParser, "_atlas_fp8_hook", False):
        return
    original = AttrParser._parse_optional_integer_or_float_type

    def parse(self):
        token = self._current_token
        if token.kind == MLIRTokenKind.BARE_IDENT and token.text in _TYPES:
            self._consume_token()
            return _TYPES[token.text]
        return original(self)

    AttrParser._parse_optional_integer_or_float_type = parse
    AttrParser._atlas_fp8_hook = True


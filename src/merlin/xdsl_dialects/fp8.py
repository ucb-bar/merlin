"""Standard FP8 types and exact constant codecs for the pinned xDSL reader.

All four TT_Float formats at Triton f797708c are builtin-style MLIR float
types. The optional ml_dtypes codec normalizes decimal constants and keeps
raw words intact, including NaNs and signed zeros. A shared parser registry
lets this standalone consumer cooperate with triton2MLIR in either import
order without imposing a dependency between the two packages.

This binding grants no numeric conversion, arithmetic, overflow, storage ABI
or target implementation policy. Native lowering remains a separate gate.
"""

from ._common import HAS_XDSL

FP8_TYPE_NAMES = ("f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ")

if HAS_XDSL:
    import sys
    from dataclasses import fields, is_dataclass
    from importlib import import_module

    from xdsl.dialects import builtin
    from xdsl.ir import ParametrizedAttribute
    from xdsl.irdl import irdl_attr_definition

    NUMPY_NAMES = dict(
        zip(FP8_TYPE_NAMES, ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"), strict=True)
    )

    class _WordFloat(float):
        def __new__(cls, value, word, format_):
            result = super().__new__(cls, value)
            result.word, result.format = word, format_
            return result

    class _Fp8Type(ParametrizedAttribute, builtin._FloatType):
        @property
        def bitwidth(self):
            return 8

        @property
        def compile_time_size(self):
            return 1

        def _dtype(self):
            try:
                import ml_dtypes
            except ImportError as error:
                raise ValueError("FP8 constant packing requires the optional ml-dtypes dependency") from error
            return getattr(ml_dtypes, NUMPY_NAMES[self.name])

        def iter_unpack(self, buffer, /):
            import numpy as np

            raw = bytes(buffer)
            for word, value in zip(raw, np.frombuffer(raw, dtype=self._dtype()), strict=True):
                yield _WordFloat(float(value), word, self.name)

        def unpack(self, buffer, num, /):
            return tuple(value for _, value in zip(range(num), self.iter_unpack(buffer)))

        def pack(self, values):
            import numpy as np

            dtype = self._dtype()
            with np.errstate(over="ignore", invalid="ignore"):
                return b"".join(
                    bytes([value.word])
                    if isinstance(value, _WordFloat) and value.format == self.name
                    else np.asarray(value, dtype=dtype).tobytes()
                    for value in values
                )

        def pack_into(self, buffer, offset, value):
            memoryview(buffer)[offset : offset + 1] = self.pack([value])

    @irdl_attr_definition
    class Float8E4M3FNType(_Fp8Type):
        name = "f8E4M3FN"

    @irdl_attr_definition
    class Float8E4M3FNUZType(_Fp8Type):
        name = "f8E4M3FNUZ"

    @irdl_attr_definition
    class Float8E5M2Type(_Fp8Type):
        name = "f8E5M2"

    @irdl_attr_definition
    class Float8E5M2FNUZType(_Fp8Type):
        name = "f8E5M2FNUZ"

    _FP8_BY_NAME = {
        type_.name: type_() for type_ in (Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type, Float8E5M2FNUZType)
    }

    def register_fp8_types():
        from xdsl.parser.attribute_parser import AttrParser
        from xdsl.utils.mlir_lexer import MLIRTokenKind

        dense = builtin.DenseIntOrFPElementsAttr
        if not getattr(dense, "_portable_exact_splat_hook", False):
            previous_splat = dense.is_splat

            def exact_splat(self):
                raw = self.data.data
                size = self.get_element_type().compile_time_size
                # Numeric equality cannot collapse distinct signed-zero words.
                # Keep the old NaN classification; only remove false splats.
                return (
                    bool(raw)
                    and len(raw) % size == 0
                    and raw.count(raw[:size]) == len(raw) // size
                    and previous_splat(self)
                )

            dense.is_splat = exact_splat
            dense._portable_exact_splat_hook = True

        # Cooperating readers use one canonical class per builtin spelling.
        registry = getattr(AttrParser, "_portable_fp8_types", None)
        if registry is None:
            registry = {}
            setattr(AttrParser, "_portable_fp8_types", registry)
        for name, type_ in tuple(_FP8_BY_NAME.items()):
            canonical = registry.setdefault(name, type_)
            _FP8_BY_NAME[name] = canonical
            globals()[type(type_).__name__] = type(canonical)
        if not getattr(AttrParser, "_portable_fp8_parse_hook", False):
            previous = AttrParser._parse_optional_integer_or_float_type

            def parse(self):
                token = self._current_token
                if token.kind == MLIRTokenKind.BARE_IDENT and token.text in registry:
                    self._consume_token()
                    return registry[token.text]
                return previous(self)

            AttrParser._parse_optional_integer_or_float_type = parse
            setattr(AttrParser, "_portable_fp8_parse_hook", True)
        if not getattr(builtin.FloatAttr, "_portable_fp8_init_hook", False):
            previous_init = builtin.FloatAttr.__init__

            def init(self, data, type):
                if isinstance(type, _Fp8Type) or any(type == item for item in registry.values()):
                    value = data.data if isinstance(data, builtin.FloatData) else data
                    data = next(type.iter_unpack(type.pack([value])))
                previous_init(self, data, type)

            builtin.FloatAttr.__init__ = init
            builtin.FloatAttr._portable_fp8_init_hook = True

    def register_fp8_float_constraints():
        from xdsl.irdl.constraints import AnyOf, BaseAttr

        # These pinned reader/builders captured AnyFloat by value. Extend the
        # actual float universe and constraints, preserving every existing class.
        for name in (
            "xdsl.dialects.builtin",
            "xdsl.parser.attribute_parser",
            "xdsl.printer",
            "xdsl.dialects.arith",
            "xdsl.dialects.math",
            "xdsl.dialects.linalg.ops",
            "xdsl.dialects.linalg.abstract_ops",
        ):
            module = import_module(name)
            if hasattr(module, "AnyFloat"):
                types = module.AnyFloat
                for type_ in _FP8_BY_NAME.values():
                    types = types | type(type_)
                module.AnyFloat = types
        if (module := sys.modules.get("merlin.common.dense_float_ir")) is not None:
            for type_ in _FP8_BY_NAME.values():
                module.AnyFloat = module.AnyFloat | type(type_)
        original_floats = {
            builtin.BFloat16Type,
            builtin.Float16Type,
            builtin.Float32Type,
            builtin.Float64Type,
            builtin.Float80Type,
            builtin.Float128Type,
        }
        seen = set()

        def extend(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, AnyOf) and original_floats <= value._based_constrs.keys():
                # IRDL has flattened copies of AnyFloatConstr in dense constants,
                # bitcasts and other definitions. Keep the public union and its
                # cached dispatch table consistent, including future constraints
                # composed from this object. Restricted float subsets stay intact.
                additions = tuple(
                    BaseAttr(type(type_)) for type_ in _FP8_BY_NAME.values() if type(type_) not in value._based_constrs
                )
                object.__setattr__(value, "attr_constrs", (*value.attr_constrs, *additions))
                for constraint in additions:
                    value._based_constrs[constraint.attr] = constraint
            if is_dataclass(value) and not isinstance(value, type):
                for field in fields(value):
                    extend(getattr(value, field.name))
            elif isinstance(value, dict):
                for child in value.values():
                    extend(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    extend(child)

        extend(builtin.AnyFloatConstr)
        for module_name, dialect_name in (
            ("builtin", "Builtin"),
            ("arith", "Arith"),
            ("math", "Math"),
            ("linalg", "Linalg"),
        ):
            dialect = getattr(import_module("xdsl.dialects." + module_name), dialect_name)
            for definition in (*dialect.attributes, *dialect.operations):
                if hasattr(definition, "get_irdl_definition"):
                    extend(definition.get_irdl_definition())

    register_fp8_types()
    register_fp8_float_constraints()
else:  # pragma: no cover - optional xDSL is absent
    _FP8_BY_NAME = {}

    def register_fp8_types():
        return

    def register_fp8_float_constraints():
        return

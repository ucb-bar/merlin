"""Invocation-local custody for explicitly selected descriptor identities.

This is ordinary source preparation, not a qualification or permission owner.
The caller must separately bind the original checked C forwarder and its objects.
"""

from __future__ import annotations

import inspect
import io
import json
from dataclasses import asdict, replace
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import TensorType
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.mlir_lexer import MLIRLexer

from merlin.common import dense_float_ir, ir_lock
from merlin.common.jsonio import write_pretty_json
from merlin.frontends import linalg_mlir

from . import descriptor_result_identity as D
from . import borrowed_buffer_calls as K
from .lowering_recipe import _identity


def _records(identities):
    if type(identities) is not tuple or not identities or len(identities) > 4096:
        raise ValueError("descriptor identities require a bounded immutable original tuple")
    rows = []
    for identity in identities:
        if (
            type(identity) is not D.DescriptorResultIdentity
            or type(identity.argument_types) is not tuple
            or type(identity.argument_access) is not tuple
            or any(not isinstance(typ, TensorType) for typ in identity.argument_types)
        ):
            raise ValueError("descriptor identities require exact typed original selections")
        D.validate_effects(identity.borrowed_effects)
        rows.append(
            {
                "symbol": identity.symbol,
                "borrowed_symbol": identity.borrowed_symbol,
                "wrapper_symbol": identity.wrapper_symbol,
                "result_argument": identity.result_argument,
                "argument_types": [str(typ) for typ in identity.argument_types],
                "argument_access": list(identity.argument_access),
                **({"argument_layouts": [str(v) for v in identity.argument_layouts]} if identity.argument_layouts else {}),
                "borrowed_effects": None if identity.borrowed_effects is None else asdict(identity.borrowed_effects),
            }
        )
    return json.dumps(rows, sort_keys=True, allow_nan=False)


def _effects_state(owner):
    if owner is None:
        return None
    if type(owner) is not K.BorrowedBufferCalls:
        raise ValueError("descriptor lifetime requires the original live borrowed-call owner")
    callbacks = tuple(K._callback_binding(callback) for callback in (owner.validate_original, owner.observe_native))
    return json.dumps(owner.record(), sort_keys=True), tuple((id(c), c) for c in owner.calls), callbacks


def _join_effects(identities, owner):
    """Bind exact original effects, never infer them from returned descriptors."""
    if owner is None:
        return
    if (type(owner) is not K.BorrowedBufferCalls or len(owner.calls) != len(identities)
            or any(type(call) is not K.BorrowedBufferCall for call in owner.calls)):
        raise ValueError("descriptor effects differ from the complete original identity roster")
    calls = {(call.wrapper_symbol, call.borrowed_symbol): call for call in owner.calls}
    if len(calls) != len(identities):
        raise ValueError("descriptor effects have ambiguous original calls")
    for identity in identities:
        call = calls.get((identity.wrapper_symbol, identity.borrowed_symbol))
        expected = tuple(str(t) for t in D.argument_buffer_types(identity))
        if (call is None or call.argument_types != expected or call.result_argument != identity.result_argument
                or call.effects is not identity.borrowed_effects):
            raise ValueError("descriptor effects differ from the exact original ABI or effect owner")


def implementation_paths():
    from merlin.xdsl_dialects import _common

    from . import lowering_recipe

    return tuple(
        dict.fromkeys(
            (
                Path(__file__),
                Path(K.__file__),
                Path(__file__).with_name("lower.py"),
                Path(lowering_recipe.__file__),
                Path(linalg_mlir.__file__),
                Path(_common.__file__),
                Path(dense_float_ir.__file__),
                Path(ir_lock.__file__),
                Path(inspect.getsourcefile(Context)),
                Path(inspect.getsourcefile(MLIRLexer)),
                Path(inspect.getsourcefile(Parser)),
                Path(inspect.getsourcefile(Printer)),
                *D.implementation_paths(),
            )
        )
    )


class DescriptorResultPreparation:
    """Freeze originals before callbacks; no cross-invocation cache or grant."""

    def __init__(self, identities, *, effects=None):
        self.identities = identities
        self._original = tuple(identities)
        self._selection = _records(identities)
        self.effects = effects
        self._original_effects = effects
        _join_effects(identities, effects)
        self._effects_state = _effects_state(effects)
        self._function_reader = self._fixed_functions
        self._functions = tuple((function, function.__code__) for function in self._function_reader())
        self._owners = tuple(_identity(path) for path in implementation_paths())
        self._products = ()
        self._report = None
        self._recipe = None
        self._upstream = None
        self._feature = None
        self.rewrites = 0

    @staticmethod
    def _fixed_functions():
        return (
            DescriptorResultPreparation.__init__,
            DescriptorResultPreparation._fixed_functions,
            DescriptorResultPreparation.revalidate,
            DescriptorResultPreparation._feature_values,
            DescriptorResultPreparation.selected_feature,
            DescriptorResultPreparation.prepare,
            DescriptorResultPreparation.native_calls,
            DescriptorResultPreparation.retain_upstream,
            DescriptorResultPreparation.bind_recipe,
            DescriptorResultPreparation.record,
            _records,
            _effects_state,
            _join_effects,
            implementation_paths,
            _identity,
            D.rewrite_descriptor_result_identities,
            D.argument_buffer_types,
            D.edit_pipeline,
            D.ensure_registered,
            D.validate_effects,
            D.BorrowedBufferEffects.__post_init__,
            K.BorrowedBufferCalls.revalidate,
            K.BorrowedBufferCalls.record,
            K.BorrowedBufferCalls.after_preprocessing,
            K._callback_binding,
            linalg_mlir.parse_mlir_text,
            linalg_mlir.layout_to_buffer_type,
            linalg_mlir.layout_to_buffer_type().verify_,
        )

    def revalidate(self):
        if (
            any(
                function is not original or function.__code__ is not code
                for function, (original, code) in zip(self._function_reader(), self._functions, strict=True)
            )
            or _records(self.identities) != self._selection
            or self.effects is not self._original_effects
            or _effects_state(self.effects) != self._effects_state
            or any(actual is not original for actual, original in zip(self.identities, self._original, strict=True))
            or tuple(_identity(Path(pin["path"])) for pin in self._owners) != self._owners
            or any(_identity(Path(pin["path"])) != pin for pin in self._products)
            or (self._recipe is not None and _identity(Path(self._recipe["path"])) != self._recipe)
            or (self._feature is not None and self._feature_values() != self._feature)
        ):
            raise ValueError("original descriptor identity selection, source or retained product changed")
        _join_effects(self.identities, self.effects)
        if self.effects is not None:
            self.effects.revalidate()
            if _effects_state(self.effects) != self._effects_state:
                raise ValueError("original descriptor effects changed during validation")

    @staticmethod
    def _feature_values():
        from .impr_features import get

        feature = get(D.FEATURE)
        return (
            feature,
            feature.edit_pipeline,
            feature.edit_schedule,
            feature.edit_cflags,
            feature.implies,
            feature.schedule_replace,
            feature.alternative_group,
            feature.requires_exactly_one_of,
        )

    def selected_feature(self):
        self.revalidate()
        D.ensure_registered()
        values = self._feature_values()
        if values[1:] != (D.edit_pipeline, None, None, frozenset(), False, None, frozenset()):
            raise ValueError("descriptor identity has a different registered pipeline hook")
        self._feature = values
        self.revalidate()
        return D.FEATURE

    def prepare(self, source, work):
        self.revalidate()
        if self._report is not None:
            raise ValueError("descriptor identity preparation can bind only once")
        module = linalg_mlir.parse_mlir_text(source)
        module.verify()
        rows = D.rewrite_descriptor_result_identities(module, self.identities)
        module.verify()
        stream = io.StringIO()
        dense_float_ir.DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
        selected = stream.getvalue() if any(row["calls"] for row in rows) else source
        original_path, selected_path = (
            work / "descriptor-identity.input.mlir",
            work / "descriptor-identity.selected.mlir",
        )
        original_path.write_text(source, encoding="utf-8")
        selected_path.write_text(selected, encoding="utf-8")
        self.rewrites = sum(row["calls"] for row in rows)
        record = {
            "schema": "merlin.descriptor_result_preparation.v1",
            "binding_stage": "after_automatic_source_passes_before_preprocessing",
            "selection": json.loads(self._selection),
            "borrowed_calls": None if self.effects is None else self.effects.record(),
            "rewrites": self.rewrites,
            "calls": rows,
            "original_source": _identity(original_path),
            "selected_source": _identity(selected_path),
            "implementation_sources": list(self._owners),
            "qualification": (
                "conditional original emitter descriptor identity; "
                "no new effect, numerical, runtime, target or authoring admission"
            ),
        }
        report = work / "descriptor-result-identity.json"
        write_pretty_json(report, record)
        self._report = _identity(report)
        self._products = (record["original_source"], record["selected_source"], self._report)
        self.revalidate()
        return selected

    def native_calls(self):
        """Transport only actually rewritten calls with the original live owner.

        Ineligible calls retain their descriptor-return route and remain opaque
        to the native ownership checker. This selection is freshly derived from
        the actual sealed preparation, never from a supplied report dictionary.
        """
        self.revalidate()
        if self.effects is None or not self.rewrites:
            return None
        record = json.loads(Path(self._report["path"]).read_text())
        selected = {row["wrapper_symbol"] for row in record["calls"] if row["calls"]}
        calls = tuple(call for call in self.effects.calls if call.wrapper_symbol in selected)
        if len(calls) != len(selected):
            raise ValueError("actual descriptor rewrite lost its original effect owner")
        pins = tuple((Path(pin["path"]), pin["sha256"]) for pin in self._products)
        result = replace(self.effects, calls=calls, pins=(*self.effects.pins, *pins), validate_original=self.revalidate)
        result.revalidate()
        return result

    def retain_upstream(self, path):
        self.revalidate()
        if self._upstream is not None:
            raise ValueError("descriptor identity preprocessing source can bind only once")
        pin = _identity(path)
        self._products = (*self._products, pin)
        self._upstream = pin

    def bind_recipe(self, path, upstream, llvm_text):
        """Join captured preparation to actual returned upstream invocation."""
        from merlin.common.digest import sha256_text

        self.revalidate()
        record = json.loads(path.read_text())
        prepared = record.get("sources", {}).get("prepared_mlir")
        if (
            self._upstream is None
            or record.get("schema") != "merlin.lowering_recipe.v1"
            or record.get("status") != "returned"
            or record.get("returned_llvm_ir") != {"sha256": sha256_text(llvm_text), "bytes": len(llvm_text.encode())}
            or not isinstance(prepared, dict)
            or _identity(Path(prepared["path"])) != prepared
        ):
            raise ValueError("descriptor identity has no matching returned lowering recipe")
        if _identity(upstream) != self._upstream:
            raise ValueError("descriptor identity preprocessing source changed")
        joined = {"report": self._report, "upstream_source": self._upstream, "actual_prepared_mlir": prepared}
        for index, pin in enumerate((*self._owners, *self._products, joined["upstream_source"])):
            record["sources"][f"descriptor_identity_{index}"] = pin
        record["descriptor_result_identity"] = joined
        write_pretty_json(path, record)
        self._recipe = _identity(path)
        self.revalidate()

    def record(self):
        self.revalidate()
        return {
            "report": dict(self._report),
            "recipe": None if self._recipe is None else dict(self._recipe),
            **json.loads(Path(self._report["path"]).read_text()),
        }

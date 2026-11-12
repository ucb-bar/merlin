"""Original-caller admission of exact source annotations, without rewriting IR.

Coverage identifies consumed attributes. It supplies no numerical accuracy,
input-domain truth, target capability or implementation-equivalence permission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from xdsl.dialects import builtin, func
from xdsl.ir import Attribute

from merlin.compiler.sourceguard import validator_identity
from merlin.compiler.transforms.certificates import ir_digest


def _source_members():
    import sys

    owner = sys.modules[__name__]
    return (
        *(
            (owner, name)
            for name in (
                "_source_members",
                "validator_identity",
                "source_semantic_attributes",
                "region_semantic_attributes",
            )
        ),
        (BoundSourceSemantics, "validate"),
        (BoundSourceSemantics, "record"),
    )


@dataclass(frozen=True)
class SourceSemanticAttribute:
    """One actual typed attribute and its immutable observed state."""

    ordinal: int
    location: str
    key: str
    attribute: Attribute
    attribute_type: type
    value: str

    def __post_init__(self):
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.location) is not str
            or not self.location
            or type(self.key) is not str
            or not self.key
            or not isinstance(self.attribute, Attribute)
            or self.attribute_type is not type(self.attribute)
            or type(self.value) is not str
            or self.value != str(self.attribute)
        ):
            raise ValueError("source semantic coverage requires exact typed immutable attribute observations")

    def record(self):
        return {
            "ordinal": self.ordinal,
            "location": self.location,
            "key": self.key,
            "type": self.attribute_type.__module__ + "." + self.attribute_type.__qualname__,
            "value": self.value,
        }


def source_semantic_attributes(module, entry) -> tuple[SourceSemanticAttribute, ...]:
    """Observe all source annotations requiring explicit caller interpretation.

    Module/function execution attributes remain separate strict intake barriers.
    Entry argument/result attributes are typed dictionary members, not generic
    permission to reinterpret a buffer or an operation's execution effects.
    """
    rows = []

    def observe(ordinal, location, attributes):
        for key, value in sorted(attributes.items()):
            if not key.startswith("prov."):
                rows.append(SourceSemanticAttribute(ordinal, location, key, value, type(value), str(value)))

    for ordinal, operation in enumerate(module.walk()):
        if isinstance(operation, builtin.ModuleOp):
            continue
        if isinstance(operation, func.FuncOp):
            if operation.sym_name.data != entry:
                continue
            for name in ("arg_attrs", "res_attrs"):
                attributes = getattr(operation, name)
                if attributes is None:
                    continue
                for index, dictionary in enumerate(attributes):
                    if not isinstance(dictionary, builtin.DictionaryAttr):
                        raise ValueError("source semantic signature attributes require typed dictionaries")
                    observe(ordinal, f"{name}:{index}", dictionary.data)
        else:
            observe(ordinal, "operation", operation.attributes)
    return tuple(rows)


def region_semantic_attributes(module, entry, operations) -> tuple[SourceSemanticAttribute, ...]:
    """Observe selected trees plus still-inherited signature/return obligations.

    Unrelated operation annotations stay on their original operations and need
    full original-caller coverage. Signature and return policies cannot acquire
    a local interpretation, domain propagation or erasure from this inventory.
    Module/function execution policies remain strict source-intake barriers.
    """
    selected = {nested for operation in operations for nested in operation.walk()}
    walked = tuple(module.walk())
    return tuple(
        row
        for row in source_semantic_attributes(module, entry)
        if row.location != "operation" or walked[row.ordinal] in selected or type(walked[row.ordinal]) is func.ReturnOp
    )


def _options_record(value):
    def check(item):
        if item is None or type(item) in (str, bool, int, float):
            return
        if type(item) is list:
            for member in item:
                check(member)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for member in item.values():
                check(member)
            return
        raise ValueError("source admission options require plain JSON values and string keys")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same_binding(left, right):
    return all(a is b for a, b in zip(left, right, strict=True))


def _same_attributes(left, right):
    return len(left) == len(right) and all(
        type(b) is SourceSemanticAttribute
        and type(b.ordinal) is int
        and type(b.location) is str
        and type(b.key) is str
        and type(b.value) is str
        and (a.ordinal, a.location, a.key, a.value) == (b.ordinal, b.location, b.key, b.value)
        and a.attribute is b.attribute
        and a.attribute_type is b.attribute_type
        for a, b in zip(left, right, strict=True)
    )


class BoundSourceSemantics:
    """One original callback/options owner bound to an unchanged source program."""

    def __init__(self, module, entry, callback, options):
        from merlin.compiler.sourceguard import observe

        self._sources = observe("semantics", _source_members)
        self.module, self.entry, self.callback, self.options = module, entry, callback, options
        self._owners = (module, callback, options, self._sources)
        self._binding = validator_identity(callback)
        self._options = _options_record(options)
        self._source = self._operations = self._attributes = None

    def validate(self, module, entry):
        self._sources.revalidate()
        if (
            module is not self.module
            or entry != self.entry
            or not _same_binding((self.module, self.callback, self.options, self._sources), self._owners)
            or not _same_binding(validator_identity(self.callback), self._binding)
            or _options_record(self.options) != self._options
        ):
            raise ValueError("original source semantic admission or options changed")
        source, operations = ir_digest(module), tuple(module.walk())
        attributes = source_semantic_attributes(module, entry)
        if self._source is not None and (
            source != self._source
            or operations != self._operations
            or not _same_attributes(self._attributes, attributes)
        ):
            raise ValueError("source semantic premises changed after admission")
        coverage = self.callback(module, entry, self.options)
        self._sources.revalidate()
        if (
            ir_digest(module) != source
            or tuple(module.walk()) != operations
            or not _same_attributes(attributes, source_semantic_attributes(module, entry))
        ):
            raise ValueError("source semantic validator changed the original IR")
        if (
            not _same_binding(validator_identity(self.callback), self._binding)
            or not _same_binding((self.module, self.callback, self.options, self._sources), self._owners)
            or _options_record(self.options) != self._options
        ):
            raise ValueError("source semantic validator changed its original binding or options")
        if type(coverage) is not tuple or not _same_attributes(attributes, coverage):
            raise ValueError("source semantic validator did not consume every exact typed source attribute")
        self._source, self._operations, self._attributes = source, operations, attributes

    def record(self):
        if self._source is None:
            raise ValueError("source semantics have not been admitted")
        callback = self._binding[1].__module__ + "." + self._binding[1].__qualname__
        return {
            "origin": "original_caller",
            "callback": callback,
            "source_sha256": self._source,
            "options": json.loads(self._options),
            "attributes": [row.record() for row in self._attributes],
            "qualification": "not_established",
            "source_observation": self._sources.record(),
        }

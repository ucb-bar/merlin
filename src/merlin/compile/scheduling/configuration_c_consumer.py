"""Explicit original index64 C-consumer selection for configuration hoisting.

This selection binds integer-domain admission to actual builtin C emission. It
issues no proof, hardware approval or native/encoder fidelity claim. Formal
hoist authority still belongs to the original selected canonical worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.sched.ir import Kernel

_LIVE = weakref.WeakKeyDictionary()
_MODULES = (
    "merlin.compile.scheduling.configuration_c_consumer",
    "merlin.sched.codegen",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.events",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sources():
    result = []
    for name in _MODULES:
        path = module_source_path(name)
        value = path.read_bytes()
        result.append({"path": str(path), "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)})
    return sorted(result, key=lambda row: row["path"])


def _functions():
    from merlin.sched import codegen
    from merlin.sched.ir import expr, integer_domain

    return (
        codegen.emit_c_function,
        codegen._check_integer_domain,
        codegen._c_value,
        codegen.render,
        integer_domain.check_integer_domain,
        expr.render,
    )


@dataclass(frozen=True)
class _Issued:
    pid: int
    thread: int
    consumer: object
    result: object
    source: Kernel
    source_text: str
    output: Kernel
    output_text: str
    selected_contracts: object
    selected_record: str
    bounds: tuple
    consumer_record: str
    proof_inputs: tuple
    transport_refs: tuple


@dataclass(frozen=True, eq=False)
class ConfigurationCConsumer:
    """Select the original actual C emitter's domain before producer execution."""

    index_bits: int
    _original: str = field(init=False, repr=False)
    _code: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.index_bits) is not int or self.index_bits != 64:
            raise ValueError("configuration C consumer requires explicitly selected integer index64")
        object.__setattr__(self, "_original", _canonical(self._record()))
        object.__setattr__(self, "_code", _functions())

    def _record(self):
        from merlin.sched.codegen import EMITTER_VERSION

        return {
            "schema": "merlin.configuration-c-consumer.index64.v1",
            "index_bits": self.index_bits,
            "emitter_version": EMITTER_VERSION,
            "sources": _sources(),
        }

    def revalidate(self):
        if (
            type(self.index_bits) is not int
            or self.index_bits != 64
            or _canonical(self._record()) != self._original
            or any(current is not original for current, original in zip(_functions(), self._code, strict=True))
        ):
            raise ValueError("original configuration C index domain or emitter sources/code changed")

    def record(self):
        self.revalidate()
        return json.loads(self._original)

    @classmethod
    def reopen(cls, record):
        # Reconstruct a descriptive request, never approval or live-result authority.
        if type(record) is not dict or set(record) != {"schema", "index_bits", "emitter_version", "sources"}:
            raise ValueError("exact original configuration C consumer record required")
        selected = cls(record["index_bits"])
        if selected.record() != record:
            raise ValueError("configuration C consumer request or source binding differs")
        return selected

    def admit(self, kernel):
        from merlin.sched.ir.expr import Expr, render
        from merlin.sched.ir.integer_domain import check_integer_domain
        from merlin.sched.ir.kernel import Call, Loop, Ptr, check_structure

        self.revalidate()
        if type(kernel) is not Kernel or check_structure(kernel):
            raise ValueError("original complete well-formed Kernel required for C index admission")
        check_integer_domain(kernel, self.index_bits)
        pending = list(kernel.body)
        while pending:
            statement = pending.pop()
            if type(statement) is Loop:
                if type(statement.extent) is not int:
                    raise ValueError("exact integer C loop extent required")
                pending.extend(statement.body)
            elif type(statement) is Call:
                for _, value in statement.args:
                    expression = value.offset if type(value) is Ptr else value
                    if isinstance(expression, Expr):
                        # Rendering checks every literal, including dead arms,
                        # comparison constants and scale factors.
                        render(expression, c=True, index_bits=self.index_bits)
            else:
                raise ValueError("unknown configuration C source statement")
        self.revalidate()

    def hoist(self, kernel, adapter, **options):
        from .configuration_loops import hoist_loop_configuration

        if "c_consumer" in options:
            raise ValueError("the original C consumer cannot be substituted")
        return hoist_loop_configuration(kernel, adapter, c_consumer=self, **options)

    def replay(self, report_path, *, lean, destination, timeout=120):
        from merlin.verify.compiler_configuration_hoist import replay_to_c_consumer

        return replay_to_c_consumer(report_path, c_consumer=self, lean=lean, destination=destination, timeout=timeout)

    def _issue(
        self,
        result,
        *,
        source,
        source_text,
        selected_contracts,
        selected_record,
        max_calls,
        max_fields,
        max_depth,
        proof_inputs=(),
        transport_refs=(),
    ):
        from .configuration import ConfigurationResult
        from .selected_contracts import SelectedSchedulingContracts

        if (
            type(self) is not ConfigurationCConsumer
            or type(result) is not ConfigurationResult
            or type(selected_contracts) is not SelectedSchedulingContracts
        ):
            raise ValueError("original typed C-consumer selection, contracts and actual configuration result required")
        selected_contracts.revalidate()
        if _canonical(selected_contracts.record()) != selected_record or source.text() != source_text:
            raise ValueError("original selected contracts or source Kernel changed during hoisting")
        self.admit(source)
        self.admit(result.kernel)
        issued = ConfigurationCResult(result.kernel, result.report)
        _LIVE[issued] = _Issued(
            os.getpid(),
            threading.get_ident(),
            self,
            result,
            source,
            source_text,
            result.kernel,
            result.kernel.text(),
            selected_contracts,
            selected_record,
            (max_calls, max_fields, max_depth),
            self._original,
            tuple(proof_inputs),
            tuple(transport_refs),
        )
        issued._revalidate()
        return issued


@dataclass(frozen=True, eq=False)
class ConfigurationCResult:
    """Live actual producer output with a controlled selected-width C consumer.

    Diagnostic report fields cannot grant mathematical or emission authority.
    Constructed/copied/serialized carriers do not represent the original live
    operation. The original emitter remains separately available to callers.
    """

    kernel: Kernel
    report: dict

    def __reduce_ex__(self, protocol):
        raise TypeError("live selected C-consumer result cannot be serialized")

    def __copy__(self):
        return ConfigurationCResult(self.kernel, self.report)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def _revalidate(self):
        from .configuration_loops import _bounded

        if type(self) is not ConfigurationCResult or self not in _LIVE:
            raise ValueError("constructed or copied C-consumer result has no original live operation")
        entry = _LIVE[self]
        owner = entry.consumer
        if type(owner) is not ConfigurationCConsumer:
            raise ValueError("original typed C-consumer issuer required")
        if (
            (entry.pid, entry.thread) != (os.getpid(), threading.get_ident())
            or self.kernel is not entry.output
            or entry.result.kernel is not entry.output
        ):
            raise ValueError("C-consumer result differs from its original host/thread/Kernel owner")
        _bounded(entry.source, *entry.bounds)
        _bounded(entry.output, *entry.bounds)
        owner.revalidate()
        entry.selected_contracts.revalidate()
        if (
            owner._original != entry.consumer_record
            or _canonical(entry.selected_contracts.record()) != entry.selected_record
            or entry.source.text() != entry.source_text
            or entry.output.text() != entry.output_text
        ):
            raise ValueError("selected C-consumer inputs/source/output changed before emission")
        for row in (*entry.proof_inputs, *entry.transport_refs):
            path = Path(row["path"])
            if path.is_symlink():
                raise ValueError("original C-bound proof/input source may not be a symlink")
            value = path.read_bytes()
            if len(value) != row["bytes"] or hashlib.sha256(value).hexdigest() != row["sha256"]:
                raise ValueError("original C-bound proof/input source changed before emission")
        owner.admit(entry.source)
        owner.admit(entry.output)
        return owner

    @property
    def index_bits(self):
        return self._revalidate().index_bits

    def emit_c(self, instruction_set, *, symbol, spell=None):
        from merlin.sched.codegen import emit_c_function

        owner = self._revalidate()
        text = emit_c_function(self.kernel, instruction_set, symbol=symbol, spell=spell, index_bits=owner.index_bits)
        self._revalidate()
        return text

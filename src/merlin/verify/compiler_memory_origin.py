"""Original live ownership for the explicitly selected scalar memory pipeline.

Records identify an observation; they cannot recreate this process/thread scope
or authorize a skipped Merlin DCE check. Source and runner custody do not
verify upstream MLIR or LLVM passes.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.common.paths import module_source_path

from . import pass_validation

_ACTIVE = ContextVar("merlin_original_memory_compile", default=None)
_LIVE = weakref.WeakKeyDictionary()
_MODULES = (
    "merlin.verify.compiler_memory_origin",
    "merlin.verify.memory_scalar_nan_admission",
    "merlin.verify.lean_graph_input_admission",
    "merlin.verify.gate_errors",
    "merlin.verify.compiler_transforms",
    "merlin.verify.pass_validation",
    "merlin.llvmlower.lower",
    "merlin.llvmlower.pipeline",
    "merlin.llvmlower.toolchain",
    "merlin.common.owned_process",
    "merlin.common.owned_children",
    "merlin.transforms.certificates",
    "merlin.xdsl_dialects._common",
    "merlin.common.dense_float_ir",
    "xdsl.parser",
    "xdsl.parser.core",
    "xdsl.parser.attribute_parser",
    "xdsl.utils.lexer",
    "xdsl.utils.mlir_lexer",
    "xdsl.printer",
    "xdsl.context",
    "xdsl.ir.core",
    "xdsl.dialects.builtin",
    "xdsl.dialects.arith",
    "xdsl.dialects.func",
    "xdsl.dialects.memref",
    "xdsl.dialects.scf",
)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sources():
    rows = []
    for name in _MODULES:
        path = module_source_path(name).resolve(strict=True)
        value = path.read_bytes()
        rows.append((str(path), len(value), hashlib.sha256(value).hexdigest()))
    return tuple(rows)


def _functions(*, canonical=False):
    import types

    from merlin.llvmlower import pipeline
    from merlin.transforms.certificates import parse_ir
    from merlin.xdsl_dialects._common import text

    from .compiler_transforms import apply, memory_lowering_pipeline
    from .lean_graph_input_admission import _attributes, _fields
    from .memory_scalar_nan_admission import (
        _admit_nan_attributes,
        _parse_bounded,
        admit_memory_scalar_attributes,
        parse_memory_module,
    )

    functions = (
        (_canonical_source, "merlin.verify.compiler_memory_origin"),
        (admit_memory_scalar_attributes, "merlin.verify.memory_scalar_nan_admission"),
        (_admit_nan_attributes, "merlin.verify.memory_scalar_nan_admission"),
        (_parse_bounded, "merlin.verify.memory_scalar_nan_admission"),
        (parse_memory_module, "merlin.verify.memory_scalar_nan_admission"),
        (_attributes, "merlin.verify.lean_graph_input_admission"),
        (_fields, "merlin.verify.lean_graph_input_admission"),
        (parse_ir, "merlin.transforms.certificates"),
        (text, "merlin.xdsl_dialects._common"),
        (apply, "merlin.verify.compiler_transforms"),
        (memory_lowering_pipeline, "merlin.verify.compiler_transforms"),
        (pipeline._bind_index_width, "merlin.llvmlower.pipeline"),
        (pass_validation.current, "merlin.verify.pass_validation"),
        (pass_validation.selected, "merlin.verify.pass_validation"),
        (pass_validation.observe, "merlin.verify.pass_validation"),
        (pass_validation.submit, "merlin.verify.pass_validation"),
    )
    if not canonical:
        return tuple((function, function.__code__) for function, _ in functions)
    for function, module in functions:
        path = module_source_path(module).resolve(strict=True)
        original = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        codes = [
            value
            for value in original.co_consts
            if isinstance(value, types.CodeType) and value.co_name == function.__name__
        ]
        if sum(function.__code__ == code for code in codes) != 1:
            raise ValueError("original memory compiler function differs from its canonical source")
    return tuple((function, function.__code__) for function, _ in functions)


def _canonical_source(raw):
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.xdsl_dialects._common import text

    from .memory_scalar_nan_admission import admit_memory_scalar_attributes, parse_memory_module

    if type(raw) is not str:
        raise ValueError("actual memory source text required")
    try:
        module = parse_memory_module(raw)
        admit_memory_scalar_attributes(module)
        # SSA hints and dictionary insertion order are printer choices. Preserve
        # every property/attribute value, symbol, type, binding and block.
        for operation in module.walk():
            operation.properties = dict(sorted(operation.properties.items()))
            operation.attributes = dict(sorted(operation.attributes.items()))
            for result in operation.results:
                result.name_hint = None
            for region in operation.regions:
                for block in region.blocks:
                    for argument in block.args:
                        argument.name_hint = None
        return text(module, generic=True)
    except (ParseError, VerifyException, RecursionError) as error:
        raise ValueError("original memory whole-module parse/render refused") from error


@dataclass(frozen=True)
class _Issued:
    pid: int
    thread: int
    validation: object
    gate: object
    tools: object
    passes: frozenset
    index_bits: int
    work: Path
    source: str
    canonical_source: str
    nonce: str
    pipeline: str
    derived64: str
    sources: tuple
    functions: tuple
    runner: str | None = None
    effective: str | None = None
    upstream: str | None = None
    input_path: Path | None = None


@dataclass(frozen=True, eq=False)
class MemoryScope:
    work: Path
    validation: pass_validation.PassValidation
    source_sha256: str
    pipeline: str
    nonce: str

    def __reduce_ex__(self, protocol):
        raise TypeError("original memory scope cannot be serialized")

    def __copy__(self):
        return MemoryScope(self.work, self.validation, self.source_sha256, self.pipeline, self.nonce)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def validate(self):
        if type(self) is not MemoryScope or self not in _LIVE or _ACTIVE.get() is not self:
            raise ValueError("memory scope is not the original active live owner")
        original = _LIVE[self]
        if (
            (original.pid, original.thread) != (os.getpid(), threading.get_ident())
            or pass_validation.current() is not original.validation
            or self.validation is not original.validation
            or original.validation.gate is not original.gate
            or original.validation.tools is not original.tools
            or original.validation.passes != original.passes
            or type(original.validation.index_bits) is not int
            or original.validation.index_bits != original.index_bits
            or not original.gate.guarding
            or self.work != original.work
            or self.nonce != original.nonce
            or self.pipeline != original.pipeline
            or self.source_sha256 != _sha(original.source)
        ):
            raise ValueError("original memory owner/validation/thread/work/source changed")
        if _sources() != original.sources or any(
            current_function is not selected_function or current_code is not selected_code
            for (current_function, current_code), (selected_function, selected_code) in zip(
                _functions(), original.functions, strict=True
            )
        ):
            raise ValueError("original memory pipeline/compiler sources or function bindings changed")
        from merlin.llvmlower.pipeline import _bind_index_width

        from .compiler_transforms import memory_lowering_pipeline

        if (
            memory_lowering_pipeline() != original.pipeline
            or _bind_index_width(original.pipeline, 64) != original.derived64
        ):
            raise ValueError("canonical original memory pipeline changed")
        for selected in original.passes:
            pass_validation.require_canonical_handler(original.validation, pass_validation.CHECKS[selected])
        return self

    def _work(self, work):
        self.validate()
        if Path(work).resolve(strict=True) != self.work:
            raise ValueError("memory operation belongs to a different original work directory")
        return _LIVE[self]

    def validate_dce_before(self, raw):
        self.validate()
        if _canonical_source(raw) != _LIVE[self].canonical_source:
            raise ValueError("DCE input differs from the original selected memory source epoch")
        return self

    def bind_upstream(self, raw):
        """Freeze the original compiler's actual post-preprocess/CIF input."""
        self.validate()
        original = _LIVE[self]
        if original.upstream is not None or original.runner is not None:
            raise ValueError("original memory upstream input was already frozen")
        if type(raw) is not str or not raw:
            raise ValueError("actual post-preprocessing upstream memory input required")
        _canonical_source(raw)  # Preserve admitted source attributes before native lowering.
        _LIVE[self] = replace(original, upstream=raw)
        return self

    def bind_input(self, upstream, work):
        original = self._work(work)
        if original.input_path is not None or original.runner is not None:
            raise ValueError("original memory input file was already bound")
        if type(upstream) is not tuple or len(upstream) != 2 or type(upstream[1]) is not str:
            raise ValueError("actual original pipeline input path/text pair required")
        path, raw = Path(upstream[0]), upstream[1]
        if (
            original.upstream is None
            or raw != original.upstream
            or path.is_symlink()
            or path.resolve(strict=True) != self.work / "model.mlir"
            or path.read_bytes() != original.upstream.encode("utf-8")
        ):
            raise ValueError("actual pipeline input differs from original frozen upstream epoch")
        _LIVE[self] = replace(original, input_path=path.resolve(strict=True))
        return self

    def upstream_text(self, work):
        original = self._work(work)
        if original.upstream is None or original.input_path is None:
            raise ValueError("original frozen upstream input is not bound")
        if original.input_path.is_symlink() or original.input_path.read_bytes() != original.upstream.encode("utf-8"):
            raise ValueError("actual original memory input file changed after freezing")
        return original.upstream

    def bind_runner(self, source, work, *, pipeline):
        original = self._work(work)
        if original.runner is not None:
            raise ValueError("original memory scope already issued its runner")
        if type(pipeline) is not str or pipeline not in (original.pipeline, original.derived64):
            raise ValueError("original memory runner needs its exact selected canonical pipeline")
        self.upstream_text(work)
        if type(source) is not str or not source:
            raise ValueError("actual original runner text required")
        _LIVE[self] = replace(original, runner=source, effective=pipeline)
        return source

    def validate_runner(self, work):
        original = self._work(work)
        self.upstream_text(work)
        runner = self.work / "run_lowering.py"
        if original.runner is None or runner.is_symlink() or runner.read_bytes() != original.runner.encode("utf-8"):
            raise ValueError("actual runner differs from the original frozen memory runner")
        return _sha(original.runner)


def current():
    selected = _ACTIVE.get()
    return selected.validate() if selected is not None else None


@contextmanager
def compiler_scope(work, *, original_source, memory_pipeline, validation):
    if _ACTIVE.get() is not None:
        raise ValueError("original memory compiler scope cannot be nested or substituted")
    if type(validation) is not pass_validation.PassValidation or pass_validation.current() is not validation:
        raise ValueError("original active typed PassValidation required for memory scope")
    if not validation.gate.guarding or type(validation.index_bits) is not int or validation.index_bits != 64:
        raise ValueError("original live 64-bit compiler validation scope required")
    if type(original_source) is not str or not original_source:
        raise ValueError("actual original selected memory source text required")
    from merlin.llvmlower.pipeline import _bind_index_width

    from .compiler_transforms import memory_lowering_pipeline

    if type(memory_pipeline) is not str or memory_pipeline != memory_lowering_pipeline():
        raise ValueError("explicit original canonical memory pipeline required")
    directory = Path(work).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("original compiler work directory required")
    nonce = secrets.token_hex(32)
    sources, functions = _sources(), _functions(canonical=True)

    derived64 = _bind_index_width(memory_pipeline, 64)
    owner = MemoryScope(directory, validation, _sha(original_source), memory_pipeline, nonce)
    _LIVE[owner] = _Issued(
        os.getpid(),
        threading.get_ident(),
        validation,
        validation.gate,
        validation.tools,
        validation.passes,
        validation.index_bits,
        directory,
        original_source,
        _canonical_source(original_source),
        nonce,
        memory_pipeline,
        derived64,
        sources,
        functions,
    )
    token = _ACTIVE.set(owner)
    try:
        owner.validate()
        yield owner
        owner.validate()
        # A successful consumption is not a lease to mutate its compiler input
        # or runner before products leave the original live scope.
        if _LIVE[owner].runner is not None:
            owner.validate_runner(directory)
        elif _LIVE[owner].input_path is not None:
            owner.upstream_text(directory)
    finally:
        _ACTIVE.reset(token)
        _LIVE.pop(owner, None)

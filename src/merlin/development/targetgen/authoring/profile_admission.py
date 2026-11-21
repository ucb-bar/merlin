"""Original frozen-profile admission of exact compiler API imports.

Filesystem isolation remains mandatory. This scope authorizes direct imports of
the admitted source leaves; it does not expose automatic checker dependencies or
replace the separate closed compiler-library contract. Saved profile dictionaries
and candidate callbacks cannot create this host scope.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import weakref
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import FunctionType, MethodType

import yaml

from merlin.common.access import module_name_for
from merlin.development.targetgen.authoring.tool_profile import verify_tool_profile
from merlin.verification.lean.contract import document_digest

# ---- compiler_profile_admission ----------------------------------------------------------------

_ACTIVE = contextvars.ContextVar("original_compiler_api_admission", default=None)
_ISSUED = weakref.WeakSet()
_OPTIONAL = frozenset({"isa_dialect_generation", "lean_proof_authoring"})


class _Admission:
    def __init__(self, *, workspace, bundle_manifest, repo):
        from merlin.common.storage import bundle_snapshots as bwrap

        self.workspace, self.bundle_manifest, self.repo = map(Path, (workspace, bundle_manifest, repo))
        if (
            self.bundle_manifest.is_symlink()
            or self.bundle_manifest.resolve() != self.bundle_manifest
            or not self.bundle_manifest.is_file()
            or self.bundle_manifest.is_relative_to(self.workspace)
        ):
            raise ValueError("compiler imports require the original archived ordinary manifest")
        self.bundle_bytes = self.bundle_manifest.read_bytes()
        self.bundle = yaml.safe_load(self.bundle_bytes)
        self.profile = verify_tool_profile(self.bundle)
        if self.profile is None:
            raise ValueError("compiler imports require the original versioned source profile")
        self.bundle_identity = document_digest(self.bundle)
        self.snapshot_sha256 = bwrap.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)[
            "content_sha256"
        ]
        self.snapshot_marker_sha256 = bwrap.snapshot_marker_digest(self.workspace)
        self.modules = frozenset((*self.profile["common_public_modules"], *_optional_modules(self)))

    def verify(self):
        from merlin.common.storage import bundle_snapshots as bwrap

        if self not in _ISSUED:
            raise ValueError("compiler imports require the original host-issued admission")
        if (
            self.bundle_manifest.read_bytes() != self.bundle_bytes
            or document_digest(self.bundle) != self.bundle_identity
            or verify_tool_profile(self.bundle) != self.profile
        ):
            raise ValueError("compiler import admission changed its original profile")
        marker = bwrap.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        if (
            marker["content_sha256"] != self.snapshot_sha256
            or bwrap.snapshot_marker_digest(self.workspace) != self.snapshot_marker_sha256
        ):
            raise ValueError("compiler import admission changed its original source snapshot")
        if self.modules != frozenset((*self.profile["common_public_modules"], *_optional_modules(self))):
            raise ValueError("compiler import admission changed its exact leaf membership")

    def permits(self, module: str) -> bool:
        return self in _ISSUED and module in self.modules


def _optional_modules(owner):
    from merlin.common.storage import bundle_snapshots as bwrap
    from merlin.development.targetgen.authoring import tool_registry

    selected = []
    for name in set(owner.profile["treatment_tools"]) & _OPTIONAL:
        for logical in tool_registry.spec(name).bundle_paths:
            module = module_name_for(logical)
            if module is None or not logical.endswith(".py") or logical.endswith("/__init__.py"):
                continue
            tail = logical.removeprefix("merlin/python/")
            matching = [
                row["path"]
                for row in owner.bundle["allowed"]
                if row["path"] == logical or row["path"].endswith("/" + tail)
            ]
            if len(matching) != 1:
                raise ValueError("optional compiler API has no exact original source grant")
            original = Path(matching[0])
            if not original.is_absolute():
                original = owner.repo / original
            selected.append((module, original))
    if not selected:
        return
    sources = bwrap.snapshot_input_paths(
        owner.workspace,
        owner.bundle,
        [original for _, original in selected],
        repo=owner.repo,
    )
    for (module, _), source in zip(selected, sources, strict=True):
        if source.is_symlink() or not source.is_file():
            raise ValueError("optional compiler API is not an original frozen source leaf")
        yield module


@contextlib.contextmanager
def original_compiler_imports(*, workspace: Path, bundle_manifest: Path, repo: Path):
    """Enter from the original common service, with its frozen grants unchanged."""
    admission = _Admission(workspace=workspace, bundle_manifest=bundle_manifest, repo=repo)
    _ISSUED.add(admission)
    token = _ACTIVE.set(admission)
    try:
        admission.verify()
        yield admission
        admission.verify()
    finally:
        _ACTIVE.reset(token)
        _ISSUED.discard(admission)


def _issued_compiler_imports():
    """Inspect issuer identity only; callers still own fresh input verification."""
    admission = _ACTIVE.get()
    if admission is not None:
        if type(admission) is not _Admission:
            raise ValueError("compiler import scope has no original issuer")
        if admission not in _ISSUED:
            raise ValueError("compiler imports require the original host-issued admission")
    return admission


def active_compiler_imports():
    admission = _issued_compiler_imports()
    if admission is not None:
        admission.verify()
    return admission


# ---- compiler_source_optimizations -------------------------------------------------------------
# Original pass-owner bridge to ordinary source-materialization choices.
#
# This selected-only adapter transports an existing live qualification owner. It
# cannot issue that qualification, restore it from data, or establish that LLVM
# implements a source rewrite. Missing production context must refuse. Frozen SDK
# inventories and the ordinary final functional guards are unchanged.

MECHANISM = "source_optimizations_v1"
PROTOCOL = "merlin.compiler_source_optimizations.v1"


def _data(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _pin(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("original source-pass input is absent or indirect")
    from merlin.common.digest import sha256_file

    return {"path": str(path), "sha256": sha256_file(path)}


def _callback(value):
    if type(value) is MethodType:
        owner, function = value.__self__, value.__func__
    elif type(value) is FunctionType:
        owner, function = None, value
    else:
        raise ValueError("original source-pass callbacks require source-bound Python functions")
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError("original source-pass callback has no implementation source")
    return (
        id(owner),
        id(function),
        function.__code__,
        _state(function.__defaults__),
        _state(function.__kwdefaults__),
        _pin(source),
    )


def _state(value):
    if value is None or type(value) in (str, bool, int, float):
        return (type(value).__name__, _data(value))
    if isinstance(value, Path):
        return ("path", str(value.absolute()))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_state(v) for v in value))
    if type(value) is dict:
        return ("dict", tuple((k, _state(v)) for k, v in sorted(value.items())))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), id(value), tuple((f.name, _state(getattr(value, f.name))) for f in fields(value)))
    if type(value) in (FunctionType, MethodType):
        return ("callback", _callback(value))
    # Existing opaque admission owners remain owned by their original consumers.
    # Identity here is not a deep state or qualification check of such owners.
    return (type(value), id(value))


def _inputs(value):
    remaining = 8192

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ValueError("original source-pass context exceeds its construction bound")
        if isinstance(item, Path):
            return _pin(item)
        if type(item) is dict:
            if len(item) > 4096 or any(type(key) is not str for key in item):
                raise ValueError("original source-pass context has unbounded or untyped fields")
            return {key: visit(child, depth + 1) for key, child in item.items()}
        if type(item) in (tuple, list):
            if len(item) > 4096:
                raise ValueError("original source-pass context exceeds its construction bound")
            return [visit(child, depth + 1) for child in item]
        if item is None or type(item) in (str, bool, int, float):
            if type(item) is str and len(item.encode()) > 256 << 10:
                raise ValueError("original source-pass context exceeds its construction bound")
            return item
        raise ValueError("original source-pass context requires bounded data and explicit file paths")

    return visit(value, 0)


def _input_files(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from _input_files(child)
    elif type(value) is list:
        for child in value:
            yield from _input_files(child)


def _policy_methods(policy, cls):
    if type(policy) is not cls or set(vars(policy)) != {f.name for f in fields(cls)}:
        raise ValueError("original source-pass policy has foreign or shadowed fields")
    methods = tuple(getattr(policy, name) for name in ("__post_init__", "record", "bind", "verify_selection"))
    if any(
        type(method) is not MethodType or method.__self__ is not policy or method.__func__ is not getattr(cls, name)
        for name, method in zip(("__post_init__", "record", "bind", "verify_selection"), methods, strict=True)
    ):
        raise ValueError("original source-pass policy method was replaced")
    return methods


def _functions(policy, build):
    from merlin.common import digest
    from merlin.compiler.dialects import xdsl_context as _common
    from merlin.compiler.frontend import linalg_mlir
    from merlin.compiler.ir import convolution as convolution_ir
    from merlin.compiler.ir import mlir_query
    from merlin.compiler.ir import xdsl_io as dense_float_ir
    from merlin.compiler.lower.host import (
        automatic_destinations,
        automatic_gathers,
        im2col_padding,
        im2col_padding_check,
        insert_slice_destination,
    )
    from merlin.compiler.lower.host import (
        checks as destination_check,
    )
    from merlin.compiler.lower.model import compilation_recipe
    from merlin.compiler.lower.options import im2col as im2col_pack
    from merlin.compiler.lower.options import optional_passes
    from merlin.compiler.lower.quant import readout as quant_round_check
    from merlin.compiler.lower.source import (
        reviewed as reviewed_pass_capabilities,
    )
    from merlin.compiler.lower.source import (
        source_optimization_request,
        source_optimization_selection,
    )
    from merlin.compiler.transforms import certificates, pipeline_state
    from merlin.development.targetgen.authoring import pass_selection as compiler_pass_selection
    from merlin.runtime.backends.baremetal import host_provider as source_optimization_build
    from merlin.verification.references import convolution_windows as convolution_verifier

    direct = []
    for module in (
        automatic_destinations,
        automatic_gathers,
        compilation_recipe,
        destination_check,
        im2col_pack,
        convolution_ir,
        convolution_verifier,
        im2col_padding,
        im2col_padding_check,
        insert_slice_destination,
        optional_passes,
        quant_round_check,
        source_optimization_request,
        source_optimization_selection,
        dense_float_ir,
        digest,
        mlir_query,
        linalg_mlir,
        certificates,
        pipeline_state,
        _common,
        compiler_pass_selection,
        reviewed_pass_capabilities,
        source_optimization_build,
    ):
        for value in vars(module).values():
            if type(value) is FunctionType:
                direct.append(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                for method in vars(value).values():
                    function = method.__func__ if type(method) in (staticmethod, classmethod) else method
                    if type(function) is FunctionType and not function.__code__.co_filename.startswith("<"):
                        direct.append(function)
                    elif type(method) is property:
                        direct.extend(
                            function for function in (method.fget, method.fset) if type(function) is FunctionType
                        )
    direct.extend(value for value in vars(build).values() if type(value) is FunctionType)
    return (
        _data,
        _pin,
        _callback,
        _state,
        _inputs,
        _input_files,
        _policy_methods,
        _functions,
        implementation_sources,
        compile_with_source_optimizations,
        policy.__post_init__,
        policy.record,
        policy.bind,
        policy.verify_selection,
        build.__init__,
        build.revalidate,
        build.accept,
        build.record,
        *direct,
    )


def implementation_sources():
    """Direct host source custody only; no transitive import or API grant."""
    from merlin.compiler.lower.host import automatic_destinations, automatic_gathers
    from merlin.compiler.lower.source.source_optimization_policy import SourceOptimizationPolicy
    from merlin.runtime.backends.baremetal.host_provider import SourceOptimizationBuild

    functions = _functions(SourceOptimizationPolicy, SourceOptimizationBuild)
    return tuple(
        dict.fromkeys(
            (
                Path(__file__).absolute(),
                *(Path(inspect.getsourcefile(function)).absolute() for function in functions),
                *automatic_gathers.implementation_paths(),
                *automatic_destinations.implementation_paths(),
                Path(inspect.getsourcefile(SourceOptimizationBuild)),
                Path(inspect.getsourcefile(SourceOptimizationPolicy)),
            )
        )
    )


def compile_with_source_optimizations(compiler, *, selection, compiler_package, original_inputs, **options):
    """Call the unchanged ordinary compiler with one original typed policy.

    Only ``source_optimizations_v1`` opts in. Its original owner must declare
    fixed ``source_optimization_permissions`` and ``source_optimization_ceiling``
    and implement bound ``source_optimization_binding(*, inputs)`` plus
    ``source_optimization_policy(package, *, inputs)`` methods. The former must
    independently rejoin actual hardware, software, numerics, runtime and library;
    it receives no selected binding to echo. The callback's name proves nothing.
    The original factory and final functional guard remain mandatory externally.

    ``original_inputs`` is an original caller's function supplying current data
    and explicit Path inputs. It must support repeated current-context reads.
    It is never read from a candidate or receipt.
    Unsupported/missing production context raises; it never silently enables a
    pass. No selected mechanism preserves the exact old compiler invocation.
    """
    if selection is None or MECHANISM not in selection.mechanisms:
        return compiler(**options)

    from merlin.compiler.lower.host import automatic_destinations, automatic_gathers
    from merlin.compiler.lower.source.reviewed import PassCapabilityBinding, ReviewedPassCapability
    from merlin.compiler.lower.source.source_optimization_policy import SourceOptimizationPolicy
    from merlin.compiler.lower.source.source_optimization_request import _PASSES, _budget
    from merlin.development.targetgen.authoring.pass_selection import CompilerPassSelection, _tree
    from merlin.runtime.backends.baremetal.host_provider import SourceOptimizationBuild

    if type(selection) is not CompilerPassSelection or "source_optimizations" in options:
        raise ValueError("source-pass build requires one explicit original selection and no override")
    if selection.mechanisms != (MECHANISM,):
        raise ValueError("source-pass build requires its single original composite mechanism")
    # Snapshot the actual original owner before registry revalidation can call it.
    members = [
        c
        for c in selection.registry._capabilities
        if type(c) is ReviewedPassCapability and c._mechanism == MECHANISM and c._binding == selection.binding
    ]
    if len(members) != 1:
        raise ValueError("source-pass build has no exact original qualified capability")
    owner = members[0]._owner
    if getattr(owner, "source_optimization_protocol", None) != PROTOCOL:
        raise ValueError("selected source-pass owner has no supported original policy protocol")
    permissions = getattr(owner, "source_optimization_permissions", None)
    ceiling = getattr(owner, "source_optimization_ceiling", None)
    if (
        type(permissions) is not tuple
        or len(set(permissions)) != len(permissions)
        or any(type(p) is not str or p not in _PASSES for p in permissions)
    ):
        raise ValueError("selected source-pass owner lacks its original closed permissions")
    _budget(ceiling)
    callbacks = tuple(
        getattr(owner, name, None)
        for name in (
            "source_optimization_binding",
            "source_optimization_policy",
            "check_compiler_binding",
            "revalidate",
        )
    )
    if any(type(c) is not MethodType or c.__self__ is not owner for c in callbacks):
        raise ValueError("selected source-pass owner lacks original binding/policy verification context")
    callback_states = tuple(_callback(c) for c in callbacks)
    compiler_state, reader_state = _callback(compiler), _callback(original_inputs)
    function_reader, function_reader_code = _functions, _functions.__code__
    functions = function_reader(SourceOptimizationPolicy, SourceOptimizationBuild)
    function_states = tuple(_callback(f) for f in functions)
    selection_callbacks = (selection.revalidate, selection.resolve)
    selection_states = tuple(_callback(f) for f in selection_callbacks)
    tree_state = _callback(_tree)
    options_state = _state(options)
    package = Path(compiler_package).absolute()
    package_state = _tree(package)
    library = tuple(
        dict.fromkeys(
            (
                Path(__file__).absolute(),
                *(Path(s[-1]["path"]) for s in (*callback_states, compiler_state, reader_state)),
                *automatic_gathers.implementation_paths(),
                *automatic_destinations.implementation_paths(),
            )
        )
    )
    pins = tuple(_pin(p) for p in library)
    implementation_pins = tuple(_pin(path) for path in implementation_sources())
    roots = tuple(Path(options[k]).absolute() for k in ("capture", "package") if k in options)
    root_states = tuple(_tree(p) for p in roots)
    build_inputs = tuple(_pin(options[k]) for k in ("board_catalog", "dts", "rtl_facts") if options.get(k) is not None)
    original_binding = _data(vars(selection.binding))
    input_pins = ()
    input_state = None
    policy = policy_state = None

    def static():
        if _functions is not function_reader or function_reader.__code__ is not function_reader_code:
            raise ValueError("original source-pass implementation reader changed")
        current = function_reader(SourceOptimizationPolicy, SourceOptimizationBuild)
        if (
            len(current) != len(functions)
            or any(a is not b for a, b in zip(current, functions, strict=True))
            or tuple(_callback(f) for f in functions) != function_states
            or _callback(compiler) != compiler_state
            or _callback(original_inputs) != reader_state
            or _callback(_tree) != tree_state
            or tuple(_callback(f) for f in (selection.revalidate, selection.resolve)) != selection_states
            or _state(options) != options_state
            or _tree(package) != package_state
            or tuple(_tree(p) for p in roots) != root_states
            or tuple(_pin(p["path"]) for p in build_inputs) != build_inputs
            or tuple(_pin(p["path"]) for p in pins) != pins
            or tuple(_pin(p["path"]) for p in implementation_pins) != implementation_pins
            or tuple(_pin(p["path"]) for p in input_pins) != input_pins
            or _data(vars(selection.binding)) != original_binding
        ):
            raise ValueError("original source-pass code, compiler, source or current inputs changed")
        for name, callback, before in zip(
            ("source_optimization_binding", "source_optimization_policy", "check_compiler_binding", "revalidate"),
            callbacks,
            callback_states,
            strict=True,
        ):
            if _callback(getattr(owner, name, None)) != before or _callback(callback) != before:
                raise ValueError("original source-pass owner callback changed")
        if (
            getattr(owner, "source_optimization_protocol", None) != PROTOCOL
            or _state(getattr(owner, "source_optimization_permissions", None)) != _state(permissions)
            or _state(getattr(owner, "source_optimization_ceiling", None)) != _state(ceiling)
        ):
            raise ValueError("original source-pass permission or ceiling changed")
        if policy is not None:
            _policy_methods(policy, SourceOptimizationPolicy)
            if _state(policy) != policy_state:
                raise ValueError("original source-pass selected policy changed")

    def context_current():
        if input_state is not None:
            current_inputs = _inputs(original_inputs())
            static()
            if _data(current_inputs) != input_state:
                raise ValueError("original source-pass current context changed during compilation")

    def checked(callback, *args, **kwargs):
        static()
        context_current()
        result = callback(*args, **kwargs)
        static()
        context_current()
        return result

    static()
    input_data = _inputs(checked(original_inputs))
    if len(_data(input_data).encode()) > 256 << 10:
        raise ValueError("original source-pass context exceeds its construction bound")
    input_state = _data(input_data)
    input_pins = tuple(_input_files(input_data))
    static()
    checked(selection_callbacks[0])
    if checked(selection_callbacks[1], MECHANISM) is not members[0]:
        raise ValueError("original source-pass capability was replaced")
    checked(callbacks[2], package)

    def binding():
        values = json.loads(input_state)
        current = checked(callbacks[0], inputs=values)
        if _data(values) != input_state or type(current) is not PassCapabilityBinding or current != selection.binding:
            raise ValueError("source-pass current hardware/software/numerics/runtime/library binding is stale")
        return current

    binding()
    values = json.loads(input_state)
    policy = checked(callbacks[1], package, inputs=values)
    if _data(values) != input_state or type(policy) is not SourceOptimizationPolicy:
        raise ValueError("original source-pass owner returned a foreign policy or changed input context")
    policy_methods = _policy_methods(policy, SourceOptimizationPolicy)
    policy_state = _state(policy)
    checked(policy_methods[0])
    if policy.permitted_passes != permissions or policy.original_maximum_added_operations != ceiling:
        raise ValueError("source-pass policy expanded original permissions or ceiling")
    original_policy = _data(checked(policy_methods[1]))
    consumer = SourceOptimizationBuild(policy)
    static()
    context_current()
    result = compiler(**options, source_optimizations=policy)
    static()
    # Retain actual returned products BEFORE calling a qualification owner.
    result_state = _data(result)
    try:
        record = result["output"]["source_optimizations"]
        executable = result["output"]["elf"]
        returned_policy = result["inputs"]["source_optimizations"]
    except (KeyError, TypeError) as error:
        raise ValueError("ordinary compiler ignored its selected original source policy") from error
    product_pins = tuple(_pin(record[k]["path"]) for k in ("input", "report", "recipe", "executable"))
    if _data(returned_policy) != original_policy:
        raise ValueError("ordinary compiler changed the selected original source policy")
    consumer.accept(record, build_dir=Path(options["output"]) / "build", executable=Path(executable))
    binding()
    checked(selection_callbacks[0])
    consumer.revalidate()
    static()
    if (
        _data(result) != result_state
        or _data(policy_methods[1]()) != original_policy
        or tuple(_pin(p["path"]) for p in product_pins) != product_pins
    ):
        raise ValueError("original source-pass returned products changed during final verification")
    return result

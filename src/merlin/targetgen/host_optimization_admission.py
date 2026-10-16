"""Original reviewed host permissions and source-bound command-buffer choices.

The request is candidate data. Permissions come only from the original software
file already sealed in the common profile. This is not a numerical, physical,
Lean or deployment qualification. Unselected builds retain their existing path.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
from pathlib import Path

import yaml

from merlin.llvmlower.host_optimization_request import (
    HostOptimizationContext,
    HostOptimizationSelection,
    request_from_record,
)

POLICY_SCHEMA = "merlin.host_optimization_admission.v1"
_ACTIVE = contextvars.ContextVar("original_host_optimization_selection", default=None)
_PREMISES = (
    "independent_element_order_unobserved",
    "floating_environment_unobserved_nontrapping",
    "sequential_execution",
    "successful_allocations",
    "non_reentrant_entry",
)


def active_selection():
    return _ACTIVE.get()


def _original_entry(admission, target, build_service):
    if build_service is not None:
        from .contract.build_service import BuildOnlyService

        if type(build_service) is not BuildOnlyService:
            raise ValueError("host choices require the original typed build service")
        build_service.verify(target)
        return build_service.recipe.require_kernel_stack_frame().entry_symbol, None
    from merlin.common.bundle_snapshot import snapshot_input_paths

    from .contract.harness_abi import from_contract
    from .target_registry import resolve

    # Ordinary operator oracles use the selected capability contract rather
    # than a BuildOnlyService. Require that exact original input, with no new
    # grant, backend callback or mutable provider fallback.
    selected = resolve(target)
    path = selected.capability_contract_path.absolute()
    [frozen] = snapshot_input_paths(admission.workspace, admission.bundle, [path], repo=admission.repo)
    payload = frozen.read_bytes()
    contract = yaml.safe_load(payload)
    if type(contract) is not dict or contract.get("name") != target or selected.load_contract() != contract:
        raise ValueError("host choices changed the original selected capability contract")
    return from_contract(contract, target=target).entry_symbol, (str(path), hashlib.sha256(payload).hexdigest())


def _original_context(admission, entry):
    from merlin.common import bundle_snapshot as bwrap

    from .tool_profile import COMMON_SUBSTRATE_V3, COMMON_SUBSTRATE_V4

    admission.verify()
    if admission.profile["common_substrate"] not in {COMMON_SUBSTRATE_V3, COMMON_SUBSTRATE_V4}:
        raise ValueError("host choices require the original substrate V3 profile")
    row = admission.profile["semantic_inputs"]["software_spec"]
    if row is None:
        raise ValueError("host choices have no original reviewed software permissions")
    [source] = bwrap.snapshot_input_paths(
        admission.workspace, admission.bundle, [Path(row["path"])], repo=admission.repo
    )
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("host permissions differ from the original software bytes")
    software = yaml.safe_load(payload)
    if software.get("schema") != "merlin.software_spec.v1" or software.get("status") != "reviewed":
        raise ValueError("host permissions require the original reviewed software semantics")
    if software.get("target") != admission.profile["semantic_inputs"]["target"]:
        raise ValueError("host permissions changed the original target selection")
    policy = software.get("host_optimization")
    required = {"schema", "permitted_stages", "permitted_optimizations"}
    if (
        type(policy) is not dict
        or not required <= set(policy)
        or set(policy) - required - {"data_layout", *_PREMISES}
        or policy["schema"] != POLICY_SCHEMA
    ):
        raise ValueError("host choices lack the original supported host permission schema")
    for key in ("permitted_stages", "permitted_optimizations"):
        if type(policy[key]) is not list:
            raise ValueError("original host permissions require explicit arrays")
    return HostOptimizationContext(
        entry=entry,
        semantic_sha256=row["sha256"],
        permitted_stages=tuple(policy["permitted_stages"]),
        permitted_optimizations=tuple(policy["permitted_optimizations"]),
        index_bits=admission.profile["host_index_bits"],
        data_layout=policy.get("data_layout"),
        **{name: policy.get(name, False) for name in _PREMISES},
    )


@contextlib.contextmanager
def selected_buffer_choices(buffer, source, *, target, build_service=None):
    """Bind a request to one actual artifact under a live original admission."""
    record = buffer.get("host_optimization_request")
    if "host_optimization_request" not in buffer:
        yield None
        return
    from .compiler_profile_admission import active_compiler_imports

    if _ACTIVE.get() is not None:
        raise ValueError("host choices cannot replace an active original selection")
    request_bytes = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    if len(request_bytes) > 256 * 1024:
        raise ValueError("host choice request exceeds the original transport budget")
    request = request_from_record(record)
    if hashlib.sha256(source.encode()).hexdigest() != request.source_sha256:
        raise ValueError("host choice request does not bind the actual terminal artifact")
    admission = active_compiler_imports()
    if admission is None or admission.profile["semantic_inputs"]["target"] != target:
        raise ValueError("host choices require the matching original compiler admission")
    entry, original_entry = _original_entry(admission, target, build_service)
    selection = HostOptimizationSelection(_original_context(admission, entry), request)
    # Recheck compatibility before any legacy build cache could skip the producer.
    from merlin.llvmlower.host_optimization_request import prepare_host_optimization

    prepare_host_optimization(source, selection, finished_llvm_only=build_service is not None)
    token = _ACTIVE.set(selection)
    try:
        yield selection
        admission.verify()
        current = json.dumps(buffer.get("host_optimization_request"), sort_keys=True, allow_nan=False).encode()
        if current != request_bytes:
            raise ValueError("host choice request changed during its actual build")
        if _original_context(admission, entry) != selection.context:
            raise ValueError("original host permissions changed during compilation")
        if _original_entry(admission, target, build_service) != (entry, original_entry):
            raise ValueError("host choices changed their original selected entry during compilation")
    finally:
        _ACTIVE.reset(token)


def host_choices_build(function):
    """Keep normal numerical/link builds on the same original scoped selection."""
    @functools.wraps(function)
    def build(buffer, source, *args, **kwargs):
        with selected_buffer_choices(
            buffer, source, target=kwargs["target"], build_service=kwargs.get("_build_service")
        ):
            return function(buffer, source, *args, **kwargs)

    return build

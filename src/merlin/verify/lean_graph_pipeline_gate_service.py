"""Trusted, process-local graph service selection and delivery authority.

Binding inspects SOURCE identities only; it does not construct a Context or
authorize native checking. Saved profile/delivery JSON never recreates authority.
Public native dispatch remains independently held in the gate backend.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import weakref
import time

from . import lean_graph_pipeline_gate as gate
from . import lean_graph_pipeline_gate_provider as provider
from .lean_graph_pipeline_gate_projection import DOMAIN, KINDS, PREMISES, RELATION, source_closure, source_ref

_BINDINGS = weakref.WeakKeyDictionary()
_DELIVERIES = OrderedDict()


@dataclass(frozen=True, eq=False)
class GraphServiceBinding:
    compiler_profile_sha256: str
    context_id: str
    workspace: str
    timeout_s: int
    sources: tuple
    context: object = field(repr=False)

    def describe(self):
        return {"schema": "merlin.graph_service_binding.v1", "compiler_profile_sha256": self.compiler_profile_sha256,
                "context_id": self.context_id, "workspace": self.workspace, "timeout_s": self.timeout_s,
                "domain": DOMAIN, "relation": RELATION, "premises": list(PREMISES), "kinds": sorted(KINDS),
                "sources": [{"path": r.path, "sha256": r.sha256, "size": r.size} for r in self.sources],
                "native_authorization": False, "saved_profile_is_authority": False}


def bind_service_context(context, *, compiler_profile_sha256, timeout_s):
    """Trusted host injection, not a request option or permission to launch.

    The context must already exist from a separately authorized construction.
    Load actual service dependencies before binding their Python source closure.
    No old toolchain selection, runtime guard, context construction or fallback.
    """
    from merlin.targetgen import common_transforms
    from merlin.transforms import pipeline
    from merlin.perf import pass_observation
    from merlin.verify import lean_checker

    # Imports intentionally bind these real entry points in source_closure.
    _ = common_transforms, pipeline, pass_observation, lean_checker
    if not gate._digest(compiler_profile_sha256) or type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
        raise ValueError("explicit compiler profile identity and bounded graph host budget required")
    provider.bound_context(context)
    if context.held is not False:
        raise ValueError("held graph context cannot be bound to a compiler service")
    sources = source_closure()
    projects = {"Merlin." + Path(ref.path).stem: (ref.path, ref.sha256, ref.size)
                for ref in sources if ref.path.endswith(".lean")}
    if set(projects) != provider._PROJECT_MODULES or projects != {
            name: (ref.path, ref.sha256, ref.bytes) for name, ref in context.originals.items()}:
        raise ValueError("service graph profile requires the exact minimal seven-module source map")
    if context.compile_audit_roots != provider._COMPILE_ROOTS:
        raise ValueError("service graph profile requires the exact dependency audit policy")
    workspace = str(context.workspace)
    if not Path(workspace).is_absolute():
        raise ValueError("graph service requires an explicit context workspace")
    result = GraphServiceBinding(compiler_profile_sha256, context.context_id, workspace, timeout_s, sources, context)
    _BINDINGS[result] = (context, result.describe())
    return result


def validate_binding(binding, *, compiler_profile_sha256, work_root=None):
    if type(binding) is not GraphServiceBinding or binding not in _BINDINGS:
        raise ValueError("live trusted graph service binding required; saved/copy/callback data refused")
    context, original = _BINDINGS[binding]
    if (binding.context is not context or binding.describe() != original
            or binding.compiler_profile_sha256 != compiler_profile_sha256
            or binding.context_id != context.context_id or str(context.workspace) != binding.workspace
            or context.held is not False):
        raise ValueError("graph service binding/context/profile changed or held")
    if work_root is not None and str(Path(work_root)) != binding.workspace:
        raise ValueError("graph request work root differs from the host-selected context")
    provider.bound_context(context)
    if any(source_ref(ref.path) != ref for ref in binding.sources):
        raise ValueError("graph service selected source bytes changed")
    projects = {"Merlin." + Path(ref.path).stem: (ref.path, ref.sha256, ref.size)
                for ref in binding.sources if ref.path.endswith(".lean")}
    if projects != {name: (ref.path, ref.sha256, ref.bytes) for name, ref in context.originals.items()}:
        raise ValueError("graph service project source map changed")
    if context.compile_audit_roots != provider._COMPILE_ROOTS:
        raise ValueError("graph service dependency audit policy changed")
    return context


def validate_options(selected):
    if any(selected.get(name, True) for name in ("normalize_shapes", "simplify_integers", "reuse_buffers")):
        raise ValueError("graph profile supports only explicit scalar CSE/DCE and LICM; legacy passes have no authority")
    if any(selected.get(name, ()) for name in ("tile_iterations", "unroll_factors", "dependency_orders")):
        raise ValueError("loop/storage/schedule transformations are outside the selected graph service profile")


def record_delivery(binding, document, source, changes, receipts, *, deadline):
    """Record the SAME live service output only after all boundary admissions."""
    from .lean_graph_pipeline_gate_projection import canonical, sha
    from merlin.transforms.certificates import ir_text, parse_ir

    validate_binding(binding, compiler_profile_sha256=binding.compiler_profile_sha256)
    if time.monotonic() >= deadline:
        raise TimeoutError("graph delivery exhausted the host work budget")
    if len(changes) != len(receipts) or not changes:
        raise ValueError("graph authority requires an actual checked changed boundary roster")
    if (document.get("source_sha256") != sha(source) or changes[0].before_ir != ir_text(parse_ir(source))
            or any(first.after_ir != second.before_ir for first, second in zip(changes, changes[1:]))):
        raise ValueError("graph delivery lacks exact original and adjacent compiler buffer joins")
    if any(change.kind not in KINDS or not gate.accepted(receipt, change, toolchain=binding.context)
           for change, receipt in zip(changes, receipts, strict=True)):
        raise ValueError("graph delivery lacks fresh whole-boundary admissions")
    if document.get("transformed_ir") != changes[-1].after_ir:
        raise ValueError("graph delivery differs from the last complete checked output")
    original = canonical(document)
    if time.monotonic() >= deadline:
        raise TimeoutError("graph delivery exhausted the host budget during revalidation")
    _DELIVERIES[id(document)] = (document, original, binding, source, tuple(changes), tuple(receipts))
    if len(_DELIVERIES) > 128:
        _DELIVERIES.popitem(last=False)


def eligible_delivery(binding, document, source):
    """Saved/copy/mutated service output cannot grant compiler eligibility."""
    from .lean_graph_pipeline_gate_projection import canonical

    found = _DELIVERIES.get(id(document))
    if found is None or found[0] is not document or found[2] is not binding or found[3] != source:
        return False
    try:
        validate_binding(binding, compiler_profile_sha256=binding.compiler_profile_sha256)
        if canonical(document) != found[1]:
            return False
        return all(gate.accepted(receipt, change, toolchain=binding.context)
                   for change, receipt in zip(found[4], found[5], strict=True))
    except (ValueError, RuntimeError, OSError, TypeError, AttributeError, KeyError):
        return False

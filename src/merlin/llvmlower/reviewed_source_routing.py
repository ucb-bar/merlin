"""Select ordinary source preparation from an explicit live reviewed registry.

This normal host injection seam supplies existing build hooks. It neither
loads an adapter by target name nor reconstructs qualification from a receipt.
No profile, numerical policy, checked library or sandbox grant is upgraded.
"""

from __future__ import annotations

import hashlib
import json

from merlin.common.digest import sha256_file
from merlin.common.paths import module_source_path

from .reviewed_pass_capabilities import PassCapabilityBinding, ReviewedPassCapabilities
from .shared_source_routing import PointwiseSourceRouting, SharedSourceRouting

MECHANISM = "source_scheduling"


def source_capability_binding(owner, *, runtime_identity):
    """Join current selected source contracts to an original host runtime identity.

    The original backend verifier owns runtime identity/effects applicability;
    the caller must supply it from that same reviewed selection. The function
    derives no runtime premise from SSA, labels, source code or test results.
    """
    if type(owner) not in (SharedSourceRouting, PointwiseSourceRouting):
        raise ValueError("source capability requires its original typed preparation owner")
    owner.revalidate()
    selected = owner._backend.contracts
    sources = dict(selected.resolution_sources)
    library = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PassCapabilityBinding(
        selected.hardware_identity,
        selected.record()["software_spec"]["sha256"],
        selected.numerical_policy,
        runtime_identity,
        library,
    )


class SelectedSourceRouting:
    """One original registry selection, rechecked at every ordinary consumer."""

    def __init__(self, registry, binding):
        if type(registry) is not ReviewedPassCapabilities or type(binding) is not PassCapabilityBinding:
            raise ValueError("ordinary source injection requires an original registry and exact binding")
        self._registry, self._binding = registry, binding
        self._sources = tuple(
            (path, sha256_file(path))
            for path in (
                module_source_path(__name__),
                module_source_path("merlin.llvmlower.reviewed_pass_capabilities"),
            )
        )
        self._registry_identity = registry.identity
        self._selection = registry.resolve(MECHANISM, binding)
        self._selection_identity = None if self._selection is None else self._selection.identity
        self._owner = None if self._selection is None else self._selection.owner
        if self._owner is not None and type(self._owner) not in (SharedSourceRouting, PointwiseSourceRouting):
            raise ValueError("source scheduling capability is not the original typed source preparation owner")
        self._revalidate()

    def _revalidate(self):
        if (
            type(self) is not SelectedSourceRouting
            or any(path.is_symlink() or sha256_file(path) != digest for path, digest in self._sources)
            or self._registry.identity != self._registry_identity
            or self._registry.resolve(MECHANISM, self._binding) is not self._selection
        ):
            raise ValueError("original source capability registry or selection changed")
        if self._selection is None:
            if self._owner is not None or self._selection_identity is not None:
                raise ValueError("missing source capability cannot be replaced with an unreviewed owner")
            return
        if (
            self._selection.identity != self._selection_identity
            or self._selection.owner is not self._owner
            or source_capability_binding(self._owner, runtime_identity=self._binding.runtime) != self._binding
        ):
            raise ValueError("original source capability facts/software/library/runtime binding changed")
        self._owner.revalidate()

    def record(self):
        self._revalidate()
        return dict(
            schema="merlin.selected_source_capability.v1",
            registry_identity=self._registry_identity,
            status="disabled" if self._selection is None else "selected_original_owner",
            refusal="no exact reviewed source_scheduling capability" if self._selection is None else None,
            capability=None if self._selection is None else self._selection.record(),
        )

    def build_arguments(self):
        """Inject only the existing ordinary preparation/provider hooks."""
        self._revalidate()
        return (
            {}
            if self._selection is None
            else {"prepared_model_transform": self.prepare, "host_provider_builder": self.build}
        )

    def prepare(self, source, output):
        self._revalidate()
        if self._owner is None:
            raise ValueError("no reviewed source capability was selected")
        selected = self._owner.prepare(source, output)
        self._revalidate()
        return selected

    def provider_for(self, context):
        self._revalidate()
        if self._owner is None:
            raise ValueError("no reviewed source capability was selected")
        result = self._owner.provider_for(context)
        self._revalidate()
        return result

    def build(self, context):
        self._revalidate()
        if self._owner is None:
            raise ValueError("no reviewed source capability was selected")
        result = self._owner.build(context)
        self._revalidate()
        return result

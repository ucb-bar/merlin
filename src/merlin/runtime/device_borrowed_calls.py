"""Carry the original sibling writers through the ordinary device build.

The writer owner's conditional effects remain conditional on implementation
qualification and final link admission. No effect is inferred from a shim,
callee name, element type or descriptor rank.
"""

import json
from dataclasses import asdict
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.llvmlower import borrowed_buffer_calls as K
from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_preparation as P
from merlin.llvmlower import sibling_source_route as S
from merlin.llvmlower.device_shim import ShimUnit
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact

from .device_descriptor_results import _bind, _build_record, _observed


class _SiblingBorrowedCalls:
    def __init__(self, routing, invocation, built, source, compilation):
        if type(routing) is not B.DeviceRouting:
            raise ValueError("sibling borrowed calls need their original typed device routing")
        P.require_device_offload_selection(routing, invocation)
        selected = S.require_route(routing, invocation.sibling_source)
        host = invocation.sibling_host
        S.require_host(host, selected)
        components = S.component_hosts(host)
        if not components:
            raise ValueError("sibling borrowed calls need actual selected component writers")
        if type(built) is not B.DeviceBuild or not built.ok or type(built.shim_unit) is not ShimUnit:
            raise ValueError("sibling borrowed calls need the actual compiled device build")
        source = Path(source).resolve(strict=True)
        if invocation.sidecar is None or (source, sha256_file(source)) not in invocation.sources:
            raise ValueError("sibling borrowed calls lost the actual prepared source")
        self.routing, self.invocation = routing, invocation
        self.selected, self.host, self.components = selected, host, components
        self.effects, self.writers = selected.effects, tuple(c.writer for c in components)
        self.route_state = self._routing_state()
        self.built, self.unit = built, built.shim_unit
        self.writer_state = json.dumps([asdict(c.writer) for c in components], sort_keys=True)
        self.build_state, self.unit_text = _build_record(built), self.unit.text
        self.source_pins = (*invocation.sources, invocation.sidecar)
        self.selection_state = selected.record()
        self.observations = {}
        self.functions = tuple((f, f.__code__) for f in self._entrypoints())
        owners = (Path(__file__), Path(S.__file__), Path(P.__file__), Path(B.__file__), Path(K.__file__),
                  Path(P.__file__).with_name("device_shim.py"),
                  Path(P.__file__).with_name("sibling_component_route.py"),
                  Path(P.__file__).with_name("head_source_route.py"),
                  Path(__file__).with_name("device_descriptor_results.py"),
                  Path(__file__).with_name("backends") / "spike_model.py")
        retained = tuple(HostLLVMArtifact.capture(p) for p in owners)
        observed, products = _observed(built.shim_invocation)
        c_path, object_path = built.shim_object.with_suffix(".c"), built.shim_object
        if (observed.get("stage") != "device_shim_compilation"
                or observed["inputs"] != [{"path": str(c_path.resolve()), "sha256": sha256_file(c_path)}]
                or observed["outputs"] != [{"path": str(object_path.resolve()), "sha256": sha256_file(object_path)}]
                or c_path.read_text() != self.unit_text):
            raise ValueError("sibling borrowed calls require the actually compiled logical shim")
        retained = (*retained, *products)
        objects = {object_path.resolve()}
        bridge_record = {
            role: {"path": str(path.resolve()), "sha256": sha256_file(path),
                   "size_bytes": path.stat().st_size}
            for role, path in (("source", c_path), ("object", object_path))
        }
        bridge_record["compilation"] = built.shim_invocation
        if set(built.kernels) != set(self.unit.symbols) or set(built.kernels.values()) != set(self.unit.kernels):
            raise ValueError("sibling borrowed calls changed the emitted device roster")
        for component in components:
            if component.writer.borrowed_effects is not selected.effects:
                raise ValueError("sibling borrowed calls require the original effects object")
            if component.device_symbol not in built.kernels:
                raise ValueError("sibling borrowed calls omitted an original device component")
            bridge = S._borrowed_bridge(self.unit, component)
            if self.unit_text.count(bridge) != 1:
                raise ValueError("sibling borrowed calls lost the exact complete result bridge")
            entry = built.custody[component.device_symbol].get("sibling_source")
            if entry != dict(source_ir_sha256=selected.source_ir_sha256,
                             interface_sha256=sha256_file_text(component.interface),
                             borrowed_effects=selected.record()["effects"],
                             original_epilogues_and_all_live_outputs_preserved=True,
                             descriptor_bridge=bridge_record):
                raise ValueError("sibling device compilation lost its original writer custody")
        for symbol in built.kernels:
            for role in ("translation", "optimization", "optimized_import", "compilation", "rename"):
                record, products = _observed(built.custody[symbol][role])
                retained = (*retained, *products)
                if role == "rename":
                    objects.update(Path(row["path"]).resolve() for row in record["outputs"])
        if objects != {Path(p).resolve() for p in built.objects}:
            raise ValueError("sibling borrowed calls lost the actual linked object roster")
        unique = {}
        for product in retained:
            if product.path in unique and unique[product.path] != product:
                raise ValueError("sibling device products have conflicting identities")
            unique[product.path] = product
        self.products = tuple(unique.values())
        self.revalidate()
        pins = (*self.source_pins, *((p.path, p.sha256) for p in self.products))
        self.calls = K.from_writers(source, (c.writer for c in components), pins,
                                    self.revalidate, self.observe_native)
        if compilation is not None:
            for i, product in enumerate(self.products):
                _bind(compilation, f"sibling_borrowed_owner_{i}", product)

    @staticmethod
    def _entrypoints():
        return (_SiblingBorrowedCalls.__init__, _SiblingBorrowedCalls.revalidate,
                _SiblingBorrowedCalls.observe_native, _SiblingBorrowedCalls._entrypoints,
                _SiblingBorrowedCalls._routing_state,
                sibling_borrowed_calls, S.component_hosts, S._borrowed_bridge, S.borrowed_bridge_source, S.require_route,
                K.from_writers, K._callback_binding, _build_record, _observed, sha256_file_text)

    def _routing_state(self):
        route = self.routing
        return (route.device, str(route.package_dir), route.operand_dtype, route.accum_dtype,
                route.granularity, json.dumps(route.numeric_policy, sort_keys=True),
                *(id(getattr(route, name)) for name in
                  ("select", "capture", "target_schedule", "linked_elf_admission", "target_configuration",
                   "llvm_transform_chain", "device_ir_budget", "intermediate_storage", "pre_group_layout")))

    def revalidate(self):
        # The complete host graph was checked once at the actual compile/input
        # boundary above. Its emitted source is pinned, rather than reparsed at
        # each native callback. The original live writer/effect roster remains
        # authoritative; equal copied effects cannot refresh it.
        if (self.invocation.sibling_host is not self.host
                or S.require_route(self.routing, self.invocation.sibling_source) is not self.selected
                or self.selected.record() != self.selection_state
                or self.selected.effects is not self.effects
                or self._routing_state() != self.route_state
                or len(S.component_hosts(self.host)) != len(self.components)
                or any(current is not original for current, original in
                       zip(S.component_hosts(self.host), self.components, strict=True))
                or any(c.writer is not w or w.borrowed_effects is not self.effects
                       for c, w in zip(self.components, self.writers, strict=True))
                or json.dumps([asdict(c.writer) for c in self.components], sort_keys=True) != self.writer_state
                or self.built.shim_unit is not self.unit or self.unit.text != self.unit_text
                or _build_record(self.built) != self.build_state
                or (*self.invocation.sources, self.invocation.sidecar) != self.source_pins
                or any(f is not original or f.__code__ is not code
                       for f, (original, code) in zip(self._entrypoints(), self.functions, strict=True))):
            raise ValueError("original sibling borrowed owner or compiled product changed")
        for path, digest in self.source_pins:
            if path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("original sibling borrowed source or sidecar changed")
        for product in self.products:
            product.validate()

    def observe_native(self, paths, initial):
        self.revalidate()
        for path in paths:
            path = Path(path).resolve(strict=True)
            pin = sha256_file(path)
            if initial:
                if path in self.observations:
                    raise ValueError("sibling native observation was already issued")
                self.observations[path] = pin
            elif self.observations.get(path) != pin:
                raise ValueError("sibling native observation changed")


def sha256_file_text(value):
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def sibling_borrowed_calls(routing, invocation, built, source, *, compilation=None):
    """Consume only the live selected route; records cannot restore this owner."""
    if getattr(routing, "sibling_source", None) is None:
        return None
    if type(invocation) is not P._DeviceOffloadInvocation or invocation.sibling_source is None:
        raise ValueError("sibling borrowed calls need their original device invocation")
    if not S.component_hosts(invocation.sibling_host):
        return None
    return _SiblingBorrowedCalls(routing, invocation, built, source, compilation).calls

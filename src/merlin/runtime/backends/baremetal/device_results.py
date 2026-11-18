"""Join generated-shim descriptor identity to ordinary model compilation.

Only Merlin's actual checked shim can supply this roster. The original kernel,
its qualification and final linked-image admission remain independently owned.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import FunctionType, MethodType

from merlin.common.digest import sha256_file
from merlin.common.process import invocation_records as invocation_record
from merlin.compiler.lower.device import build as B
from merlin.compiler.lower.device import device_build as _merlin_device_build_1
from merlin.compiler.lower.device import device_shim as _merlin_device_shim_2
from merlin.compiler.lower.device import preparation as P
from merlin.compiler.lower.device import shim as D
from merlin.compiler.lower.device import sibling_source_route as S
from merlin.compiler.lower.device.shim import ShimUnit
from merlin.compiler.lower.host import borrowed_buffer_calls as K
from merlin.compiler.lower.host.transforms import HostLLVMArtifact
from merlin.compiler.lower.options import descriptor_result_identity as R

# ---- device_descriptor_results -----------------------------------------------------------------


def _selection(identities):
    return tuple(
        (
            id(value),
            value.symbol,
            value.borrowed_symbol,
            value.wrapper_symbol,
            value.result_argument,
            tuple(str(t) for t in value.argument_types),
            value.argument_access,
            tuple(str(v) for v in value.argument_layouts),
            None if getattr(value, "borrowed_effects", None) is None else asdict(value.borrowed_effects),
        )
        for value in identities
    )


def _callback_identity(value):
    function = getattr(value, "__func__", value)
    return id(value), id(function), id(getattr(function, "__code__", None))


def _callback_source(callback):
    function = callback.__func__ if type(callback) is MethodType else callback
    if type(function) is not FunctionType or inspect.getsourcefile(function) is None:
        raise ValueError("descriptor effects require an original source-bound Python owner")
    return Path(inspect.getsourcefile(function)).resolve()


def _effects_record(owner):
    return (
        json.dumps(owner.record(), sort_keys=True, allow_nan=False),
        _callback_identity(owner.validate_original),
        _callback_identity(owner.observe_native),
    )


def _build_record(built):
    return json.dumps(
        {
            "objects": [str(p) for p in built.objects],
            "shim_object": str(built.shim_object),
            "kernels": built.kernels,
            "custody": built.custody,
            "shim_invocation": built.shim_invocation,
            "built_from": built.built_from,
        },
        sort_keys=True,
        allow_nan=False,
    )


def _observed(pin):
    path = Path(pin["path"])
    if sha256_file(path) != pin["sha256"]:
        raise ValueError("original device compilation receipt changed")
    observed = invocation_record.verify(path)
    rows = (
        *observed["inputs"],
        *observed["outputs"],
        *observed["dependencies"],
        observed["executable"],
        observed["stdout"],
        observed["stderr"],
    )
    products = tuple(
        HostLLVMArtifact(Path(row["path"]), row["sha256"], Path(row["path"]).stat().st_size) for row in rows
    )
    return observed, (HostLLVMArtifact.capture(path), *products)


def _bind(compilation, name, artifact):
    expected = artifact.to_dict()
    for original in compilation.record.get("preparation", {}).values():
        if Path(original["path"]).resolve() == artifact.path:
            if original != expected:
                raise ValueError("descriptor preparation conflicts with an original bound identity")
            return
    compilation.bind_preparation(name, artifact.path)


class DeviceDescriptorResults:
    """One original route, actual shim and consumed preparation; no permission grant."""

    def __init__(self, routing):
        self.routing = routing
        self._original_routing = routing
        self.selected = getattr(routing, "descriptor_results", False)
        if type(self.selected) is not bool:
            raise ValueError("descriptor results require an original boolean")
        self._owners, self._functions, self._products = (), (), ()
        self._unit = self._roster = self._report = self._invocation = self._unit_text = None
        self.identities = ()
        self._input_bound = False
        self._views_prepared = False
        self._inputs = ()
        self._build = self._build_snapshot = None
        self._effects = self._effects_snapshot = self._effects_check = self._qualified_selection = None
        if not self.selected:
            return
        if type(routing) is not _merlin_device_build_1.DeviceRouting:
            raise ValueError("descriptor identity requires the original typed device routing")
        if getattr(routing, "granularity", "contraction") not in ("contraction", "group") or any(
            getattr(routing, name, None) is not None
            for name in (
                "catalog_manifest",
                "catalog_object",
                "catalog_builder",
                "post_offload_transform",
                "connected_spatial",
            )
        ):
            raise ValueError(
                "descriptor identity requires the original generated contraction or logical group shim route"
            )
        if routing.sibling_source is not None:
            from merlin.compiler.lower.device.head import HeadBatchSourceRoute
            from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute

            if routing.granularity != "group" or type(routing.sibling_source) not in (
                HeadBatchSourceRoute,
                SiblingSourceRoute,
            ):
                raise ValueError("mixed descriptor identity requires its original typed logical sibling route")
            routing.sibling_source.record()
        if routing.descriptor_effects is None:
            raise ValueError(
                "descriptor identity requires an original qualified borrowed-buffer effects owner; "
                "descriptor equality does not establish no retention, no releases or completion"
            )
        factory_source = _callback_source(routing.descriptor_effects)
        from merlin.compiler.lower.options.descriptor_result_preparation import implementation_paths

        directory = Path(B.__file__).parent
        self._owners = tuple(
            HostLLVMArtifact.capture(path)
            for path in dict.fromkeys(
                (
                    Path(__file__),
                    Path(D.__file__),
                    Path(B.__file__),
                    Path(K.__file__),
                    factory_source,
                    directory / "device_preparation.py",
                    directory / "quantized_sum_route.py",
                    directory / "quantized_sum_source.py",
                    directory / "device_input_views.py",
                    Path(__file__).with_name("image_link.py"),
                    *implementation_paths(),
                )
            )
        )
        self._function_reader = DeviceDescriptorResults._entrypoints
        self._functions = tuple((f, f.__code__) for f in self._function_reader())
        self._route = self._routing_snapshot()
        self.revalidate()

    def _routing_snapshot(self):
        route = self.routing
        scalar = tuple(
            getattr(route, name)
            for name in ("device", "operand_dtype", "accum_dtype", "granularity", "require_complete_groups")
        )
        objects = tuple(
            id(getattr(route, name))
            for name in (
                "exact_selection",
                "offload_selection",
                "linked_elf_admission",
                "target_configuration",
                "device_ir_budget",
                "catalog_manifest",
                "catalog_object",
                "catalog_builder",
                "post_offload_transform",
                "connected_spatial",
                "sibling_source",
                "quantized_sum",
            )
        )
        callbacks = tuple(
            _callback_identity(getattr(route, name))
            for name in (
                "select",
                "prepared_transform",
                "catalog_builder",
                "post_offload_transform",
                "final_elf_audit",
                "descriptor_effects",
            )
        )
        policy = json.dumps(route.numeric_policy, sort_keys=True, allow_nan=False)
        if len(policy.encode()) > 65536:
            raise ValueError("descriptor route numerical policy exceeds its byte budget")
        return (
            scalar,
            str(route.package_dir),
            objects,
            callbacks,
            policy,
            (
                None
                if route.quantized_sum is None
                else json.dumps(route.quantized_sum.record(), sort_keys=True, allow_nan=False)
            ),
        )

    @staticmethod
    def _entrypoints():
        from merlin.compiler.frontend.linalg_mlir import layout_to_buffer_type, parse_mlir_text
        from merlin.compiler.lower.device import device_input_views as V
        from merlin.compiler.lower.device import outline as Q
        from merlin.compiler.lower.device import outline as QS
        from merlin.compiler.lower.options import descriptor_result_preparation as P

        return (
            Q.QuantizedSumRoute.record,
            Q.freeze_route,
            Q.require_route,
            Q.group_quantized_sum,
            Q.prepare_host,
            Q.quantized_sum_interface,
            QS.plan_quantized_sum_source,
            QS.QuantizedSumSource.revalidate,
            D.emit_translation_unit,
            D._with_descriptor_results,
            D._borrowed_result_wrapper,
            _merlin_device_shim_2.emit_logical_translation_unit,
            _merlin_device_shim_2._logical_entry,
            _merlin_device_shim_2._logical_borrowed_result_wrapper,
            D.descriptor_result_marker,
            B.build_device_objects,
            R.rewrite_descriptor_result_identities,
            R.argument_buffer_types,
            V.prepare,
            V.rewrite_readonly_dense_offsets,
            R.validate_effects,
            R.BorrowedBufferEffects.__post_init__,
            R.edit_pipeline,
            R.ensure_registered,
            K.validate_effects,
            K.validate_calls,
            K.BorrowedBufferEffects.__post_init__,
            K.BorrowedBufferCall.__post_init__,
            K.BorrowedBufferCalls.__post_init__,
            K.BorrowedBufferCalls.revalidate,
            K.BorrowedBufferCalls.record,
            K.BorrowedBufferCalls.after_preprocessing,
            K._callback_binding,
            parse_mlir_text,
            layout_to_buffer_type,
            layout_to_buffer_type().verify_,
            P._records,
            P._effects_state,
            P._join_effects,
            P.implementation_paths,
            P.DescriptorResultPreparation.__init__,
            P.DescriptorResultPreparation._fixed_functions,
            P.DescriptorResultPreparation._feature_values,
            P.DescriptorResultPreparation.selected_feature,
            P.DescriptorResultPreparation.prepare,
            P.DescriptorResultPreparation.native_calls,
            P.DescriptorResultPreparation.revalidate,
            P.DescriptorResultPreparation.retain_upstream,
            P.DescriptorResultPreparation.bind_recipe,
            P.DescriptorResultPreparation.record,
            DeviceDescriptorResults.__init__,
            DeviceDescriptorResults._entrypoints,
            DeviceDescriptorResults._routing_snapshot,
            DeviceDescriptorResults.revalidate,
            DeviceDescriptorResults._require_static,
            DeviceDescriptorResults._bind_effects,
            DeviceDescriptorResults.bind_sources,
            DeviceDescriptorResults.bind_input,
            DeviceDescriptorResults.prepare_dense_input_views,
            DeviceDescriptorResults.bind_shim,
            DeviceDescriptorResults.options,
            DeviceDescriptorResults.consumed,
            DeviceDescriptorResults.identity,
            DeviceDescriptorResults.record,
            _selection,
            _callback_identity,
            _build_record,
            _observed,
            _bind,
            _callback_source,
            _effects_record,
        )

    def _require_static(self):
        if (
            self.routing is not self._original_routing
            or getattr(self.routing, "descriptor_results", False) is not self.selected
        ):
            raise ValueError("original descriptor-result routing changed")
        if not self.selected:
            return
        if self._routing_snapshot() != self._route:
            raise ValueError("original descriptor-result route, policy or callback changed")
        for artifact in (*self._owners, *self._products):
            artifact.validate()
        if any(
            f is not original or f.__code__ is not code
            for f, (original, code) in zip(self._function_reader(), self._functions, strict=True)
        ):
            raise ValueError("original descriptor-result implementation changed")
        if self._unit is not None and (
            self._unit.text != self._unit_text or _selection(self._unit.result_identities) != self._roster
        ):
            raise ValueError("original emitted descriptor identity changed")
        if self._build is not None and (
            self._build.shim_unit is not self._unit or _build_record(self._build) != self._build_snapshot
        ):
            raise ValueError("original retained device build or object roster changed")
        if self._qualified_selection is not None and _selection(self.identities) != self._qualified_selection:
            raise ValueError("original qualified descriptor effects changed")

    def revalidate(self):
        self._require_static()
        if self._effects is not None:
            if _effects_record(self._effects) != self._effects_snapshot:
                raise ValueError("original borrowed effects owner changed")
            self._effects_check(self._inputs[0].path.read_text())
            if _effects_record(self._effects) != self._effects_snapshot:
                raise ValueError("original borrowed effects changed during qualification validation")
            self._require_static()

    def bind_sources(self, compilation):
        self.revalidate()
        for index, source in enumerate(self._owners):
            _bind(compilation, f"descriptor_result_owner_{index}", source)

    def bind_shim(self, built, compilation):
        """Consume the actual single device build, never a reconstructed unit."""
        self.revalidate()
        if not self.selected or self._unit is not None or not self._input_bound:
            raise ValueError("descriptor shim needs one original selected build")
        if (
            type(built) is not _merlin_device_build_1.DeviceBuild
            or not built.ok
            or type(built.shim_unit) is not D.ShimUnit
        ):
            raise ValueError("descriptor identity needs its actual generated shim")
        unit, pin = built.shim_unit, built.shim_invocation
        if not isinstance(pin, dict) or sha256_file(pin["path"]) != pin["sha256"]:
            raise ValueError("descriptor shim compilation receipt changed")
        observed, retained = _observed(pin)
        c_path, object_path = built.shim_object.with_suffix(".c"), built.shim_object
        if (
            observed.get("stage") != "device_shim_compilation"
            or observed.get("inputs") != [{"path": str(c_path.resolve()), "sha256": sha256_file(c_path)}]
            or observed.get("outputs") != [{"path": str(object_path.resolve()), "sha256": sha256_file(object_path)}]
            or c_path.read_text() != unit.text
        ):
            raise ValueError("descriptor identity is not the actually compiled shim")
        if any(
            type(value) is not R.DescriptorResultIdentity
            or getattr(value, "borrowed_effects", None) is not None
            or value.argument_layouts
            or value.symbol not in built.kernels
            or value.symbol not in unit.symbols
            for value in unit.result_identities
        ):
            raise ValueError("generated shim lost its typed descriptor identities")
        if (
            unit.text.count("/* merlin.descriptor_result_identities.v1 ") != bool(unit.result_identities)
            or unit.result_identities
            and unit.text.count(D.descriptor_result_marker(unit.result_identities)) != 1
        ):
            raise ValueError("compiled shim differs from its original issued descriptor roster")
        owners = {row["path"] for row in observed["dependencies"]}
        if not {str(path) for path in D.descriptor_result_dependencies(unit)} <= owners:
            raise ValueError("descriptor shim compilation omitted its implementation owners")
        for value in unit.result_identities:
            forwarder = (
                _merlin_device_shim_2._logical_borrowed_result_wrapper(value)
                if self.routing.granularity == "group"
                else D._borrowed_result_wrapper(value.symbol, 2, value.borrowed_symbol)
            )
            if unit.text.count(forwarder) != 1:
                raise ValueError("generated shim lost its checked descriptor forwarder")
        self._unit, self._roster = unit, _selection(unit.result_identities)
        self._unit_text = unit.text
        self._invocation = Path(pin["path"])
        object_paths = [object_path.resolve()]
        if set(built.kernels) != set(unit.symbols) or set(built.kernels.values()) != set(unit.kernels):
            raise ValueError("actual device build differs from emitted shim symbols")
        for symbol in built.kernels:
            for role in ("translation", "optimization", "optimized_import", "compilation", "rename"):
                original, products = _observed(built.custody[symbol][role])
                retained = (*retained, *products)
                if role == "rename":
                    object_paths.extend(Path(row["path"]) for row in original["outputs"])
        if len(set(object_paths)) != len(object_paths) or set(object_paths) != {
            Path(path).resolve() for path in built.objects
        }:
            raise ValueError("actual device build lost its original observed objects")
        self._build, self._build_snapshot = built, _build_record(built)
        unique = {}
        for product in retained:
            if product.path in unique and unique[product.path] != product:
                raise ValueError("device invocation roster has inconsistent original products")
            unique[product.path] = product
        products = tuple(unique.values())
        self._products = (*self._products, *products)
        self.revalidate()
        for index, source in enumerate(products):
            _bind(compilation, f"descriptor_result_shim_{index}", source)
        if unit.result_identities:
            self._bind_effects(compilation)

    def _bind_effects(self, compilation):
        from xdsl.dialects.builtin import NoneAttr

        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        self.revalidate()
        owner = self.routing.descriptor_effects(self._inputs[0].path, self._build)
        self.revalidate()
        if type(owner) is not K.BorrowedBufferCalls:
            raise ValueError("original descriptor effects owner must return its live typed borrowed-call selection")
        owner.__post_init__()
        source = self._inputs[0]
        if (
            owner.source_path.resolve() != source.path
            or owner.source_sha256 != source.sha256
            or owner.input_sha256 != source.sha256
        ):
            raise ValueError("borrowed effects owner is stale for the actual prepared source")
        required = {
            path.resolve(): sha256_file(path)
            for path in (*self._build.objects, self._build.shim_object.with_suffix(".c"))
        }
        if not required.items() <= dict((path.resolve(), sha) for path, sha in owner.pins).items():
            raise ValueError("borrowed effects owner does not bind the actual kernel objects and checked shim")
        calls = {(call.wrapper_symbol, call.borrowed_symbol): call for call in owner.calls}
        identities = self._unit.result_identities
        if set(calls) != {(v.wrapper_symbol, v.borrowed_symbol) for v in identities}:
            raise ValueError("borrowed effects owner differs from actual emitted descriptor calls")
        selected = []
        declarations = {
            op.sym_name.data: op
            for op in parse_mlir_text(source.path.read_text()).body.block.ops
            if op.name == "func.func"
        }
        for value in identities:
            call = calls[value.wrapper_symbol, value.borrowed_symbol]
            declaration = declarations.get(value.symbol)
            if declaration is None or declaration.arg_attrs is None:
                raise ValueError("descriptor effects lost the current original declaration")
            layouts = tuple(attr.data.get("bufferization.buffer_layout", NoneAttr()) for attr in declaration.arg_attrs)
            if any(layout != NoneAttr() for layout in layouts):
                value = replace(value, argument_layouts=layouts)
            expected = tuple(str(t) for t in R.argument_buffer_types(value))
            if call.argument_types != expected or call.result_argument != value.result_argument:
                raise ValueError("borrowed effects owner differs from actual positional buffer ABI")
            selected.append(replace(value, borrowed_effects=call.effects))
        self._effects, self._effects_snapshot = owner, _effects_record(owner)
        self._effects_check = owner.revalidate
        self._products = (
            *self._products,
            HostLLVMArtifact.capture(_callback_source(owner.validate_original)),
            *(HostLLVMArtifact(Path(path).resolve(), sha, Path(path).stat().st_size) for path, sha in owner.pins),
        )
        self.identities = tuple(selected)
        self._qualified_selection = _selection(self.identities)
        self.revalidate()
        record = self._build.shim_object.parent / "descriptor-result-effects.json"
        record.write_text(self._effects_snapshot[0] + "\n")
        self._products = (*self._products, HostLLVMArtifact.capture(record))
        self.revalidate()
        for index, product in enumerate(self._products):
            _bind(compilation, f"descriptor_result_qualified_{index}", product)

    def bind_input(self, source, sidecar, compilation):
        """Freeze actual prepared IR and routing before device package callbacks."""
        self.revalidate()
        if not self.selected or self._products:
            raise ValueError("descriptor identity needs one original prepared input")
        self._products = tuple(HostLLVMArtifact.capture(path) for path in (source, sidecar))
        self._inputs = self._products
        self._input_bound = True
        self.revalidate()

    def prepare_dense_input_views(self, built, work, compilation, *, sibling_source=None):
        """Own the existing checked derivation, retaining its original input pins."""
        from merlin.compiler.lower.device.device_input_views import prepare

        self.revalidate()
        if (
            not self.selected
            or not self._input_bound
            or self._unit is not None
            or self._views_prepared
            or len(self._inputs) != 2
        ):
            raise ValueError("descriptor views require the original pre-build input")
        source, sidecar = self._inputs
        if (sibling_source is None) != (self.routing.sibling_source is None):
            raise ValueError("mixed descriptor views require the original sibling source owner")
        if sibling_source is not None and sibling_source[0] is not self.routing.sibling_source:
            raise ValueError("mixed descriptor views changed the original sibling selection")
        selected = prepare(source.path, sidecar.path, built, work, compilation, sibling_source=sibling_source)
        self.revalidate()
        product = HostLLVMArtifact.capture(selected)
        receipt = HostLLVMArtifact.capture(Path(work) / "selection.json")
        self._products = (*self._products, product, receipt)
        self._inputs = (product, sidecar)
        self._views_prepared = True
        self.revalidate()
        return selected

    def options(self):
        self.revalidate()
        return (
            {}
            if not self.identities
            else {
                "descriptor_result_identities": self.identities,
                "descriptor_result_effects": self._effects,
            }
        )

    def consumed(self, result, compilation):
        self.revalidate()
        for index, product in enumerate(self._inputs):
            _bind(compilation, f"descriptor_result_input_{index}", product)
        if not self.identities:
            return
        record = result.stats.get("descriptor_result_identity")
        if not isinstance(record, dict) or record.get("selection") != [
            {
                "symbol": v.symbol,
                "borrowed_symbol": v.borrowed_symbol,
                "wrapper_symbol": v.wrapper_symbol,
                "result_argument": v.result_argument,
                "argument_types": [str(t) for t in v.argument_types],
                "argument_access": list(v.argument_access),
                **({"argument_layouts": [str(layout) for layout in v.argument_layouts]} if v.argument_layouts else {}),
                "borrowed_effects": asdict(v.borrowed_effects),
            }
            for v in self.identities
        ]:
            raise ValueError("ordinary lowering did not consume original emitted descriptor identities")
        if (
            record.get("borrowed_calls") != self._effects.record()
            or record.get("rewrites")
            and getattr(result, "borrowed_arena_lifetime", None) is None
        ):
            raise ValueError("ordinary descriptor rewrite lost its original native lifetime owner")
        paths = [
            record["report"]["path"],
            record["recipe"]["path"],
            record["original_source"]["path"],
            record["selected_source"]["path"],
        ]
        expected = [
            result.workdir / name
            for name in (
                "descriptor-result-identity.json",
                "lowering_recipe.json",
                "descriptor-identity.input.mlir",
                "descriptor-identity.selected.mlir",
            )
        ]
        if tuple(Path(path).resolve() for path in paths) != tuple(path.resolve() for path in expected):
            raise ValueError("descriptor identity returned artifacts outside actual lowering")
        for pin in (record["report"], record["recipe"], record["original_source"], record["selected_source"]):
            artifact = HostLLVMArtifact(Path(pin["path"]), pin["sha256"], pin["bytes"])
            artifact.validate()
        if json.loads(Path(record["report"]["path"]).read_text()).get("selection") != record["selection"]:
            raise ValueError("descriptor identity report differs from consumed original selection")
        recipe = json.loads(Path(paths[1]).read_text())
        if (
            recipe.get("status") != "returned"
            or recipe.get("descriptor_result_identity", {}).get("report") != record["report"]
            or recipe.get("returned_llvm_ir")
            != {"sha256": sha256_file(result.ll_path), "bytes": result.ll_path.stat().st_size}
        ):
            raise ValueError("descriptor identity is absent from actual returned LLVM lowering")
        joined = recipe["descriptor_result_identity"]
        for name in ("upstream_source", "actual_prepared_mlir"):
            pin = joined[name]
            HostLLVMArtifact(Path(pin["path"]), pin["sha256"], pin["bytes"]).validate()
            paths.append(pin["path"])
        paths.append(str(result.ll_path))
        self._products = (*self._products, *(HostLLVMArtifact.capture(path) for path in paths))
        self._report = record
        for index, path in enumerate(paths):
            _bind(compilation, f"descriptor_result_consumed_{index}", HostLLVMArtifact.capture(path))
        self.revalidate()

    def identity(self):
        self.revalidate()
        return (
            b""
            if not self.selected
            else json.dumps(
                {
                    "descriptor_results": True,
                    "owners": [pin.to_dict() for pin in self._owners],
                    "borrowed_calls": None if self._effects_snapshot is None else json.loads(self._effects_snapshot[0]),
                },
                sort_keys=True,
            ).encode()
        )

    def record(self):
        """Retain the actual consumed selection; never reconstruct its owner."""
        self.revalidate()
        if not self.selected:
            return None
        return {
            "selected": True,
            "lowering": self._report,
            "borrowed_calls": None if self._effects is None else self._effects.record(),
        }


# ---- device_borrowed_calls ---------------------------------------------------------------------
# Carry the original sibling writers through the ordinary device build.
#
# The writer owner's conditional effects remain conditional on implementation
# qualification and final link admission. No effect is inferred from a shim,
# callee name, element type or descriptor rank.


class _SiblingBorrowedCalls:
    def __init__(self, routing, invocation, built, source, compilation):
        if type(routing) is not _merlin_device_build_1.DeviceRouting:
            raise ValueError("sibling borrowed calls need their original typed device routing")
        P.require_device_offload_selection(routing, invocation)
        selected = S.require_route(routing, invocation.sibling_source)
        host = invocation.sibling_host
        S.require_host(host, selected)
        components = S.component_hosts(host)
        if not components:
            raise ValueError("sibling borrowed calls need actual selected component writers")
        if (
            type(built) is not _merlin_device_build_1.DeviceBuild
            or not built.ok
            or type(built.shim_unit) is not ShimUnit
        ):
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
        owners = (
            Path(__file__),
            Path(S.__file__),
            Path(P.__file__),
            Path(B.__file__),
            Path(K.__file__),
            Path(P.__file__).with_name("device_shim.py"),
            Path(P.__file__).with_name("sibling_component_route.py"),
            Path(P.__file__).with_name("head_source_route.py"),
            Path(__file__),
            Path(__file__).with_name("image_link.py"),
        )
        retained = tuple(HostLLVMArtifact.capture(p) for p in owners)
        observed, products = _observed(built.shim_invocation)
        c_path, object_path = built.shim_object.with_suffix(".c"), built.shim_object
        if (
            observed.get("stage") != "device_shim_compilation"
            or observed["inputs"] != [{"path": str(c_path.resolve()), "sha256": sha256_file(c_path)}]
            or observed["outputs"] != [{"path": str(object_path.resolve()), "sha256": sha256_file(object_path)}]
            or c_path.read_text() != self.unit_text
        ):
            raise ValueError("sibling borrowed calls require the actually compiled logical shim")
        retained = (*retained, *products)
        objects = {object_path.resolve()}
        bridge_record = {
            role: {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
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
            if entry != dict(
                source_ir_sha256=selected.source_ir_sha256,
                interface_sha256=sha256_file_text(component.interface),
                borrowed_effects=selected.record()["effects"],
                original_epilogues_and_all_live_outputs_preserved=True,
                descriptor_bridge=bridge_record,
            ):
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
        self.calls = K.from_writers(source, (c.writer for c in components), pins, self.revalidate, self.observe_native)
        if compilation is not None:
            for i, product in enumerate(self.products):
                _bind(compilation, f"sibling_borrowed_owner_{i}", product)

    @staticmethod
    def _entrypoints():
        return (
            _SiblingBorrowedCalls.__init__,
            _SiblingBorrowedCalls.revalidate,
            _SiblingBorrowedCalls.observe_native,
            _SiblingBorrowedCalls._entrypoints,
            _SiblingBorrowedCalls._routing_state,
            sibling_borrowed_calls,
            S.component_hosts,
            S._borrowed_bridge,
            S.borrowed_bridge_source,
            S.require_route,
            K.from_writers,
            K._callback_binding,
            _build_record,
            _observed,
            sha256_file_text,
        )

    def _routing_state(self):
        route = self.routing
        return (
            route.device,
            str(route.package_dir),
            route.operand_dtype,
            route.accum_dtype,
            route.granularity,
            json.dumps(route.numeric_policy, sort_keys=True),
            *(
                id(getattr(route, name))
                for name in (
                    "select",
                    "capture",
                    "target_schedule",
                    "linked_elf_admission",
                    "target_configuration",
                    "llvm_transform_chain",
                    "device_ir_budget",
                    "intermediate_storage",
                    "pre_group_layout",
                )
            ),
        )

    def revalidate(self):
        # The complete host graph was checked once at the actual compile/input
        # boundary above. Its emitted source is pinned, rather than reparsed at
        # each native callback. The original live writer/effect roster remains
        # authoritative; equal copied effects cannot refresh it.
        if (
            self.invocation.sibling_host is not self.host
            or S.require_route(self.routing, self.invocation.sibling_source) is not self.selected
            or self.selected.record() != self.selection_state
            or self.selected.effects is not self.effects
            or self._routing_state() != self.route_state
            or len(S.component_hosts(self.host)) != len(self.components)
            or any(
                current is not original
                for current, original in zip(S.component_hosts(self.host), self.components, strict=True)
            )
            or any(
                c.writer is not w or w.borrowed_effects is not self.effects
                for c, w in zip(self.components, self.writers, strict=True)
            )
            or json.dumps([asdict(c.writer) for c in self.components], sort_keys=True) != self.writer_state
            or self.built.shim_unit is not self.unit
            or self.unit.text != self.unit_text
            or _build_record(self.built) != self.build_state
            or (*self.invocation.sources, self.invocation.sidecar) != self.source_pins
            or any(
                f is not original or f.__code__ is not code
                for f, (original, code) in zip(self._entrypoints(), self.functions, strict=True)
            )
        ):
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

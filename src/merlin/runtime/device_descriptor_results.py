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

from merlin.common import invocation_record
from merlin.common.digest import sha256_file
from merlin.llvmlower import borrowed_buffer_calls as K
from merlin.llvmlower import descriptor_result_identity as R
from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_shim as D
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact


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
        if type(routing) is not B.DeviceRouting:
            raise ValueError("descriptor identity requires the original typed device routing")
        if getattr(routing, "granularity", "contraction") not in ("contraction", "group") or any(
            getattr(routing, name, None) is not None
            for name in ("catalog_manifest", "catalog_object", "catalog_builder", "post_offload_transform",
                         "connected_spatial", "sibling_source")
        ):
            raise ValueError("descriptor identity requires the original generated contraction or logical group shim route")
        if routing.descriptor_effects is None:
            raise ValueError(
                "descriptor identity requires an original qualified borrowed-buffer effects owner; "
                "descriptor equality does not establish no retention, no releases or completion"
            )
        factory_source = _callback_source(routing.descriptor_effects)
        from merlin.llvmlower.descriptor_result_preparation import implementation_paths

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
                    directory / "device_input_views.py",
                    Path(__file__).with_name("backends") / "spike_model.py",
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
        return scalar, str(route.package_dir), objects, callbacks, policy

    @staticmethod
    def _entrypoints():
        from merlin.frontends.linalg_mlir import parse_mlir_text, layout_to_buffer_type
        from merlin.llvmlower import descriptor_result_preparation as P
        from merlin.llvmlower import device_input_views as V

        return (
            D.emit_translation_unit,
            D._with_descriptor_results,
            D._borrowed_result_wrapper,
            D.emit_logical_translation_unit,
            D._logical_entry,
            D._logical_borrowed_result_wrapper,
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
        if type(built) is not B.DeviceBuild or not built.ok or type(built.shim_unit) is not D.ShimUnit:
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
            forwarder = (D._logical_borrowed_result_wrapper(value) if self.routing.granularity == "group"
                         else D._borrowed_result_wrapper(value.symbol, 2, value.borrowed_symbol))
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
        from merlin.frontends.linalg_mlir import parse_mlir_text

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
        declarations = {op.sym_name.data: op for op in parse_mlir_text(source.path.read_text()).body.block.ops
                        if op.name == "func.func"}
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

    def prepare_dense_input_views(self, built, work, compilation):
        """Own the existing checked derivation, retaining its original input pins."""
        from merlin.llvmlower.device_input_views import prepare

        self.revalidate()
        if (not self.selected or not self._input_bound or self._unit is not None
                or self._views_prepared or len(self._inputs) != 2):
            raise ValueError("descriptor views require the original pre-build input")
        source, sidecar = self._inputs
        selected = prepare(source.path, sidecar.path, built, work, compilation)
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
        return {} if not self.identities else {
            "descriptor_result_identities": self.identities,
            "descriptor_result_effects": self._effects,
        }

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
        if (record.get("borrowed_calls") != self._effects.record()
                or record.get("rewrites") and getattr(result, "borrowed_arena_lifetime", None) is None):
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
        return {"selected": True, "lowering": self._report,
                "borrowed_calls": None if self._effects is None else self._effects.record()}

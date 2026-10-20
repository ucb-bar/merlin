"""Join generated-shim descriptor identity to ordinary model compilation.

Only Merlin's actual checked shim can supply this roster. The original kernel,
its qualification and final linked-image admission remain independently owned.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.digest import sha256_file
from merlin.llvmlower import descriptor_result_identity as R
from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_shim as D
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact


def _selection(unit):
    return tuple(
        (
            id(value),
            value.symbol,
            value.borrowed_symbol,
            value.wrapper_symbol,
            value.result_argument,
            tuple(str(t) for t in value.argument_types),
            value.argument_access,
        )
        for value in unit.result_identities
    )


def _callback_identity(value):
    function = getattr(value, "__func__", value)
    return id(value), id(function), id(getattr(function, "__code__", None))


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
        self._inputs = ()
        self._build = self._build_snapshot = None
        if not self.selected:
            return
        if type(routing) is not B.DeviceRouting:
            raise ValueError("descriptor identity requires the original typed device routing")
        if getattr(routing, "granularity", "contraction") != "contraction" or any(
            getattr(routing, name, None) is not None
            for name in ("catalog_manifest", "catalog_object", "catalog_builder", "post_offload_transform")
        ):
            raise ValueError("descriptor identity requires the original generated contraction shim route")
        from merlin.llvmlower.descriptor_result_preparation import implementation_paths

        directory = Path(B.__file__).parent
        self._owners = tuple(
            HostLLVMArtifact.capture(path)
            for path in dict.fromkeys(
                (
                    Path(__file__),
                    Path(D.__file__),
                    Path(B.__file__),
                    directory / "device_preparation.py",
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
            )
        )
        callbacks = tuple(
            _callback_identity(getattr(route, name))
            for name in ("select", "prepared_transform", "catalog_builder", "post_offload_transform", "final_elf_audit")
        )
        policy = json.dumps(route.numeric_policy, sort_keys=True, allow_nan=False)
        if len(policy.encode()) > 65536:
            raise ValueError("descriptor route numerical policy exceeds its byte budget")
        return scalar, str(route.package_dir), objects, callbacks, policy

    @staticmethod
    def _entrypoints():
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.llvmlower import descriptor_result_preparation as P

        return (
            D.emit_translation_unit,
            D._with_descriptor_results,
            D._borrowed_result_wrapper,
            D.descriptor_result_marker,
            B.build_device_objects,
            R.rewrite_descriptor_result_identities,
            R.edit_pipeline,
            R.ensure_registered,
            parse_mlir_text,
            P._records,
            P.implementation_paths,
            P.DescriptorResultPreparation.__init__,
            P.DescriptorResultPreparation._fixed_functions,
            P.DescriptorResultPreparation._feature_values,
            P.DescriptorResultPreparation.selected_feature,
            P.DescriptorResultPreparation.prepare,
            P.DescriptorResultPreparation.revalidate,
            P.DescriptorResultPreparation.retain_upstream,
            P.DescriptorResultPreparation.bind_recipe,
            P.DescriptorResultPreparation.record,
            DeviceDescriptorResults.__init__,
            DeviceDescriptorResults._entrypoints,
            DeviceDescriptorResults._routing_snapshot,
            DeviceDescriptorResults.revalidate,
            DeviceDescriptorResults.bind_sources,
            DeviceDescriptorResults.bind_input,
            DeviceDescriptorResults.bind_shim,
            DeviceDescriptorResults.options,
            DeviceDescriptorResults.consumed,
            DeviceDescriptorResults.identity,
            _selection,
            _callback_identity,
            _build_record,
            _observed,
            _bind,
        )

    def revalidate(self):
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
            self._unit.text != self._unit_text
            or _selection(self._unit) != self._roster
            or any(a is not b for a, b in zip(self.identities, self._unit.result_identities, strict=True))
        ):
            raise ValueError("original emitted descriptor identity changed")
        if self._build is not None and (
            self._build.shim_unit is not self._unit or _build_record(self._build) != self._build_snapshot
        ):
            raise ValueError("original retained device build or object roster changed")

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
            if unit.text.count(D._borrowed_result_wrapper(value.symbol, 2, value.borrowed_symbol)) != 1:
                raise ValueError("generated shim lost its checked descriptor forwarder")
        self._unit, self._roster, self.identities = unit, _selection(unit), unit.result_identities
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

    def bind_input(self, source, sidecar, compilation):
        """Freeze actual prepared IR and routing before device package callbacks."""
        self.revalidate()
        if not self.selected or self._products:
            raise ValueError("descriptor identity needs one original prepared input")
        self._products = tuple(HostLLVMArtifact.capture(path) for path in (source, sidecar))
        self._inputs = self._products
        self._input_bound = True
        self.revalidate()

    def options(self):
        self.revalidate()
        return {} if not self.identities else {"descriptor_result_identities": self.identities}

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
            }
            for v in self.identities
        ]:
            raise ValueError("ordinary lowering did not consume original emitted descriptor identities")
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
                {"descriptor_results": True, "owners": [pin.to_dict() for pin in self._owners]}, sort_keys=True
            ).encode()
        )

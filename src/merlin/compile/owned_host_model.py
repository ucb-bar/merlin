"""Compile original model IR with a fixed host provider and a separate device package.

The service is selected by the trusted caller. Its per-model interface accepts
no host compiler, submitted whole-program LLVM, or submitted host arithmetic.
It uses the ordinary source-to-device routing and saved-model build, retaining
those APIs' capture, linked-ELF, and execution requirements. This is a build
service, not evidence that an experiment has passed its independent gates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from merlin.compile.model_execution_inputs import strict_tree_sha256


@dataclass(frozen=True)
class OwnedHostModelCompiler:
    host_package: Path
    host_package_identity: dict[str, Any]
    target: str
    board_catalog: Path
    board: str
    dts: Path

    @classmethod
    def select(cls, *, host_package, target, board_catalog, board, dts):
        """Bind the trusted host provider once, outside candidate invocation."""
        package = Path(host_package)
        if package.is_symlink():
            raise ValueError("owned host provider must not be a symlink")
        package = package.resolve(strict=True)
        return cls(package, strict_tree_sha256(package), target, Path(board_catalog), board, Path(dts))

    def compile(
        self,
        *,
        capture,
        candidate_package,
        datapath,
        linked_elf_admission,
        output,
        arena_mb,
        timeout_s=300,
        run="none",
        rtl_facts=None,
        reference_file=None,
        tolerance=None,
        math_archive_symbols=None,
        readback="full",
        materialized_inputs=None,
        group_profile=False,
        op_profile=False,
        require_all_accelerator_groups=False,
        connected_spatial=None,
        sibling_source=None,
        independent_heads=None,
        pre_group_layout=None,
        intermediate_storage=None,
        prepack_weights=False,
        share_exact_tensor_preparations=False,
        dense_input_views=False,
        descriptor_results=False,
        descriptor_effects=None,
        host_math_policy="native",
        host_bounded_rne=False,
        host_roundeven=False,
        host_elementwise_fusion=False,
    ):
        """Build original source; the candidate supplies closed device groups only.

        The caller must independently select the accelerator contract and linked
        ELF admission. The resulting ordinary receipt retains the actual source,
        provider, device objects, and complete-output execution evidence.
        An explicit full-source contract selects every independently admitted
        accelerator group; ordinary host-only groups retain their placement.
        An original caller may select the existing typed connected-spatial route;
        normal preparation checks its source pair, ordered readouts, workspace
        ownership and closed helper before the ordinary host/object build.
        ``host_math_policy`` explicitly selects an existing supplied host recipe
        for both ordinary single-forward and later chain composition; numerical
        qualification remains required for the selected arithmetic.
        ``share_exact_tensor_preparations`` selects the existing source-exact
        view/constant preparation pass; no arithmetic or target effects are inferred.
        ``host_bounded_rne`` requests the existing source-proved scalar RV64GC
        late legalizer. It supplies no old observer/effect permissions; missing
        original bounded chains or an unsupported CPU lane refuse normally.
        ``host_roundeven`` selects the existing exact scalar RV64GC/lp64d
        round-even runtime helper. ``host_elementwise_fusion`` selects the
        existing second guarded elementwise fusion pass after generalization.
        Both retain normal CPU/DTS, numerical and actual-product checks.
        ``op_profile`` selects the ordinary typed operation profiler, retaining
        its source table and execution observations. Instrumentation can change
        optimization and scheduling; compare a separate uninstrumented control.
        ``pre_group_layout`` consumes an original software-owned HostGraphPolicy
        before complete group binding, retaining its exact source/report in the
        ordinary compilation recipe. Missing permissions or savings keep source
        unchanged. Head/sibling routes consume it on the checked projected host,
        retaining opaque device-call boundaries. Partial/spatial binding remains
        unsupported until its source correspondence is supplied independently.
        ``intermediate_storage`` consumes the same original policy's explicit
        sequential/successful-allocation/non-reentrant LLVM permissions. It
        reuses only independently checked final allocation lifetimes, after
        source/device binding, retaining unsupported sites and recurrence.
        ``descriptor_results`` selects checked single-result logical shims only
        with the original source-bound ``descriptor_effects`` factory. Descriptor
        equality supplies no noescape, release or completion permission.
        """
        from merlin.compile.baremetal_model import compile_saved_model
        from merlin.compile.route_before_build import plan_before_build
        from merlin.llvmlower.device_offload import BY_GROUP
        from merlin.runtime.host_math import host_math_recipe

        # Original-owner arithmetic choice, validated before routing/building.
        # Changing it requires independent numerical qualification as usual.
        host_math_recipe(host_math_policy)
        if type(host_bounded_rne) is not bool:
            raise ValueError("host bounded RNE must be an explicit Boolean choice")
        for name, value in (("host roundeven", host_roundeven),
                            ("host elementwise fusion", host_elementwise_fusion)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be an explicit Boolean choice")
        if type(op_profile) is not bool:
            raise ValueError("operation profiling must be an explicit Boolean choice")
        if type(dense_input_views) is not bool:
            raise ValueError("dense input views must be an explicit Boolean choice")
        if type(descriptor_results) is not bool:
            raise ValueError("descriptor results must be an explicit Boolean choice")
        if descriptor_results and descriptor_effects is None:
            raise ValueError("descriptor results require the original borrowed-call effects factory")
        if type(share_exact_tensor_preparations) is not bool:
            raise ValueError("tensor preparation sharing must be an explicit Boolean choice")
        if intermediate_storage is not None:
            from merlin.llvmlower.host_graph_policy import HostGraphPolicy

            if type(intermediate_storage) is not HostGraphPolicy or intermediate_storage.target != self.target:
                raise TypeError("intermediate storage needs the original selected target software policy")
            intermediate_storage.revalidate()
        if pre_group_layout is not None:
            from merlin.llvmlower.host_graph_policy import HostGraphPolicy

            if type(pre_group_layout) is not HostGraphPolicy or pre_group_layout.target != self.target:
                raise TypeError("pre-group layout needs the original selected target software policy")
            pre_group_layout.revalidate()
            if not require_all_accelerator_groups or connected_spatial is not None:
                raise ValueError("pre-group layout deferred: unsupported partial or spatial source binding")

        if linked_elf_admission is None:
            raise ValueError("an independently selected linked-ELF admission is required")
        if connected_spatial is not None:
            from merlin.llvmlower.connected_spatial_route import ConnectedSpatialRoute

            if type(connected_spatial) is not ConnectedSpatialRoute:
                raise TypeError("owned model connected lowering requires an original typed spatial route")
            connected_spatial.record()
        if sibling_source is not None:
            from merlin.llvmlower.sibling_source_route import SiblingSourceRoute

            if type(sibling_source) is not SiblingSourceRoute:
                raise TypeError("owned model sibling lowering requires an original typed sibling route")
            sibling_source.record()
            if connected_spatial is not None:
                raise ValueError("original sibling and spatial routes are mutually exclusive")
        if independent_heads is not None:
            from merlin.llvmlower.head_source_route import HeadBatchSourceRoute

            if type(independent_heads) is not HeadBatchSourceRoute:
                raise TypeError("owned head lowering requires its original typed source selection")
            independent_heads.record()
            if sibling_source is not None or connected_spatial is not None:
                raise ValueError("head, sibling and spatial source routes are separately selected")
        candidate = Path(candidate_package)
        if candidate.is_symlink():
            raise ValueError("candidate device package must not be a symlink")
        candidate = candidate.resolve(strict=True)
        if (
            candidate == self.host_package
            or candidate.is_relative_to(self.host_package)
            or self.host_package.is_relative_to(candidate)
        ):
            raise ValueError("candidate device package must be separate from the owned host provider")
        if strict_tree_sha256(self.host_package) != self.host_package_identity:
            raise ValueError("selected owned host provider changed")
        source = Path(capture)
        routed = plan_before_build(
            self.target,
            (source / "model.mlir").read_text(encoding="utf-8"),
            datapath=datapath,
            device_package=str(candidate),
            capture=source / "model.mlir",
            granularity=BY_GROUP,
            linked_elf_admission=linked_elf_admission,
            require_all_accelerator_groups=require_all_accelerator_groups,
        )
        device = routed.get("device_routing")
        if device is None or routed.get("offload") is None:
            raise ValueError(f"no buildable source-bound device route: {routed.get('device_routing_why')}")
        if pre_group_layout is not None or intermediate_storage is not None:
            device = replace(device, pre_group_layout=pre_group_layout, intermediate_storage=intermediate_storage)
        if descriptor_results or descriptor_effects is not None:
            device = replace(device, descriptor_results=descriptor_results, descriptor_effects=descriptor_effects)
        if connected_spatial is not None:
            from merlin.llvmlower.device_build import DeviceRouting

            if type(device) is not DeviceRouting:
                raise TypeError("owned model connected lowering requires ordinary device routing")
            if device.connected_spatial is not None and device.connected_spatial is not connected_spatial:
                raise ValueError("owned model already has a distinct connected spatial selection")
            # Dataclass construction runs the existing route/freeze validation.
            # Source/plan/readout/workspace checks run in normal preparation.
            device = replace(device, connected_spatial=connected_spatial)
        if sibling_source is not None:
            from merlin.llvmlower.device_build import DeviceRouting

            if type(device) is not DeviceRouting:
                raise TypeError("owned model sibling lowering requires ordinary device routing")
            if device.sibling_source is not None and device.sibling_source is not sibling_source:
                raise ValueError("owned model already has a distinct sibling source selection")
            device = replace(device, sibling_source=sibling_source)
        if independent_heads is not None:
            from merlin.llvmlower.device_build import DeviceRouting

            if type(device) is not DeviceRouting or device.sibling_source is not None:
                raise TypeError("owned head lowering needs an unchanged ordinary device route")
            device = replace(device, sibling_source=independent_heads)
        receipt = compile_saved_model(
            capture=source,
            package=self.host_package,
            board_catalog=self.board_catalog,
            board=self.board,
            dts=self.dts,
            output=output,
            target=self.target,
            run=run,
            arena_mb=arena_mb,
            timeout_s=timeout_s,
            device=device,
            rtl_facts=rtl_facts,
            reference_file=reference_file,
            tolerance=tolerance,
            math_archive_symbols=math_archive_symbols,
            readback=readback,
            materialized_inputs=materialized_inputs,
            respect_captured_quantization_scope=True,
            group_profile=group_profile,
            op_profile=op_profile,
            prepack_weights=prepack_weights,
            share_exact_tensor_preparations=share_exact_tensor_preparations,
            dense_input_views=dense_input_views,
            host_math_policy=host_math_policy,
            host_bounded_rne=host_bounded_rne,
            **({"host_roundeven": True} if host_roundeven else {}),
            **({"host_elementwise_fusion": True} if host_elementwise_fusion else {}),
        )
        if strict_tree_sha256(self.host_package) != self.host_package_identity:
            raise ValueError("selected owned host provider changed during compilation")
        return receipt

"""Compile original model IR with a fixed host provider and a separate device package.

The service is selected by the trusted caller. Its per-model interface accepts
no host compiler, submitted whole-program LLVM, or submitted host arithmetic.
It uses the ordinary source-to-device routing and saved-model build, retaining
those APIs' capture, linked-ELF, and execution requirements. This is a build
service, not evidence that an experiment has passed its independent gates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from merlin.runtime.execution.model_inputs import strict_tree_sha256

# ---- owned -------------------------------------------------------------------------------------


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
        quantized_sum=None,
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
        from merlin.compiler.lower.device.offload import BY_GROUP
        from merlin.compiler.planning.model import plan_before_build
        from merlin.model import compile_saved_model
        from merlin.runtime.host_math import host_math_recipe

        # Original-owner arithmetic choice, validated before routing/building.
        # Changing it requires independent numerical qualification as usual.
        host_math_recipe(host_math_policy)
        if type(host_bounded_rne) is not bool:
            raise ValueError("host bounded RNE must be an explicit Boolean choice")
        for name, value in (("host roundeven", host_roundeven), ("host elementwise fusion", host_elementwise_fusion)):
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
            from merlin.compiler.lower.host.graph import HostGraphPolicy

            if type(intermediate_storage) is not HostGraphPolicy or intermediate_storage.target != self.target:
                raise TypeError("intermediate storage needs the original selected target software policy")
            intermediate_storage.revalidate()
        if pre_group_layout is not None:
            from merlin.compiler.lower.host.graph import HostGraphPolicy

            if type(pre_group_layout) is not HostGraphPolicy or pre_group_layout.target != self.target:
                raise TypeError("pre-group layout needs the original selected target software policy")
            pre_group_layout.revalidate()
            if not require_all_accelerator_groups or connected_spatial is not None:
                raise ValueError("pre-group layout deferred: unsupported partial or spatial source binding")

        if quantized_sum is not None:
            from merlin.compiler.lower.device.outline import QuantizedSumRoute

            if type(quantized_sum) is not QuantizedSumRoute:
                raise TypeError("owned model quantized sum needs its original typed source route")
            quantized_sum.record()
            if any(v is not None for v in (connected_spatial, sibling_source, independent_heads)):
                raise ValueError("quantized sum source grouping has a separately selected route")
        if linked_elf_admission is None:
            raise ValueError("an independently selected linked-ELF admission is required")
        if connected_spatial is not None:
            from merlin.compiler.lower.device.spatial import ConnectedSpatialRoute

            if type(connected_spatial) is not ConnectedSpatialRoute:
                raise TypeError("owned model connected lowering requires an original typed spatial route")
            connected_spatial.record()
        if sibling_source is not None:
            from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute

            if type(sibling_source) is not SiblingSourceRoute:
                raise TypeError("owned model sibling lowering requires an original typed sibling route")
            sibling_source.record()
            if connected_spatial is not None:
                raise ValueError("original sibling and spatial routes are mutually exclusive")
        if independent_heads is not None:
            from merlin.compiler.lower.device.head import HeadBatchSourceRoute

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
        if quantized_sum is not None:
            device = replace(device, quantized_sum=quantized_sum)
        if connected_spatial is not None:
            from merlin.compiler.lower.device.device_build import DeviceRouting

            if type(device) is not DeviceRouting:
                raise TypeError("owned model connected lowering requires ordinary device routing")
            if device.connected_spatial is not None and device.connected_spatial is not connected_spatial:
                raise ValueError("owned model already has a distinct connected spatial selection")
            # Dataclass construction runs the existing route/freeze validation.
            # Source/plan/readout/workspace checks run in normal preparation.
            device = replace(device, connected_spatial=connected_spatial)
        if sibling_source is not None:
            from merlin.compiler.lower.device.device_build import DeviceRouting

            if type(device) is not DeviceRouting:
                raise TypeError("owned model sibling lowering requires ordinary device routing")
            if device.sibling_source is not None and device.sibling_source is not sibling_source:
                raise ValueError("owned model already has a distinct sibling source selection")
            device = replace(device, sibling_source=sibling_source)
        if independent_heads is not None:
            from merlin.compiler.lower.device.device_build import DeviceRouting

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


# ---- siblings ----------------------------------------------------------------------------------
# Original captured sibling source owners with private device-only compilation.
#
# The frozen source, placement and ABI choose the host. The candidate compiles only
# an exact raw integer source projection. Every original host quantizer and live
# result remains in the linked original owner. Ordinary capsule numerical, trace,
# object, memory and engine gates are still mandatory; no model credit is granted.

SCHEMA = "merlin.owned_sibling_source.v1"
RECEIPT = "merlin.owned_sibling_source_compilation.v1"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def selected(capsule):
    return (capsule.get("owned_host_source") or {}).get("schema") == SCHEMA


def source_document_fields(selection):
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "source_graph", "plan"}
        or selection["schema"] != SCHEMA
    ):
        raise ValueError("original sibling source needs its complete frozen document selection")
    return ("source_graph", "plan")


def _read(capsule, *, target):
    import yaml

    from merlin.compiler.build.spatial_sources import _local_document
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
    from merlin.compiler.lower.source.region import prepare_sibling_source_host, tensor_abi
    from merlin.development.targetgen.execution.host.frozen_placement import read as placement

    root = Path(capsule["__dir__"]).absolute()
    if root != root.resolve() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("original source owner must be a direct capsule custody directory")
    declaration_path = root / "capsule.yaml"
    if declaration_path.is_symlink() or not declaration_path.is_file():
        raise ValueError("original sibling source declaration is absent or indirect")
    raw = declaration_path.read_bytes()
    declaration = yaml.safe_load(raw)
    for key in (
        "name",
        "kind",
        "label",
        "inputs",
        "operation",
        "numeric_policy",
        "expected",
        "semantic",
        "interface_mlir",
        "linalg_mlir",
        "required_oracle_tiers",
        "owned_host_source",
        "software_screen",
        "component_program",
        "integer_partial_sum_bound",
    ):
        if declaration.get(key) != capsule.get(key):
            raise ValueError("original source owner differs from its frozen capsule declaration")
    selection = declaration["owned_host_source"]
    paths = [_local_document(root, selection[k]) for k in source_document_fields(selection)]
    source_path, source_bytes = paths[0]
    plan_path, plan_bytes = paths[1]
    interface = declaration["interface_mlir"]
    if Path(interface).name != interface or declaration.get("linalg_mlir") != interface:
        raise ValueError("original source owner requires its complete local linalg program")
    interface_path = root / interface
    if interface_path.is_symlink() or interface_path.read_bytes() != source_bytes:
        raise ValueError("original source graph differs from its frozen complete public interface")
    frozen = placement(declaration, root)
    plan = json.loads(plan_bytes)
    if (
        type(plan) is not dict
        or set(plan) != {"schema", "source_sha256", "preparation", "borrowed_effects"}
        or plan["schema"] != "merlin.original_sibling_source_plan.v1"
        or plan["source_sha256"] != _sha(source_bytes)
    ):
        raise ValueError("original sibling source plan differs from its exact source")
    effects = BorrowedBufferEffects(**plan["borrowed_effects"])
    source = parse_mlir_text(source_bytes.decode())
    prepared = prepare_sibling_source_host(source, effects=effects)
    if prepared.receipt != plan["preparation"]:
        raise ValueError("original source preparation differs from its frozen derived source plan")
    ins, outs = tensor_abi(source)
    if declaration["inputs"] != list(ins):
        raise ValueError("original source inputs differ from the frozen complete signature")
    program = declaration.get("component_program")
    if program != {
        "schema": "merlin.original_ssa_program.v1",
        "source_sha256": _sha(source_bytes),
        "outputs": list(outs),
    }:
        raise ValueError("original source outputs differ from the frozen complete result binding")
    # Reopen independent source eligibility. No emitted field may create a
    # placement, dtype or supporting-device grant. This initial route supports
    # exactly the source's signed integer sibling roots as accelerator work.
    from merlin.development.targetgen.execution.native.model_policy import (
        census as independent_frozen_source_eligibility,
    )

    certificate = independent_frozen_source_eligibility({**declaration, "__dir__": str(root)}, target=target)
    eligible = [row for row in certificate.get("regions", []) if row.get("target_eligible") is True]
    if (
        certificate.get("source_mlir_sha256") != _sha(source_bytes)
        or certificate.get("n_eligible") != len(prepared.receipt["outline"]["source_geometries"])
        or len(eligible) != certificate.get("n_eligible")
        or any(
            row.get("semantic_family") != "contraction"
            or row.get("carrier_op") not in {"linalg.generic", "linalg.matmul"}
            for row in eligible
        )
    ):
        raise ValueError("original source eligibility differs from its complete exact sibling projection")
    # Frozen and fresh admission must name the same accelerator carrier census.
    from collections import Counter

    actual = dict(Counter(row["carrier_op"] for row in eligible))
    if actual != frozen["accelerator_operations"]:
        raise ValueError("independent source eligibility differs from frozen accelerator placement")
    pins = [(declaration_path, raw), (interface_path, source_bytes), *paths]

    def validate():
        if any(path.is_symlink() or path.read_bytes() != payload for path, payload in pins):
            raise ValueError("original source owner inputs changed during compiler execution")

    validate()
    return prepared, declaration, pins, validate, certificate


def compiler_input(capsule, generated, *, target):
    """Derive the private compiler interface from the exact frozen original SSA."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import StringAttr

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.target.contract.interfaces.linalg import bind_logical_io_metadata

    prepared, declaration, pins, validate, certificate = _read(capsule, target=target)
    core = prepared.core.clone()
    f = next(op for op in core.body.block.ops if isinstance(op, func.FuncOp))
    f.sym_name = StringAttr("forward")
    core.attributes["prov.level"] = StringAttr("linalg-on-tensors")
    ins = [
        {
            "name": "core_arg" + str(i),
            "shape": list(t.get_shape()),
            "dtype": str(t.element_type),
            "role": "input" if i == 0 else "weight",
        }
        for i, t in enumerate(f.function_type.inputs)
    ]
    outs = [
        {"name": "core_output" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
        for i, t in enumerate(f.function_type.outputs)
    ]
    interface, binding = bind_logical_io_metadata(text(core, generic=True), ins, outs)
    root = Path(generated) / "original-source-projection"
    root.mkdir(parents=True, exist_ok=False)
    path = root / "interface.mlir"
    path.write_text(interface)
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "merlin.original_sibling_source_projection.v1",
                "source_sha256": _sha(pins[1][1]),
                "preparation": prepared.receipt,
                "binding": binding,
                "interface_sha256": _sha(interface.encode()),
                "full_source_outputs": declaration["component_program"]["outputs"],
                "independent_source_eligibility": certificate,
                "execution": False,
            },
            indent=2,
        )
        + "\n"
    )
    validate()
    return path, validate


def full_command_buffer(capsule, private):
    """Retain the accelerator projection while publishing original host pointers."""
    from copy import deepcopy

    from merlin.compiler.build.spatial_sources import _local_document
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.lower.source.region import tensor_abi

    path, raw = _local_document(Path(capsule["__dir__"]), capsule["owned_host_source"]["source_graph"])
    ins, outs = tensor_abi(parse_mlir_text(raw.decode()))
    if set(private.get("tensors", {})).intersection(row["name"] for row in (*ins, *outs)):
        raise ValueError("private source projection aliases the original public pointer names")
    cb = {"abi_version": private["abi_version"], "target": private["target"], "tensors": {}, "commands": []}
    cb["tensors"].update({row["name"]: {k: v for k, v in row.items() if k != "name"} for row in ins})
    cb["tensors"].update(
        {row["name"]: {"shape": row["shape"], "dtype": row["dtype"], "role": "output"} for row in outs}
    )
    cb["outputs"] = [row["name"] for row in outs]
    cb["kernel_abi"] = {
        "kind": "whole_program",
        "args": [
            {"tensor": row["name"], "access": "read" if i < len(ins) else "write"}
            for i, row in enumerate((*ins, *outs))
        ],
        "outputs": cb["outputs"],
    }
    cb.setdefault("params", {})["source_owned_host_projection"] = {
        "schema": SCHEMA,
        "source_sha256": _sha(raw),
        "device_projection_only": True,
        "host_work_volume": "unknown; original host source retained separately",
        "device_command_buffer": deepcopy(private),
    }
    return cb


def emit_selected(capsule, candidate_llvm, command_buffer, output, *, target, timeout, candidate_package):
    """Compile the independently frozen host around the actual closed helper."""
    from dataclasses import asdict

    from xdsl.dialects import func, llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.parser import Parser

    from merlin.compiler.build.spatial_sources import normalize_unused_void_asm_results
    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.ir.llvm_context import make_llvm_context
    from merlin.compiler.lower.device.device_shim import LogicalBuffer, emit_logical_translation_unit
    from merlin.compiler.lower.host.arena import heap_demand
    from merlin.compiler.lower.host.borrowed_buffer_calls import from_writers
    from merlin.compiler.lower.host.builder import PTR, FnBuilder, append_void_region
    from merlin.compiler.lower.model import codegen, target_data_layout, toolchain
    from merlin.compiler.lower.model.compilation_recipe import CompilationRecipe
    from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir
    from merlin.compiler.lower.source.region import emit_ranked_source_entry
    from merlin.runtime.backends.registry import harness_build_recipe
    from merlin.target.contract.interfaces.abi import for_target

    prepared, declaration, pins, validate, certificate = _read(capsule, target=target)
    if command_buffer.get("target") != target:
        raise ValueError("private source compiler returned another target")
    output = Path(output).absolute()
    if output != output.resolve() or any(p.is_symlink() for p in output.parents):
        raise ValueError("source host output must be a direct compilation owner")
    output.mkdir(parents=True, exist_ok=False)
    # Re-derive the exact interface used by the ordinary source projection.
    projected, project_validate = compiler_input(capsule, output, target=target)
    callee = next(op for op in prepared.core.body.block.ops if isinstance(op, func.FuncOp))
    expected = {
        "inputs": [
            {"name": "core_arg" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
            for i, t in enumerate(callee.function_type.inputs)
        ],
        "outputs": [
            {"name": "core_output" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
            for i, t in enumerate(callee.function_type.outputs)
        ],
    }
    input_names = [r["name"] for r in expected["inputs"]]
    output_names = [r["name"] for r in expected["outputs"]]
    expected_args = [{"tensor": n, "access": "read"} for n in input_names] + [
        {"tensor": n, "access": "write"} for n in output_names
    ]
    if command_buffer.get("kernel_abi") != {"kind": "whole_program", "args": expected_args, "outputs": output_names}:
        raise ValueError("actual private helper differs from the source-derived complete device ABI")
    for row in (*expected["inputs"], *expected["outputs"]):
        spec = command_buffer.get("tensors", {}).get(row["name"], {})
        if spec.get("shape") != row["shape"] or spec.get("dtype") != row["dtype"]:
            raise ValueError("actual private helper changes a source tensor format or shape")
    recipe = CompilationRecipe(output, producer=Path(__file__))
    for path, _ in pins:
        recipe.bind_preparation(path.name, path)
    candidate_source = output / "candidate-input.llvm.mlir"
    candidate_source.write_text(candidate_llvm)
    recipe.bind_preparation("candidate_llvm", candidate_source)
    source = output / "original-host.mlir"
    source.write_text(text(prepared.module, generic=True) + "\n")
    recipe.bind_preparation("prepared_source", source)
    selected_sha = _sha(candidate_llvm.encode())

    def unchanged():
        validate()
        project_validate()
        if _sha(candidate_source.read_bytes()) != selected_sha:
            raise ValueError("selected candidate helper changed during host compilation")

    roster = from_writers(
        source,
        (prepared.writer,),
        tuple((path, _sha(raw)) for path, raw in pins) + ((candidate_source, selected_sha),),
        unchanged,
    )

    def run(argv, inputs, dest):
        recipe.run(
            argv,
            runner=lambda args: codegen._run(args, inputs=inputs, outputs=(dest,), timeout_s=timeout),
            inputs=inputs,
            output=dest,
        )

    abi = for_target(target)
    build = harness_build_recipe(target).with_effective_abi()
    if not abi.fence_symbol:
        raise ValueError("source host requires the selected separate completion ABI")
    layout = target_data_layout.observe_index_width(
        toolchain.clang(), [codegen.RISCV_FLAGS[0], build.march(), build.mabi()]
    )
    if layout["index_bits"] != 64:
        raise ValueError("selected source transport requires independently observed 64-bit descriptors")
    host_ll = output / "host.ll"
    host_ll.write_text(
        lower_to_llvm_ir(
            source.read_text(),
            workdir=output / "host-lowering",
            timeout=timeout,
            hoist_static_allocs=False,
            index_bits=64,
            data_layout=layout["data_layout"],
            borrowed_buffer_calls=roster,
        )
    )
    callee = next(op for op in prepared.core.body.block.ops if isinstance(op, func.FuncOp))
    types = (*callee.function_type.inputs, *callee.function_type.outputs)
    names = (*input_names, *output_names)
    buffers = tuple(
        LogicalBuffer(
            name, "input" if i < len(input_names) else "output", name, tuple(t.get_shape()), str(t.element_type)
        )
        for i, (name, t) in enumerate(zip(names, types, strict=True))
    )
    shim = emit_logical_translation_unit(
        target,
        {
            "original_source_transport": (
                buffers,
                [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers],
            )
        },
        kernel_symbol_for=lambda _: "original_source_device",
    )
    if shim.skipped or shim.symbols != ("original_source_transport",):
        raise ValueError("ordinary descriptor transport did not cover the complete source core")
    n = len(types)
    bridge = (
        shim.text
        + "\nvoid _mlir_ciface_"
        + prepared.writer.borrowed_symbol
        + "("
        + ",".join("const merlin_lmemref_2d*d" + str(i) for i in range(n))
        + ") {original_source_transport_results result;_mlir_ciface_original_source_transport(&result,"
        + ",".join("d" + str(i) for i in range(n))
        + ");}\n"
    )
    producer = Parser(make_llvm_context(), candidate_llvm).parse_module()
    llvm_inputs = [host_ll]
    region_receipts = {}
    for original, renamed, nargs in (
        (abi.entry_symbol, "original_source_device", n),
        (abi.fence_symbol, abi.fence_symbol, 0),
    ):
        fs = [f for f in producer.body.block.ops if isinstance(f, llvm.FuncOp) and f.sym_name.data == original]
        if len(fs) != 1 or fs[0].function_type != llvm.LLVMFunctionType([PTR] * nargs, llvm.LLVMVoidType()):
            raise ValueError("actual authored helper/completion differs from the selected pointer ABI")
        b = FnBuilder([PTR] * nargs)
        region_receipts[renamed] = append_void_region(b, fs[0], b.entry.args)
        m = ModuleOp([llvm.FuncOp(renamed, fs[0].function_type, linkage=llvm.LinkageAttr("external"), body=b.finish())])
        m.verify()
        p, ll = output / (renamed + ".mlir"), output / (renamed + ".ll")
        p.write_text(text(m) + "\n")
        run([toolchain.mlir_translate(), "--mlir-to-llvmir", p, "-o", ll], (p,), ll)
        llvm_inputs.append(ll)
    # The selected compiler's headers are software ABI, not accelerator facts.
    # Use the original harness compiler's declared sysroot observation.
    from merlin.runtime.backends.baremetal.image_link import _mlir_runtime_compiler

    flags = [*codegen.RISCV_FLAGS[1:], build.march(), build.mabi()]
    compiler = _mlir_runtime_compiler(toolchain.clang(), build.compiler, flags)
    for name, code in (
        ("descriptor-transport", bridge),
        ("original-entry", emit_ranked_source_entry(prepared.source, entry_symbol=abi.entry_symbol)),
    ):
        p, ll = output / (name + ".c"), output / (name + ".ll")
        p.write_text(code)
        run([*compiler, "-S", "-emit-llvm", p, "-o", ll], (p,), ll)
        llvm_inputs.append(ll)
    # Exact existing shared software runtime, privately selected by the source
    # owner. Its bounded storage is a BSS object, never a guessed board address.
    from merlin.common.paths import runtime_dir

    runtime = runtime_dir()
    allocator = runtime / "baremetal/spike/merlin_malloc.c"
    math_runtime = runtime / "abi/mlir_runtime.c"
    demand = heap_demand([host_ll.read_text()])
    if demand.unbounded:
        raise ValueError("source host has an unbounded allocation demand")
    budget = ((demand.bytes + demand.allocations * 128 + 63) // 64) * 64
    if not 0 < budget < 1 << 63:
        raise ValueError("source host needs a bounded nonempty software workspace")
    runtime_c, runtime_ll = output / "source-runtime.c", output / "source-runtime.ll"
    runtime_c.write_text(
        "#include <stdint.h>\nstatic unsigned char original_source_arena["
        + str(budget)
        + "] __attribute__((aligned(64)));\n"
        + "#define MERLIN_ARENA_BASE_ADDR ((uintptr_t)original_source_arena)\n"
        + "#define MERLIN_ARENA_SIZE_BYTES "
        + str(budget)
        + 'ULL\n#include "'
        + str(allocator)
        + '"\n#include "'
        + str(math_runtime)
        + '"\n'
    )
    run(
        [
            *compiler,
            *[token for path in build.include_roots for token in ("-I", str(path))],
            "-S",
            "-emit-llvm",
            runtime_c,
            "-o",
            runtime_ll,
        ],
        (runtime_c, allocator, math_runtime),
        runtime_ll,
    )
    linked = output / "original-linked.ll"
    run([toolchain.llvm_install() / "bin/llvm-link", "-S", *llvm_inputs, "-o", linked], tuple(llvm_inputs), linked)
    resolved = output / "original-runtime-linked.ll"
    run(
        [toolchain.llvm_install() / "bin/llvm-link", "-S", linked, "--only-needed", runtime_ll, "-o", resolved],
        (linked, runtime_ll),
        resolved,
    )
    imported = output / "original-host.stock-import.mlir"
    run(
        [
            toolchain.mlir_translate(),
            "--import-llvm",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            resolved,
            "-o",
            imported,
        ],
        (resolved,),
        imported,
    )
    final = Parser(make_llvm_context(), imported.read_text()).parse_module()
    void_results = normalize_unused_void_asm_results(final)
    artifact = output / "original-host.llvm.mlir"
    artifact.write_text(text(final, generic=True) + "\n")
    verified = output / "original-host.verified.ll"
    run([toolchain.mlir_translate(), "--mlir-to-llvmir", artifact, "-o", verified], (artifact,), verified)
    unchanged()
    recipe.completed_product(artifact, kind="llvm_ir")
    report = {
        "schema": RECEIPT,
        "source_sha256": _sha(pins[1][1]),
        "artifact_sha256": _sha(artifact.read_bytes()),
        "candidate_llvm_sha256": selected_sha,
        "preparation": prepared.receipt,
        "writer": asdict(prepared.writer),
        "source_eligibility": certificate,
        "closed_regions": region_receipts,
        "workspace": {
            "heap_demand": demand.to_dict(),
            "arena_bytes": budget,
            "address": "ordinary linked aligned BSS",
            "reentrant": False,
        },
        "runtime": {str(p): _sha(p.read_bytes()) for p in (allocator, math_runtime)},
        "index_layout": layout,
        "unused_void_asm_results_normalized": void_results,
        "target_execution": False,
        "whole_model_gate_credit": False,
        "qualification": (
            "source/host compilation only; ordinary original-output numerical, trace, memory "
            "and engine gates remain required"
        ),
    }
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return artifact.read_text(), report

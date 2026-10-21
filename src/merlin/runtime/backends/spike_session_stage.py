"""One session stage through the ordinary preparation and device-object APIs."""

from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.llvmlower import device_preparation as D
from merlin.llvmlower import session_bundle, session_objects
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.device_build import (
    freeze_device_elf_admission,
    freeze_device_ir_budget,
    freeze_device_target_configuration,
)
from merlin.llvmlower.lower import lower_model_file

from . import spike_model as M
from . import zephyr_model as Z


def freeze_routes(devices, count):
    """Require a single original final-image owner across all selected stages."""
    from merlin.llvmlower.device_build import DeviceRouting

    if type(devices) is not tuple or len(devices) != count:
        raise ValueError("session requires one original routing or None per stage")
    frozen, owner = [], None
    for device in devices:
        if device is not None and type(device) is not DeviceRouting:
            raise ValueError("session device routes require original DeviceRouting values")
        if device is not None and device.descriptor_results:
            raise ValueError("session descriptor-result forwarding is not implemented")
        admission = freeze_device_elf_admission(device)
        if D.device_route_requested(device):
            if admission is None:
                raise ValueError("selected session device stage requires original final ELF admission")
            current = admission[:2]
            if owner is not None and (owner[0] != current[0] or owner[1] is not current[1]):
                raise ValueError("session stages need the same target and exact live final ELF admission service")
            owner = current
        frozen.append(
            dict(
                admission=admission,
                offload=D.freeze_device_offload_selection(device),
                configuration=freeze_device_target_configuration(device),
                budget=freeze_device_ir_budget(device),
            )
        )
    return tuple(frozen)


def build_stage(program, row, work, *, device, frozen, plan, features, int8_compute, gcc, check_current):
    """Compile actual prepared source and partial-link before helper namespacing."""
    work.mkdir(parents=True)
    compilation = CompilationRecipe(work, producer=Path(__file__))
    original = program.bundle / "model.mlir"
    prepared = original
    effective = features
    if int8_compute or features or D.device_route_requested(device):
        prepared, effective = Z.prepare_for_lowering(
            original,
            work,
            int8_compute=int8_compute,
            features=features,
            device=device,
            device_offload_invocation=frozen["offload"],
        )
    check_current()
    selected = M.selected_model_compiler_plan(
        backend="scalar", cflags_override=plan["clang_cflags"], vlen=None, features=effective
    )
    abi = ("compiler_resolved", "compiler_sha256", "data_layout", "index_bits")
    if (
        any(selected["observation"][key] != plan["observation"][key] for key in abi)
        or selected["gcc_cflags"] != plan["gcc_cflags"]
    ):
        raise ValueError("session stage preparation requires a different original compiler/runtime ABI")
    plan = selected
    inputs, outputs = session_bundle.forward_signature(prepared)
    if tuple(inputs) != program.inputs or tuple(outputs) != program.outputs:
        raise ValueError("session stage preparation changed the captured public tensor ABI")
    D.retain_device_offload_source(frozen["offload"], prepared, compilation=compilation)
    renamed = work / "model.prepared.renamed.mlir"
    renamed.write_text(session_bundle.rename_forward(prepared.read_text(), row["entrypoint"]))
    compilation.bind_preparation(f"stage_{row['index']}_prepared", prepared)
    compilation.bind_preparation(f"stage_{row['index']}_renamed", renamed)
    observation = plan["observation"]
    lowered = lower_model_file(
        renamed,
        work / "lower",
        targets=(),
        textual=False,
        hoist_static_allocs=False,
        features=effective,
        data_layout=observation["data_layout"],
        index_bits=observation["index_bits"],
    )
    index = lowered.stats.get("index_lowering", {})
    if (index.get("index_bits"), index.get("data_layout")) != (observation["index_bits"], observation["data_layout"]):
        raise ValueError("session stage lowering lost the original compiler/index selection")
    clang = Path(observation["compiler_resolved"])
    compilation.run(
        [clang, M.CLANG_TARGET, *plan["model_cflags"], "-c", lowered.ll_path, "-o", work / "model.o"],
        runner=M._run,
        inputs=[lowered.ll_path],
        output=work / "model.o",
    )
    pins = {path: sha256_file(path) for path in (original, prepared, renamed, lowered.ll_path, work / "model.o")}
    check_current()
    sidecar, arguments, built, artifacts = D.build_prepared_device_objects(
        device,
        work,
        offload_selection=frozen["offload"],
        device_admission_selection=frozen["admission"],
        target_configuration_selection=frozen["configuration"],
        device_ir_budget=frozen["budget"],
        shim_cflags=lambda: M._mlir_runtime_compiler(clang, gcc, plan["clang_cflags"])[1:],
    )
    check_current()
    _reopen(pins)
    objects = [work / "model.o", *(built.objects if built is not None else ())]
    combined = work / "combined.o"
    compilation.run(
        [gcc.with_name("riscv64-unknown-elf-ld"), "-r", *objects, "-o", combined],
        runner=M._run,
        inputs=objects,
        output=combined,
    )
    pins[combined] = sha256_file(combined)
    _reopen(pins)
    selected = work / "namespaced.o"
    tools = clang.parent
    session_objects.namespace_stage_object(
        combined,
        selected,
        index=row["index"],
        entrypoint=row["entrypoint"],
        nm=tools / "llvm-nm",
        objcopy=tools / "llvm-objcopy",
        runner=M._run,
        compilation=compilation,
    )
    _reopen(pins)
    check_current()
    _reopen(pins)
    compilation.completed_product(selected, kind="object")
    return dict(
        object=selected,
        arguments=arguments,
        artifacts=artifacts,
        source=prepared,
        llvm=lowered.ll_path,
        recipe=compilation.path,
        compiler_plan=plan,
    )


def _reopen(pins):
    for path, digest in pins.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("session stage original source or produced host object changed")

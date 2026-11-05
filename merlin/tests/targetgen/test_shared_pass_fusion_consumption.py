"""Neutral ordinary target-IR fusion; no generated target/hardware qualification."""

import ctypes
import inspect
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import test_shared_pass_fusion_provider as F
from test_readout_model_fusion import inputs_and_reference, reference_outputs

from merlin.common.digest import sha256_file
from merlin.llvmlower.compilation_recipe import (
    CompilationRecipe,
    verify_completed_recipe,
)
from merlin.targetgen import capsule_common as C
from merlin.targetgen import package_runtime as P
from merlin.targetgen.shared_pass_contract import shared_pass_requirement

DRIVER = r"""
import json,sys
from pathlib import Path
import test_shared_pass_fusion_provider as F
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from dataclasses import replace
command,source,*rest=sys.argv[1:]
source=Path(source)
if command=='parse':
    assert source.read_bytes()==Path('original.target.mlir').read_bytes()
elif command=='lower_interface_to_target':
    sys.stdout.write(source.read_text())
elif command=='emit_target_bundle':
    configuration=json.loads(Path('fixture.json').read_text())
    selected=read_scheduling_contracts(*configuration['paths'],target='neutral')
    # Only selected ISA constants/calls are parsed; fixture model geometry is
    # original host-selected and separately checked against the complete ABI.
    from test_readout_model_fusion import graph
    original=graph(*configuration['shape'],configuration['width'],configuration['separate'])
    adapter=replace(original[3],configuration_identity=selected.hardware_identity)
    codec=replace(original[5],program_decoder=replace(original[5].program_decoder,configuration_identity=selected.hardware_identity))
    model=F.place_source_model((selected,original[0],original[1],adapter,original[4],codec,original[6],original[7]),configuration['shape'],configuration['width'],configuration['separate'])
    text=source.read_text()
    if configuration.get('ignore_selected'):
        text,_=F.transform(Path('original.target.mlir').read_text(),model,enabled=False)
    image=F.image_from_target(text,model)
    if configuration.get('wrong_operand'):
        image=bytearray(image);image[(len(image)//F.WORD.size-1)*F.WORD.size+1+4*F.KEYS.index('bias')]^=1;image=bytes(image)
    Path(rest[0]).write_bytes(Path('buffer.json').read_bytes())
    sys.stdout.write(F.llvm_artifact(image))
else:
    raise RuntimeError('ordinary selected consumer must not bypass target IR')
"""


def package(directory, model, shape, width, separate, *, fault=None):
    directory.mkdir()
    selected, kernel, *_ = model
    (directory / "driver.py").write_text(DRIVER)
    (directory / "original.target.mlir").write_text(F.target_text(kernel, selected))
    (directory / "fixture.json").write_text(
        json.dumps(
            dict(
                paths=[str(selected.facts_path), str(selected.software_path)],
                shape=shape,
                width=width,
                separate=separate,
                **({fault: True} if fault else {}),
            )
        )
    )
    tensor = {
        arg.name: dict(
            shape=list(arg.shape),
            dtype=arg.dtype,
            role="input" if arg.access == "read" else "output",
        )
        for arg in kernel.args
    }
    buffer = dict(
        abi_version="0.1",
        target="neutral",
        commands=[],
        tensors=tensor,
        kernel_abi=dict(
            kind="whole_program",
            args=[dict(tensor=arg.name, access=arg.access) for arg in kernel.args],
            outputs=[arg.name for arg in kernel.args if arg.access == "write"],
        ),
    )
    (directory / "buffer.json").write_text(json.dumps(buffer))
    requirement = shared_pass_requirement(version=2, required_supported=("joint_layout_fusion",))
    from merlin.targetgen.generate_prompt import shared_pass_manifest_template

    manifest = shared_pass_manifest_template(requirement)
    contract = manifest["shared_pass_contract"]
    for boundary in contract["boundaries"].values():
        boundary["contracts"].update(
            semantics=model[3].numerical_policy_identity,
            effects="neutral-fixed-byte-instruction-executor-v1",
            capabilities=selected.hardware_identity,
            resources=selected.hardware_identity,
            completion="neutral-synchronous-call-return-v1",
        )
    family = next(row for row in contract["families"] if row["name"] == "joint_layout_fusion")
    family.update(
        status="supported",
        boundary="device",
        reason="Neutral contiguous readout/epilogue physical composition; other layouts require separate qualification",
    )
    contract["controls"]["default_policy"] = ["joint_layout_fusion"]
    contract["controls"]["choices"]["joint_layout_fusion"]["enabled"] = [False, True]
    manifest.update(
        target="neutral",
        artifact_type="mlir_oot_target_backend",
        language="python",
        authoring=dict(mode="hand_curated"),
        integrity_exempt=False,
        entrypoints=dict(tool="driver.py"),
    )
    for name in (
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_target_artifact",
        "lower_target_to_llvm",
        "emit_target_bundle",
    ):
        manifest["commands"][name] = dict(
            argv=[
                sys.executable,
                "-B",
                "driver.py",
                name,
                "{input_mlir}",
                *(["{output_json}"] if name in ("emit_command_buffer", "emit_target_bundle") else []),
            ]
        )
    (directory / "manifest.yaml").write_text(json.dumps(manifest))
    return P.load_package(directory, shared_pass_requirement=requirement), requirement


def guarded_execute(image, model, shape, width, separate, *, numeric_fault=False):
    """The loaded model carries real output bytes through three state epochs."""
    data, _ = inputs_and_reference(*shape, width)
    kernel = model[1]
    sizes = {arg.name: arg.nbytes for arg in kernel.args}
    guards = []

    def array(payload):
        raw = (ctypes.c_uint8 * (len(payload) + 32))(*([193] * (len(payload) + 32)))
        view = (ctypes.c_uint8 * len(payload)).from_buffer(raw, 16)
        view[:] = payload
        guards.append((raw, len(payload)))
        return view

    local0 = array(bytes([163]) * 1024)
    local1 = array(bytes([163]) * 1024)
    local2 = array(bytes([163]) * 1024) if separate else local1
    buffers = [local0, local1, local2] + [
        array(data[name].tobytes() if name in data else bytes([91]) * sizes[name]) for name in F.SPACES[3:]
    ]
    bytep = ctypes.POINTER(ctypes.c_uint8)
    pointers = (bytep * 10)(*(ctypes.cast(b, bytep) for b in buffers))
    capacities = (ctypes.c_uint64 * 10)(1024, 1024, 1024, *(sizes[name] for name in F.SPACES[3:]))
    initial = bytes((3 * i + 127) % 256 for i in range(sizes["y"]))
    state = array(initial)
    position = array((7).to_bytes(4, "little"))
    hidden = array(bytes(3 * sizes["hidden"]))
    output = array(bytes(3 * sizes["y"]))
    counts = array(bytes(24))
    image.fixture_model.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int]
    image.fixture_model.restype = ctypes.c_int
    assert (
        image.fixture_model(
            pointers,
            capacities,
            state,
            position,
            hidden,
            output,
            counts,
            int(numeric_fault),
        )
        == 0
    )
    expected_hidden = []
    expected_output = []
    carry = initial
    mask = (1 << (8 * width)) - 1
    for epoch in range(7, 10):
        original = data["x"].reshape(-1)
        values = [(int(value) + carry[i % len(carry)] + epoch) & mask for i, value in enumerate(original)]
        signed = [value - (mask + 1) if value > (mask // 2) else value for value in values]
        current = dict(
            data,
            x=np.asarray(signed, dtype="i1" if width == 1 else "<i2").reshape(data["x"].shape),
        )
        expected = reference_outputs(current)
        expected_hidden.append(expected["hidden"])
        expected_output.append(expected["y"])
        carry = expected["y"]
    assert bytes(hidden) == b"".join(expected_hidden), "actual complete hidden history differs from original arithmetic"
    assert bytes(output) == b"".join(expected_output), "actual complete output history differs from original arithmetic"
    assert bytes(state) == carry and int.from_bytes(position, "little") == 10
    for index, name in enumerate(F.SPACES[3:], 3):
        if name in data:
            assert bytes(buffers[index]) == data[name].tobytes()
    assert all(bytes(raw[:16]) == bytes(raw[n + 16 :]) == bytes([193]) * 16 for raw, n in guards)
    return dict(
        hidden=bytes(hidden).hex(),
        output=bytes(output).hex(),
        state=bytes(state).hex(),
        position=int.from_bytes(position, "little"),
        counts=list((ctypes.c_uint64 * 3).from_buffer(counts)),
        initial_state=initial.hex(),
        readonly_before={name: data[name].tobytes().hex() for name in data},
        readonly_after={name: bytes(buffers[F.SPACES.index(name)]).hex() for name in data},
        expected=dict(
            hidden=b"".join(expected_hidden).hex(),
            output=b"".join(expected_output).hex(),
            state=carry.hex(),
            position=10,
        ),
        guards=[dict(prefix=bytes(raw[:16]).hex(), suffix=bytes(raw[n + 16 :]).hex()) for raw, n in guards],
    )


def test_fixture_parser_fusion_and_actual_llvm_image_join(tmp_path):
    model = F.fixture(tmp_path, (3, 7, 5, 4), 1, False)
    source = F.target_text(model[1], model[0])
    selected, result = F.transform(source, model)
    assert len(result.kernel.body) == 5
    image = F.image_from_target(selected, model)
    assert F.verify_artifact(selected, F.llvm_artifact(image), model) == image
    # A matching dead global cannot qualify a function that uses another one.
    with pytest.raises(AssertionError):
        F.verify_artifact(selected, F.llvm_artifact(image, wrong_pointer=True), model)


def test_missing_physical_route_retains_original_complete_calls(tmp_path):
    model = F.fixture(tmp_path, (2, 5, 11, 3), 2, True)
    source = F.target_text(model[1], model[0])
    selected, result = F.transform(source, model, physical=False)
    assert len(result.kernel.body) == 7 and sum(c.instr == "readout" for c in result.kernel.body) == 2
    assert not result.report["storage"].get("fusion", {}).get("checked_steps")


def build_arm(
    tmp_path,
    shape,
    width,
    separate,
    *,
    enabled,
    fault=None,
    physical=True,
    numeric_fault=False,
    use_schedule=True,
):
    from merlin.common.invocation_record import verify
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.target_schedule import TargetScheduleSelection

    tools = (
        Path(toolchain.mlir_translate()),
        Path(toolchain.clang()),
        Path("/usr/bin/time"),
    )
    if not all(path.is_file() for path in tools):
        pytest.skip("selected native translator/compiler required")
    facts = tmp_path / "facts"
    facts.mkdir()
    model = F.fixture(facts, shape, width, separate)
    selected, kernel, objects, adapter, cap, encoding, limits, options = model
    pkg, requirement = package(tmp_path / "package", model, shape, width, separate, fault=fault)
    source = tmp_path / "source.mlir"
    source.write_text(F.target_text(kernel, selected))
    runtime = tmp_path / "runtime.c"
    runtime.write_text(F.recurrent_runtime(model))
    owners = [
        Path(inspect.getsourcefile(function))
        for function in (
            build_arm,
            guarded_execute,
            F.fixture,
            F.verify_artifact,
            F.read_target,
            F.recurrent_runtime,
            F.inputs,
            F.compile_case,
            F.params,
            inputs_and_reference,
            reference_outputs,
            C.lower_interface,
            TargetScheduleSelection,
        )
    ]
    files = [
        source,
        runtime,
        Path(C.__file__).resolve(),
        *tools,
        selected.facts_path,
        selected.software_path,
        *pkg.directory.iterdir(),
        *owners,
    ]
    before = {str(path): sha256_file(path) for path in files if path.is_file()}
    selection = (
        TargetScheduleSelection(
            selected,
            objects,
            adapter,
            encoding,
            limits,
            parameters=dict(
                budgets=ScheduleBudgets(max_fusions=8 if enabled else 0),
                fusion_emitters={cap.readout.selector: cap.emit} if physical else None,
                fusion_qualification_identity="independent-neutral-two-layer-v1",
                **options,
            ),
        )
        if use_schedule
        else None
    )
    generated = tmp_path / "build"
    compile_started = time.perf_counter()
    _, artifact = C.lower_interface(
        pkg,
        source,
        generated,
        contract=None,
        timeout=30,
        target_schedule=selection,
        shared_pass_requirement=requirement,
    )
    receipt = generated / ("target_schedule.json" if use_schedule else "shared_pass_target.json")
    receipt_sha = sha256_file(receipt)
    selected_text = (generated / "selected.target.mlir").read_text()
    payload = F.artifact_image(artifact)
    mlir, ll, obj, library = [generated / name for name in ("lowered.llvm.mlir", "kernel.ll", "kernel.o", "model.so")]
    recipe = CompilationRecipe(generated, producer=Path(inspect.getsourcefile(build_arm)))
    for i, path in enumerate((*map(Path, before), receipt, generated / "selected.target.mlir", mlir)):
        recipe.bind_preparation(str(i), path)

    measurements = []

    def run(argv):
        resource = generated / f"native-resource-{len(measurements)}.txt"
        started = time.perf_counter()
        result = subprocess.run(
            [tools[2], "-v", "-o", resource, "--", *argv],
            capture_output=True,
            text=True,
        )
        text = resource.read_text()
        rss = [line.split(":", 1)[1].strip() for line in text.splitlines() if "Maximum resident set size" in line]
        assert len(rss) == 1
        measurements.append(
            dict(
                argv=[str(v) for v in argv],
                wall_seconds=time.perf_counter() - started,
                peak_rss_kib=int(rss[0]),
                resource=str(resource),
            )
        )
        return result

    recipe.run(
        [tools[0], "--mlir-to-llvmir", mlir, "-o", ll],
        runner=run,
        inputs=(mlir,),
        output=ll,
    )
    recipe.run(
        [tools[1], "-O2", "-fPIC", "-c", ll, "-o", obj],
        runner=run,
        inputs=(ll,),
        output=obj,
    )
    recipe.run(
        [tools[1], "-O2", "-shared", "-fPIC", obj, runtime, "-o", library],
        runner=run,
        inputs=(obj, runtime),
        output=library,
    )
    compile_wall_seconds = time.perf_counter() - compile_started
    recipe.completed(library)
    recipe_sha = sha256_file(recipe.path)
    image_sha = sha256_file(library)
    products = {str(path): sha256_file(path) for path in generated.rglob("*") if path.is_file()}
    loaded = ctypes.CDLL(str(library))
    actual_image = bytes((ctypes.c_uint8 * len(payload)).in_dll(loaded, "fixture_commands"))
    assert actual_image == payload
    actual = guarded_execute(loaded, model, shape, width, separate, numeric_fault=numeric_fault)
    assert bytes((ctypes.c_uint8 * len(payload)).in_dll(loaded, "fixture_commands")) == payload
    if selection is not None:
        selection.revalidate()
    assert sha256_file(receipt) == receipt_sha
    verify_completed_recipe(recipe.path, executable=library, expected_recipe_sha256=recipe_sha)
    assert sha256_file(library) == image_sha
    assert before == {path: sha256_file(Path(path)) for path in before}
    assert products == {path: sha256_file(Path(path)) for path in products}
    for path in generated.rglob("invocation.json"):
        verify(path)
    return dict(
        model=model,
        selected_text=selected_text,
        artifact=artifact,
        payload=payload,
        actual=actual,
        pins={**before, **products},
        recipe=str(recipe.path),
        image=str(library),
        receipt_sha256=receipt_sha,
        compilation=dict(
            wall_seconds=compile_wall_seconds,
            native_tools=measurements,
            peak_rss_scope="each native translation/compiler process and its children; package lowering peak unknown",
        ),
        static_image_bytes=library.stat().st_size,
        command_image_bytes=len(payload),
    )


@pytest.mark.parametrize("shape,width,separate", [((3, 7, 5, 4), 1, False), ((2, 5, 11, 3), 2, True)])
def test_ordinary_target_fusion_reaches_loaded_bytes_and_complete_recurrent_outputs(tmp_path, shape, width, separate):
    records = []
    for enabled in (False, True):
        arm = tmp_path / str(enabled)
        arm.mkdir()
        record = build_arm(arm, shape, width, separate, enabled=enabled)
        assert F.verify_artifact(record["selected_text"], record["artifact"], record["model"]) == record["payload"]
        raw = [F.WORD.unpack_from(record["payload"], i) for i in range(0, len(record["payload"]), F.WORD.size)]
        assert sum(row[0] == 1 for row in raw) == (0 if enabled else 2)
        assert sum(row[0] == 3 for row in raw) == (2 if enabled else 0)
        assert record["actual"]["counts"][0] == 3 * (5 if enabled else 7)
        records.append(record)
    assert {k: v for k, v in records[0]["actual"].items() if k != "counts"} == {
        k: v for k, v in records[1]["actual"].items() if k != "counts"
    }
    m, _, n, p = shape
    for counter in (1, 2):
        assert records[0]["actual"]["counts"][counter] - records[1]["actual"]["counts"][counter] == 3 * 4 * m * (n + p)
    for record in records:
        assert record["pins"] == {path: sha256_file(Path(path)) for path in record["pins"]}
    (tmp_path / "pair.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k not in ("model", "payload")} for r in records],
            indent=2,
        )
    )


def test_ignoring_selected_target_is_rejected_even_with_correct_full_outputs(tmp_path):
    record = build_arm(tmp_path, (3, 7, 5, 4), 1, False, enabled=True, fault="ignore_selected")
    assert record["actual"]["counts"][0] == 21
    with pytest.raises(AssertionError):
        F.verify_artifact(record["selected_text"], record["artifact"], record["model"])


def test_wrong_rounding_or_wrap_boundary_fails_independent_numerical_reference(
    tmp_path,
):
    # The primitive's built-in wrong mode evaluates the i32 addition in wide
    # arithmetic before clamping, instead of preserving the source wrap boundary.
    with pytest.raises(AssertionError, match="actual complete .* history differs"):
        build_arm(tmp_path, (3, 7, 5, 4), 1, False, enabled=True, numeric_fault=True)


def test_pure_actual_operand_replacement_is_not_qualified_by_equal_command_counts(
    tmp_path,
):
    model = F.fixture(tmp_path, (3, 7, 5, 4), 1, False)
    selected, result = F.transform(F.target_text(model[1], model[0]), model)
    image = F.image_from_target(selected, model)
    words = [list(F.WORD.unpack_from(image, i)) for i in range(0, len(image), F.WORD.size)]
    assert words[-1][0] == 3
    words[-1][1 + F.KEYS.index("bias")] = 6
    changed = b"".join(F.WORD.pack(*word) for word in words)
    assert len(changed) == len(image) and sum(word[0] == 3 for word in words) == 2
    with pytest.raises(AssertionError, match="actual emitted call operands differ"):
        F.verify_artifact(selected, F.llvm_artifact(changed), model)


def test_source_intermediate_fanout_prevents_its_physical_fusion(tmp_path):
    model = F.fixture(tmp_path, (3, 7, 5, 4), 1, False)
    original = model[1]
    read = F.command(
        "readout",
        **(
            F.params(original.body[1])
            | dict(
                src=1,
                dst=0,
                source=F.params(original.body[1])["destination"],
                destination=F.params(original.body[0])["destination"],
            )
        ),
    )
    source = replace(original, body=(*original.body[:3], read, *original.body[3:]))
    modified = (model[0], source, *model[2:])
    _, result = F.transform(F.target_text(source, model[0]), modified)
    # The second independent readout still fuses; the observed first stage must remain.
    assert sum(call.instr == "readout" for call in result.kernel.body) == 2
    assert len(result.report["storage"]["fusion"]["checked_steps"]) == 1


def test_missing_physical_route_executes_original_program_with_complete_outputs(
    tmp_path,
):
    record = build_arm(tmp_path, (2, 5, 11, 3), 2, True, enabled=True, physical=False)
    assert record["actual"]["counts"][0] == 21
    assert F.verify_artifact(record["selected_text"], record["artifact"], record["model"]) == record["payload"]


def test_original_source_slots_execute_without_any_schedule_optimizer(tmp_path):
    record = build_arm(tmp_path, (3, 7, 5, 4), 1, False, enabled=False, use_schedule=False)
    assert record["actual"]["counts"][0] == 21
    assert F.verify_artifact(record["selected_text"], record["artifact"], record["model"]) == record["payload"]

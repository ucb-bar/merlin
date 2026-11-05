"""Ordinary selected-IR copy pipelining reaches an actual delayed native queue."""

import ctypes
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import test_shared_pass_fusion_provider as F
import test_shared_pass_pipeline_provider as P

from merlin.common.digest import sha256_file
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.targetgen import capsule_common as C
from merlin.targetgen import package_runtime as R
from merlin.targetgen.generate_prompt import shared_pass_manifest_template
from merlin.targetgen.shared_pass_contract import shared_pass_requirement

DRIVER = r"""
import json,sys
from pathlib import Path
import test_shared_pass_pipeline_provider as P
import test_shared_pass_fusion_provider as F
command,source,*rest=sys.argv[1:]
text=Path(source).read_text()
if command=='parse':
 assert text==Path('original.target.mlir').read_text()
elif command=='lower_interface_to_target':
 sys.stdout.write(text)
elif command=='emit_target_bundle':
 from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
 from merlin.targetgen.isa_dialect_gen import encode_instruction
 from merlin.sched.ir import evaluate
 config=json.loads(Path('fixture.json').read_text())
 selected=read_scheduling_contracts(*config['paths'],target='neutral')
 # Actual selected SSA values are the sole emitter instruction operands.
 model=(selected,None,None,None,None)
 from merlin.sched.ir import Kernel
 model=(selected,Kernel('target_program',(),()),None,None,None)
 actual=P.read_target(Path('original.target.mlir').read_text() if config.get('ignore') else text,model)
 calls=list(actual.body)
 if config.get('drop_wait'):
  calls.pop(next(i for i,c in enumerate(calls) if c.instr=='await'))
 image=b''.join(encode_instruction(selected.isa,c.instr,{k:evaluate(v,{}) for k,v in c.args}).to_bytes(
  selected.isa.word_bits//8,selected.instruction_byteorder) for c in calls)
 Path(rest[0]).write_bytes(Path('buffer.json').read_bytes())
 sys.stdout.write(F.llvm_artifact(image))
else:
 raise RuntimeError('ordinary emitter requires selected target IR')
"""


def package(directory, model, *, fault=None):
    directory.mkdir()
    selected, kernel = model[:2]
    (directory / "driver.py").write_text(DRIVER)
    (directory / "original.target.mlir").write_text(P.target_text(kernel, selected))
    (directory / "fixture.json").write_text(
        json.dumps(
            dict(paths=[str(selected.facts_path), str(selected.software_path)], **({fault: True} if fault else {}))
        )
    )
    buffer = dict(
        abi_version="0.1",
        target="neutral",
        commands=[],
        tensors={
            a.name: dict(shape=list(a.shape), dtype=a.dtype, role="input" if a.access == "read" else "output")
            for a in kernel.args
        },
        kernel_abi=dict(
            kind="whole_program", args=[dict(tensor=a.name, access=a.access) for a in kernel.args], outputs=["output"]
        ),
    )
    (directory / "buffer.json").write_text(json.dumps(buffer))
    requirement = shared_pass_requirement(version=2, required_supported=("buffering_pipelining",))
    manifest = shared_pass_manifest_template(requirement)
    shared = manifest["shared_pass_contract"]
    for boundary in shared["boundaries"].values():
        boundary["contracts"].update(
            semantics="exact-i8",
            effects="neutral-delayed-copy-v1",
            capabilities=selected.hardware_identity,
            resources=selected.hardware_identity,
            completion="actual-token-wait-v1",
        )
    family = next(r for r in shared["families"] if r["name"] == "buffering_pipelining")
    family.update(
        status="supported",
        boundary="device",
        reason="Neutral disjoint byte copies with bounded pending tokens; other execution laws remain unknown",
    )
    shared["controls"]["default_policy"] = ["buffering_pipelining"]
    shared["controls"]["choices"]["buffering_pipelining"]["enabled"] = [False, True]
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
    return R.load_package(directory, shared_pass_requirement=requirement), requirement


def execute(library, model, *, fault=0):
    kernel, banks = model[1], model[8]
    guards = []

    def buffer(data):
        raw = (ctypes.c_uint8 * (len(data) + 32))(*([193] * (len(data) + 32)))
        view = (ctypes.c_uint8 * len(data)).from_buffer(raw, 16)
        view[:] = data
        guards.append((raw, len(data)))
        return view

    first = buffer(bytes([131]) * banks[0].capacity)
    second = first if banks[1].backing == banks[0].backing else buffer(bytes([131]) * banks[1].capacity)
    source = buffer(bytes((37 * i + 253) % 256 for i in range(kernel.args[0].nbytes)))
    output = buffer(bytes([197]) * kernel.args[1].nbytes)
    pointers = (ctypes.c_void_p * 4)(*(ctypes.addressof(x) for x in (first, second, source, output)))
    caps = (ctypes.c_uint64 * 4)(len(first), len(second), len(source), len(output))
    library.fixture_kernel.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
    library.fixture_kernel.restype = ctypes.c_int
    carry = bytes(source)
    histories = []
    for _ in range(3):
        source[:] = carry
        output[:] = bytes([197]) * len(output)
        counts = (ctypes.c_uint64 * 6)()
        status = library.fixture_kernel(pointers, caps, counts, fault)
        actual = bytes(output)
        expected = bytes([197, 197]) + bytes((3 * x + 7) % 256 for x in carry) + bytes([197, 197])
        assert bytes(source) == carry, "actual readonly input changed"
        assert all(bytes(raw[:16]) == bytes(raw[n + 16 :]) == bytes([193]) * 16 for raw, n in guards), (
            "actual canary overwritten"
        )
        histories.append(
            dict(
                input=carry.hex(),
                output=actual.hex(),
                expected=expected.hex(),
                status=status,
                counts=list(counts),
                first=bytes(first).hex(),
                second=bytes(second).hex(),
            )
        )
        if status:
            break
        assert actual == expected, "actual delayed-copy numerical result differs"
        carry = actual[2:-2]
    return histories


def build(tmp_path, *, enabled=True, admitted=True, fault=None, native_fault=0, bits=64, big=False, alias=True):
    from merlin.common.invocation_record import verify
    from merlin.compile.scheduling.copy_pipeline import CopyPipelinePolicy, copy_pipeline_motion
    from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.target_schedule import TargetScheduleSelection

    tools = (Path(toolchain.mlir_translate()), Path(toolchain.clang()), Path("/usr/bin/time"))
    if not all(p.is_file() for p in tools):
        pytest.skip("explicit selected translator/native compiler required")
    facts = tmp_path / "facts"
    facts.mkdir()
    model = P.fixture(facts, width=7, panels=7, slots=2, bits=bits, big=big, alias=alias)
    selected, kernel, objects, adapter, codec, limits, admission, options, *_ = model
    pkg, requirement = package(tmp_path / "package", model, fault=fault)
    source = tmp_path / "source.mlir"
    source.write_text(P.target_text(kernel, selected))
    runtime = tmp_path / "runtime.c"
    runtime.write_text(P.runtime(model))
    budgets = ScheduleBudgets(eliminate_copies=False, optimize_storage=False, max_fusions=0)
    motion = (
        copy_pipeline_motion(
            admission if admitted else None,
            policy=CopyPipelinePolicy(distance=12, max_moves=32, max_pending_copies=2),
            budgets=budgets,
        )
        if enabled
        else None
    )
    owners = [
        Path(inspect.getsourcefile(f))
        for f in (
            build,
            execute,
            P.fixture,
            P.project,
            P.read_target,
            P.verify_artifact,
            P.runtime,
            F.llvm_artifact,
            F.artifact_image,
            TargetScheduleSelection,
            C.lower_interface,
        )
    ]
    before = {
        str(p): sha256_file(p)
        for p in (
            source,
            runtime,
            Path(C.__file__).resolve(),
            *tools,
            selected.facts_path,
            selected.software_path,
            *pkg.directory.iterdir(),
            *owners,
        )
        if p.is_file()
    }
    selection = TargetScheduleSelection(
        selected, objects, adapter, codec, limits, parameters=dict(budgets=budgets, copy_pipeline=motion, **options)
    )
    out = tmp_path / "build"
    started = time.perf_counter()
    _, artifact = C.lower_interface(
        pkg, source, out, contract=None, timeout=30, target_schedule=selection, shared_pass_requirement=requirement
    )
    selected_text = (out / "selected.target.mlir").read_text()
    payload = F.artifact_image(artifact)
    mlir, ll, obj, so = (out / name for name in ("lowered.llvm.mlir", "kernel.ll", "kernel.o", "model.so"))
    recipe = CompilationRecipe(out, producer=Path(inspect.getsourcefile(build)))
    for i, p in enumerate((*map(Path, before), out / "target_schedule.json", out / "selected.target.mlir", mlir)):
        recipe.bind_preparation(str(i), p)
    measurements = []

    def run(argv):
        resource = out / f"native-resource-{len(measurements)}.txt"
        start = time.perf_counter()
        result = subprocess.run([tools[2], "-v", "-o", resource, "--", *argv], capture_output=True, text=True)
        rss = [
            line.split(":", 1)[1].strip()
            for line in resource.read_text().splitlines()
            if "Maximum resident set size" in line
        ]
        assert len(rss) == 1
        measurements.append(
            dict(
                argv=[str(a) for a in argv],
                wall_seconds=time.perf_counter() - start,
                peak_rss_kib=int(rss[0]),
                resource=str(resource),
            )
        )
        return result

    recipe.run([tools[0], "--mlir-to-llvmir", mlir, "-o", ll], runner=run, inputs=(mlir,), output=ll)
    recipe.run([tools[1], "-O2", "-fPIC", "-c", ll, "-o", obj], runner=run, inputs=(ll,), output=obj)
    recipe.run(
        [tools[1], "-O2", "-shared", "-fPIC", obj, runtime, "-o", so], runner=run, inputs=(obj, runtime), output=so
    )
    wall = time.perf_counter() - started
    recipe.completed(so)
    recipe_sha = sha256_file(recipe.path)
    products = {str(p): sha256_file(p) for p in out.rglob("*") if p.is_file()}
    image = ctypes.CDLL(str(so))
    assert bytes((ctypes.c_uint8 * len(payload)).in_dll(image, "fixture_commands")) == payload
    actual = execute(image, model, fault=native_fault)
    assert bytes((ctypes.c_uint8 * len(payload)).in_dll(image, "fixture_commands")) == payload
    selection.revalidate()
    verify_completed_recipe(recipe.path, executable=so, expected_recipe_sha256=recipe_sha)
    assert before == {p: sha256_file(Path(p)) for p in before}
    assert products == {p: sha256_file(Path(p)) for p in products}
    for p in out.rglob("invocation.json"):
        verify(p)
    record = dict(
        selected=selected_text,
        artifact=artifact,
        payload=payload.hex(),
        actual=actual,
        pins={**before, **products},
        recipe=str(recipe.path),
        image=str(so),
        static_image_bytes=so.stat().st_size,
        command_image_bytes=len(payload),
        compilation=dict(
            wall_seconds=wall,
            native_tools=measurements,
            peak_rss_scope="each native tool and children; package peak unknown",
        ),
        timing="UNKNOWN",
    )
    (tmp_path / "execution.json").write_text(json.dumps(record, indent=2))
    return model, record


@pytest.mark.parametrize("bits,big,alias", [(64, False, True), (32, True, False)])
def test_selected_copy_pipeline_reaches_loaded_multipending_native_machine(tmp_path, bits, big, alias):
    records = []
    for enabled in (False, True):
        arm = tmp_path / str(enabled)
        arm.mkdir()
        model, record = build(arm, enabled=enabled, bits=bits, big=big, alias=alias)
        assert P.verify_artifact(record["selected"], record["artifact"], model).hex() == record["payload"]
        assert len(record["actual"]) == 3
        for row in record["actual"]:
            assert row["status"] == 0 and row["counts"][0] == 29
            assert row["counts"][3] == (2 if enabled else 1)
            assert row["counts"][4:] == [7, 7]
        records.append(record)
    for before, after in zip(records[0]["actual"], records[1]["actual"], strict=True):
        assert {k: v for k, v in before.items() if k != "counts"} == {k: v for k, v in after.items() if k != "counts"}
        assert before["counts"][:3] == after["counts"][:3]
    assert records[0]["payload"] != records[1]["payload"]
    for record in records:
        assert record["pins"] == {p: sha256_file(Path(p)) for p in record["pins"]}
    (tmp_path / "pair.json").write_text(json.dumps(records, indent=2))


def test_missing_original_copy_admission_keeps_executable_baseline(tmp_path):
    model, record = build(tmp_path, admitted=False)
    assert P.verify_artifact(record["selected"], record["artifact"], model).hex() == record["payload"]
    assert all(row["status"] == 0 and row["counts"][3] == 1 for row in record["actual"])


def test_ignored_selected_order_fails_even_when_all_numerical_outputs_match(tmp_path):
    model, record = build(tmp_path, fault="ignore")
    assert all(row["status"] == 0 and row["counts"][3] == 1 for row in record["actual"])
    with pytest.raises(AssertionError, match="actual emitted call operands differ"):
        P.verify_artifact(record["selected"], record["artifact"], model)


def test_actual_missing_wait_refuses_pending_read_and_incomplete_lifetime(tmp_path):
    model, record = build(tmp_path, fault="drop_wait")
    assert record["actual"][0]["status"] == 7
    with pytest.raises(AssertionError, match="actual emitted call operands differ"):
        P.verify_artifact(record["selected"], record["artifact"], model)


def test_actual_broken_wait_visibility_fails_complete_output_reference(tmp_path):
    with pytest.raises(AssertionError, match="actual delayed-copy numerical result differs"):
        build(tmp_path, native_fault=1)

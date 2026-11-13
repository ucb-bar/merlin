"""Real tiny-application test for validated native math component text replacement.

Set EXECUTION_HOST_MATH_ARCHIVE, EXECUTION_HOST_MATH_PRODUCTION_ARCHIVE,
EXECUTION_HOST_MATH_EQUIVALENCE and EXECUTION_HOST_MATH_RV64_OUTPUT to explicit
optional evidence inputs; missing evidence skips with a specific reason.
"""
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest

import compile_capsule
from merlin.compile.c_unary_math import emit_unary_math_c
from merlin.compile.semantic_application import CompiledCRegion, _render
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import (
    KernelRequest, SemanticNode, TensorType, pointwise_identity_maps,
)
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation, CycleInterval, GlobalPlan, RegionAlternative, ValueRepresentation,
)


PRODUCTION_SHA = "0626df7b1ce6426945820fd03dbd6752931c7594676c94f2043d5673588d178b"


@pytest.fixture
def tiny_math_application(tmp_path):
    names = ('EXECUTION_HOST_MATH_ARCHIVE', 'EXECUTION_HOST_MATH_PRODUCTION_ARCHIVE',
             'EXECUTION_HOST_MATH_EQUIVALENCE', 'EXECUTION_HOST_MATH_RV64_OUTPUT')
    if any(not os.environ.get(name) for name in names):
        pytest.skip('optional native math packaging requires explicit evidence paths: ' + ', '.join(names))
    archive, production_archive, equivalence, rv64_output = (Path(os.environ[name]) for name in names)
    if platform.machine() != "x86_64" or not shutil.which("cc"):
        pytest.skip("real native math packaging requires x86_64 and cc")
    for path in (archive, production_archive, equivalence, rv64_output):
        if not path.is_file():
            pytest.skip(f"optional execution-host math evidence is missing: {path}")
    native_sha = json.loads(equivalence.read_text())['execution_host_archive_sha256']
    assert sha256(archive.read_bytes()).hexdigest() == native_sha
    assert sha256(production_archive.read_bytes()).hexdigest() == PRODUCTION_SHA
    tensor = TensorType((2,), "f32", "raw-bit-pattern")
    x = SemanticNode("angles", "input", (), tensor, effect="input")
    operations = tuple(SemanticNode(
        operation, f"source_f32_{operation}_unqualified", (x.id,), tensor,
        (("numeric_semantics", f"source-f32-{operation}-unqualified-v1"),),
        index_maps=pointwise_identity_maps(1, 1),
    ) for operation in ("sin", "cos"))
    source = KernelRequest((x, *operations), tuple(node.id for node in operations),
                           ("external", "external"), ((x.id, "external"),), "synthetic")
    representation = ValueRepresentation("memory", "dense", "f32")
    plan = GlobalPlan(tuple(RegionAlternative(
        node.id, (index,), "selected_math", "host", CycleInterval.unknown("test"),
        (BufferRepresentation(x.id, representation),),
        (BufferRepresentation(node.id, representation),),
    ) for index, node in enumerate(operations)), (), CycleInterval.unknown("test"))
    linkage = dict(schema="merlin.host_linkage_contract.v1", supplier="selected_math_archive",
                   archive_sha256=PRODUCTION_SHA, symbols=["cosf", "sinf"])
    policy = dict(schema="merlin.compiled_unary_math.v1", policy="selected-libm-rne-finite-v1",
                  isa="rv64gc", abi="lp64d", effects="enabled-fpu-rne-unobserved-flags-and-errno-v1",
                  operations=[node.op for node in operations], linkage=linkage)
    compiled = []
    for index, node in enumerate(operations):
        region = extract_semantic_region(source, (node.id,))
        name = f"region_{index:03d}"
        description, generated = emit_unary_math_c(region, name, policy)
        compiled.append(CompiledCRegion(name, name + ".h", (x.id,), (node.id,),
                        {member: data.encode() for member, data in generated.items()}, (), (), description))
    _, regions, interface, links, files = _render(
        source, SimpleNamespace(plan=lambda _: plan), "capsule", 4096, tuple(compiled))
    application = tmp_path / "application"
    for member, data in files.items():
        path = application / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = dict(schema="merlin.semantic_application.v1", application_name="capsule",
                    application_interface=interface, application_memory_bytes=4096,
                    request_digest=source.digest(), regions=regions, segments=[], link_sources=links,
                    stage_artifacts={member: sha256(data).hexdigest() for member, data in files.items()})
    (application / "manifest.json").write_text(json.dumps(manifest))
    before = {member: (application / member).read_bytes() for member in (*files, "manifest.json")}
    production_bytes = production_archive.read_bytes()
    assert interface["host_linkage"] == [linkage]
    for index in range(2):
        raw = before[f"c-package/regions/region_{index:03d}/region_{index:03d}.c"]
        assert b"#if !defined(__riscv)" in raw and b'__asm__ volatile("frrm' in raw
    return SimpleNamespace(application=application, manifest=manifest, before=before,
                           archive=archive, equivalence=equivalence, rv64_output=rv64_output,
                           production_archive=production_archive, production_bytes=production_bytes,
                           runtime=tmp_path / "runtime", native_sha=native_sha)


def _assert_production_unchanged(case):
    assert {member: (case.application / member).read_bytes() for member in case.before} == case.before
    assert case.production_archive.read_bytes() == case.production_bytes


def test_real_tiny_application_selects_native_archive_and_preserves_production(tiny_math_application, monkeypatch):
    case = tiny_math_application
    shutil.copytree(case.application / "c-package", case.runtime)
    real_run, commands = subprocess.run, []

    def compile_native(command, **kwargs):
        commands.append(list(command))
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            return real_run(["nice", "-n", "10", *command], env=environment, **kwargs)
        except subprocess.CalledProcessError as error:
            pytest.fail(f"actual tiny application compiler failed:\n{error.stderr}")

    with monkeypatch.context() as patch:
        patch.setattr(compile_capsule.subprocess, "run", compile_native)
        compile_capsule.build_host_library(case.application, case.manifest, package=case.runtime,
            execution_host_math={"functional_validation": True, "archive": case.archive,
                                 "equivalence": case.equivalence, "rv64_output": case.rv64_output})
    assert len(commands) == 1
    command = commands[0]
    assert str(case.archive) in command
    assert "-Wl,-Bsymbolic-functions" in command and "-Wl,--exclude-libs,ALL" in command
    assert not any(item.startswith("-D__riscv") for item in command)
    selection = json.loads((case.runtime / "execution-host-math.json").read_text())
    assert selection["validation_only"] is True and selection["deployment_qualified"] is False
    assert selection["production_linkage"] == case.manifest["application_interface"]["host_linkage"]
    assert selection["production_linkage"][0]["archive_sha256"] == PRODUCTION_SHA
    assert selection["execution_linkage"] == [{**selection["production_linkage"][0], "archive_sha256": case.native_sha}]
    assert len(selection["components"]) == 2
    assert {component["symbol"] for component in selection["components"]} == {"sinf", "cosf"}
    for component in selection["components"]:
        assert component["elements"] == 2
        assert component["source_request_digest"] == component["execution_source_request_digest"]
        assert component["production_component_sha256"] != component["execution_component_sha256"]
    observation = selection["link_supplier_observation"]
    assert observation["schema"] == "merlin.link_supplier_trace.v1"
    assert set(observation["symbols"]) == {"cosf", "sinf"}
    for supplier in observation["symbols"].values():
        assert supplier["selected_input"]["sha256"] == case.native_sha
        assert supplier["selected_input"]["path"] == str(case.archive.resolve())
        assert supplier["definition"] == str(case.archive.resolve()) + "(wrapper.o)"
    library = case.runtime / "capsule.so"
    assert selection["library_sha256"] == sha256(library.read_bytes()).hexdigest()
    loaded = ct.CDLL(str(library))
    loaded.run.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    loaded.run.restype = ct.c_int
    incoming = (ct.c_uint32 * 2)(0x00000000, 0x3f800000)
    sine, cosine = (ct.c_uint32 * 2)(), (ct.c_uint32 * 2)()
    assert loaded.run((ct.c_void_p * 1)(ct.addressof(incoming)),
                      (ct.c_void_p * 2)(ct.addressof(sine), ct.addressof(cosine))) == 0
    rows = list(struct.iter_unpack("<III", case.rv64_output.read_bytes()))
    assert tuple(incoming) == (rows[0][0], rows[32][0])
    assert tuple(sine) == (rows[0][1], rows[32][1])
    assert tuple(cosine) == (rows[0][2], rows[32][2])
    _assert_production_unchanged(case)


def test_default_selected_math_refuses_before_package_write_or_compiler(tiny_math_application, monkeypatch):
    case = tiny_math_application

    def unexpected_compile(*args, **kwargs):
        pytest.fail("default selected-math refusal must precede compiler invocation")

    monkeypatch.setattr(compile_capsule.subprocess, "run", unexpected_compile)
    with pytest.raises(ValueError, match="selected host-platform execution support"):
        compile_capsule.build_host_library(case.application, case.manifest, package=case.runtime)
    assert not case.runtime.exists()
    _assert_production_unchanged(case)

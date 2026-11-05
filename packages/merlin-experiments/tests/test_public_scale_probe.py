"""Original public shape semantics and actual emission observations; no native jobs."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import public_scale_sources as SOURCES
from merlin_experiments.phase1.brokers import selfcheck as BROKER
from merlin_experiments.phase1.feedback import compile_only as PROBE
from merlin_experiments.phase1.feedback import selfcheck as WORKER
from merlin_experiments.phase1.tools import selfcheck as CLIENT

from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir


def member(tmp_path, *, kind="MATMUL_RESIDENT", label="public", dtype="i8"):
    root = tmp_path / kind
    root.mkdir()
    if kind == "CONV2D":
        tensors = {"A": {"shape": [1, 5, 7, 3], "dtype": dtype, "role": "input"},
                   "W": {"shape": [18, 5], "dtype": dtype, "role": "weight"},
                   "Y": {"shape": [6, 5], "dtype": "i32", "role": "output"}}
        work = {"opcode": kind, "operands": {"ifm": "A", "weight": "R", "dst": "Y"},
                "attributes": {"layout": "nhwc", "kernel": [2, 3, 3, 5], "stride": [2, 2],
                               "padding": [0, 0, 0, 0], "dilation": [1, 1]}}
    elif kind == "MOVEMENT":
        tensors = {"A": {"shape": [3, 5], "dtype": dtype, "role": "input"},
                   "Y": {"shape": [3, 5], "dtype": "i32", "role": "output"}}
        work = {"opcode": kind, "operands": {"src": "A", "dst": "Y"}, "attributes": {}}
    else:
        tensors = {"A": {"shape": [3, 5], "dtype": dtype, "role": "input"},
                   "W": {"shape": [5, 7], "dtype": dtype, "role": "weight"}}
        work = {"opcode": kind, "operands": {"lhs": "A", "rhs": "R", "dst": "X"}, "attributes": {}}
    commands = [] if kind == "MOVEMENT" else [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"}, "attributes": {"layout": "packed_rhs"}}]
    commands.append(work)
    if kind == "MATMUL_RESIDENT":
        commands.append({"opcode": "COMMIT", "operands": {"src": "X", "dst": "Y"},
                         "attributes": {"output_dtype": "i32", "epilogue": []}})
    if kind != "MOVEMENT":
        commands.append({"opcode": "EVICT", "operands": {"handle": "R"}})
    cb = {"abi_version": "0.1", "target": "fixture", "tensors": tensors, "commands": commands}
    source = emit_interface_mlir(cb)
    descriptor = {"name": kind, "kind": "isa", "label": label, "interface_mlir": "interface.mlir"}
    (root / "interface.mlir").write_text(source)
    (root / "capsule.yaml").write_text(json.dumps(descriptor))
    return {**descriptor, "__dir__": str(root)}, cb


@pytest.mark.parametrize("kind", ["MOVEMENT", "CONV2D"])
def test_named_source_emitter_retains_original_operation_and_output(tmp_path, kind):
    cap, cb = member(tmp_path, kind=kind)
    parsed = parse_interface_mlir((Path(cap["__dir__"]) / "interface.mlir").read_text())
    assert parsed["commands"] == cb["commands"]
    assert parsed["tensors"] == cb["tensors"]


def test_contraction_scale_dimension_joins_and_hardware_tails_without_values(tmp_path):
    cap, _ = member(tmp_path)
    cases, record = SOURCES.derive([cap], index_bits=64, tile_edge=16)
    assert [case.scale for case in cases] == [1, 8, 8, 8, 64]
    assert cases[0].source_ir == (Path(cap["__dir__"]) / "interface.mlir").read_text()
    for case in cases:
        abi = case.original_abi.record()
        a, w = abi["inputs"]
        y = abi["outputs"][0]
        assert a["shape"][1] == w["shape"][0]
        assert y["shape"] == [a["shape"][0], w["shape"][1]]
        assert len(case.source_ir) < 4096
    assert cases[2].original_abi.inputs[0].shape == (31, 47)
    assert cases[3].original_abi.inputs[0].shape == (33, 49)
    assert not record["unavailable"]
    assert not list(tmp_path.rglob("*.npy")) and not list(tmp_path.rglob("*.npz"))


def test_convolution_preserves_stride_padding_dilation_and_weight_geometry(tmp_path):
    from merlin.runtime.commandbuffer import conv_out_dims

    cap, cb = member(tmp_path, kind="CONV2D")
    cases, record = SOURCES.derive([cap], index_bits=64, tile_edge=16)
    assert len(cases) == 5 and not record["unavailable"]
    for case in cases:
        parsed = parse_interface_mlir(case.source_ir)
        work = next(cmd for cmd in parsed["commands"] if cmd["opcode"] == "CONV2D")
        attrs = work["attributes"]
        for name in ("stride", "padding", "dilation", "layout"):
            assert attrs[name] == cb["commands"][1]["attributes"][name]
        a, w = case.original_abi.inputs
        n, h, width, ci = a.shape
        kh, kw, channels, co = attrs["kernel"]
        oh, ow = conv_out_dims(h, width, kh, kw, attrs["stride"], attrs["padding"], attrs["dilation"])
        assert ci == channels and w.shape == (kh * kw * ci, co)
        assert case.original_abi.outputs[0].shape == (n * oh * ow, co)


def test_precision_and_compositions_are_not_reconstructed(tmp_path):
    cap, _ = member(tmp_path, kind="MOVEMENT", dtype="f32")
    cases, record = SOURCES.derive([cap], index_bits=64)
    assert len(cases) == 1 and not record["unavailable"]
    assert cases[0].record()["derivation"]["scale_family"] == "retained_only"


@pytest.mark.parametrize("source,accepted", [
    ('// !merlin_iface.acc<f32>\n "!merlin_iface.acc<i64>" !merlin_iface.acc < i32 >', True),
    ('!merlin_iface.acc< // ignored\n i32 >', True),
    ('!merlin_iface.acc< f32 >', False),
    ('!merlin_iface.acc<si32>', False),
    ('!merlin_iface.acc<tensor<4xi32>>', False),
    ('!merlin_iface.acc<i32, i64>', False),
    ('!merlin_iface.acc<i32', False),
    ('!merlin_iface.acc', False),
])
def test_accumulator_precision_is_structurally_parsed(source, accepted):
    assert SOURCES._supported_accumulator_types(source) is accepted


def test_nonstandard_precision_whitespace_stays_exact_original(tmp_path):
    cap, _ = member(tmp_path)
    path = Path(cap["__dir__"]) / "interface.mlir"
    changed = path.read_text().replace("!merlin_iface.acc<i32>", "!merlin_iface.acc< f32 >")
    path.write_text(changed)
    cases, _ = SOURCES.derive([cap], index_bits=64)
    assert len(cases) == 1 and cases[0].source_ir == changed
    assert cases[0].record()["derivation"]["scale_family"] == "retained_only"


def test_private_selection_and_public_source_mutation_refuse(tmp_path):
    cap, _ = member(tmp_path)
    hidden = {"name": "withheld", "label": "hidden", "__dir__": str(tmp_path / "absent-private")}
    with pytest.raises(ValueError, match="non-public"):
        SOURCES.derive([cap, hidden], selected="withheld", index_bits=64)
    cases, _ = SOURCES.derive([cap, hidden], index_bits=64)
    (Path(cap["__dir__"]) / "interface.mlir").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        cases[0].verify()


def test_index_bounds_and_roster_refuse_transactionally(tmp_path, monkeypatch):
    cap, _ = member(tmp_path)
    cases, record = SOURCES.derive([cap], index_bits=8, tile_edge=16)
    assert not cases and record["unavailable"]
    monkeypatch.setattr(SOURCES, "MAX_CASES", 2)
    assert not SOURCES.derive([cap], index_bits=64)[0]


def test_scale_flag_is_diagnostic_only_and_requires_compile_only(capsys):
    assert PROBE.SCALE_PROBE_FLAG == BROKER.FORWARDED_PROBES["scale_probe"] == "--scale-probe"
    for main in (CLIENT.main, WORKER.main):
        with pytest.raises(SystemExit) as error:
            main(["--scale-probe"])
        assert error.value.code == 2
    assert "--compile-only" in capsys.readouterr().err


@pytest.fixture
def emitted_case(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.targetgen import capsule_common as CC

    candidate, contract = tmp_path / "candidate", tmp_path / "contract"
    candidate.mkdir()
    contract.mkdir()
    source = tmp_path / "source.mlir"
    source.write_text("original public typed source")
    package = SimpleNamespace(target="fixture")
    monkeypatch.setattr(PROBE.P, "load_package", lambda *args, **kwargs: package)
    monkeypatch.setattr(PROBE.P, "integrity_scan", lambda *args: None)
    monkeypatch.setattr(PROBE.P, "build_package", lambda *args, **kwargs: None)
    artifact = "module { llvm.func @kernel() { llvm.return } }\n"

    def lower(package, source, generated, **kwargs):
        assert PROBE.P.active_package_executor() is not None
        for name in ("input.interface.mlir", "command_buffer.json", "lowered.target.mlir", "lowered.llvm.mlir"):
            (generated / name).write_text(artifact)
        return {}, artifact

    monkeypatch.setattr(CC, "lower_interface", lower)
    translator = tmp_path / "translator"
    llvm = 'define void @kernel() {\nentry:\n  call void asm sideeffect "actual effect", ""()\n  ret void\n}\n'
    translator.write_text(f"#!{sys.executable}\nimport pathlib,sys\n"
                          f"pathlib.Path(sys.argv[-1]).write_text({llvm!r})\n")
    translator.chmod(0o700)
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: translator)
    return candidate, contract, source, tmp_path / "products"


def test_original_emission_actual_llvm_counts_and_explicit_resource_refusal(emitted_case):
    from merlin.llvmlower.device_ir_scalability import DeviceIRBudget

    candidate, contract, source, output = emitted_case
    with PROBE.P.scoped_package_executor(SimpleNamespace(run_entrypoint=lambda *a, **k: None,
                                                         build_package=lambda *a, **k: None)):
        result = PROBE._emit_scale_case(candidate, source, output, contract=contract,
            target="fixture", timeout_s=10, ir_budget=DeviceIRBudget(mode="enforce", max_code_bytes=1))
    assert result["code_growth"]["inline_asm_sites"] == 1
    assert result["code_growth"]["functions"] == 1
    assert result["lowering_status"] == "resource_refused"
    assert result["compilation_status"] == "not_attempted"
    assert result["resource_budget"]["status"] == "refused"
    assert not list(output.rglob("*.o")) and not list(output.rglob("*.elf"))
    assert len(result["emission_invocation_sha256"]) == len(result["translation_invocation_sha256"]) == 64


def test_scale_package_build_and_emit_share_one_deadline(emitted_case, monkeypatch):
    from merlin.targetgen import capsule_common

    candidate, contract, source, output = emitted_case
    now, observed = [100.0], []
    monkeypatch.setattr(PROBE.time, "monotonic", lambda: now[0])

    def build(*args, timeout):
        observed.append(("build", timeout))
        now[0] += 7

    original_lower = capsule_common.lower_interface

    def lower(*args, **kwargs):
        observed.append(("lower", kwargs["timeout"]))
        kwargs["invoke"]("fixture", "emit", source)
        return original_lower(*args, **kwargs)

    def entry(*args, timeout, **kwargs):
        observed.append(("entrypoint", timeout))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(PROBE.P, "build_package", build)
    monkeypatch.setattr(PROBE.P, "run_entrypoint", entry)
    monkeypatch.setattr(capsule_common, "lower_interface", lower)
    with PROBE.P.scoped_package_executor(SimpleNamespace(
        run_entrypoint=lambda *a, **kw: None, build_package=lambda *a, **kw: None,
    )):
        result = PROBE._emit_scale_case(candidate, source, output, contract=contract,
                                       target="fixture", timeout_s=10, ir_budget=None)
    assert observed == [("build", 10.0), ("lower", 3.0), ("entrypoint", 3.0)]
    assert result["lowering_status"] == "emitted"


def test_scale_build_that_spends_budget_cannot_start_emission(emitted_case, monkeypatch):
    from merlin.targetgen import capsule_common

    candidate, contract, source, output = emitted_case
    now = [100.0]
    monkeypatch.setattr(PROBE.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(PROBE.P, "build_package", lambda *a, **kw: now.__setitem__(0, 111.0))
    monkeypatch.setattr(
        capsule_common, "lower_interface", lambda *a, **kw: pytest.fail("expired build started emission")
    )
    with (
        PROBE.P.scoped_package_executor(SimpleNamespace(
            run_entrypoint=lambda *a, **kw: None, build_package=lambda *a, **kw: None,
        )),
        pytest.raises(TimeoutError, match="deadline expired"),
    ):
        PROBE._emit_scale_case(candidate, source, output, contract=contract,
                               target="fixture", timeout_s=10, ir_budget=None)


def test_scale_cpu_queue_wait_cannot_restart_translation_budget(emitted_case, monkeypatch):
    from contextlib import contextmanager

    from merlin.common import cpu_pool

    candidate, contract, source, output = emitted_case
    now = [100.0]
    monkeypatch.setattr(PROBE.time, "monotonic", lambda: now[0])

    @contextmanager
    def queue(*, timeout_s):
        assert timeout_s == 10.0
        now[0] += 11
        yield

    monkeypatch.setattr(cpu_pool, "shared_compute_pool", queue)
    monkeypatch.setattr(PROBE.subprocess, "run", lambda *a, **kw: pytest.fail("expired queue started translation"))
    with (
        PROBE.P.scoped_package_executor(SimpleNamespace(
            run_entrypoint=lambda *a, **kw: None, build_package=lambda *a, **kw: None,
        )),
        pytest.raises(TimeoutError, match="deadline expired"),
    ):
        PROBE._emit_scale_case(candidate, source, output, contract=contract,
                               target="fixture", timeout_s=10, ir_budget=None)


def test_emission_requires_original_scoped_executor_and_typed_policy(emitted_case):
    candidate, contract, source, output = emitted_case
    with pytest.raises(ValueError, match="original scoped"):
        PROBE._emit_scale_case(candidate, source, output, contract=contract,
            target="fixture", timeout_s=10, ir_budget=None)
    with pytest.raises(ValueError, match="typed host"):
        PROBE._emit_scale_case(candidate, source, output, contract=contract,
            target="fixture", timeout_s=10, ir_budget={"mode": "enforce"})


def test_mutated_emitted_mlir_is_refused_before_counts_are_published(emitted_case, monkeypatch):
    from merlin.llvmlower import toolchain

    candidate, contract, source, output = emitted_case
    translator = toolchain.mlir_translate()
    text = translator.read_text()
    translator.write_text(text + "pathlib.Path(sys.argv[2]).write_text('altered emitted IR')\n")
    with PROBE.P.scoped_package_executor(SimpleNamespace(run_entrypoint=lambda *a, **k: None,
                                                         build_package=lambda *a, **k: None)):
        with pytest.raises(ValueError, match="unchanged observed inputs"):
            PROBE._emit_scale_case(candidate, source, output, contract=contract,
                target="fixture", timeout_s=10, ir_budget=None)


def test_scale_observer_does_not_require_object_policy_or_give_qualification(tmp_path, monkeypatch):
    cap, _ = member(tmp_path, kind="MOVEMENT")
    candidate, contract, facts_dir = (tmp_path / name for name in ("candidate", "contract", "facts"))
    for path in (candidate, contract, facts_dir):
        path.mkdir()
    (candidate / "compiler.py").write_text("original = True\n")
    fact = facts_dir / "facts.json"
    fact.write_text('{"facts":{"target":"fixture","arrays":[{"cols":16}]}}')
    owner = object.__new__(PROBE.CommonTransformService)
    owner.profile = {"sha256": "a" * 64, "host_index_bits": 64}
    owner._verify = lambda: None
    owner.compiler_imports = nullcontext
    called = []

    def emit(candidate, source, root, **kwargs):
        assert PROBE.P.active_package_executor() is not None
        called.append(source)
        return {"lowering_status": "emitted", "compilation_status": "not_attempted",
                "code_growth": {"inline_asm_sites": 2}}

    monkeypatch.setattr(PROBE, "_emit_scale_case", emit)
    monkeypatch.setattr(PROBE.CO, "compile_source_only", lambda **kwargs: pytest.fail("object path used"))
    result = PROBE.observe(candidate=candidate, target_experiment=SimpleNamespace(target="fixture"),
        capsules=[cap], contract=contract, facts_path=fact, common_service=owner,
        work_root=tmp_path / "products", timeout_s=30, scale_probe=True)
    assert len(called) == 5
    assert result["mode"] == "scale_emission_only" and result["all_pass"] is False
    assert result["numeric"] == result["rtl"] == "not_attempted"
    assert all(set(row["static_obligations"].values()) == {"UNKNOWN"} for row in result["cases"])

    def refused(candidate, source, root, **kwargs):
        generated = root / "generated"
        generated.mkdir(parents=True)
        (generated / "lowered.llvm.mlir").write_text("unverified emitted growth")
        raise TimeoutError("original deadline")

    monkeypatch.setattr(PROBE, "_emit_scale_case", refused)
    result = PROBE.observe(candidate=candidate, target_experiment=SimpleNamespace(target="fixture"),
        capsules=[cap], contract=contract, facts_path=fact, common_service=owner,
        work_root=tmp_path / "refused-products", timeout_s=30, scale_probe=True)
    assert result["all_pass"] is False
    assert all(row["lowering_status"] == "unavailable" and row["reason"] == "TimeoutError"
               and row["partial_emission"]["bytes"] == 25 and "code_growth" not in row
               for row in result["cases"])

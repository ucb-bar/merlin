"""Input relinks reuse verified weights without writing another checkpoint blob."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.llvmlower import c_runtime
from merlin.model_check import full_build as FB


@pytest.fixture
def image(tmp_path, monkeypatch, request):
    from merlin.common.paths import module_source_path
    from merlin.compile import baremetal_model
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.targetgen.rtl.facts import observed_facts

    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "model.mlir").write_text(
        "module { func.func @forward(%w: tensor<2xf32>, %x: tensor<2xf32>) -> tensor<2xf32> "
        "{ return %x : tensor<2xf32> } }\n"
    )
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", 2, 3))
    (stage / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "param", "weight": "weight"},
                "1": {"kind": "input", "name": "x"},
            }
        )
    )
    np.savez(stage / "inputs.npz", in0=np.array([1, 2], np.float32))
    host = tmp_path / "host"
    host.mkdir()
    (host / "manifest.yaml").write_text("host package")
    board, dts = tmp_path / "boards.yaml", tmp_path / "cpu.dts"
    board.write_text("board")
    dts.write_text("host ISA")
    tool = tmp_path / "inert-tool"
    tool.write_text("inert unit-test tool; never executed")
    tool.chmod(0o700)

    def compiler(**kwargs):
        build = kwargs["output"] / "build"
        c_runtime.generate(stage, build / "cgen", stage / "inputs.npz")
        (build / "model_main.c").write_text("/* fixture driver source; never compiled */\n")
        (build / "weights_blob.o").write_bytes(b"weight object")
        recipe = CompilationRecipe(build, producer=module_source_path("merlin.model_check.full_build"))

        def returned(argv, **kwargs):
            Path(argv[argv.index("-o") + 1]).write_bytes(b"observed fixture output")
            return SimpleNamespace(returncode=0)

        recipe.run([tool, build / "model_main.c", "-I", build / "cgen", "-o", build / "model_main.o"],
                   runner=returned, inputs=[build / "model_main.c"], output=build / "model_main.o", cwd=build)
        elf = build / "model.elf"
        recipe.run([tool, build / "model_main.o", build / "weights_blob.o", "-o", elf], runner=returned,
                   inputs=[build / "model_main.o", build / "weights_blob.o"], output=elf, cwd=build)
        recipe.completed(elf)
        return {"output": {"elf": str(elf), "elf_sha256": FB.sha256(elf), "build_hash": "unit-observation",
                           "compilation_recipe": {"path": str(recipe.path), "sha256": FB.sha256(recipe.path)}}}

    monkeypatch.setattr(baremetal_model, "compile_saved_model", compiler)
    monkeypatch.setattr(FB, "elf_sections", lambda path: {})
    monkeypatch.setattr(FB, "_routing", lambda *args: "unit device route; no hardware qualification")
    with observed_facts("neutral", {"facts": {}}):
        record = FB.build(stage, target="neutral", label="unit", submission=host,
                          submission_sha256="unused", settings=FB.BuildSettings("board", board, dts, "typed", 1, host),
                          directory=tmp_path / "image-cache", offload=bool(getattr(request, "param", False)))
        yield stage, Path(record["build_dir"]), record


@pytest.mark.parametrize("change", [None, "weights", "size", "call"])
def test_relink_checks_blob_identity_and_call_before_reusing_objects(image, tmp_path, monkeypatch, change):
    stage, build, record = image
    if change == "weights":
        path = stage / "weights.safetensors"
        path.write_bytes(path.read_bytes()[:-4] + struct.pack("<f", 4))
    elif change == "size":
        record["weights_bin_bytes"] += 1
    elif change == "call":
        (build / "cgen/model_call.c").write_text("different argument-table call")
    commands = []
    monkeypatch.setattr(FB, "_run", lambda argv, cwd: commands.append(([str(a) for a in argv], cwd)))
    inputs = {"in0": np.array([8, 9], np.float32)}
    directory = tmp_path / "step"
    if change:
        with pytest.raises(FB.FullBuildError,
                           match="weights blob|argument-table call|inputs changed|stored build origin"):
            FB.relink(record, stage, inputs, directory)
        assert not commands
    else:
        def run(argv, cwd):
            commands.append(([str(a) for a in argv], cwd))
            Path(argv[argv.index("-o") + 1]).write_bytes(b"relinked fixture output")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(FB, "_run", run)
        assert FB.relink(record, stage, inputs, directory) == directory / "model.elf"
        assert len(commands) == 2
        assert str(build / "weights_blob.o") in commands[1][0]
        assert str(directory / "model_main.o") in commands[1][0]
        assert str(directory / "cgen") in commands[0][0]
        assert (directory / "cgen/model_io.h").read_bytes() != (build / "cgen/model_io.h").read_bytes()
        with np.load(directory / "capture/inputs.npz") as bound:
            np.testing.assert_array_equal(bound["in0"], inputs["in0"])
    assert not (directory / "cgen/weights.bin").exists()
    assert (build / "cgen/weights.bin").read_bytes() == struct.pack("<ff", 2, 3)


def test_stage_reuses_only_identical_weights_and_regeneration_cannot_mutate_source(image, tmp_path):
    stage, build, _ = image
    source = build / "cgen/weights.bin"
    destination = tmp_path / "second_stage"
    info = c_runtime.generate(stage, destination, stage / "inputs.npz", weights_source=source)
    assert (destination / "weights.bin").samefile(source)
    assert info["weights_bytes"] == 8 and info["weights_sha256"] == FB.sha256(source)
    weights = stage / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-4] + struct.pack("<f", 4))
    c_runtime.generate(stage, destination, stage / "inputs.npz")
    assert not (destination / "weights.bin").samefile(source)
    assert source.read_bytes() == struct.pack("<ff", 2, 3)
    assert (destination / "weights.bin").read_bytes() == struct.pack("<ff", 2, 4)
    with pytest.raises(ValueError, match="differs from this stage"):
        c_runtime.generate(stage, tmp_path / "mismatch", stage / "inputs.npz", weights_source=source)
    assert not (tmp_path / "mismatch/weights.bin").exists()


def test_weight_source_cannot_be_requested_in_hash_only_mode(image, tmp_path):
    stage, build, _ = image
    with pytest.raises(ValueError, match="materialize_weights=True"):
        c_runtime.generate(
            stage,
            tmp_path / "conflicting_options",
            stage / "inputs.npz",
            materialize_weights=False,
            weights_source=build / "cgen/weights.bin",
        )


@pytest.mark.parametrize("mutation", ["source", "object", "recipe", "tool", "header", "preparation", "board", "host"])
def test_changed_origin_cannot_relink_even_when_original_elf_is_unchanged(image, tmp_path, monkeypatch, mutation):
    stage, build, record = image
    original = FB.sha256(Path(record["elf"]))
    if mutation == "source":
        path = stage / "model.mlir"
        path.write_text(path.read_text().replace("return %x", "return %w"))
    elif mutation == "preparation":
        (build / "blob_weights.json").write_text('{"args":[],"dead_args":[]}')
    else:
        path = {
            "object": build / "weights_blob.o", "recipe": build / "compilation_recipe.json",
            "tool": tmp_path / "inert-tool", "header": build / "cgen/model_gen.h",
            "board": tmp_path / "boards.yaml", "host": tmp_path / "host/manifest.yaml",
        }[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
    commands = []
    monkeypatch.setattr(FB, "_run", lambda *args: commands.append(args))
    with pytest.raises(FB.FullBuildError, match="changed|rebuild"):
        FB.relink(record, stage, {"in0": np.array([8, 9], np.float32)}, tmp_path / "step")
    assert not commands and FB.sha256(Path(record["elf"])) == original
    assert not (tmp_path / "step").exists()


@pytest.mark.parametrize("selection", ["legacy", "record", "controls"])
def test_missing_or_changed_origin_selection_cannot_authorize_relink(image, tmp_path, monkeypatch, selection):
    stage, _, record = image
    if selection == "legacy":
        record.pop("relink_binding")
    elif selection == "record":
        record["key"] = {**record["key"], "target": "other"}
    else:
        monkeypatch.setenv("MERLIN_NEW_SCHEDULING_CONTROL", "changed")
    monkeypatch.setattr(FB, "_run", lambda *args: pytest.fail("stale record launched a tool"))
    with pytest.raises(FB.FullBuildError, match="changed|rebuild|differs"):
        FB.relink(record, stage, {"in0": np.array([8, 9], np.float32)}, tmp_path / "step")


@pytest.mark.parametrize("location", ["capture", "build", "cache", "ancestor", "symlink"])
def test_relink_output_cannot_remove_or_overlap_original_inputs(image, tmp_path, location):
    stage, build, record = image
    output = {"capture": stage, "build": build, "cache": Path(record["cache_directory"]),
              "ancestor": tmp_path, "symlink": tmp_path / "link"}[location]
    if location == "symlink":
        output.symlink_to(stage, target_is_directory=True)
    before = FB.sha256(stage / "weights.safetensors")
    with pytest.raises(FB.FullBuildError, match="overlaps|symlink"):
        FB.relink(record, stage, {"in0": np.array([8, 9], np.float32)}, output)
    assert FB.sha256(stage / "weights.safetensors") == before
    assert (build / "model.elf").is_file()


@pytest.mark.parametrize("during", ["generation", "compile", "link"])
def test_origin_mutation_during_relink_never_returns_an_executable(image, tmp_path, monkeypatch, during):
    stage, build, record = image
    commands = []
    if during == "generation":
        generate = c_runtime.generate

        def changed(*args, **kwargs):
            result = generate(*args, **kwargs)
            (build / "weights_blob.o").write_bytes(b"corrupted object")
            return result

        monkeypatch.setattr(c_runtime, "generate", changed)

    def run(argv, cwd):
        commands.append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"new fixture output")
        if (during == "compile" and len(commands) == 1) or (during == "link" and len(commands) == 2):
            (build / "weights_blob.o").write_bytes(b"corrupted object")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(FB, "_run", run)
    with pytest.raises(FB.FullBuildError, match="changed|rebuild"):
        FB.relink(record, stage, {"in0": np.array([8, 9], np.float32)}, tmp_path / "step")
    assert len(commands) == {"generation": 0, "compile": 1, "link": 2}[during]


def _policy(*, elf, evidence_root, status, seen):
    evidence_root.mkdir(parents=True, exist_ok=True)
    report = {"status": status, "elf_sha256": FB.sha256(elf)}
    seen.append((Path(elf), report["elf_sha256"]))
    path = evidence_root / "policy.json"
    path.write_text(json.dumps(report))
    return {**report, "report_path": str(path), "report_sha256": FB.sha256(path)}


@pytest.mark.parametrize("image", [True], indirect=True)
@pytest.mark.parametrize("status", [None, "accepted", "refused"])
def test_device_relink_requires_current_policy_on_actual_new_image(image, tmp_path, monkeypatch, status):
    import inspect
    from functools import partial

    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    stage, _, record = image
    commands, seen = [], []

    def run(argv, cwd):
        commands.append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"new fixture executable")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(FB, "_run", run)
    service = None
    if status is not None:
        owner = Path(inspect.getsourcefile(_policy)).resolve()
        service = LinkedElfAdmissionService("neutral", partial(_policy, status=status, seen=seen),
                                            ((str(owner), FB.sha256(owner)),))
    inputs, directory = {"in0": np.array([8, 9], np.float32)}, tmp_path / "step"
    if status == "accepted":
        elf = FB.relink(record, stage, inputs, directory, admission=service)
        assert seen == [(elf, FB.sha256(elf))] and seen[0][1] != FB.sha256(Path(record["elf"]))
    else:
        with pytest.raises(FB.FullBuildError, match="currently selected|refused"):
            FB.relink(record, stage, inputs, directory, admission=service)
        assert len(commands) == (0 if status is None else 2)


@pytest.mark.parametrize("artifact", ["weights_blob.o", "cgen/model_gen.h", "compilation_recipe.json"])
def test_ordinary_build_repairs_stale_relink_products_instead_of_reusing_image(image, tmp_path, artifact):
    stage, build, record = image
    path = build / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    repaired = FB.build(stage, target="neutral", label="unit", submission=tmp_path / "host",
                        submission_sha256="unused", settings=FB.BuildSettings(**record["key"]["settings"]),
                        directory=Path(record["cache_directory"]), offload=False)
    assert not repaired["reused"] and repaired["relink_binding"]
    assert FB._verify_relink(repaired, stage)["status"] == "completed"


def test_relink_waits_for_build_lock_and_checks_origin_after_acquiring_it(image, tmp_path, monkeypatch):
    import contextvars
    import fcntl
    import select
    import subprocess
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    stage, build, record = image
    script = '''import fcntl,sys
with open(sys.argv[1], 'r') as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    print('locked', flush=True)
    sys.stdin.readline()
'''
    process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", script,
                                str(Path(record["cache_directory"]) / ".lock")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    attempted, checked = threading.Event(), threading.Event()
    original_lock, original_check = fcntl.flock, FB._verify_relink

    def flock(*args):
        attempted.set()
        return original_lock(*args)

    def verify(*args, **kwargs):
        checked.set()
        return original_check(*args, **kwargs)

    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(FB, "_verify_relink", verify)
    monkeypatch.setattr(FB, "_run", lambda *args, **kwargs: pytest.fail("changed origin launched a tool"))
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline().strip() == "locked"
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                future = pool.submit(contextvars.copy_context().run, FB.relink, record, stage,
                                     {"in0": np.array([8, 9], np.float32)}, tmp_path / "step")
                assert attempted.wait(5) and not checked.is_set() and not future.done()
                (build / "weights_blob.o").write_bytes(b"changed while waiting for the build lock")
            finally:
                process.communicate("\n", timeout=5)
            with pytest.raises(FB.FullBuildError, match="changed|rebuild"):
                future.result(timeout=5)
            assert checked.is_set()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _host_policy(*, elf, evidence_root, status):
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / "host-policy.json"
    result = {"status": status, "elf_sha256": FB.sha256(elf)}
    path.write_text(json.dumps(result))
    return {**result, "report_path": str(path), "report_sha256": FB.sha256(path)}


@pytest.mark.parametrize("status", ["accepted", "refused"])
def test_explicit_host_policy_checks_actual_relinked_image(image, tmp_path, monkeypatch, status):
    from functools import partial

    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    stage, _, record = image
    owner = Path(__file__).resolve()
    policy = LinkedElfAdmissionService(
        "neutral", partial(_host_policy, status=status), ((str(owner), FB.sha256(owner)),)
    )
    commands = []

    def run(argv, cwd):
        commands.append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"actual relinked fixture bytes")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(FB, "_run", run)
    directory = tmp_path / "checked-relink"
    if status == "refused":
        with pytest.raises(FB.FullBuildError, match="admission refused"):
            FB.relink(record, stage, {"in0": np.array([8, 9], np.float32)}, directory, admission=policy)
        assert json.loads((directory / "compilation_recipe.json").read_text())["status"] == "invoked"
    else:
        result = FB.relink(record, stage, {"in0": np.array([8, 9], np.float32)}, directory, admission=policy)
        assert result.read_bytes() == b"actual relinked fixture bytes"
    assert len(commands) == 2
    report = json.loads((directory / "cache-admission/host-policy.json").read_text())
    assert report == {"status": status, "elf_sha256": FB.sha256(directory / "model.elf")}

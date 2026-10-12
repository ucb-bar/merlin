"""Shared original-build routing controls; fake tools are never executed.

These controls establish file closure/selection and preserve the original core
owners. They deliberately issue no positive namespace or process observation.
"""

import copy
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import codegen as C
from merlin.runtime import _native_launch as L
from merlin.runtime import build_runtime as B
from merlin.targetgen.contract import compile as K


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "managed-output"))
    system = tmp_path / "system"
    (system / "bin").mkdir(parents=True)
    (system / "usr/bin").mkdir(parents=True)
    tool, bwrap = tmp_path / "compiler", tmp_path / "bwrap"
    for path in (system / "bin/bash", system / "usr/bin/env", tool, bwrap):
        path.write_bytes(b"fake executable bytes; never loaded")
        path.chmod(0o555)
    output = tmp_path / "output"
    output.mkdir()
    original_cwd = tmp_path / "original-cwd"
    original_cwd.mkdir()
    kwargs = dict(
        tools=(tool,),
        readonly_roots=((system / "bin", Path("/bin")), (system / "usr", Path("/usr"))),
        output_root=output,
        bwrap=bwrap,
        max_files=256,
        max_bytes=1024 * 1024,
        max_output_files=32,
        max_output_bytes=4096,
        max_console_bytes=4096,
    )
    owner = B.prepare_build_runtime_view(tmp_path / "snapshot", **kwargs)
    yield SimpleNamespace(**locals())
    try:
        owner.close()
    except ValueError:
        pass  # An integrity refusal retains the existing parent storage pin.


def selection(f, name, *, extra=(), environment=None, storage=False):
    job = f.output / name
    job.mkdir()
    (job / "tmp").mkdir()
    (job / "slots").mkdir()
    work = job / "build"
    work.mkdir()
    source = work / "kernel.ll"
    source.write_text("define void @entry() { ret void }\n")
    output = work / "kernel.o"
    env = {"TMPDIR": str(job / "tmp"), "SLOT_STORAGE": str(job / "slots"), "LANG": "C.UTF-8"}
    if environment is not None:
        env.update(environment)
    return f.owner.selection(
        [str(f.tool), "-O2", *extra, "-c", str(source), "-o", str(output)],
        inputs=(source,),
        outputs=(output,),
        cwd=f.original_cwd,
        environment=env,
        directory=work,
        stage="object",
        text=True,
        storage_roots=(job / "slots",) if storage else (),
    )


def test_original_absolute_source_uses_closed_cwd_without_grader_tree(frozen):
    selected = selection(frozen, "first", storage=True)
    assert selected is not None
    assert selected["identity"]["guest_cwd"] == "/run/work"
    assert not B._guest_member(frozen.owner.image, str(frozen.original_cwd)).exists()
    assert "/run/merlin/inputs/p0/kernel.ll" in selected["identity"]["guest_argv"]
    assert selected["child_environment"]["TMPDIR"] == "/tmp"
    assert selected["child_environment"]["SLOT_STORAGE"] == "/tmp/storage-0"
    assert selected["identity"]["storage_directories"] == ["/tmp/storage-0"]
    assert selected["identity"]["streams"]["newline"] is None
    assert frozen.owner.reopen_selection(selected) == selected


def test_executed_storage_relocation_hits_and_relevant_changes_miss(frozen):
    first, second = selection(frozen, "one", storage=True), selection(frozen, "two", storage=True)
    assert first["key"] == second["key"]
    assert first["request"]["environment"] != second["request"]["environment"]
    assert first["original_environment"] != second["original_environment"]
    changed_env = selection(frozen, "env", storage=True, environment={"LANG": "different"})
    changed_flags = selection(frozen, "flags", storage=True, extra=("-fno-builtin",))
    assert first["key"] != changed_env["key"] != changed_flags["key"]
    request = dict(second["request"])
    Path(request["inputs"][0]).write_text("define void @other() { ret void }\n")
    assert frozen.owner.selection(**request)["key"] != first["key"]
    with pytest.raises(ValueError, match="input bytes changed"):
        frozen.owner.reopen_selection(second)
    # Unselected host storage paths are neither omitted nor called normalized.
    assert selection(frozen, "unselected-one")["key"] != selection(frozen, "unselected-two")["key"]


def test_closed_multiple_fresh_inputs_keep_order_and_content(frozen):
    work = frozen.output / "link"
    work.mkdir()
    sources = tuple(work / name for name in ("harness.o", "kernel.o", "abi.ld", "operands.bin"))
    for index, source in enumerate(sources):
        source.write_bytes(bytes([index]))
    outputs = (work / "program.elf",)
    request = dict(
        argv=[str(frozen.tool), *map(str, sources), "-o", str(outputs[0])],
        inputs=sources,
        outputs=outputs,
        cwd=work,
        environment={},
        directory=work,
        stage="elf",
        text=True,
    )
    selected = frozen.owner.selection(**request)
    assert selected is not None and len(selected["identity"]["inputs"]) == 4
    assert selected["identity"]["guest_cwd"] == "/run/merlin/inputs/p0"
    sources[2].write_bytes(b"changed ABI")
    with pytest.raises(ValueError, match="input bytes changed"):
        frozen.owner.reopen_selection(selected)


@pytest.mark.parametrize(
    "extra",
    [
        ("@options",),
        ("-fplugin=unknown",),
        ("-fprofile-use=unknown",),
        ("-Irelative",),
        ("-MD",),
        ("-save-temps",),
        ("-unknown-build-choice",),
        ("-gsplit-dwarf",),
    ],
)
def test_unsupported_lookup_or_product_producer_stays_fresh(frozen, extra):
    assert selection(frozen, "unknown", extra=extra) is None


def test_products_collect_complete_object_and_stack_report_only(frozen):
    selected = selection(frozen, "products", extra=("-fstack-usage",), storage=True)
    request = dict(selected["request"])
    request["outputs"].append(str(Path(request["outputs"][0]).with_suffix(".su")))
    selected = frozen.owner.selection(**request)
    scratch = frozen.output / "scratch"
    scratch.mkdir()
    (scratch / "kernel.o").write_bytes(b"object")
    with pytest.raises(ValueError, match="missing product"):
        frozen.owner.collect_products(selected, scratch)
    assert not Path(request["outputs"][0]).exists()
    (scratch / "kernel.su").write_bytes(b"stack report")
    (scratch / "kernel.o").chmod(0o2640)
    (scratch / "kernel.su").chmod(0o600)
    (scratch / "extra").write_bytes(b"undeclared")
    with pytest.raises(ValueError, match="undeclared"):
        frozen.owner.collect_products(selected, scratch)
    assert not Path(request["outputs"][0]).exists()
    (scratch / "extra").unlink()
    frozen.owner.collect_products(selected, scratch)
    assert [Path(p).read_bytes() for p in request["outputs"]] == [b"object", b"stack report"]
    assert [stat.S_IMODE(Path(p).stat().st_mode) for p in request["outputs"]] == [0o640, 0o600]
    parent_identity = Path(request["directory"]).stat()
    replaced_inode = Path(request["outputs"][0]).stat().st_ino
    (scratch / "kernel.o").write_bytes(b"repaired object")
    (scratch / "kernel.o").chmod(0o644)
    frozen.owner.collect_products(selected, scratch)
    product = Path(request["outputs"][0])
    assert product.read_bytes() == b"repaired object" and stat.S_IMODE(product.stat().st_mode) == 0o644
    assert product.stat().st_ino != replaced_inode
    assert Path(request["directory"]).stat().st_ino == parent_identity.st_ino
    # Cache staging reopens selection after products exist; their bytes remain
    # independently authenticated by the actual process observation.
    assert frozen.owner.reopen_selection(selected) == selected
    assert frozen.owner.selection(**request) is None


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
def test_product_replacement_refuses_linked_or_special_destinations(frozen, kind):
    selected = selection(frozen, "unsafe-product", storage=True)
    product = Path(selected["request"]["outputs"][0])
    scratch = frozen.output / "scratch"
    scratch.mkdir()
    (scratch / "kernel.o").write_bytes(b"new object")
    original = frozen.output / "original"
    original.write_bytes(b"preserved original")
    if kind == "symlink":
        product.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, product)
    elif kind == "directory":
        product.mkdir()
    else:
        os.mkfifo(product)
    with pytest.raises(ValueError, match="unlinked regular|canonical and unlinked"):
        frozen.owner.collect_products(selected, scratch)
    assert original.read_bytes() == b"preserved original"


def test_budget_drift_saved_owner_and_forged_observation_refuse(frozen):
    with pytest.raises(ValueError, match="exact live"):
        copy.copy(frozen.owner).verify()
    with pytest.raises(ValueError, match="exact live"):
        with B.executing_build_runtime(frozen.owner.verify()):
            pass
    with pytest.raises(ValueError, match="original completed"):
        frozen.owner.reopen_observation({"body": {"returncode": 0}, "signature": "fabricated"})
    for field, value in (("max_files", 1), ("max_bytes", 1)):
        with pytest.raises(ValueError, match="bound"):
            B.prepare_build_runtime_view(frozen.tmp_path / field, **{**frozen.kwargs, field: value})
    frozen.tool.chmod(0o755)
    frozen.tool.write_bytes(b"changed selected compiler")
    with pytest.raises(ValueError, match="original build tool"):
        frozen.owner.verify()


def test_snapshot_membership_and_private_product_budget_refuse(frozen):
    work = frozen.output / "oversized"
    work.mkdir()
    (work / "product").write_bytes(b"12345")
    with pytest.raises(ValueError, match="byte budget"):
        L._measure_writable((work,), 8, 4)
    frozen.owner.image.chmod(0o755)
    (frozen.owner.image / "injected").write_text("changed membership")
    with pytest.raises(ValueError, match="snapshot"):
        frozen.owner.verify()


def test_uncovered_subprocess_marks_whole_build_ineligible(frozen):
    with B.executing_build_runtime(frozen.owner):
        scope = frozen.owner.begin_build()
        assert (
            B.run_in_build_view(
                [str(frozen.tool)],
                inputs=(),
                outputs=(),
                directory=frozen.output,
                stage="unsupported",
                timeout=1,
                text=True,
            )
            is None
        )
        assert frozen.owner.end_build(scope)["eligible"] is False


def test_core_codegen_reuse_hook_precedes_observer_and_declares_stack(monkeypatch, tmp_path):
    calls = []

    def selected(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "output\n", "")

    def forbidden(*args, **kwargs):
        pytest.fail("reuse must not fabricate a fresh subprocess observation")

    monkeypatch.setattr(B, "run_in_build_view", selected)
    monkeypatch.setattr(I, "observe", forbidden)
    obj = tmp_path / "kernel.o"
    C._run(
        ["selected-tool", "-fstack-usage", "-c", "kernel.ll", "-o", str(obj)],
        inputs=(tmp_path / "kernel.ll",),
        outputs=(obj,),
        timeout_s=1,
    )
    assert calls[0][1]["outputs"] == (obj, obj.with_suffix(".su"))
    assert calls[0][1]["env"] == dict(os.environ)
    monkeypatch.setattr(B, "run_in_build_view", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "refused"))
    with pytest.raises(C.CodegenError, match="refused"):
        C._run(["tool", "-o", str(obj)], inputs=("source",), outputs=(obj,), timeout_s=1)


def test_contract_hook_and_default_both_preserve_original_selection(monkeypatch, tmp_path):
    monkeypatch.delenv("MERLIN_BUILD_CPUS", raising=False)
    calls = []
    expected = subprocess.CompletedProcess(["tool"], 0, "result", "")
    monkeypatch.setattr(B, "run_in_build_view", lambda argv, **kw: (calls.append(kw), expected)[1])

    def forbidden(*args, **kwargs):
        pytest.fail("selected result must not create another invocation")

    monkeypatch.setattr(I, "run", forbidden)
    assert (
        K._observed_run(
            ["tool"],
            workdir=tmp_path,
            stage="elf",
            inputs=("a", "b"),
            outputs=("out",),
            capture_output=True,
            text=True,
            timeout=1,
        )
        is expected
    )
    assert calls[0]["inputs"] == ("a", "b") and calls[0]["stage"] == "elf"
    monkeypatch.setattr(B, "run_in_build_view", lambda *args, **kwargs: None)
    fresh = []
    monkeypatch.setattr(I, "run", lambda argv, **kw: (fresh.append((argv, kw)), expected)[1])
    assert (
        K._observed_run(
            ["tool"], workdir=tmp_path, stage="elf", inputs=("a", "b"), outputs=("out",), capture_output=True, text=True
        )
        is expected
    )
    assert fresh[0][0] == ["tool"]
    assert fresh[0][1]["inputs"] == ("a", "b") and fresh[0][1]["text"] is True


def test_native_wrapper_retains_original_mount_roster_and_stack(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(L, "run_process", lambda *args, **kw: (calls.append((args, kw)), "original result")[1])
    owner, selected, elf = object(), {}, tmp_path / "program.elf"
    assert L.run(owner, selected, elf, timeout=1, directory=tmp_path, unlimited_stack=True) == "original result"
    assert calls[0] == (
        (owner, selected),
        dict(
            readonly_mounts=((elf, "/run/merlin/input.elf"),),
            inputs=(elf,),
            timeout=1,
            directory=tmp_path,
            unlimited_stack=True,
        ),
    )


def test_effective_text_decoder_preserves_subprocess_newlines():
    result = subprocess.CompletedProcess([], 0, b"one\r\ntwo\rthree\n", b"warning\r\n")
    B.BuildRuntimeView._decode(result, {"request": {"text": True, "encoding": "utf-8", "errors": "strict"}})
    assert result.stdout == "one\ntwo\nthree\n" and result.stderr == "warning\n"


@pytest.mark.parametrize("stage", ["object", "harness_object", "elf"])
def test_shared_launcher_observes_original_selected_step_before_exec(frozen, monkeypatch, stage):
    """Stop at the real record boundary; no process or namespace is started."""
    for module in (B, I, L):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd() / "src")
    selected = selection(frozen, "stage")
    selected = frozen.owner.selection(**{**selected["request"], "stage": stage})
    calls = []

    class BeforeExec(RuntimeError):
        pass

    def observe(directory, **kwargs):
        invocation = I.Invocation(directory, **kwargs)
        calls.append((kwargs, invocation.path))
        raise BeforeExec("file-only control stops before Popen")

    monkeypatch.setattr(I, "observe", observe)
    with pytest.raises(BeforeExec):
        L.run_process(
            frozen.owner,
            selected,
            readonly_mounts=(),
            inputs=tuple(map(Path, selected["request"]["inputs"])),
            outputs=tuple(map(Path, selected["request"]["outputs"])),
            timeout=1,
            directory=Path(selected["request"]["directory"]),
            build=True,
        )
    kwargs, path = calls[0]
    record = json.loads(path.read_text())
    assert kwargs["stage"] == record["stage"] == stage
    assert record["kind"] == "subprocess" and record["status"] == "running"
    assert record["inputs"] == selected["input_pins"]
    assert kwargs["outputs"] == tuple(map(Path, selected["request"]["outputs"]))
    assert not any(Path(p).exists() for p in selected["request"]["outputs"])
    with pytest.raises(ValueError, match="did not complete"):
        I.verify(path)


def test_shared_launcher_refuses_detached_saved_stage_before_record(frozen, monkeypatch):
    selected = selection(frozen, "changed-stage")
    selected["identity"]["stage"] = "elf"
    monkeypatch.setattr(I, "observe", lambda *args, **kwargs: pytest.fail("changed selection reached observer"))
    with pytest.raises(ValueError, match="selection or fresh input bytes changed"):
        L.run_process(
            frozen.owner,
            selected,
            readonly_mounts=(),
            inputs=(),
            outputs=(),
            timeout=1,
            directory=frozen.output,
            build=True,
        )


@pytest.mark.parametrize("damage", [None, "stage", "outputs"])
def test_completed_build_joins_original_step_and_complete_products(frozen, monkeypatch, damage):
    """Synthetic completion checks joins only; it issues no kernel authority."""
    selected = selection(frozen, "completed")
    selected = frozen.owner.selection(**{**selected["request"], "stage": "elf"})

    def synthetic_process(owner, current, *, inputs, outputs, directory, **kwargs):
        assert current == selected
        for output in outputs:
            output.write_bytes(b"synthetic original product")
        if damage == "outputs":
            other = directory / "other.elf"
            other.write_bytes(b"unrelated output")
            outputs = (other,)
        result = subprocess.CompletedProcess(current["request"]["argv"], 0, b"link output\n", b"")
        with I.observe(
            directory,
            stage="object" if damage == "stage" else "elf",
            argv=result.args,
            cwd=directory,
            env={},
            inputs=inputs,
            outputs=outputs,
            dependencies=owner.observation_sources(),
        ) as recorded:
            recorded.complete(result)
        return result, recorded.path, {"control": "synthetic; no namespace observation"}

    monkeypatch.setattr(L, "run_process", synthetic_process)
    if damage is not None:
        with pytest.raises(ValueError, match="actual build stage" if damage == "stage" else "actual build products"):
            frozen.owner.run(selected, timeout=1)
        return
    result, observation = frozen.owner.run(selected, timeout=1)
    body = frozen.owner.reopen_observation(observation, selection=selected)
    original = I.verify(Path(body["record"]["path"]))
    assert original["stage"] == body["identity"]["stage"] == "elf"
    assert original["outputs"] == body["outputs"]
    assert result.stdout == "link output\n"
    changed = copy.deepcopy(selected)
    changed["identity"]["stage"] = "object"
    with pytest.raises(ValueError, match="selection or fresh input bytes changed"):
        frozen.owner.reopen_observation(observation, selection=changed)

"""Source-only cache routing and byte controls, without namespace/tool execution."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import codegen
from merlin.runtime import _native_files as F
from merlin.targetgen.contract import build_products as B


def _selection(tmp_path, monkeypatch, *, flags=(), stack=False):
    tool, source = tmp_path / "selected-clang", tmp_path / "kernel.ll"
    tool.write_text("source-only tool bytes; never executed")
    tool.chmod(0o755)
    source.write_text("; freshly emitted LLVM input")
    obj = tmp_path / "kernel.o"
    inputs, outputs = (source,), (obj, obj.with_suffix(".su")) if stack else (obj,)
    monkeypatch.setattr(codegen, "clang", lambda: tool)
    source_owner = Path(codegen.__file__).resolve()
    selected = {
        "key": "1" * 64,
        "identity": {
            "stage": "object",
            "tool": {"path": str(tool), "requested_path": str(tool)},
            "owner_source_pins": [F.pin(source_owner, max_bytes=1024 * 1024)],
            "guest_argv": [
                str(tool),
                "--target=explicit-unknown-elf",
                "-march=explicit_arch",
                "-mabi=explicit_abi",
                *flags,
                *(["-fstack-usage"] if stack else []),
                "-c",
                "/run/input/kernel.ll",
                "-o",
                "/run/work/kernel.o",
            ],
        },
        "request": {
            "argv": [str(tool)],
            "cwd": str(tmp_path),
            "environment": {},
            "inputs": list(map(str, inputs)),
            "outputs": list(map(str, outputs)),
            "directory": str(tmp_path),
            "text": True,
            "encoding": "utf-8",
        },
    }
    return selected, inputs, outputs


@pytest.mark.parametrize("stack", [False, True])
def test_original_explicit_llvm_object_scope_is_target_agnostic(tmp_path, monkeypatch, stack):
    selection, inputs, outputs = _selection(tmp_path, monkeypatch, flags=("-O2", "-ffreestanding"), stack=stack)
    assert B._eligible(selection, inputs, outputs)
    if stack:
        assert not B._eligible(selection, inputs, outputs[:1])


@pytest.mark.parametrize(
    "flag", ["-march=native", "@config.rsp", "-fplugin=external.so", "-fprofile-use=profile", "-Xclang", "-unknown"]
)
def test_unknown_cpu_plugin_profile_response_and_flags_stay_fresh(tmp_path, monkeypatch, flag):
    selection, inputs, outputs = _selection(tmp_path, monkeypatch, flags=(flag,))
    assert not B._eligible(selection, inputs, outputs)


@pytest.mark.parametrize("change", ["source", "stage", "tool", "implicit_abi"])
def test_unissued_original_object_choices_stay_fresh(tmp_path, monkeypatch, change):
    selection, inputs, outputs = _selection(tmp_path, monkeypatch)
    if change == "source":
        inputs = (inputs[0].with_suffix(".c"),)
    elif change == "stage":
        selection["identity"]["stage"] = "elf"
    elif change == "tool":
        selection["identity"]["tool"]["requested_path"] = "different-selector"
    else:
        selection["identity"]["guest_argv"] = [
            x for x in selection["identity"]["guest_argv"] if not x.startswith("-mabi=")
        ]
    assert not B._eligible(selection, inputs, outputs)


def test_restore_streams_exact_bytes_into_new_inode_and_preserves_parent(tmp_path):
    source, output = tmp_path / "original.o", tmp_path / "restored.o"
    source.write_bytes(b"\x00original compiler bytes\xff")
    source.chmod(0o755)
    output.write_text("old output")
    inode, parent = source.stat().st_ino, tmp_path.stat().st_ino
    expected = {"pin": F.pin(source, max_bytes=1024), "bytes": source.stat().st_size, "mode": 0o755}
    B._restore(source, output, expected, max_bytes=1024)
    assert output.read_bytes() == source.read_bytes()
    assert output.stat().st_ino != inode and output.stat().st_nlink == 1
    assert output.stat().st_mode & 0o777 == 0o755 and tmp_path.stat().st_ino == parent


@pytest.mark.parametrize("change", ["symlink", "hardlink", "directory", "bytes", "bound"])
def test_restore_refuses_unrelated_linked_special_or_changed_bytes(tmp_path, change):
    source, output = tmp_path / "original.o", tmp_path / "restored.o"
    source.write_bytes(b"original")
    expected = {"pin": F.pin(source, max_bytes=1024), "bytes": source.stat().st_size, "mode": 0o644}
    if change == "symlink":
        output.symlink_to(source)
    elif change == "hardlink":
        os.link(source, output)
    elif change == "directory":
        output.mkdir()
    elif change == "bytes":
        source.write_bytes(b"mutation")
    with pytest.raises(ValueError):
        B._restore(source, output, expected, max_bytes=1 if change == "bound" else 1024)
    assert not tuple(tmp_path.glob(".reused-*"))


def test_default_calibration_scope_has_no_product_reuse_or_publication(tmp_path):
    with B.executing_build_products():
        assert B.lookup_build_products(None, {}, (), (), tmp_path, "object") is None
        B.stage_build_products(None, {}, (), (), {})
    with pytest.raises(ValueError, match="exact original index"):
        with B.executing_build_products({"saved": "owner"}, job_root=tmp_path):
            pass


def test_reuse_receipt_cannot_be_read_as_a_fresh_invocation(tmp_path):
    record = tmp_path / "artifact_reuse.json"
    record.write_text(
        '{"schema":"merlin.artifact_reuse.v1","kind":"artifact_reuse","status":"restored","new_process":false}'
    )
    with pytest.raises(ValueError, match="did not complete"):
        I.verify(record)


def test_owner_imports_are_selected_from_this_managed_source():
    for module in (B, I, codegen):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd() / "src")


@pytest.fixture
def synthetic_archive(tmp_path, monkeypatch):
    """Stub only the real view/process gate to test host cache lifecycle.

    No process is invoked and no namespace/enforcement qualification is claimed.
    Production factory admission and real process adoption belong to the view.
    """
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    job = tmp_path / "original-job"
    job.mkdir(mode=0o700)
    selection, inputs, outputs = _selection(job, monkeypatch)
    outputs[0].write_bytes(b"complete synthetic original object bytes")
    with I.observe(
        job,
        stage="build_runtime_process",
        argv=[selection["identity"]["tool"]["path"]],
        inputs=inputs,
        outputs=outputs,
        env={},
    ) as observed:
        observed.complete(subprocess.CompletedProcess([], 0, b"original stdout\r\n", b"original stderr\r\n"))
    recorded = I.verify(observed.path)
    body = {
        "selection_key": selection["key"],
        "identity": selection["identity"],
        "record": F.pin(observed.path, max_bytes=1024 * 1024),
        "inputs": recorded["inputs"],
        "outputs": [F.pin(path, max_bytes=1024) for path in outputs],
        "stdout": recorded["stdout"],
        "stderr": recorded["stderr"],
        "returncode": 0,
    }
    observation = {"body": body, "signature": "synthetic gate only"}

    class ViewGate:
        output_root = tmp_path

        def verify(self):
            return {"view_sha256": "synthetic routing fixture only"}

        def reopen_selection(self, chosen):
            if chosen["identity"] != selection["identity"]:
                raise ValueError("selection changed")
            for path in chosen["request"]["inputs"]:
                if F.digest(path) != body["inputs"][0]["sha256"]:
                    raise ValueError("fresh input changed")

        def reopen_observation(self, value, *, selection=None, outputs=None):
            if value != observation:
                raise ValueError("no authenticated original process")
            I.verify(Path(value["body"]["record"]["path"]))
            return value["body"]

    root = tmp_path / "archive"
    root.mkdir(mode=0o700)
    for name in ("pending", "entries", "reads"):
        (root / name).mkdir(mode=0o700)
    issuer, view = object(), ViewGate()
    index = B.BuildProductIndex(
        root, context={}, build_owner=view, max_entries=4, max_bytes=8 * 1024 * 1024, max_files=256, issuer=issuer
    )
    monkeypatch.setitem(B._ISSUED, issuer, index)
    yield index, view, selection, inputs, outputs, observation, job
    # An unknown/damaged fixture deliberately retains its pins.
    try:
        index.close()
    except ValueError:
        pass


def test_synthetic_parent_publication_handoff_has_a_real_byte_hit_and_no_new_invocation(synthetic_archive, tmp_path):
    index, view, selection, inputs, outputs, observation, job = synthetic_archive
    index.stage_observation(view, selection, inputs=inputs, outputs=outputs, observation=observation, job_root=job)
    assert (
        index.lookup(view, selection, inputs=inputs, outputs=outputs, job_root=job, directory=job, stage="object")
        is None
    )
    assert index.publish_completed_job(job) == 1
    current = tmp_path / "current-job"
    current.mkdir(mode=0o700)
    fresh_input, restored = current / "kernel.ll", current / "kernel.o"
    fresh_input.write_bytes(inputs[0].read_bytes())
    chosen = copy.deepcopy(selection)
    chosen["request"].update(directory=str(current), inputs=[str(fresh_input)], outputs=[str(restored)])
    with B.executing_build_products(index, job_root=current):
        result = B.lookup_build_products(view, chosen, (fresh_input,), (restored,), current, "object")
    assert result is not None and result.stdout == "original stdout\n"
    assert result.stderr == "original stderr\n" and restored.read_bytes() == outputs[0].read_bytes()
    receipt = I.verify_artifact_reuse(Path(result.artifact_reuse_receipt))
    assert receipt["kind"] == "artifact_reuse" and receipt["new_process"] is False
    assert receipt["original_invocation"] == observation["body"]["record"]
    assert not tuple(current.rglob("invocation.json"))
    assert tuple((index.root / "reads").iterdir())
    assert index.discard_cleaned_job(current) == 0
    assert not tuple((index.root / "reads").iterdir())


@pytest.mark.parametrize("change", ["record", "product", "fresh_input", "signature"])
def test_synthetic_mutation_or_unknown_observation_cannot_hit(synthetic_archive, tmp_path, change):
    index, view, selection, inputs, outputs, observation, job = synthetic_archive
    if change == "signature":
        with pytest.raises(ValueError, match="authenticated"):
            index.stage_observation(
                view,
                selection,
                inputs=inputs,
                outputs=outputs,
                observation={"body": observation["body"], "signature": "saved JSON"},
                job_root=job,
            )
        assert not tuple((index.root / "pending").iterdir())
        return
    index.stage_observation(view, selection, inputs=inputs, outputs=outputs, observation=observation, job_root=job)
    index.publish_completed_job(job)
    if change == "record":
        Path(observation["body"]["record"]["path"]).write_text("{}")
    elif change == "product":
        outputs[0].write_bytes(b"changed original")
    else:
        inputs[0].write_bytes(b"changed fresh input")
    with B.executing_build_products(index, job_root=job):
        assert B.lookup_build_products(view, selection, inputs, outputs, job, "object") is None


def test_synthetic_pending_pin_discard_requires_creating_parent_and_known_evidence(synthetic_archive, monkeypatch):
    index, view, selection, inputs, outputs, observation, job = synthetic_archive
    index.stage_observation(view, selection, inputs=inputs, outputs=outputs, observation=observation, job_root=job)
    creator = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: creator + 1)
    with pytest.raises(ValueError, match="creating host parent"):
        index.publish_completed_job(job)
    with pytest.raises(ValueError, match="creating host parent"):
        index.discard_cleaned_job(job)
    assert tuple((index.root / "pending").iterdir())
    monkeypatch.setattr(os, "getpid", lambda: creator)
    assert index.discard_cleaned_job(job) == 1
    assert not tuple((index.root / "pending").iterdir())


def test_synthetic_saved_records_and_reconstructed_handles_cannot_issue_cache_authority(synthetic_archive):
    index, _, _, _, _, _, _ = synthetic_archive
    with pytest.raises(ValueError, match="exact live"):
        copy.copy(index).verify()
    path = index.root / "entries" / ("f" * 64 + ".json")
    path.write_text(json.dumps({"body": {}, "signature": "saved JSON"}))
    with pytest.raises(ValueError, match="not issued"):
        index._reopen(path)
    path.unlink()


def test_original_stack_repair_overwrite_is_discarded_only_after_parent_cleanup(synthetic_archive):
    from merlin.common import storage_lifecycle as storage

    index, view, selection, inputs, outputs, observation, job = synthetic_archive
    index.stage_observation(view, selection, inputs=inputs, outputs=outputs, observation=observation, job_root=job)
    assert storage.blockers(job)
    outputs[0].write_bytes(b"original stack-arena repair replaces the kernel object")
    # Pending bytes never become a hit or a publication. No cleanup/status JSON
    # supplies this parent seam; the actual pool calls it after real reaping.
    assert index.publish_completed_job(job) == 0
    assert not tuple((index.root / "entries").iterdir())
    assert not tuple((index.root / "pending").iterdir())
    assert not storage.blockers(job)


def test_observation_pins_stream_and_refuse_drift_without_full_file_loading(tmp_path, monkeypatch):
    source = tmp_path / "large-product.o"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("observation loaded a whole file"))
    assert I._pin(source)["sha256"] == F.digest(source)
    opened = Path.open

    class DriftingStream:
        def __init__(self, stream):
            self.stream, self.changed = stream, False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            chunk = self.stream.read(size)
            if not self.changed:
                self.changed = True
                source.chmod(0o600)  # ctime drift suffices even with unchanged byte count.
            return chunk

    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: DriftingStream(opened(self, *a, **kw)))
    assert I._pin(source)["sha256"] is None


def test_actual_issued_view_factory_joins_index_but_saved_completion_stays_unavailable(tmp_path, monkeypatch):
    from merlin.runtime import build_runtime as runtime

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "storage"))
    system = tmp_path / "system"
    (system / "bin").mkdir(parents=True)
    (system / "usr/bin").mkdir(parents=True)
    for tool in (system / "bin/bash", system / "usr/bin/env", tmp_path / "clang", tmp_path / "bwrap"):
        tool.write_bytes(b"source fixture only; never executed")
        tool.chmod(0o555)
    output = tmp_path / "output"
    output.mkdir()
    tool = tmp_path / "clang"
    monkeypatch.setattr(codegen, "clang", lambda: tool)
    view = runtime.prepare_build_runtime_view(
        tmp_path / "view",
        tools=(tool,),
        readonly_roots=((system / "bin", Path("/bin")), (system / "usr", Path("/usr"))),
        output_root=output,
        bwrap=tmp_path / "bwrap",
        max_files=256,
        max_bytes=1024 * 1024,
        max_output_files=16,
        max_output_bytes=4096,
        max_console_bytes=4096,
    )
    index = B.prepare_build_product_index(
        output / "index", context={}, build_owner=view, max_entries=4, max_bytes=1024 * 1024, max_files=256
    )
    assert index.verify(build_owner=view, context={})["build_view"] == view.verify()
    job = output / "job"
    job.mkdir(mode=0o700)
    source, product = job / "kernel.ll", job / "kernel.o"
    source.write_text("define void @entry() { ret void }\n")
    selected = view.selection(
        [
            str(tool),
            "--target=explicit-unknown-elf",
            "-march=explicit_arch",
            "-mabi=explicit_abi",
            "-O2",
            "-c",
            str(source),
            "-o",
            str(product),
        ],
        inputs=(source,),
        outputs=(product,),
        cwd=job,
        environment={},
        directory=job,
        stage="object",
        text=True,
    )
    assert selected is not None and B._eligible(selected, (source,), (product,))
    assert (
        index.lookup(view, selected, inputs=(source,), outputs=(product,), job_root=job, directory=job, stage="object")
        is None
    )
    with pytest.raises(ValueError, match="original completed"):
        index.stage_observation(
            view,
            selected,
            inputs=(source,),
            outputs=(product,),
            observation={"body": {"returncode": 0}, "signature": "saved completion JSON"},
            job_root=job,
        )
    assert not tuple((index.root / "pending").iterdir())
    assert not tuple(job.rglob("invocation.json")) and not product.exists()
    index.close()
    view.close()

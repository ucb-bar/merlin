"""Real repeated invocation replay retains every file/role and call boundary."""

import concurrent.futures
import copy
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common import selected_pin_replay as P

ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@pytest.fixture
def actual(tmp_path):
    tool = tmp_path / "cat"
    shutil.copyfile("/usr/bin/cat", tool)
    tool.chmod(0o700)
    rows = []
    for slot in range(3):
        source = tmp_path / ("input-" + str(slot))
        source.write_bytes(bytes(range(slot, slot + 32)))
        result = I.run(
            [str(tool), str(source)],
            directory=tmp_path / str(slot),
            cwd=tmp_path,
            stage="independent_complete_output_copy",
            inputs=[source],
            dependencies=[tool],
            env=ENV,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0 and result.stdout == source.read_bytes()
        record = next((tmp_path / str(slot) / "invocations").glob("*/invocation.json"))
        rows.append((source, record))
    return tool, rows


def test_actual_roster_shares_only_explicit_tool_and_rereads_both_boundaries(actual, monkeypatch):
    tool, rows = actual
    reads, ordinary = [], []
    fresh, pin = P._fresh_pin, I._pin
    monkeypatch.setattr(P, "_fresh_pin", lambda path: reads.append(path) or fresh(path))
    monkeypatch.setattr(I, "_pin", lambda path: ordinary.append(Path(path)) or pin(path))
    with P.replay_selected_pins([tool], max_pins=1) as owner:
        for source, record in rows:
            actual_record = I.require_environment(record, environment=ENV, pin_replay=owner)
            assert actual_record["argv"] == [str(tool), str(source)]
            assert Path(actual_record["stdout"]["path"]).read_bytes() == source.read_bytes()
    assert reads == [tool, tool]
    assert tool not in ordinary
    assert all(source in ordinary for source, _ in rows)
    for _, record in rows:
        I.require_environment(record, environment=ENV)
    assert ordinary.count(tool) == 2 * len(rows)


@pytest.mark.parametrize("when", ["preexisting", "during", "exceptional"])
def test_tool_content_drift_refuses_and_revokes_owner(actual, when):
    tool, rows = actual
    if when == "preexisting":
        tool.write_bytes(tool.read_bytes() + b"altered tool")
    owner = None
    with pytest.raises(ValueError, match="changed"):
        with P.replay_selected_pins([tool], max_pins=1) as owner:
            if when == "preexisting":
                I.verify(rows[0][1], pin_replay=owner)
            else:
                I.verify(rows[0][1], pin_replay=owner)
                tool.write_bytes(tool.read_bytes() + b"altered tool")
                if when == "exceptional":
                    raise RuntimeError("original caller also failed")
    with pytest.raises(ValueError, match="live"):
        owner.get(tool)


def test_unchanged_exception_propagates_and_revokes_owner(actual):
    tool, _ = actual
    with pytest.raises(RuntimeError, match="original caller"):
        with P.replay_selected_pins([tool], max_pins=1) as owner:
            raise RuntimeError("original caller")
    with pytest.raises(ValueError, match="live"):
        owner.get(tool)


def test_same_size_content_change_with_restored_mtime_still_refuses(actual):
    tool, _ = actual
    stat = tool.stat()
    with pytest.raises(ValueError, match="content changed"):
        with P.replay_selected_pins([tool], max_pins=1):
            data = bytearray(tool.read_bytes())
            data[-1] ^= 1
            tool.write_bytes(data)
            os.utime(tool, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert tool.stat().st_size == stat.st_size and tool.stat().st_mtime_ns == stat.st_mtime_ns


def test_named_complete_native_output_is_always_fresh(tmp_path):
    tool = tmp_path / "copy"
    shutil.copyfile("/usr/bin/cp", tool)
    tool.chmod(0o700)
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_bytes(bytes(range(64)))
    result = I.run(
        [str(tool), str(source), str(output)],
        directory=tmp_path / "execution",
        cwd=tmp_path,
        stage="ordinary_named_complete_output",
        inputs=[source],
        outputs=[output],
        dependencies=[tool],
        env=ENV,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0 and output.read_bytes() == source.read_bytes()
    record = next((tmp_path / "execution/invocations").glob("*/invocation.json"))
    with P.replay_selected_pins([tool], max_pins=1) as owner:
        I.verify(record, pin_replay=owner)
        output.write_bytes(b"different complete output")
        with pytest.raises(ValueError, match="changed"):
            I.verify(record, pin_replay=owner)


@pytest.mark.parametrize("role", ["input", "stdout", "stderr", "environment", "record_status"])
def test_each_unselected_product_and_record_field_is_still_reopened(actual, role):
    tool, rows = actual
    source, path = rows[0]
    document = json.loads(path.read_bytes())
    with P.replay_selected_pins([tool], max_pins=1) as owner:
        if role == "input":
            source.write_bytes(b"changed input")
        elif role in {"stdout", "stderr"}:
            Path(document[role]["path"]).write_bytes(b"changed product")
        elif role == "environment":
            document["environment"]["sha256"] = "0" * 64
            path.write_text(json.dumps(document))
        else:
            document["inputs_unchanged"] = False
            path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            I.require_environment(path, environment=ENV, pin_replay=owner)


def test_nested_scopes_have_explicit_distinct_members_and_fresh_content_reads(actual, tmp_path, monkeypatch):
    tool, _ = actual
    other = tmp_path / "other"
    other.write_bytes(b"independent second selection")
    reads, fresh = [], P._fresh_pin
    monkeypatch.setattr(P, "_fresh_pin", lambda path: reads.append(path) or fresh(path))
    with P.replay_selected_pins([tool], max_pins=1) as outer:
        with P.replay_selected_pins([other], max_pins=1) as inner:
            assert outer.get(other) is None and inner.get(tool) is None
            assert outer.get(tool) == I._pin(tool) and inner.get(other) == I._pin(other)
            returned = outer.get(tool)
            returned["sha256"] = "0" * 64
            assert outer.get(tool) == I._pin(tool)
        with pytest.raises(ValueError, match="live"):
            inner.get(other)
        assert outer.get(tool) == I._pin(tool)
    assert reads == [tool, other, other, tool]


def test_thread_and_copied_and_expired_owners_cannot_reuse_scope(actual):
    tool, rows = actual
    with P.replay_selected_pins([tool], max_pins=1) as owner:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(ValueError, match="same-thread"):
                pool.submit(I.verify, rows[0][1], pin_replay=owner).result()
        with pytest.raises(ValueError, match="live"):
            P.replayed_pin(copy.copy(owner), tool)
        with pytest.raises(ValueError, match="substitute"):
            P.replayed_pin({"path": str(tool)}, tool)
    with pytest.raises(ValueError, match="live"):
        I.verify(rows[0][1], pin_replay=owner)


def test_distinct_thread_cannot_inherit_an_unclosed_dead_thread_scope(tmp_path):
    tool = tmp_path / "selected-tool"
    tool.write_bytes(b"original selected bytes")
    retained = {}

    def issue():
        retained["thread"] = threading.current_thread()
        retained["context"] = P.replay_selected_pins([tool], max_pins=1)
        retained["owner"] = retained["context"].__enter__()

    original = threading.Thread(target=issue)
    original.start()
    original.join()

    def access():
        assert threading.current_thread() is not retained["thread"]
        try:
            retained["owner"].get(tool)
        except ValueError as error:
            retained["error"] = str(error)

    try:
        later = threading.Thread(target=access)
        later.start()
        later.join()
        assert "same-thread" in retained.get("error", "")
    finally:
        retained["context"].__exit__(None, None, None)
    with pytest.raises(ValueError, match="live"):
        retained["owner"].get(tool)


def test_forked_process_cannot_inherit_scope_but_observes_fresh_native_output(tmp_path):
    tool, source = tmp_path / "cat", tmp_path / "input"
    shutil.copyfile("/usr/bin/cat", tool)
    tool.chmod(0o700)
    source.write_bytes(bytes(range(64)))
    read_fd, write_fd = os.pipe()
    with P.replay_selected_pins([tool], max_pins=1) as owner:
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            report = {}
            try:
                try:
                    owner.get(tool)
                except ValueError as error:
                    report["refusal"] = str(error)
                result = I.run(
                    [str(tool), str(source)],
                    directory=tmp_path / "fresh-child",
                    cwd=tmp_path,
                    stage="ordinary_fork_observation_is_fresh",
                    inputs=[source],
                    dependencies=[tool],
                    env=ENV,
                    capture_output=True,
                    timeout=10,
                )
                report["returncode"] = result.returncode
                report["complete_output_equal"] = result.stdout == source.read_bytes()
            except Exception as error:
                report["error"] = type(error).__name__
            os.write(write_fd, json.dumps(report).encode())
            os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        report = json.loads(os.read(read_fd, 65536))
        os.close(read_fd)
        waited, status = os.waitpid(pid, 0)
        assert waited == pid and status == 0
        assert "same-process" in report.get("refusal", "")
        assert report["returncode"] == 0 and report["complete_output_equal"] is True
        record = next((tmp_path / "fresh-child/invocations").glob("*/invocation.json"))
        observed = I.require_environment(record, environment=ENV)
        assert observed["executable"] == I._pin(tool)
        assert observed["dependencies"] == [I._pin(tool)]
        assert owner.get(tool) == I._pin(tool)


def test_actual_observation_frames_always_hash_fresh_inside_selected_scope(actual, tmp_path):
    tool, _ = actual
    with pytest.raises(ValueError, match="content changed"):
        with P.replay_selected_pins([tool], max_pins=1) as owner:
            initial = owner.get(tool)
            tool.write_bytes(tool.read_bytes() + b"different executable frame")
            source = tmp_path / "fresh-input"
            source.write_bytes(b"ordinary actual output")
            result = I.run(
                [str(tool), str(source)],
                directory=tmp_path / "fresh",
                cwd=tmp_path,
                stage="ordinary_observation_is_fresh",
                inputs=[source],
                dependencies=[tool],
                env=ENV,
                capture_output=True,
                timeout=10,
            )
            assert result.returncode == 0 and result.stdout == source.read_bytes()
            record = next((tmp_path / "fresh/invocations").glob("*/invocation.json"))
            observed = json.loads(record.read_bytes())
            assert observed["executable"] == I._pin(tool) != initial
            assert observed["executable_unchanged"] is True and observed["dependencies_unchanged"] is True
            with pytest.raises(ValueError, match="changed"):
                I.verify(record, pin_replay=owner)


def test_all_selected_files_reread_on_failed_exceptional_exit(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("first", "second")]
    for path in paths:
        path.write_bytes(b"original bytes")
    reads, fresh = [], P._fresh_pin
    monkeypatch.setattr(P, "_fresh_pin", lambda path: reads.append(path) or fresh(path))
    with pytest.raises(ValueError, match="content changed"):
        with P.replay_selected_pins(paths, max_pins=2):
            paths[0].unlink()
            raise RuntimeError("caller failure")
    assert reads == [*paths, *paths]


@pytest.mark.parametrize("defect", ["bool", "over_budget", "duplicate", "relative", "symlink", "missing", "empty"])
def test_selection_is_bounded_and_canonical_before_hashing(tmp_path, monkeypatch, defect):
    source = tmp_path / "tool"
    source.write_bytes(b"original bytes")
    selected, limit = [source], 1
    if defect == "bool":
        limit = True
    elif defect == "over_budget":
        selected = [source, source]
    elif defect == "duplicate":
        selected, limit = [source, source], 2
    elif defect == "relative":
        selected = [Path("relative")]
    elif defect == "symlink":
        link = tmp_path / "alias"
        link.symlink_to(source)
        selected = [link]
    elif defect == "missing":
        source.unlink()
    else:
        selected = []
    monkeypatch.setattr(P, "_fresh_pin", lambda _: pytest.fail("invalid scope hashed a file"))
    with pytest.raises(ValueError):
        with P.replay_selected_pins(selected, max_pins=limit):
            pytest.fail("invalid scope was issued")


def test_actual_declared_stock_tools_replay_every_original_parser_row(tmp_path, monkeypatch):
    selection = os.environ.get("MERLIN_TEST_PIN_REPLAY_TOOLS")
    if selection is None:
        pytest.skip("actual stock parser tools require an explicit native selection")
    tools = json.loads(selection)
    assert isinstance(tools, list) and tools and all(type(path) is str for path in tools)
    for ordinal, path in enumerate(tools):
        tool = Path(path)
        owner = tmp_path / str(ordinal)
        owner.mkdir()
        rows = []
        for slot in range(3):
            source, output = owner / (str(slot) + ".mlir"), owner / (str(slot) + "-parsed.mlir")
            source.write_text(
                "module { hw.module @direct(in %a : i"
                + str(slot + 3)
                + ", out y : i"
                + str(slot + 3)
                + ") { hw.output %a : i"
                + str(slot + 3)
                + " } }\n"
            )
            argv = [str(tool), str(source), "--verify-each", "-o", str(output)]
            result = I.run(
                argv,
                directory=owner / str(slot),
                cwd=owner,
                stage="ordinary_exact_stock_parser",
                inputs=[source],
                outputs=[output],
                dependencies=[tool],
                env=ENV,
                capture_output=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr.decode()
            assert output.is_file() and "hw.output" in output.read_text()
            rows.append((source, output, argv, next((owner / str(slot) / "invocations").glob("*/invocation.json"))))
        reads, fresh = [], P._fresh_pin
        with monkeypatch.context() as patch:
            patch.setattr(P, "_fresh_pin", lambda path: reads.append(path) or fresh(path))
            with P.replay_selected_pins([tool], max_pins=1) as scope:
                for source, output, argv, path in rows:
                    observed = I.require_environment(path, environment=ENV, pin_replay=scope)
                    assert observed["argv"] == argv
                    assert observed["inputs"] == [I._pin(source)] and observed["outputs"] == [I._pin(output)]
                    assert observed["dependencies"] == [I._pin(tool)]
            assert reads == [tool, tool]

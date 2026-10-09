"""Isolated queue source controls; no shared daemon or real FireSim dispatch."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE_SHA = "3319b5e3aaaf9d6a7ebf70486130848a0d9ba85b0a7276cb2b9d7f3d1181e566"
OWNER = Path(__file__).resolve().parent
REPO = OWNER.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path, name, payload=b"independent-test-bytes", *, link=False):
    with tarfile.open(path, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../excluded"
            stream.addfile(member)
        else:
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))


@pytest.fixture
def queue(tmp_path, monkeypatch):
    selected = os.environ.get("MERLIN_TEST_QUEUE_SOURCE")
    if not selected:
        pytest.skip("explicit source-pinned external queue selection required")
    original = Path(selected)
    assert digest(original) == SOURCE_SHA
    install = tmp_path / "install"
    install.mkdir()
    shutil.copyfile(original, install / "firesim_queue.py")
    subprocess.run(
        ["/usr/bin/patch", "--batch", "--fuzz=0", "-p1", "-i", str(OWNER / "committed-inputs.patch")],
        cwd=install,
        check=True,
        capture_output=True,
    )
    shutil.copyfile(REPO / "src/merlin/common/pinned_files.py", install / "pinned_files.py")
    monkeypatch.setenv("FIRESIM_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("FIRESIM_QUEUE_HWDB_SNAPSHOT_ROOT", str(tmp_path / "private"))
    spec = importlib.util.spec_from_file_location("isolated_queue_control", install / "firesim_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POLL_INTERVAL_SECONDS = 0.01
    module.HEARTBEAT_INTERVAL_SECONDS = 0.01
    monkeypatch.setattr(module, "_require_shared_group", lambda: None)
    return module


def inputs(tmp_path):
    elf = tmp_path / "control.elf"
    elf.write_bytes(b"original independent control ELF fixture")
    bit = tmp_path / "bit.tgz"
    driver = tmp_path / "driver.tgz"
    archive(bit, "platform/image.bin")
    archive(driver, "driver.bin")
    hwdb = tmp_path / "hwdb.yaml"
    hwdb.write_text(f"selected:\n  bitstream_tar: {bit.as_uri()}\n  driver_tar: {driver.as_uri()}\n")
    files = [[str(x), digest(x)] for x in (elf, bit, driver)]
    uris = [["bitstream_tar", bit.as_uri(), "firesim.tar.gz"], ["driver_tar", driver.as_uri(), "driver-bundle.tar.gz"]]
    slots = [["owned-slot", "job-control.elf"]]
    return elf, bit, driver, hwdb, files, uris, slots


def declaration(queue, items):
    elf, _, _, hwdb, files, uris, slots = items
    return queue._committed_declaration(
        files, slots, uris, stage_from=str(elf), bootbinary=elf.name, hw_config="selected", hwdb_raw=hwdb.read_bytes()
    )


def test_exact_private_uri_members_and_source_mutation(queue, tmp_path):
    items = inputs(tmp_path)
    selection = declaration(queue, items)
    root = queue._prepare_hwdb_snapshot_root()
    original = root / "original.yaml"
    original.write_bytes(items[3].read_bytes())
    state = queue._private_committed_inputs(selection, root, 1, original, "selected", str(items[0]))
    assert state["effective"].sha256 != state["original_hwdb"].sha256
    assert state["root"].stat().st_mode & 0o777 == 0o700
    assert all(pin.path.stat().st_mode & 0o777 == 0o444 for pin in state["private_files"])
    for selected in state["archives"]:
        assert selected["private_uri"] != selected["source_uri"]
        assert str(state["root"]) in selected["private_uri"]
    items[0].write_bytes(b"changed after snapshot")
    staged = tmp_path / "staged.elf"
    shutil.copyfile(state["staged"].path, staged)
    (tmp_path / "simulation").mkdir()
    queue._verify_committed_inputs(state, staged, tmp_path / "simulation", "staging", consumed=False)


@pytest.mark.parametrize(
    "defect",
    [
        "stale",
        "duplicates",
        "missing-slot",
        "wrong-uri",
        "missing-archive",
        "foreign-file",
        "slot-escape",
        "slot-alias",
        "helper",
    ],
)
def test_declaration_refuses_changed_or_incomplete_membership(queue, tmp_path, defect):
    items = inputs(tmp_path)
    if defect == "stale":
        items[0].write_bytes(b"different")
    elif defect == "duplicates":
        items[4].append(items[4][0])
    elif defect == "missing-slot":
        items[6].clear()
    elif defect == "wrong-uri":
        items[5][0][1] = items[2].as_uri()
    elif defect == "missing-archive":
        items[5].pop()
    elif defect == "foreign-file":
        extra = tmp_path / "foreign"
        extra.write_bytes(b"foreign")
        items[4].append([str(extra), digest(extra)])
    elif defect == "slot-escape":
        items[6][0][0] = "../outside"
    elif defect == "slot-alias":
        items[6].append(items[6][0])
    elif defect == "helper":
        Path(queue.__file__).with_name("pinned_files.py").write_text("raise RuntimeError('unselected')")
    with pytest.raises((RuntimeError, ValueError)):
        declaration(queue, items)


@pytest.mark.parametrize(
    "raw",
    [
        b"selected: {bitstream_tar: one, bitstream_tar: two}",
        b"selected: &x {driver_tar: a}\nother: *x",
        b"selected: {}\nforeign: {}",
        b"!!python/object:foreign {}",
    ],
)
def test_structural_hwdb_ambiguous_or_unsafe_refusal(queue, raw):
    with pytest.raises(Exception):
        queue._committed_hwdb(raw, "selected")


@pytest.mark.parametrize("name,link", [("../escape", False), ("/absolute", False), ("link", True)])
def test_archive_paths_and_links_refused(queue, tmp_path, name, link):
    path = tmp_path / "bad.tgz"
    archive(path, name, link=link)
    with pytest.raises(RuntimeError):
        queue._archive_members(path)


FAKE_FIRESIM = r"""#!/usr/bin/python3
import fcntl, hashlib, json, os, pathlib, shutil, sys, tarfile
import yaml
cfg = json.loads(pathlib.Path(os.environ["OWNED_FAKE_CONFIG"]).read_text())
command = sys.argv[-1]
with pathlib.Path(cfg["lock"]).open("r+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held = True
    else:
        held = False
assert held, "lifecycle lock was not held during subprocess"
with pathlib.Path(cfg["events"]).open("a") as out:
    out.write(json.dumps({"command": command, "lock_held": held}) + "\n")
if cfg["mode"] == "legacy":
    sys.exit(0)
runtime = yaml.safe_load(pathlib.Path(sys.argv[sys.argv.index("-c")+1]).read_text())
simulation = pathlib.Path(runtime["run_farm"]["recipe_arg_overrides"]["default_simulation_dir"])
slot = simulation / "owned-slot"
if command == "infrasetup":
    hwdb = yaml.safe_load(pathlib.Path(sys.argv[sys.argv.index("-a")+1]).read_text())["selected"]
    slot.mkdir()
    shutil.copyfile(cfg["staged"], slot / "job-control.elf")
    cache = pathlib.Path(cfg["cache"])
    cache.mkdir(exist_ok=True)
    for field, basename in (("bitstream_tar", "firesim.tar.gz"), ("driver_tar", "driver-bundle.tar.gz")):
        uri = hwdb[field]
        cached = cache / hashlib.sha256(uri.encode()).hexdigest()
        if not cached.exists():
            shutil.copyfile(pathlib.Path(uri.removeprefix("file://")), cached)
        shutil.copyfile(cached, slot / basename)
        with tarfile.open(cached) as archive:
            for member in archive:
                output = slot / member.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.extractfile(member).read())
    if cfg["mode"] == "wrong-elf":
        (slot / "job-control.elf").write_bytes(b"wrong elf")
    if cfg["mode"] == "wrong-member":
        (slot / "driver.bin").write_bytes(b"wrong extracted member")
    if cfg["mode"] == "wrong-archive":
        (slot / "driver-bundle.tar.gz").write_bytes(b"wrong archive")
    if cfg["mode"] == "slot-alias":
        renamed = simulation / "other-slot"
        slot.rename(renamed)
        slot.symlink_to(renamed)
    if cfg["mode"] == "wrong-staged":
        pathlib.Path(cfg["staged"]).write_bytes(b"changed staged elf")
    if cfg["mode"] == "runtime-mutation":
        with pathlib.Path(sys.argv[sys.argv.index("-c")+1]).open("a") as out:
            out.write("\n# changed runtime bytes\n")
    if cfg["mode"] == "hwdb-mutation":
        with pathlib.Path(sys.argv[sys.argv.index("-a")+1]).open("a") as out:
            out.write("\n# changed effective HWDB bytes\n")
if command == "runworkload" and cfg["mode"] == "run-mutation":
    (slot / "driver.bin").write_bytes(b"changed during execution")
if command == "kill" and slot.exists() and cfg["mode"] == "teardown-mutation":
    (slot / "job-control.elf").write_bytes(b"changed during teardown")
"""


def submit_and_run(queue, tmp_path, mode, *, committed=True, after_submit=None):
    items = inputs(tmp_path)
    elf, bit, driver, hwdb, files, uris, slots = items
    chipyard = tmp_path / "chipyard"
    deploy = chipyard / "sims/firesim/deploy"
    (deploy / "workloads").mkdir(parents=True)
    (chipyard / "env.sh").write_text(f"export PATH={tmp_path / 'bin'}:/usr/bin:/bin\n")
    (deploy.parent / "sourceme-manager.sh").write_text("")
    (deploy / "workloads/control.json").write_text(json.dumps({"common_bootbinary": elf.name}))
    (deploy / "config_runtime.yaml").write_text(
        "run_farm:\n  recipe_arg_overrides:\n    default_simulation_dir: ignored\n"
        "workload:\n    workload_name: ignored\n    suffix_tag: ignored\n"
    )
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin/firesim"
    fake.write_text(FAKE_FIRESIM)
    fake.chmod(0o755)
    cfg = tmp_path / "fake.json"
    events = tmp_path / "events.jsonl"
    cache = tmp_path / "cache"
    cache.mkdir()
    for archive_file in (bit, driver):
        (cache / hashlib.sha256(archive_file.as_uri().encode()).hexdigest()).write_bytes(b"stale original URI cache")
    cfg.write_text(
        json.dumps(
            {
                "mode": mode,
                "lock": str(queue.FPGA_LOCK),
                "events": str(events),
                "cache": str(cache),
                "staged": str(deploy / "workloads/control" / elf.name),
            }
        )
    )
    # The exact process environment is explicitly stored in the isolated job.
    args = SimpleNamespace(
        chipyard=str(chipyard),
        workload="control",
        bootbinary=elf.name,
        stage_from=str(elf),
        hw_config="selected" if committed else None,
        hwdb_config_artifact=str(hwdb) if committed else None,
        committed_file=files if committed else None,
        consumed_slot=slots if committed else None,
        hwdb_local_uri=uris if committed else None,
        user="owned-test",
        priority=5,
        project="test",
        timeout=5,
        background=True,
    )
    assert queue.cmd_runworkload_full(args) == 0
    conn = queue._connect()
    conn.row_factory = __import__("sqlite3").Row
    job = dict(conn.execute("SELECT * FROM jobs").fetchone())
    job["env_json"] = json.dumps({"OWNED_FAKE_CONFIG": str(cfg), "PATH": "/usr/bin:/bin"})
    if after_submit:
        after_submit(items)
    result = queue._run_one_job_runworkload_full(conn, job)
    row = dict(conn.execute("SELECT * FROM jobs").fetchone())
    observed = [json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
    conn.close()
    return result, row, observed, items


def test_actual_lifecycle_private_cache_and_full_consumption(queue, tmp_path):
    rc, row, events, _ = submit_and_run(queue, tmp_path, "positive")
    assert rc == 0 and row["state"] == "DONE"
    assert [x["command"] for x in events] == ["kill", "infrasetup", "runworkload", "kill"]
    assert all(x["lock_held"] for x in events)
    root = queue.HWDB_SNAPSHOT_ROOT / "job-1-inputs"
    for phase in ("before_leading_kill", "before_infrasetup", "before_runworkload", "after_teardown"):
        report = json.loads((root / f"observed-{phase}.json").read_text())
        assert report["original_hwdb_sha256"] != report["effective_hwdb_sha256"]
    assert json.loads((root / "observed-before_runworkload.json").read_text())["consumed"]


@pytest.mark.parametrize(
    "mode",
    ["wrong-elf", "wrong-member", "wrong-archive", "slot-alias", "wrong-staged", "runtime-mutation", "hwdb-mutation"],
)
def test_actual_bad_staged_consumption_refuses_run_but_tears_down(queue, tmp_path, mode):
    rc, row, events, _ = submit_and_run(queue, tmp_path, mode)
    assert rc != 0 and row["state"] == "FAILED"
    assert [x["command"] for x in events] == ["kill", "infrasetup", "kill"]
    assert all(x["lock_held"] for x in events)
    assert (queue.HWDB_SNAPSHOT_ROOT / "job-1-inputs/refused-lifecycle.json").is_file()


@pytest.mark.parametrize("mode", ["run-mutation", "teardown-mutation"])
def test_actual_late_byte_change_retains_failed_outcome_after_teardown(queue, tmp_path, mode):
    rc, row, events, _ = submit_and_run(queue, tmp_path, mode)
    assert rc != 0 and row["state"] == "FAILED"
    assert [x["command"] for x in events] == ["kill", "infrasetup", "runworkload", "kill"]
    assert (queue.HWDB_SNAPSHOT_ROOT / "job-1-inputs/refused-teardown.json").is_file()


@pytest.mark.parametrize("index", [0, 1, 3])
def test_actual_source_change_after_submission_launches_no_phase(queue, tmp_path, index):
    rc, row, events, _ = submit_and_run(
        queue, tmp_path, "positive", after_submit=lambda items: items[index].write_bytes(b"changed before dispatch")
    )
    assert rc != 0 and row["state"] == "FAILED" and not events
    assert (queue.HWDB_SNAPSHOT_ROOT / "job-1-refused-lifecycle.json").is_file()


def test_simulation_root_alias_refused_before_any_fire_sim(queue, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()

    def replace_root(items):
        (queue.JOBS_DIR / "1/simulation").symlink_to(outside)

    rc, row, events, _ = submit_and_run(queue, tmp_path, "positive", after_submit=replace_root)
    assert rc != 0 and row["state"] == "FAILED" and not events


def test_optional_private_renderer_does_not_write_shared_yaml_alias(queue, tmp_path):
    excluded = tmp_path / "excluded-owned-data"
    excluded.write_bytes(b"excluded original bytes")

    def alias_runtime(items):
        (queue.JOBS_DIR / "1/config_runtime.yaml").symlink_to(excluded)

    rc, row, events, _ = submit_and_run(queue, tmp_path, "positive", after_submit=alias_runtime)
    assert rc == 0 and row["state"] == "DONE"
    assert excluded.read_bytes() == b"excluded original bytes"
    assert [x["command"] for x in events] == ["kill", "infrasetup", "runworkload", "kill"]


def test_legacy_without_selection_does_not_load_helper(queue, tmp_path):
    Path(queue.__file__).with_name("pinned_files.py").unlink()
    rc, row, events, _ = submit_and_run(queue, tmp_path, "legacy", committed=False)
    assert rc == 0 and row["state"] == "DONE"
    assert [x["command"] for x in events] == ["kill", "infrasetup", "runworkload", "kill"]

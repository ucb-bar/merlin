"""The compiler-side knob helper reads exactly what the idle tuner's broker override writes.

``merlin.compile.tuning_knobs`` is staged with the editable compiler library: an author's backend calls
``knob(...)`` where it makes a choice, and ``declare(...)`` writes ``TUNING_KNOBS.json``. These tests hold
it to the broker's own rules (:mod:`merlin_experiments.phase1.brokers.selfcheck_tuning`): the same
declarations are accepted and refused, the same single-knob alternatives are listed, and a compiler run
on the broker's private copy -- after ``apply_request`` or through the real broker loop -- reads the
overridden value while ``submission/`` keeps its own.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1.brokers import selfcheck_tuning as broker_tuning
from merlin_experiments.phase1.context import InvocationContext

from merlin.compile import tuning_knobs as K

BACKEND = """\
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "compiler_library"))  # the staged editable library, as an entrypoint does
from merlin.compile.tuning_knobs import knob
assert Path(sys.modules["merlin.compile.tuning_knobs"].__file__).resolve().is_relative_to(ROOT)
print(json.dumps({"overlap": knob("overlap", default="off", values=("off", "on"), root=ROOT),
                  "block": knob("block", default=2, values=(1, 2, 4), root=ROOT)}))
"""


@pytest.fixture(autouse=True)
def _fresh_state():
    K.use_root(None)
    K._CACHE.clear()
    K._CONSULTED.clear()
    yield
    K.use_root(None)


def _submission(tmp_path: Path) -> Path:
    root = tmp_path / "ws" / "submission"
    root.mkdir(parents=True)
    (root / "backend.py").write_text(BACKEND)
    library = root / "compiler_library" / "merlin" / "compile"
    library.mkdir(parents=True)
    for package in (library.parent, library):
        (package / "__init__.py").write_text("")
    shutil.copy2(K.__file__, library / "tuning_knobs.py")  # stdlib only: runs from the staged copy alone
    K.declare(root, {"overlap": K.Knob(("off", "on"), default="off", mechanism="readout_overlap"),
                     "block": K.Knob((1, 2, 4), default=2, note="rows per step")})
    return root


def _run_backend(root: Path) -> dict:
    out = subprocess.run([sys.executable, str(root / "backend.py")], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_declare_writes_a_declaration_the_broker_accepts_and_defaults_the_settings(tmp_path):
    root = _submission(tmp_path)
    declared = broker_tuning.declaration(root)
    assert declared == K.declaration(root)
    assert declared["settings_file"] == K.DEFAULT_SETTINGS
    assert declared["knobs"]["overlap"]["mechanism"] == "readout_overlap"
    assert json.loads((root / K.DEFAULT_SETTINGS).read_text()) == {"block": 2, "overlap": "off"}
    assert K.alternatives(root) == [{k: v for k, v in alt.items() if k != "note"}
                                    for alt in broker_tuning.alternatives(root)]
    assert [(a["knob"], a["value"]) for a in K.alternatives(root)] == [("overlap", "on"), ("block", 1), ("block", 4)]
    report = K.check(root)
    assert report["valid"] and report["current"] == {"overlap": "off", "block": 2}


def test_declare_keeps_existing_valid_settings_and_other_keys(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "knobs.json").write_text(json.dumps({"overlap": "on", "block": 3, "unrelated": [1]}))
    K.declare(root, {"overlap": K.Knob(("off", "on"), default="off"), "block": K.Knob((1, 2), default=1)},
              settings_file="knobs.json")
    assert json.loads((root / "knobs.json").read_text()) == {"overlap": "on", "block": 1, "unrelated": [1]}


def test_knob_reads_the_setting_and_falls_back_to_the_default(tmp_path):
    root = _submission(tmp_path)
    assert K.knob("overlap", default="off", values=("off", "on"), root=root) == "off"
    K.set_values(root, {"overlap": "on"})
    assert K.knob("overlap", default="off", values=("off", "on"), root=root) == "on"
    assert K.knob("undeclared", default=7, values=(7, 8), root=root) == 7
    assert K.knob("overlap", default="off", values=("off", "on"), root=tmp_path / "nothing") == "off"
    assert K.consulted() == {"overlap": "off", "undeclared": 7}


def test_knob_refuses_values_the_call_site_does_not_accept(tmp_path):
    root = _submission(tmp_path)
    with pytest.raises(K.KnobError, match="not accepted"):
        K.knob("block", default=2, values=(1, 2), root=root)  # 4 is declared, so tooling may set it
    (root / K.DEFAULT_SETTINGS).write_text(json.dumps({"overlap": True, "block": 2}))
    with pytest.raises(K.KnobError, match="not one of"):
        K.knob("overlap", default="off", values=("off", "on"), root=root)  # True is not "on"
    with pytest.raises(K.KnobError, match="must be one of"):
        K.knob("overlap", default="maybe", values=("off", "on"), root=root)
    with pytest.raises(K.KnobError, match="not a declared knob value"):
        K.set_values(root, {"block": 3})


def test_typed_comparison_matches_the_broker(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    K.declare(root, {"flag": K.Knob((False, True), default=False)})
    (root / K.DEFAULT_SETTINGS).write_text(json.dumps({"flag": 0}))
    assert K.check(root)["problems"] == ["flag: current setting 0 is not a declared value"]
    with pytest.raises(K.KnobError):
        K.knob("flag", default=False, values=(False, True), root=root)
    assert [a["value"] for a in K.alternatives(root)] == [a["value"] for a in broker_tuning.alternatives(root)]


@pytest.mark.parametrize("document", [
    {"schema": "other", "settings_file": "s.json", "knobs": {"a": {"values": [1]}}},
    {"schema": K.SCHEMA, "settings_file": "../s.json", "knobs": {"a": {"values": [1]}}},
    {"schema": K.SCHEMA, "settings_file": ".hidden/s.json", "knobs": {"a": {"values": [1]}}},
    {"schema": K.SCHEMA, "settings_file": "/abs/s.json", "knobs": {"a": {"values": [1]}}},
    {"schema": K.SCHEMA, "settings_file": "s.yaml", "knobs": {"a": {"values": [1]}}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {f"k{i}": {"values": [1]} for i in range(17)}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {"a": {"values": list(range(9))}}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {"a": {"values": [[1]]}}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {"a": {"values": ["x" * 65]}}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {"a" * 65: {"values": [1]}}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {"a": [1]}},
    {"schema": K.SCHEMA, "settings_file": "s.json", "knobs": {"a": {"values": [1], "mechanism": 3}}},
    {"schema": K.SCHEMA, "settings_file": "dir/s.json", "knobs": {"a": {"values": [None, 1.5, "x", True]}}},
])
def test_declarations_are_accepted_and_refused_exactly_as_the_broker_does(tmp_path, document):
    root = tmp_path / "tree"
    root.mkdir()
    (root / K.DECLARATION).write_text(json.dumps(document))
    accepted = broker_tuning.declaration(root) is not None
    assert (not K.declaration_problems(document)) is accepted
    if accepted:
        assert K.declaration(root) == broker_tuning.declaration(root)
    else:
        with pytest.raises(K.KnobError):
            K.declaration(root)
        assert not K.check(root)["valid"]


def test_the_limits_are_the_brokers(tmp_path):
    for name in ("SCHEMA", "DECLARATION", "MAX_KNOBS", "MAX_VALUES", "MAX_TEXT", "MAX_BYTES"):
        assert getattr(K, name) == getattr(broker_tuning, name), name


def test_root_comes_from_use_root_or_the_running_program(tmp_path, monkeypatch):
    root = _submission(tmp_path)
    K.set_values(root, {"block": 4})
    monkeypatch.setattr(sys, "argv", [str(root / "backend.py")])
    assert K.current_root() == root.resolve()
    assert K.knob("block", default=2, values=(1, 2, 4)) == 4
    other = tmp_path / "other"
    other.mkdir()
    K.declare(other, {"block": K.Knob((1, 2, 4), default=1)})
    K.use_root(other)
    assert K.knob("block", default=2, values=(1, 2, 4)) == 1
    assert K.find_root(root / "backend.py") == root.resolve() and K.find_root(tmp_path) is None


def test_the_command_line_checks_and_sets(tmp_path, capsys):
    root = _submission(tmp_path)
    assert K.main(["check", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert K.main(["set", str(root), "overlap=on", "block=4"]) == 0
    assert json.loads(capsys.readouterr().out) == {"block": 4, "overlap": "on"}
    assert K.main(["set", str(root), "block=3"]) == 2
    (root / K.DECLARATION).unlink()
    assert K.main(["check", str(root)]) == 1


def test_a_broker_override_reaches_the_compiler_in_the_private_copy_only(tmp_path):
    root = _submission(tmp_path)
    snapshot = tmp_path / "snapshot" / "submission"
    shutil.copytree(root, snapshot)
    request = broker_tuning.apply_request(snapshot, {"tuning_overrides": {"overlap": "on", "block": 4}},
                                          digest=lambda path: "f" * 64)
    assert request[broker_tuning.APPLIED_KEY]["overrides"] == {"overlap": "on", "block": 4}
    assert _run_backend(snapshot) == {"overlap": "on", "block": 4}
    assert _run_backend(root) == {"overlap": "off", "block": 2}
    assert K.check(snapshot)["valid"]


# ---------------------------------------------------------------------------------------------
# The real broker loop, with a stand-in self-check that compiles the graded tree
# ---------------------------------------------------------------------------------------------

FAKE_SELFCHECK = r"""
import argparse, json, pathlib, subprocess, sys
p = argparse.ArgumentParser()
for flag in ("--out", "--submission", "--sim", "--capsules", "--workers", "--timeout", "--progress-out"):
    p.add_argument(flag)
a, _ = p.parse_known_args()
tree = pathlib.Path(a.submission)
run = subprocess.run([sys.executable, str(tree / "backend.py")], capture_output=True, text=True)
compiled = json.loads(run.stdout) if run.returncode == 0 else run.stderr[-2000:]
pathlib.Path(a.out).write_text(json.dumps({"all_pass": True, "per_capsule": [{"capsule": "c1", "pass": True}],
                                           "compiled_with": compiled}))
"""


def _context(root: Path) -> InvocationContext:
    return InvocationContext(root, root / "target.yaml", root, "fixture", root / "runs", root / "reports",
                             root / "bundles", ())


@contextlib.contextmanager
def _broker(tmp_path: Path, ws: Path, monkeypatch):
    from merlin_experiments.phase1.brokers import selfcheck as broker
    from merlin_experiments.phase1.feedback import promotion

    fake = tmp_path / "fake_selfcheck.py"
    fake.write_text(FAKE_SELFCHECK)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    monkeypatch.setattr(broker, "worker_command", lambda *a, **k: [sys.executable, str(fake)])
    monkeypatch.setattr(broker, "_freeze_passed_source", lambda *a, **k: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: (None, None, None))
    thread = threading.Thread(target=broker.main, args=(["--ws", str(ws), "--poll", "0.01"],),
                              kwargs={"context": _context(tmp_path), "capsules_root": tmp_path})
    thread.start()
    try:
        yield broker
    finally:
        (ws / ".qa_channel" / "STOP").write_text("stop")
        thread.join(timeout=20)
    assert not thread.is_alive()


def _wait(channel: Path, rid: str) -> dict:
    limit = time.monotonic() + 30
    while time.monotonic() < limit and not (channel / f"done_{rid}").exists():
        time.sleep(0.02)
    return json.loads((channel / f"resp_{rid}.json").read_text())


def test_an_idle_tuning_request_compiles_the_overridden_value_end_to_end(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import idle_performance, idle_tuning

    root = _submission(tmp_path)
    ws = root.parent
    (ws / ".qa_channel").mkdir()
    with _broker(tmp_path, ws, monkeypatch) as broker:
        official = broker._submission_digest(root)
        rid = idle_performance.submit(ws / ".qa_channel", ["c1"], official, origin=idle_tuning.ORIGIN,
                                      prefix=idle_tuning.REQUEST_PREFIX, timeout=30,
                                      extra={"suite": "core", "sim": None,
                                             broker_tuning.REQUEST_KEY: {"overlap": "on"}})
        doc = _wait(ws / ".qa_channel", rid)
    assert doc.get("compiled_with") == {"overlap": "on", "block": 2}, doc.get("compiled_with")
    assert doc["variant"]["kind"] == "knob_overrides" and doc["official"] is False
    assert json.loads((root / K.DEFAULT_SETTINGS).read_text())["overlap"] == "off"

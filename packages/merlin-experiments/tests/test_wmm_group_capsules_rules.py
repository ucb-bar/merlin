"""Per-group perf capsules, the orchestration and every refusal rule: each arm to its own builder with
its own recipe, a one-group program asking the package for its own group only, an unanswered group
refused before any emulator time with the package's own words, board counts admitted only on their own
evidence with a steady control.  Builders, the emulator and the board are faked at their seams."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.whole_model_measured import group_capsules as G
from merlin_experiments.phase2.whole_model_measured import group_capsules_board as GB

from merlin.perf import whole_model_group_timing as T
from merlin.perf import whole_model_verdict as V


def test_each_arm_goes_to_its_own_builder_with_its_own_recipe(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        T,
        "build_group_programs",
        lambda pkg, groups, **kw: calls.setdefault("pkg", (pkg, groups, kw)) and {7: {"group": 7}},
    )
    monkeypatch.setattr(
        T,
        "build_reference_group_programs",
        lambda groups, **kw: calls.setdefault("ref", (groups, kw)) and {7: {"group": 7}},
    )
    package = G.Arm("package", "board_a", "/h", prohibited_roles=("r",), harness_overrides=("/o",))
    reference = G.Arm("reference", "board_a", "/h", harness_overrides=("/f",))
    got = G.build_arm_programs(package, [7], package_dir="/p", model_capsule="/c", target="t", out=tmp_path)
    G.build_arm_programs(reference, [7], package_dir="/ignored", model_capsule="/c", target="t", out=tmp_path)
    assert calls["pkg"][0] == "/p" and calls["pkg"][1] == [7]
    assert calls["pkg"][2]["prohibited_roles"] == ("r",) and calls["pkg"][2]["harness_overrides"] == ("/o",)
    assert calls["ref"][1]["harness_overrides"] == ("/f",) and "prohibited_roles" not in calls["ref"][1]
    assert got[7]["arm"] == "package" and got[7]["arm_recipe"]["machine"] == "board_a"
    with pytest.raises(G.GroupCapsuleError, match="package directory"):
        G.build_arm_programs(package, [7], package_dir=None, model_capsule="/c", target="t", out=tmp_path)


def test_the_board_spec_drops_the_operators_preparation_and_keeps_the_shared_lock():
    spec = GB.board_machine_spec(
        {"kind": "firesim", "hw_config": "hw", "lock_path": "/l", "queue_command": ["q"], "prepare_command": ["x"]}
    )
    assert "prepare_command" not in spec and spec["lock_path"] == "/l"
    with pytest.raises(G.GroupCapsuleError, match="lock_path"):
        GB.board_machine_spec({"kind": "firesim", "hw_config": "hw", "queue_command": ["q"]})


def test_in_model_counts_are_read_from_a_whole_model_result():
    got = G.in_model_cycles(
        {
            "objective_cycles": 9,
            "device": {"artifact": "board", "abi_header_sha256": "h"},
            "verdict": {"groups": [{"group": "1", "cycles": 5, "correct": True}]},
        }
    )
    assert got["groups"] == {"1": 5} and got["correct"] == {"1": True} and got["device"] == "board"


# ------------------------------------------------------------------------------------ board batch


def _block(group: str, cycles: int, digest: int, *, whole: int | None = None) -> str:
    templates = V.PROTOCOL_TEMPLATES
    lines = [
        templates["invocations"],
        templates["group"].format(group=group, kind="k", cycles=cycles, sum="UNKNOWN", checksum="UNKNOWN"),
        templates["words"].format(group=group, bytes=8, digest=digest),
    ]
    if whole is not None:
        lines.append(templates["full_model"].format(cycles=whole))
    return "\n".join(lines) + "\n"


def _fake_board(tmp_path, monkeypatch, blocks: dict[int, str]):
    from merlin_experiments.phase2.whole_model_measured import machines as M

    from merlin.runtime.backends import base as backends

    linked = {}

    def link_batch(variants, out, **_kw):
        linked["labels"] = [v["label"] for v in variants]
        return {"elf": str(tmp_path / "batch.elf"), "variants": linked["labels"]}

    program = SimpleNamespace(link_batch=link_batch, split_batch=lambda text, n: dict(blocks))
    monkeypatch.setattr(backends, "whole_model_driver", lambda target: SimpleNamespace(program=program))
    uart = tmp_path / "uart.log"
    uart.write_text("console")

    class Board:
        def run(self, elf, workdir, *, timeout_s):
            return {"completed": True, "uart_log": str(uart)}

    monkeypatch.setattr(M, "machine_from_spec", lambda spec: Board())
    return linked


def _control(tmp_path, solo_cycles: int, digest: int) -> dict:
    solo_uart = tmp_path / "solo.log"
    solo_uart.write_text(_block("1", 100, digest, whole=solo_cycles))
    request = tmp_path / "board_request.json"
    request.write_text(
        json.dumps(
            {
                "variant": {
                    "program": {"compiler": "cc", "flags": [], "link_flags": [], "link_script": None},
                    "supports": [],
                }
            }
        )
    )
    result = tmp_path / "solo_result.json"
    result.write_text(
        json.dumps({"verdict": {"whole_window_cycles": solo_cycles}, "run": {"uart_log": str(solo_uart)}})
    )
    return {"board_request": str(request), "solo_result": str(result), "cycles_tolerance": 0.02}


def _entry(group: int, arm: str, words):
    variant = {"program": {"compiler": "cc", "flags": [], "link_flags": [], "link_script": None}, "supports": []}
    return {
        "timing": {"group": group, "arm": arm, "variant": variant},
        "local": {"correct": True, "words": words},
    }


def test_a_board_count_is_admitted_only_with_matching_words_and_a_steady_control(tmp_path, monkeypatch):
    control = _control(tmp_path, 1000, 11)
    blocks = {1: _block("1", 100, 11, whole=1005), 2: _block("4", 777, 42), 3: _block("4", 555, 43)}
    linked = _fake_board(tmp_path, monkeypatch, blocks)
    entries = {"a_package_g4": _entry(4, "package", [8, 42]), "b_reference_g4": _entry(4, "reference", [8, 99])}
    record = GB.board_batch(entries, control=control, machine={}, target="t", out=tmp_path / "out")
    assert linked["labels"] == ["control", "a_package_g4", "b_reference_g4"]
    assert record["control"]["ok"] is True
    assert record["rows"]["a_package_g4"]["admitted"] and record["rows"]["a_package_g4"]["cycles"] == 777
    wrong = record["rows"]["b_reference_g4"]
    assert not wrong["admitted"] and "words differ" in wrong["refusal"]


def test_a_drifted_control_admits_nothing_in_its_batch(tmp_path, monkeypatch):
    control = _control(tmp_path, 1000, 11)
    blocks = {1: _block("1", 100, 11, whole=1100), 2: _block("4", 777, 42)}
    _fake_board(tmp_path, monkeypatch, blocks)
    record = GB.board_batch(
        {"a": _entry(4, "package", [8, 42])}, control=control, machine={}, target="t", out=tmp_path / "o"
    )
    assert record["control"]["ok"] is False
    assert not record["rows"]["a"]["admitted"] and "control drifted" in record["rows"]["a"]["refusal"]


def test_a_refused_program_never_reaches_the_board(tmp_path, monkeypatch):
    _fake_board(tmp_path, monkeypatch, {})
    with pytest.raises(G.GroupCapsuleError, match="refused"):
        GB.board_batch(
            {"a": {"timing": {"group": 1, "refusal": "no"}}}, control=None, machine={}, target="t", out=tmp_path
        )


def test_a_one_group_program_asks_the_package_only_for_its_own_group(tmp_path, monkeypatch):
    """ask_only states every other group by the reference (declined before the ask), so a try costs one
    question; the asked group's statement is the full statement's (verified byte-identical on a model)."""
    from merlin.perf import whole_model_build as W

    calls = []

    def fake_state(capsule, *, target, package_dir=None, decline=(), **kw):
        calls.append((package_dir, list(decline)))
        return {"whole_program": {"per_group": [{"group": g} for g in (1, 2, 3)]}}

    monkeypatch.setattr(W, "load_model_capsule", lambda path: SimpleNamespace(interface="i", weights="w"))
    monkeypatch.setattr(W, "state", fake_state)
    monkeypatch.setattr(W, "bind_groups", lambda buffer: [])
    monkeypatch.setattr(W, "decline_ops", lambda rows, decline: rows)
    monkeypatch.setattr(W, "_kernel_objects", lambda rows, **kw: None)
    from merlin.runtime.backends import base as backends

    driver = SimpleNamespace(program=SimpleNamespace(extract=lambda *a, **k: {}))
    monkeypatch.setattr(backends, "whole_model_driver", lambda target: driver)
    model = SimpleNamespace(inputs={"x": 1}, outputs={"y": 2}, interface="i", weights_manifest="m", weights="w")
    monkeypatch.setattr(W, "load_model_capsule", lambda path: model)
    T._prepare(
        "/pkg", model_capsule="/c", target="t", work=tmp_path, groups=[2], decline=(), timeout=1, jobs=1, ask_only=True
    )
    assert calls == [(None, []), ("/pkg", [1, 3])]
    calls.clear()
    T._prepare("/pkg", model_capsule="/c", target="t", work=tmp_path, groups=[2], decline=(), timeout=1, jobs=1)
    assert calls == [("/pkg", [])]


def test_a_group_the_package_does_not_answer_is_refused_before_any_emulator_time(tmp_path, monkeypatch):
    """A cell that times the library in the package's place spends hours (measured: a declined stem
    convolution ran its host fallback to the emulator's cycle bound) for a number nothing may use."""
    arm = G.Arm("package", "m", "/h")
    monkeypatch.setattr(
        G,
        "build_arm_programs",
        lambda arm, groups, **kw: {
            1: {"group": 1, "arm": "package", "linked": "vendor", "cause": "package_declined", "elf": "e"},
            2: {"group": 2, "arm": "package", "linked": "submission", "elf": "e"},
        },
    )
    timed = {}

    def fake_time(programs, **kw):
        timed.update(programs)
        return {label: {"status": "graded", "cycles": 5, "correct": True} for label in programs}

    monkeypatch.setattr(G, "time_on_gsim", fake_time)
    document = G.measure_on_gsim(
        {"package": arm}, [1, 2], package_dir="/p", model_capsule="/c", target="t", out=tmp_path, require_package=True
    )
    rows = {r["group"]: r for r in document["rows"]}
    assert set(timed) == {"package_g2"}
    assert rows[1]["status"] == "refused" and "package_declined" in rows[1]["refusal"]
    assert rows[2]["cycles"] == 5


def test_an_unanswered_group_is_refused_with_the_packages_own_words(tmp_path, monkeypatch):
    monkeypatch.setattr(
        G,
        "build_arm_programs",
        lambda arm, groups, **kw: {
            3: {"group": 3, "arm": "package", "linked": "vendor", "cause": "package_declined", "why": "NameError: x"}
        },
    )
    monkeypatch.setattr(G, "time_on_gsim", lambda programs, **kw: {})
    document = G.measure_on_gsim(
        {"package": G.Arm("package", "m", "/h")}, [3], package_dir="/p", model_capsule="/c", target="t",
        out=tmp_path, require_package=True,
    )  # fmt: skip
    (row,) = document["rows"]
    assert "NameError: x" in row["refusal"] and row["why"] == "NameError: x"


def test_a_group_capsule_measurement_records_the_contract_it_graded_under(tmp_path, monkeypatch):
    from merlin.perf import exactness as EX

    seen = {}

    def fake_build(arm, groups, **kw):
        seen["contract"] = kw.get("exactness")
        return {}

    monkeypatch.setattr(G, "build_arm_programs", fake_build)
    monkeypatch.setattr(G, "time_on_gsim", lambda programs, **kw: {})
    document = G.measure_on_gsim(
        {"package": G.Arm("package", "m", "/h")}, [3], package_dir="/p", model_capsule="/c", target="t", out=tmp_path
    )
    # No contract named: the default one, every form exact, and the measurement says so.
    assert seen["contract"].semantics_sha256 == EX.DEFAULT_SEMANTICS_SHA256
    assert document["exactness"]["semantics_sha256"] == EX.DEFAULT_SEMANTICS_SHA256

"""The completion-order check through the real capsule ladder: L2 structural evidence.

The ladder's front half and adapters are the tier-cache tests' stubs; the facts are the derived
Gemmini ordering record and the executed stream is handed to the real checker in place of a Spike
commit log, so what is under test is the wiring: a hazard fails the functional tier with its reason,
a clear or unknown verdict is recorded without failing, the verdict is cached by the executable's
bytes, and it replaces the accumulator-only rule as the memory-perturbation sweep's trigger.
"""

from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import completion_order as C
from merlin.targetgen import load_order as LO

sys.path.insert(0, str(merlin_dir() / "tests" / "targetgen"))
from test_tier_certificate_cache import _grade, _isolated_store  # noqa: E402,F401
from test_tier_certificate_cache import _ladder as ladder_fixture

_ladder = ladder_fixture

DATA = merlin_dir() / "tests" / "data" / "completion_order"
MASKS = {"scratchpad": 0x3FFF, "accumulator": 0x3FF}
ZERO_THEN_DATA = [
    None,
    (0, 0x100001, 0x40),
    (2, 0xFF << 48, (16 << 48) | (16 << 32)),
    (2, 1 << 48, (1 << 48) | (16 << 32) | 3),
    None,
]
FENCED = [
    None,
    (0, 0x100001, 0x40),
    (2, 0xFF << 48, (16 << 48) | (16 << 32)),
    None,
    (2, 1 << 48, (1 << 48) | (16 << 32) | 3),
    None,
]


@pytest.fixture
def facts(monkeypatch, tmp_path):
    """Derived ordering facts for the stub target, and a commit log replaced by ``stream``."""
    from merlin.targetgen import completion_order_gate as G
    from merlin.targetgen.rtl import facts as RF

    record = json.loads((DATA / "gemmini_command_ordering.json").read_text())
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(RF, "load_facts", lambda target, **_k: {"facts": {"command_ordering": record}})
    monkeypatch.setattr(
        C.OrderingModel,
        "for_target",
        classmethod(lambda cls, target, body=None: cls.from_facts(record, row_masks=MASKS)),
    )
    state = {"stream": ZERO_THEN_DATA, "runs": 0, "record": record, "engine": "original", "decoder": "original"}
    monkeypatch.setattr(G, "_observation_identity", lambda *_: {"engine": state["engine"]})
    monkeypatch.setattr(G, "_checker_digest", lambda: state["decoder"])

    def check_elf(target, elf, *, model=None, timeout_s=0, max_examples=8):
        state["runs"] += 1
        return C.check_commands(state["stream"], model)

    monkeypatch.setattr(C, "check_elf", check_elf)
    return state


def test_a_hazard_fails_the_functional_tier_with_its_reason(tmp_path, _ladder, facts):
    res = _grade(tmp_path, "racy", b"\x7fELF-racy", {})
    assert res["status"] == "fail"
    assert res["failure"]["plane"] == "completion_order"
    assert res["failure"]["category"] == "protocol_violation"
    assert "no ordering point" in res["failure"]["detail"] and "#1" in res["failure"]["detail"]
    assert res["tiers"]["L2"]["status"] == "fail" and "memory-ordering hazard" in res["tiers"]["L2"]["reason"]
    assert res["completion_order"]["verdict"] == "hazard" and res["completion_order"]["tier"] == "L2"


def test_a_clear_stream_passes_and_is_carried_by_its_bytes(tmp_path, _ladder, facts):
    facts["stream"] = FENCED
    first = _grade(tmp_path, "one", b"\x7fELF-fenced", {})
    assert first["status"] == "pass", first.get("failure")
    assert first["completion_order"]["verdict"] == "clear" and first["completion_order"]["cache"] == "measured"
    second = _grade(tmp_path, "two", b"\x7fELF-fenced", {})
    assert second["completion_order"]["cache"] == "carried" and facts["runs"] == 1
    third = _grade(tmp_path, "three", b"\x7fELF-fenced-but-rebuilt", {})
    assert third["completion_order"]["cache"] == "measured" and facts["runs"] == 2


def test_unknown_facts_are_recorded_never_silently_passed(tmp_path, _ladder, monkeypatch):
    from merlin.targetgen.rtl import facts as RF

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(RF, "load_facts", lambda target, **_k: {"facts": {}})
    res = _grade(tmp_path, "nofacts", b"\x7fELF-x", {})
    assert res["status"] == "pass", res.get("failure")
    assert res["completion_order"]["verdict"] == "unknown"
    assert "command_ordering" in res["completion_order"]["detail"]


def test_switching_the_check_off_is_recorded(tmp_path, _ladder, facts, monkeypatch):
    from merlin.targetgen import completion_order_gate as G

    monkeypatch.setenv(G.SWITCH_ENV, "off")
    res = _grade(tmp_path, "off", b"\x7fELF-racy", {})
    assert res["status"] == "pass" and res["completion_order"]["verdict"] == "skipped"


def test_the_completion_order_verdict_triggers_the_perturbation_sweep(tmp_path, monkeypatch):
    """A clear executed stream is never swept even when the accumulator-only rule would fire, and the
    sweep needs no accumulator facts when the trigger is supplied."""
    from merlin.targetgen import completion_order_gate as G

    monkeypatch.setenv(LO.SEEDS_ENV, "4")
    monkeypatch.setattr(LO, "facts_for", lambda target: (None, "no accumulator facts"))
    out = LO.run_order_check(
        target="t",
        trace=None,
        elf=tmp_path / "x.elf",
        judge=lambda rc, text: "pass",
        static=G.sweep_trigger({"verdict": "clear", "hazard_pairs": 0}),
    )
    assert out.record["status"] == "static_clear" and out.record["static"]["source"] == "completion_order"
    out = LO.run_order_check(
        target="t",
        trace=None,
        elf=tmp_path / "x.elf",
        judge=lambda rc, text: "pass",
        engine=(None, "no engine"),
        static=G.sweep_trigger({"verdict": "unknown"}),
    )
    assert out.record["status"] == "unavailable" and out.record["static"]["verdict"] == "unresolved"
    # without a trigger the accumulator-only rule still decides, and needs its facts
    out = LO.run_order_check(target="t", trace=None, elf=tmp_path / "x.elf", judge=lambda rc, text: "pass")
    assert out.record["status"] == "not_applicable"


@pytest.mark.parametrize("changed", ["facts", "geometry", "engine", "decoder"])
def test_reuse_requires_all_facts_geometry_decoder_and_engine(tmp_path, _ladder, facts, monkeypatch, changed):
    facts["stream"] = FENCED
    assert _grade(tmp_path, "before", b"\x7fELF-same", {})["completion_order"]["cache"] == "measured"
    if changed == "facts":
        facts["record"]["additional_selected_fact"] = 17
    elif changed == "geometry":
        monkeypatch.setattr(
            C.OrderingModel,
            "for_target",
            classmethod(
                lambda cls, target, body=None: cls.from_facts(facts["record"], row_masks={**MASKS, "scratchpad": 255})
            ),
        )
    else:
        facts[changed] = "new"
    assert _grade(tmp_path, "after", b"\x7fELF-same", {})["completion_order"]["cache"] == "measured"
    assert facts["runs"] == 2


def test_unidentified_engine_never_reuses_a_verdict(tmp_path, _ladder, facts, monkeypatch):
    from merlin.targetgen import completion_order_gate as G

    monkeypatch.setattr(G, "_observation_identity", lambda *_: None)
    facts["stream"] = FENCED
    _grade(tmp_path, "one", b"\x7fELF-same", {})
    _grade(tmp_path, "two", b"\x7fELF-same", {})
    assert facts["runs"] == 2


def test_selected_engine_receiver_and_isa_are_reopened(tmp_path, monkeypatch):
    from merlin.perf import executed_trace
    from merlin.targetgen import completion_order_gate as G

    engine, receiver, elf = (tmp_path / name for name in ("engine", "receiver.so", "program.elf"))
    engine.write_bytes(b"engine-v1")
    engine.chmod(0o700)
    receiver.write_bytes(b"receiver-v1")
    elf.write_bytes(b"elf")
    monkeypatch.setattr(
        executed_trace,
        "trace_invocation",
        lambda *_: ([str(engine), f"--extlib={receiver}", str(elf)], {"LD_LIBRARY_PATH": str(tmp_path)}),
    )
    isa = {"CUSTOM_OPCODE": 123}
    monkeypatch.setattr(C, "trace_isa", lambda *_: dict(isa))
    before = G._observation_identity("t", elf)
    receiver.write_bytes(b"receiver-v2")
    assert G._observation_identity("t", elf) != before
    receiver.write_bytes(b"receiver-v1")
    engine.write_bytes(b"engine-v2")
    assert G._observation_identity("t", elf) != before
    engine.write_bytes(b"engine-v1")
    isa["CUSTOM_OPCODE"] = 11
    assert G._observation_identity("t", elf) != before


def test_unknown_completion_relation_does_not_fail_l2(tmp_path, _ladder, facts):
    facts["record"]["completion_order"]["load"] = {"order": "unknown"}
    result = _grade(tmp_path, "uncertain", b"\x7fELF-uncertain", {})
    assert result["status"] == "pass"
    assert result["tiers"]["L2"]["status"] == "pass"
    assert result["completion_order"]["verdict"] == "unknown"


def test_explicit_ordering_selection_supplies_missing_facts_without_mutating_them(
    tmp_path, _ladder, facts, monkeypatch
):
    from merlin.targetgen.rtl import command_ordering_selection as S
    from merlin.targetgen.rtl import facts as RF

    original = {"facts": {"unchanged": 17}}
    monkeypatch.setattr(RF, "load_facts", lambda *_a, **_k: original)
    pin = {"path": "/owner/ordering-selection.json", "sha256": "a" * 64}
    identity = {"selection": pin, "ordering_sha256": "b" * 64}

    def read(selected, *, target, document):
        assert selected == pin and document == original
        return facts["record"], identity

    monkeypatch.setattr(S, "read", read)
    monkeypatch.setenv(S.ENV, json.dumps(pin))
    facts["stream"] = FENCED
    result = _grade(tmp_path, "sidecar", b"\x7fELF-sidecar", {})
    assert result["completion_order"]["verdict"] == "clear"
    assert result["completion_order"]["ordering_selection"] == identity
    assert original == {"facts": {"unchanged": 17}}


def test_invalid_explicit_selection_never_falls_back_to_other_facts(tmp_path, _ladder, facts, monkeypatch):
    from merlin.targetgen.rtl import command_ordering_selection as S

    monkeypatch.setenv(S.ENV, '{"path":"/missing-selection.json","sha256":"missing"}')
    result = _grade(tmp_path, "badsidecar", b"\x7fELF-invalid-sidecar", {})
    assert result["completion_order"]["verdict"] == "unknown"
    assert "selection unavailable" in result["completion_order"]["detail"]
    assert facts["runs"] == 0

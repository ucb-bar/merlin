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


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    """These tests replace the checker's inputs in this process; a worker process would not see them."""
    from merlin.targetgen import completion_order_gate as G

    monkeypatch.setenv(G.WORKER_ENV, "off")


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


@pytest.mark.parametrize("elf", [False, True])
def test_the_worker_process_returns_the_in_process_record(tmp_path, monkeypatch, elf):
    """The check runs in a worker process with this process's import path and environment."""
    from merlin.targetgen import completion_order_gate as G
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    from merlin.targetgen.rtl import command_ordering_selection as S

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv(S.ENV, '{"path":"/missing-selection.json","sha256":"missing"}')
    generated = tmp_path / "generated"
    generated.mkdir()
    if elf:
        (generated / PACKAGE_ELF_NAME).write_bytes(b"\x7fELF-worker")
    expected = G._check("gemmini", generated)
    monkeypatch.setenv(G.WORKER_ENV, "on")
    assert G.check("gemmini", generated) == expected
    assert expected["verdict"] == "unknown"


def test_a_failed_worker_is_recorded_never_a_pass(tmp_path, monkeypatch):
    from merlin.targetgen import completion_order_gate as G

    monkeypatch.setenv(G.WORKER_ENV, "on")
    monkeypatch.setattr(G, "_WORKER_BOOTSTRAP", "import sys; sys.stderr.write('broken'); sys.exit(3)")
    result = G.check("gemmini", tmp_path)
    assert result["verdict"] == "unknown"
    assert result["detail"] == "check crashed: RuntimeError: worker exited 3: broken"


def test_a_worker_past_its_deadline_is_stopped_with_its_engine(tmp_path, monkeypatch):
    import os
    import time

    from merlin.targetgen import completion_order_gate as G

    marker = tmp_path / "engine.pid"
    monkeypatch.setenv(G.WORKER_ENV, "on")
    monkeypatch.setenv(G.TIMEOUT_ENV, "0.2")
    monkeypatch.setattr(G, "_WORKER_GRACE_S", 0.5)
    monkeypatch.setattr(
        G,
        "_WORKER_BOOTSTRAP",
        "import subprocess, sys, time; "
        f"p = subprocess.Popen(['sleep', '60']); open({str(marker)!r}, 'w').write(str(p.pid)); time.sleep(60)",
    )
    result = G.check("gemmini", tmp_path)
    assert result["verdict"] == "unknown" and result["detail"].startswith("check crashed: TimeoutExpired")
    engine = int(marker.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{engine}/stat") as stat:
                if stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                    break
        except FileNotFoundError:
            break
        time.sleep(0.05)
    else:
        os.kill(engine, 9)
        raise AssertionError("the worker's engine outlived the stopped worker")


def test_content_identity_reuses_relocated_files_but_keeps_semantic_environment(tmp_path, monkeypatch):
    from merlin.perf import executed_trace
    from merlin.targetgen import completion_order_gate as G

    files = []
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        engine, library, elf = (directory / leaf for leaf in ("engine", "receiver.so", "program.elf"))
        engine.write_bytes(b"engine")
        engine.chmod(0o700)
        library.write_bytes(b"receiver")
        elf.write_bytes(b"executable")
        files.append((engine, library, elf))
    state = {"which": 0, "mode": "first"}

    def invocation(_target, elf):
        engine, receiver, _elf = files[state["which"]]
        return [str(engine), f"--extlib={receiver}", str(elf)], {
            "TMPDIR": str(elf.parent), "MERLIN_OUT_ROOT": str(elf.parent),
            "LD_LIBRARY_PATH": str(elf.parent), "CUSTOM_EXECUTION_MODE": state["mode"],
        }

    monkeypatch.setattr(executed_trace, "trace_invocation", invocation)
    monkeypatch.setattr(C, "trace_isa", lambda *_: {"CUSTOM_OPCODE": 11})
    original = G._observation_identity("t", files[0][2])
    state["which"] = 1
    assert G._observation_identity("t", files[1][2]) == original
    state["mode"] = "second"
    assert G._observation_identity("t", files[1][2]) != original
    state["mode"] = "first"
    (files[1][1].parent / "dependent.so").write_bytes(b"new dependency")
    assert G._observation_identity("t", files[1][2]) != original


@pytest.mark.parametrize("changed", ["elf", "decoder", "engine"])
def test_changed_observation_inputs_never_publish_a_complete_verdict(tmp_path, _ladder, facts, monkeypatch, changed):
    from merlin.targetgen import completion_order_gate as G

    old_check = C.check_elf

    def change(target, elf, **kwargs):
        result = old_check(target, elf, **kwargs)
        if changed == "elf":
            elf.write_bytes(b"changed during observation")
        else:
            facts[changed] = "changed during observation"
        return result

    monkeypatch.setattr(C, "check_elf", change)
    result = _grade(tmp_path, "changed", b"\x7fELF-original", {})
    record = result["completion_order"]
    assert record["verdict"] == "unknown" and "inputs changed" in record["detail"]


def test_checker_digest_covers_recorder_source_and_python_reader(tmp_path, monkeypatch):
    from merlin.perf import command_recorder as R
    from merlin.targetgen import completion_order_gate as G

    before = G._checker_digest()
    replacement = tmp_path / "recorder.cc"
    replacement.write_bytes(R.SOURCE.read_bytes() + b"\n// changed\n")
    monkeypatch.setattr(R, "SOURCE", replacement)
    assert G._checker_digest() != before


def test_in_process_check_holds_one_compute_lease(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from merlin.common import cpu_pool
    from merlin.targetgen import completion_order_gate as G

    state = {"held": False}

    @contextmanager
    def lease(**_kwargs):
        assert not state["held"]
        state["held"] = True
        try:
            yield
        finally:
            state["held"] = False

    monkeypatch.setattr(cpu_pool, "shared_compute_pool", lease)
    monkeypatch.setattr(G, "_check", lambda *_: {"leased": state["held"]})
    assert G.check("t", tmp_path) == {"leased": True}
    assert not state["held"]


def test_worker_holds_and_releases_a_cpu_lease_in_the_callers_class(tmp_path, monkeypatch):
    import os

    from merlin.common import cpu_pool as P
    from merlin.targetgen import completion_order_gate as G

    cpu = min(os.sched_getaffinity(0))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    monkeypatch.delenv(P.PRIORITY_ENV, raising=False)
    monkeypatch.setenv(G.WORKER_ENV, "on")
    monkeypatch.setattr(G, "_WORKER_BOOTSTRAP", (
        "import json, os, sys; sys.path[:] = json.loads(sys.argv[1]); "
        "from merlin.common import cpu_pool as P; "
        "from merlin.targetgen import completion_order_gate as G; "
        "G._check = lambda *_: {'leased': P.current_thread_owns_cpu_lease(), "
        "'priority': P.lease_priority(), 'affinity': sorted(os.sched_getaffinity(0))}; "
        "G._worker_main()"
    ))
    with P.cpu_priority(P.AUXILIARY):
        assert G.check("t", tmp_path) == {"leased": True, "priority": P.AUXILIARY, "affinity": [cpu]}
    with P.shared_compute_pool(timeout_s=0):
        assert P.current_thread_owns_cpu_lease()


def test_the_completion_order_check_is_its_own_host_stage(tmp_path, monkeypatch):
    """Minutes of commit-log tracing after the numeric comparison must show up as a named stage."""
    from merlin_experiments.phase1.feedback import setup_timing

    from merlin.targetgen import completion_order_gate as gate

    monkeypatch.setattr(gate, "after_functional_pass", lambda *a, **k: ({"verdict": "clear"}, None))
    with setup_timing.selected(tmp_path, "s" * 64) as timeline:
        record, failure = gate.ladder_step("synthetic", "L2", {}, tmp_path)
    assert (record, failure) == ({"verdict": "clear"}, None)
    assert [row["stage"] for row in timeline.rows] == ["completion_order"]
    assert timeline.rows[0]["status"] == "returned"


def _exhausting(facts, monkeypatch, *, exhausted=True):
    """check_elf replaced by a run that ran out of its budget (or failed for another reason)."""
    def check_elf(target, elf, *, model=None, timeout_s=0, max_examples=8):
        facts["runs"] += 1
        facts["budgets"].append(timeout_s)
        record = {"schema": C.SCHEMA, "verdict": "unknown", "detail": f"the functional run exited -9 (timeout {timeout_s:g}s)"}
        if exhausted:
            record["budget_exhausted"] = {"timeout_s": float(timeout_s)}
        return record

    facts["budgets"] = []
    monkeypatch.setattr(C, "check_elf", check_elf)


def test_a_budget_exhausted_unknown_is_carried_for_the_same_bytes_and_budget(tmp_path, _ladder, facts, monkeypatch):
    from merlin.targetgen import completion_order_gate as G

    _exhausting(facts, monkeypatch)
    first = _grade(tmp_path, "one", b"\x7fELF-huge", {})["completion_order"]
    assert first["verdict"] == "unknown" and first["cache"] == "measured"
    second = _grade(tmp_path, "two", b"\x7fELF-huge", {})["completion_order"]
    # Same conclusion, not recomputed: "unknown" stays "unknown", never a clear by omission.
    assert second["verdict"] == "unknown" and second["cache"] == "carried" and facts["runs"] == 1
    assert second["detail"] == first["detail"]
    # A different budget is a different key, and so are other bytes or another checker version.
    monkeypatch.setenv(G.TIMEOUT_ENV, "2400")
    assert _grade(tmp_path, "three", b"\x7fELF-huge", {})["completion_order"]["cache"] == "measured"
    assert facts["budgets"] == [1200.0, 2400.0]
    assert _grade(tmp_path, "four", b"\x7fELF-huge", {})["completion_order"]["cache"] == "carried"
    assert _grade(tmp_path, "five", b"\x7fELF-huge-rebuilt", {})["completion_order"]["cache"] == "measured"
    facts["decoder"] = "new"
    assert _grade(tmp_path, "six", b"\x7fELF-huge", {})["completion_order"]["cache"] == "measured"
    assert facts["runs"] == 4


def test_other_unknown_outcomes_are_never_carried(tmp_path, _ladder, facts, monkeypatch):
    _exhausting(facts, monkeypatch, exhausted=False)
    _grade(tmp_path, "one", b"\x7fELF-crash", {})
    assert _grade(tmp_path, "two", b"\x7fELF-crash", {})["completion_order"]["cache"] == "measured"
    assert facts["runs"] == 2


def test_check_elf_marks_only_a_budget_kill_as_exhausted(tmp_path, monkeypatch):
    """The structured mark comes from the timer that killed the run, not from the exit status alone."""
    import sys

    from merlin.perf import command_recorder
    from merlin.perf import executed_trace

    model = C.OrderingModel.from_facts(json.loads((DATA / "gemmini_command_ordering.json").read_text()), row_masks=MASKS)
    monkeypatch.setattr(command_recorder, "selected", lambda: None)
    monkeypatch.setattr(C, "trace_isa", lambda target: {"CUSTOM_OPCODE": 0x7B})
    for program, budget, exhausted in (("import time; time.sleep(30)", 0.5, True), ("raise SystemExit(3)", 30, False)):
        monkeypatch.setattr(executed_trace, "trace_invocation", lambda *_a, p=program: ([sys.executable, "-c", p], {}))
        record = C.check_elf("t", tmp_path / "x.elf", model=model, timeout_s=budget)
        assert record["verdict"] == "unknown"
        assert ("budget_exhausted" in record) is exhausted
        if exhausted:
            assert record["budget_exhausted"] == {"timeout_s": 0.5}


def _development_grade(tmp_path, run_id, program):
    """An L2-only development screen: the materialized ceiling keeps the RTL tier out of its ladder."""
    from test_tier_certificate_cache import CR, _adapters, _capsule, _two_tier_config

    capsule = _capsule()
    capsule["required_oracle_tiers"] = ["L2"]
    capsule["oracle_tier_ceiling"] = "L2"
    return CR.run_capsule(
        capsule, "unused-package", runs_root=tmp_path / run_id, run_id=run_id,
        config=_two_tier_config(), oracle_adapters=_adapters(program, {}),
    )


def test_a_development_grade_defers_the_check_to_certification(tmp_path, _ladder, facts):
    res = _development_grade(tmp_path, "dev", b"\x7fELF-racy")
    assert res["status"] == "pass", res.get("failure")
    assert res["tiers"]["L2"]["status"] == "pass"
    assert res["completion_order"]["verdict"] == "deferred" and res["completion_order"]["tier"] == "L2"
    assert "deferred to certification" in res["completion_order"]["detail"]
    assert facts["runs"] == 0, "a development screen must not pay for the commit-log trace"


def test_a_spike_only_selfcheck_never_starts_the_trace(tmp_path, _ladder, facts, monkeypatch):
    """The scalability self-check's shape: the contract declares the RTL tier, the request holds only
    the Spike adapter, and no ceiling is materialized. Counting the declared RTL tier ran the trace."""
    from test_tier_certificate_cache import CR, _adapters, _capsule, _two_tier_config

    from merlin.targetgen import completion_order_gate as G

    def trace(*_a, **_k):
        raise AssertionError("an L2-only self-check started the commit-log trace")

    monkeypatch.setattr(G, "after_functional_pass", trace)
    capsule = _capsule()
    capsule["required_oracle_tiers"] = ["L2"]
    config = _two_tier_config()
    assert config.rtl_tiers & set(config.oracle_tiers), "the contract still declares an RTL tier"
    spike_only = {"L2": _adapters(b"\x7fELF-racy", {})["L2"]}
    res = CR.run_capsule(
        capsule, "unused-package", runs_root=tmp_path / "selfcheck", run_id="selfcheck",
        config=config, oracle_adapters=spike_only,
    )
    assert res["status"] == "pass", res.get("failure")
    assert res["completion_order"]["verdict"] == "deferred" and res["completion_order"]["tier"] == "L2"
    assert facts["runs"] == 0


def test_a_certification_grade_still_fails_the_hazard_on_its_functional_tier(tmp_path, _ladder, facts):
    from merlin.targetgen import completion_order_gate as G

    with G.certification():
        res = _development_grade(tmp_path, "final", b"\x7fELF-racy")
    assert res["status"] == "fail" and res["failure"]["plane"] == "completion_order"
    assert res["completion_order"]["verdict"] == "hazard" and facts["runs"] == 1
    assert G.certifies(["L2"], {"L3"}) is False, "the scope ends with its block"


def test_a_ladder_that_reaches_an_rtl_tier_runs_the_check(tmp_path, _ladder, facts):
    from merlin.targetgen import completion_order_gate as G

    assert G.certifies(["L2", "L3"], {"L3"}) and not G.certifies(["L2"], {"L3"})
    res = _grade(tmp_path, "cert", b"\x7fELF-racy", {})
    assert res["completion_order"]["verdict"] == "hazard" and facts["runs"] == 1


def test_after_coverage_checks_published_passes_only_inside_the_remaining_time(tmp_path, monkeypatch):
    """The final race-check pass annotates coverage: bounded by the deadline, never a failure by omission."""
    from merlin.targetgen import completion_order_gate as G

    calls = []

    def check(target, generated, *, timeout_s=None):
        calls.append((generated.parent.name, timeout_s))
        return {"verdict": "hazard" if generated.parent.name == "racy" else "clear", "detail": "d"}

    monkeypatch.setattr(G, "check", check)
    results = {name: tmp_path / name / "capsule_result.json" for name in ("clean", "racy")}
    left = [100.0]
    summary = G.after_coverage("t", results, remaining=lambda: left[0], workers=1)
    assert summary["status"] == "complete" and summary["hazards"] == ["racy"] and not summary["not_checked"]
    assert calls == [("clean", 70.0), ("racy", 70.0)], "each engine budget ends before the deadline"
    left[0] = 10.0
    summary = G.after_coverage("t", results, remaining=lambda: left[0], workers=2)
    assert summary["status"] == "not_run" and summary["not_checked"] == ["clean", "racy"] and len(calls) == 2


def test_the_worker_engine_budget_follows_an_explicit_timeout(monkeypatch):
    import pickle
    import subprocess

    from merlin.targetgen import completion_order_gate as G

    seen = {}

    class Worker:
        pid, returncode = 0, 0

        def __init__(self, command, env, **_kwargs):
            seen["env"] = env

        def communicate(self, _payload=None, timeout=None):
            seen["timeout"] = timeout
            return pickle.dumps({"verdict": "clear"}), b""

    monkeypatch.setattr(subprocess, "Popen", Worker)
    assert G._in_worker("t", "g", 42.0) == {"verdict": "clear"}
    assert seen["env"][G.TIMEOUT_ENV] == "42.0" and seen["timeout"] == 72.0

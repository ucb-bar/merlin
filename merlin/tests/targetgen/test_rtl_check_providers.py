"""Shared RTL checks must use the selected protocol, not a transport-shaped default."""

import json
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import circt_gate
from merlin.targetgen import rtl_check_compiler as compiler
from merlin.targetgen import rtl_check_runner as runner
from merlin.targetgen import rtl_checks as checks


class ProtocolChecks:
    """Synthetic support: two transports with deliberately incompatible ordering."""

    def __init__(self, before, after):
        self.before, self.after = before, after
        self.calls = []

    def load_default_facts(self, target):
        return {"source": target}

    def project_facts(self, facts):
        self.calls.append("project")
        return {"limit": facts.get("limit", 1)}

    def screen(self, trace, capsule=None, rtl_facts=None, *, target, command_buffer=None):
        self.calls.append("screen")
        names = [row["class"] for row in trace["instructions"]]
        valid = self.before in names and self.after in names and names.index(self.before) < names.index(self.after)
        return checks.CheckReport(
            (capsule or {}).get("name"),
            trace.get("source"),
            rtl_facts or {},
            [checks.Check("protocol.order", "T0", "error", "pass" if valid else "fail", target)],
        )

    def compile_trace_checks(self, facts, capsule, prefix="TRACE"):
        self.calls.append("compile")
        return f"// {prefix}: {self.before} {self.after} {capsule.get('count', 1)} {facts.get('limit', 1)}\n"

    def render_trace(self, trace, facts):
        self.calls.append("render")
        return f"{self.before} {self.after}\n"


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    import socket
    import subprocess

    def refused(*args, **kwargs):
        pytest.fail("provider dispatch tests cannot launch a process or listener")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def selected(monkeypatch):
    owners = {"first": ProtocolChecks("LOAD", "ISSUE"), "second": ProtocolChecks("ISSUE", "LOAD")}
    backends = {
        name: SimpleNamespace(rocc_semantics=SimpleNamespace(rtl_checks=owner)) for name, owner in owners.items()
    }
    monkeypatch.setattr(base, "get_backend", lambda target: backends[target])
    monkeypatch.setattr(compiler, "_is_rocc_target", lambda *_: True)
    monkeypatch.setattr(compiler, "_provenance", lambda *_: {})
    return owners, backends


def trace(*names):
    return {"source": "synthetic", "instructions": [{"class": name} for name in names]}


def test_same_transport_does_not_imply_same_protocol(selected):
    owners, _ = selected
    stream = trace("LOAD", "ISSUE")
    assert checks.screen(stream, target="first").verdict == "ok"
    assert checks.screen(stream, target="second").verdict == "reject"
    assert checks.screen(trace("ISSUE", "LOAD"), target="second").verdict == "ok"
    for target, owner in owners.items():
        compiled = compiler.compile_trace_checks({}, {}, target=target)
        rendered = runner.render_trace(stream, {}, target=target)
        assert rendered.strip() in compiled
        assert checks.load_default_facts(target) == {"source": target}
        assert owner.calls


@pytest.mark.parametrize(
    "member", ["load_default_facts", "project_facts", "screen", "compile_trace_checks", "render_trace"]
)
def test_incomplete_support_never_uses_another_protocol(selected, member):
    owners, _ = selected
    checks.screen(trace("LOAD", "ISSUE"), target="first")
    setattr(owners["first"], member, None)
    for invoke in (
        lambda: checks.screen(trace("LOAD", "ISSUE"), target="first"),
        lambda: compiler.compile_trace_checks({}, {}, target="first"),
        lambda: runner.render_trace(trace(), {}, target="first"),
    ):
        with pytest.raises(checks.RtlChecksUnavailable, match="capability"):
            invoke()


def test_assertions_do_not_cache_by_capsule_name_or_fact_object(selected):
    owners, _ = selected
    capsule, facts = {"name": "unchanged-name", "count": 1}, {"limit": 2}
    first = runner.compiled_checks(facts, capsule, "first")["trace"]
    capsule["count"] = 3
    second = runner.compiled_checks(facts, capsule, "first")["trace"]
    facts["limit"] = 4
    third = runner.compiled_checks(facts, capsule, "first")["trace"]
    owners["first"].before = "REPLACED"
    fourth = runner.compiled_checks(facts, capsule, "first")["trace"]
    assert len({first, second, third, fourth}) == 4


@pytest.mark.parametrize("bad", [None, {}, "pass"])
def test_malformed_provider_report_refuses(selected, bad):
    owners, _ = selected
    owners["first"].screen = lambda *args, **kwargs: bad
    with pytest.raises(checks.RtlChecksUnavailable, match="report"):
        checks.screen(trace(), target="first")


@pytest.mark.parametrize(
    "field,value", [("status", "success"), ("severity", "optional"), ("tier", "unknown"), ("id", "")]
)
def test_malformed_check_refuses(selected, field, value):
    owners, _ = selected
    report = owners["first"].screen(trace("LOAD", "ISSUE"), target="first")
    setattr(report.checks[0], field, value)
    owners["first"].screen = lambda *args, **kwargs: report
    with pytest.raises(checks.RtlChecksUnavailable, match="checks"):
        checks.screen(trace(), target="first")


@pytest.mark.parametrize("operation", ["facts", "compile", "render"])
def test_malformed_provider_output_refuses(selected, operation):
    owners, _ = selected
    if operation == "facts":
        owners["first"].load_default_facts = lambda *_: []

        def invoke():
            return checks.load_default_facts("first")
    elif operation == "compile":
        owners["first"].compile_trace_checks = lambda *_: []

        def invoke():
            return compiler.compile_trace_checks({}, {}, target="first")
    else:
        owners["first"].render_trace = lambda *_: ""

        def invoke():
            return runner.render_trace(trace(), {}, target="first")

    with pytest.raises(checks.RtlChecksUnavailable, match="malformed"):
        invoke()


def test_runner_uses_one_selected_owner_for_all_three_operations(selected, monkeypatch, tmp_path):
    owners, backends = selected
    resolutions = []

    def resolve(target):
        resolutions.append(target)
        assert len(resolutions) == 1
        return backends[target]

    monkeypatch.setattr(base, "get_backend", resolve)
    monkeypatch.setattr(runner, "_load_capsule", lambda *_: {"name": "fixture"})
    monkeypatch.setattr(runner, "run_filecheck", lambda *args: (True, "synthetic comparator"))
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "instruction_trace.json").write_text(json.dumps(trace("LOAD", "ISSUE")))
    result = runner.screen_run(tmp_path, {}, {}, "synthetic-filecheck", target="first")
    assert result["verdict"] == "ok"
    assert owners["first"].calls == ["compile", "render", "project", "screen"]
    assert resolutions == ["first"]


@pytest.mark.parametrize("mode", ["missing", "raises", "facts-error", "reject"])
def test_advisory_screen_never_suppresses_oracle_or_claims_missing_is_ok(selected, monkeypatch, tmp_path, mode):
    owners, backends = selected
    if mode == "missing":
        backends["first"] = SimpleNamespace()
    elif mode == "raises":

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic check failure")

        owners["first"].screen = fail
    elif mode == "facts-error":

        def missing_facts(*args):
            raise ValueError("synthetic unavailable facts")

        monkeypatch.setattr(circt_gate, "load_facts", missing_facts)
    monkeypatch.setattr(circt_gate.RD, "decode_text", lambda *args, **kwargs: trace("ISSUE", "LOAD"))
    calls, records = [], []

    def oracle(*args):
        calls.append(args)
        return "oracle-result"

    wrapped = circt_gate.gated_adapter(oracle, log=records, target="first", facts=None if mode == "facts-error" else {})
    assert wrapped({}, "synthetic", tmp_path, 1) == "oracle-result"
    assert len(calls) == 1 and records[0]["sim_skipped"] is False
    assert records[0]["verdict"] == ("reject" if mode == "reject" else "unavailable")
    assert ("reason" in records[0]) == (mode != "reject")

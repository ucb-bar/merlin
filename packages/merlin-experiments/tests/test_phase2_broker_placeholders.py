"""Historical ASCII brace-token semantics, without sockets or child execution."""

import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP


@pytest.mark.parametrize(
    "template,names,rendered",
    [
        ("plain", [], "plain"),
        ("", [], ""),
        ("{}", [], "{}"),
        ("{a}", ["a"], "A"),
        ("{Z9_}", ["Z9_"], "Z"),
        ("{a}{a}/{b}", ["a", "a", "b"], "AA/B"),
        ("{{a}}", ["a"], "{A}"),
        ("{{{a}}}", ["a"], "{{A}}"),
        ("{outer{b}}", ["b"], "{outerB}"),
        ("{a{b}tail}", ["b"], "{aBtail}"),
        ("{a}tail}", ["a"], "Atail}"),
        ("{a}{", ["a"], "A{"),
        ("{a", [], "{a"),
        ("a}", [], "a}"),
        ("{", [], "{"),
        ("{{", [], "{{"),
        ("{_a}", [], "{_a}"),
        ("{1a}", [], "{1a}"),
        ("{a-b}", [], "{a-b}"),
        ("{a.b}", [], "{a.b}"),
        ("{ a}", [], "{ a}"),
        ("{a }", [], "{a }"),
        ("{a\nb}", [], "{a\nb}"),
        ("{é}{aé}{a١}", [], "{é}{aé}{a١}"),
        ("{a\x00}{b}", ["b"], "{a\x00}B"),
        ("{_bad{a}}{a-b{b}}", ["a", "b"], "{_badA}{a-bB}"),
    ],
)
def test_find_and_render_exact_historical_tokens(template, names, rendered):
    tokens = list(B._placeholder_tokens(template))
    assert [name for _, _, name in tokens] == names
    assert all(template[start:end] == "{" + name + "}" for start, end, name in tokens)
    assert B._render_placeholders(template, {"a": "A", "b": "B", "Z9_": "Z"}) == rendered


def test_replacements_are_literal_and_not_recursively_expanded():
    assert B._render_placeholders("{a}/{b}", {"a": "{b}\\1", "b": "done"}) == "{b}\\1/done"


def test_first_missing_key_preserves_key_error():
    with pytest.raises(KeyError) as error:
        B._render_placeholders("{a}/{missing}/{later}", {"a": "A"})
    assert error.value.args == ("missing",)


@pytest.mark.parametrize(
    "template,names",
    [
        ("{{a}}/{a}", ["a"]),
        ("{bad-name}/{_bad}", []),
        ("{outer{b}}", ["b"]),
        ("{candidate}/tool/{a}", ["a"]),
    ],
)
def test_actual_registry_discovery_uses_same_tokens(tmp_path, template, names):
    row = {"name": "fixture", "argv_template": [template], "placeholders": names, "purpose": "test", "required": False}
    [action] = B.actions_from_registry_contract([row], tmp_path)
    assert action.placeholders == tuple(names)
    if template.startswith("{candidate}/"):
        assert action.argv_template[0] == str(tmp_path) + "/tool/{a}"
    else:
        assert action.argv_template == (template,)
    with pytest.raises(B.StageGateError, match="binding contract"):
        B.actions_from_registry_contract([{**row, "placeholders": [*names, "undeclared"]}], tmp_path)


def test_actual_execute_renders_before_dispatch_without_network(tmp_path, monkeypatch):
    action = B.BrokerAction("fixture", ("{{a}}/{a}", "{bad-name}"), ("a",), "test", False)
    broker = B.Broker(
        SimpleNamespace(argv=(), network="unused", clear_environment=True),
        SimpleNamespace(),
        tmp_path,
        (action,),
        tmp_path / "receipts.jsonl",
        deadline=time.monotonic() + 10,
        max_calls=1,
        max_tool_seconds=1,
        workflow=BP.select_workflow(
            BP.CORPUS_FEEDBACK_V1,
            candidate=tmp_path,
            target_experiment=SimpleNamespace(),
            receipt_path=tmp_path / "receipts.jsonl",
        ),
    )
    seen = []

    def dispatch(request, action_name, bindings, raw_argv, *rest):
        seen.append(raw_argv)
        return {"returncode": 0}

    monkeypatch.setattr(broker, "_execute_allocated", dispatch)
    assert broker.execute({"action": "fixture", "bindings": {"a": "bound"}}) == {"returncode": 0}
    assert seen == [["{bound}/bound", "{bad-name}"]]
